#include "wiGPUSortLib.h"
#include "wiRenderer.h"
#include "wiResourceManager.h"
#include "shaders/ShaderInterop_GPUSortLib.h"
#include "wiEventHandler.h"
#include "wiTimer.h"
#include "wiBacklog.h"

using namespace wi::graphics;

namespace wi::gpusortlib
{
	static GPUBuffer indirectBuffer;
	static Shader kickoffSortCS;
	static Shader sortCS;
	static Shader sortInnerCS;
	static Shader sortStepCS;
	// GG: the same with the keys read through a UAV (COMPARISON_RW)
	static Shader sortCS_RW;
	static Shader sortInnerCS_RW;
	static Shader sortStepCS_RW;


	void LoadShaders()
	{
		wi::renderer::LoadShader(ShaderStage::CS, kickoffSortCS, "gpusortlib_kickoffSortCS.cso");
		wi::renderer::LoadShader(ShaderStage::CS, sortCS, "gpusortlib_sortCS.cso");
		wi::renderer::LoadShader(ShaderStage::CS, sortInnerCS, "gpusortlib_sortInnerCS.cso");
		wi::renderer::LoadShader(ShaderStage::CS, sortStepCS, "gpusortlib_sortStepCS.cso");
		wi::renderer::LoadShader(ShaderStage::CS, sortCS_RW, "gpusortlib_sortCS.cso", ShaderModel::SM_6_0, { "COMPARISON_RW" });
		wi::renderer::LoadShader(ShaderStage::CS, sortInnerCS_RW, "gpusortlib_sortInnerCS.cso", ShaderModel::SM_6_0, { "COMPARISON_RW" });
		wi::renderer::LoadShader(ShaderStage::CS, sortStepCS_RW, "gpusortlib_sortStepCS.cso", ShaderModel::SM_6_0, { "COMPARISON_RW" });

	}

	void Initialize()
	{
		wi::Timer timer;

		GPUBufferDesc bd;
		bd.usage = Usage::DEFAULT;
		bd.bind_flags = BindFlag::UNORDERED_ACCESS;
		bd.misc_flags = ResourceMiscFlag::INDIRECT_ARGS | ResourceMiscFlag::BUFFER_RAW;
		bd.size = sizeof(IndirectDispatchArgs);
		wi::graphics::GetDevice()->CreateBuffer(&bd, nullptr, &indirectBuffer);

		static wi::eventhandler::Handle handle = wi::eventhandler::Subscribe(wi::eventhandler::EVENT_RELOAD_SHADERS, [](uint64_t userdata) { LoadShaders(); });
		LoadShaders();

		wilog("wi::gpusortlib Initialized (%d ms)", (int)std::round(timer.elapsed()));
	}


	void SortSmall(
		const GPUBuffer& comparisonBuffer_read,
		const GPUBuffer& counterBuffer_read,
		uint32_t counterReadOffset,
		const GPUBuffer& indexBuffer_write,
		CommandList cmd,
		int comparison_subresource,
		int index_subresource,
		bool comparison_rw)
	{
		GraphicsDevice* device = wi::graphics::GetDevice();
		SortConstants sort;
		sort.counterReadOffset = counterReadOffset;
		device->BindComputeShader(comparison_rw ? &sortCS_RW : &sortCS, cmd);
		device->BindResource(&counterBuffer_read, 0, cmd);
		if (comparison_rw)
			device->BindUAV(&comparisonBuffer_read, 1, cmd, comparison_subresource);
		else
			device->BindResource(&comparisonBuffer_read, 1, cmd, comparison_subresource);
		device->BindUAV(&indexBuffer_write, 0, cmd, index_subresource);
		device->PushConstants(&sort, sizeof(sort), cmd);
		device->Dispatch(1, 1, 1, cmd);
	}

	void Sort(
		uint32_t maxCount, 
		const GPUBuffer& comparisonBuffer_read, 
		const GPUBuffer& counterBuffer_read, 
		uint32_t counterReadOffset, 
		const GPUBuffer& indexBuffer_write,
		CommandList cmd,
		int comparison_subresource,
		int index_subresource,
		bool comparison_rw)
	{
		GraphicsDevice* device = wi::graphics::GetDevice();

		device->EventBegin("GPUSortLib", cmd);


		SortConstants sort;
		sort.counterReadOffset = counterReadOffset;

		// initialize sorting arguments:
		{
			device->BindComputeShader(&kickoffSortCS, cmd);

			const GPUResource* res[] = {
				&counterBuffer_read,
			};
			device->BindResources(res, 0, arraysize(res), cmd);

			const GPUResource* uavs[] = {
				&indirectBuffer,
			};
			device->BindUAVs(uavs, 0, arraysize(uavs), cmd);

			{
				GPUBarrier barriers[] = {
					GPUBarrier::Buffer(&indirectBuffer, ResourceState::INDIRECT_ARGUMENT, ResourceState::UNORDERED_ACCESS)
				};
				device->Barrier(barriers, arraysize(barriers), cmd);
			}

			device->PushConstants(&sort, sizeof(sort), cmd);
			device->Dispatch(1, 1, 1, cmd);

			{
				GPUBarrier barriers[] = {
					GPUBarrier::Memory(),
					GPUBarrier::Buffer(&indirectBuffer, ResourceState::UNORDERED_ACCESS, ResourceState::INDIRECT_ARGUMENT)
				};
				device->Barrier(barriers, arraysize(barriers), cmd);
			}

		}


		device->BindUAV(&indexBuffer_write, 0, cmd, index_subresource);
		device->BindResource(&counterBuffer_read, 0, cmd);
		if (comparison_rw)
			device->BindUAV(&comparisonBuffer_read, 1, cmd, comparison_subresource);
		else
			device->BindResource(&comparisonBuffer_read, 1, cmd, comparison_subresource);

		// initial sorting:
		bool bDone = true;
		{
			// calculate how many threads we'll require:
			//   we'll sort 512 elements per CU (threadgroupsize 256)
			//     maybe need to optimize this or make it changeable during init
			//     TGS=256 is a good intermediate value

			unsigned int numThreadGroups = ((maxCount - 1) >> 9) + 1;

			//assert(numThreadGroups <= 1024);

			if (numThreadGroups > 1)
			{
				bDone = false;
			}

			// sort all buffers of size 512 (and presort bigger ones)
			device->BindComputeShader(comparison_rw ? &sortCS_RW : &sortCS, cmd);
			device->PushConstants(&sort, sizeof(sort), cmd);
			device->DispatchIndirect(&indirectBuffer, 0, cmd);

			GPUBarrier barriers[] = {
				GPUBarrier::Memory(),
			};
			device->Barrier(barriers, arraysize(barriers), cmd);
		}

		int presorted = 512;
		while (!bDone)
		{
			// Incremental sorting:

			bDone = true;
			device->BindComputeShader(comparison_rw ? &sortStepCS_RW : &sortStepCS, cmd);

			// prepare thread group description data
			uint32_t numThreadGroups = 0;

			if (maxCount > (uint32_t)presorted)
			{
				if (maxCount > (uint32_t)presorted * 2)
					bDone = false;

				uint32_t pow2 = presorted;
				while (pow2 < maxCount)
					pow2 *= 2;
				numThreadGroups = pow2 >> 9;
			}

			uint32_t nMergeSize = presorted * 2;
			for (uint32_t nMergeSubSize = nMergeSize >> 1; nMergeSubSize > 256; nMergeSubSize = nMergeSubSize >> 1)
			{
				SortConstants sort;
				sort.job_params.x = nMergeSubSize;
				if (nMergeSubSize == nMergeSize >> 1)
				{
					sort.job_params.y = (2 * nMergeSubSize - 1);
					sort.job_params.z = -1;
				}
				else
				{
					sort.job_params.y = nMergeSubSize;
					sort.job_params.z = 1;
				}
				sort.counterReadOffset = counterReadOffset;

				device->PushConstants(&sort, sizeof(sort), cmd);
				device->Dispatch(numThreadGroups, 1, 1, cmd);

				GPUBarrier barriers[] = {
					GPUBarrier::Memory(),
				};
				device->Barrier(barriers, arraysize(barriers), cmd);
			}

			device->BindComputeShader(comparison_rw ? &sortInnerCS_RW : &sortInnerCS, cmd);
			device->PushConstants(&sort, sizeof(sort), cmd);
			device->Dispatch(numThreadGroups, 1, 1, cmd);

			GPUBarrier barriers[] = {
				GPUBarrier::Memory(),
			};
			device->Barrier(barriers, arraysize(barriers), cmd);

			presorted *= 2;
		}



		device->EventEnd(cmd);
	}

}
