#include "wiEmittedParticle.h"
#include "wiScene.h"
#include "wiRenderer.h"
#include "wiResourceManager.h"
#include "wiPrimitive.h"
#include "wiRandom.h"
#include "wiArchive.h"
#include "wiTextureHelper.h"
#include "wiGPUSortLib.h"
#include "wiProfiler.h"
#include "wiBacklog.h"
#include "wiEventHandler.h"
#include "wiTimer.h"
#include "wiVector.h"

#include <algorithm>

using namespace wi::graphics;
using namespace wi::scene;
using namespace wi::enums;

namespace wi
{
	static Shader vertexShader;
	static Shader meshShader;
	static Shader pixelShader[EmittedParticleSystem::PARTICLESHADERTYPE_COUNT];
	static Shader kickoffUpdateCS;
	static Shader finishUpdateCS;
	static Shader emitCS;
	static Shader emitCS_VOLUME;
	static Shader emitCS_FROMMESH;
	static Shader sphpartitionCS;
	static Shader sphcellallocationCS;
	static Shader sphbinningCS;
	static Shader sphdensityCS;
	static Shader sphforceCS;
	static Shader simulateCS;
	static Shader simulateCS_SORTING;
	static Shader simulateCS_DEPTHCOLLISIONS;
	static Shader simulateCS_SORTING_DEPTHCOLLISIONS;

	static BlendState			blendStates[BLENDMODE_COUNT];
	static RasterizerState		rasterizerState;
	static RasterizerState		wireFrameRS;
	static DepthStencilState	depthStencilState;
	static PipelineState		PSO[BLENDMODE_COUNT][EmittedParticleSystem::PARTICLESHADERTYPE_COUNT];
	static PipelineState		PSO_wire;
	// GG: into the half resolution target (wi::renderer::DrawSoftParticles_Half), no depth test (the depth fade hides it behind the
	// scene), each blend keeping the colour premultiplied and in alpha how much of the scene shows through
	static PipelineState		PSO_half[BLENDMODE_COUNT][EmittedParticleSystem::PARTICLESHADERTYPE_COUNT];
	static BlendState			halfBlendStates[BLENDMODE_COUNT];
	static DepthStencilState	depthStencilStateNone;

	static bool ALLOW_MESH_SHADER = false;


	void EmittedParticleSystem::SetMaxParticleCount(uint32_t value)
	{
		MAX_PARTICLES = value;
		counterBuffer = {}; // will be recreated
	}

	void EmittedParticleSystem::CreateSelfBuffers()
	{
		GraphicsDevice* device = wi::graphics::GetDevice();

		if (particleBuffer.desc.size < MAX_PARTICLES * sizeof(Particle))
		{
			BLAS = {};

			GPUBufferDesc bd;
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			bd.misc_flags = ResourceMiscFlag::BUFFER_STRUCTURED;

			// Particle buffer:
			bd.stride = sizeof(Particle);
			bd.size = bd.stride * MAX_PARTICLES;
			device->CreateBuffer(&bd, nullptr, &particleBuffer);
			device->SetName(&particleBuffer, "EmittedParticleSystem::particleBuffer");

			// Alive index lists (double buffered):
			bd.stride = sizeof(uint32_t);
			bd.size = bd.stride * MAX_PARTICLES;
			device->CreateBuffer(&bd, nullptr, &aliveList[0]);
			device->SetName(&aliveList[0], "EmittedParticleSystem::aliveList[0]");
			device->CreateBuffer(&bd, nullptr, &aliveList[1]);
			device->SetName(&aliveList[1], "EmittedParticleSystem::aliveList[1]");
			device->CreateBuffer(&bd, nullptr, &culledIndirectionBuffer);
			device->SetName(&culledIndirectionBuffer, "EmittedParticleSystem::culledIndirectionBuffer");
			device->CreateBuffer(&bd, nullptr, &culledIndirectionBuffer2);
			device->SetName(&culledIndirectionBuffer2, "EmittedParticleSystem::culledIndirectionBuffer2");

			// Dead index list:
			auto fill_dead_indices = [&](void* dest) {
				uint32_t* indices = (uint32_t*)dest;
				for (uint32_t i = 0; i < MAX_PARTICLES; ++i)
				{
					std::memcpy(indices + i, &i, sizeof(uint32_t));
				}
			};
			device->CreateBuffer2(&bd, fill_dead_indices, &deadList);
			device->SetName(&deadList, "EmittedParticleSystem::deadList");

			bd.misc_flags = ResourceMiscFlag::BUFFER_RAW | ResourceMiscFlag::TYPED_FORMAT_CASTING | ResourceMiscFlag::NO_DEFAULT_DESCRIPTORS;
			if (device->CheckCapability(GraphicsDeviceCapability::RAYTRACING))
			{
				bd.misc_flags |= ResourceMiscFlag::RAY_TRACING;
			}
			const uint64_t alignment = device->GetMinOffsetAlignment(&bd);

			vb_pos.size = sizeof(MeshComponent::Vertex_POS32W) * 4 * MAX_PARTICLES;
			vb_nor.size = sizeof(MeshComponent::Vertex_NOR) * 4 * MAX_PARTICLES;
			vb_uvs.size = sizeof(MeshComponent::Vertex_UVS) * 4 * MAX_PARTICLES;
			vb_col.size = sizeof(MeshComponent::Vertex_COL) * 4 * MAX_PARTICLES;

			bd.size =
				AlignTo(vb_pos.size, alignment) +
				AlignTo(vb_nor.size, alignment) +
				AlignTo(vb_uvs.size, alignment) +
				AlignTo(vb_col.size, alignment)
			;

			device->CreateBuffer(&bd, nullptr, &generalBuffer);
			device->SetName(&generalBuffer, "EmittedParticleSystem::generalBuffer");

			uint64_t buffer_offset = 0ull;

			buffer_offset = AlignTo(buffer_offset, alignment);
			vb_pos.offset = buffer_offset;
			vb_pos.subresource_srv = device->CreateSubresource(&generalBuffer, SubresourceType::SRV, vb_pos.offset, vb_pos.size, &MeshComponent::Vertex_POS32W::FORMAT);
			vb_pos.subresource_uav = device->CreateSubresource(&generalBuffer, SubresourceType::UAV, vb_pos.offset, vb_pos.size, &MeshComponent::Vertex_POS32W::FORMAT); // UAV can't have RGB32_F format!
			vb_pos.descriptor_srv = device->GetDescriptorIndex(&generalBuffer, SubresourceType::SRV, vb_pos.subresource_srv);
			vb_pos.descriptor_uav = device->GetDescriptorIndex(&generalBuffer, SubresourceType::UAV, vb_pos.subresource_uav);
			buffer_offset += vb_pos.size;

			buffer_offset = AlignTo(buffer_offset, alignment);
			vb_nor.offset = buffer_offset;
			vb_nor.subresource_srv = device->CreateSubresource(&generalBuffer, SubresourceType::SRV, vb_nor.offset, vb_nor.size, &MeshComponent::Vertex_NOR::FORMAT);
			vb_nor.subresource_uav = device->CreateSubresource(&generalBuffer, SubresourceType::UAV, vb_nor.offset, vb_nor.size, &MeshComponent::Vertex_NOR::FORMAT);
			vb_nor.descriptor_srv = device->GetDescriptorIndex(&generalBuffer, SubresourceType::SRV, vb_nor.subresource_srv);
			vb_nor.descriptor_uav = device->GetDescriptorIndex(&generalBuffer, SubresourceType::UAV, vb_nor.subresource_uav);
			buffer_offset += vb_nor.size;

			buffer_offset = AlignTo(buffer_offset, alignment);
			vb_uvs.offset = buffer_offset;
			vb_uvs.subresource_srv = device->CreateSubresource(&generalBuffer, SubresourceType::SRV, vb_uvs.offset, vb_uvs.size, &MeshComponent::Vertex_UVS::FORMAT);
			vb_uvs.subresource_uav = device->CreateSubresource(&generalBuffer, SubresourceType::UAV, vb_uvs.offset, vb_uvs.size, &MeshComponent::Vertex_UVS::FORMAT);
			vb_uvs.descriptor_srv = device->GetDescriptorIndex(&generalBuffer, SubresourceType::SRV, vb_uvs.subresource_srv);
			vb_uvs.descriptor_uav = device->GetDescriptorIndex(&generalBuffer, SubresourceType::UAV, vb_uvs.subresource_uav);
			buffer_offset += vb_uvs.size;

			buffer_offset = AlignTo(buffer_offset, alignment);
			vb_col.offset = buffer_offset;
			vb_col.subresource_srv = device->CreateSubresource(&generalBuffer, SubresourceType::SRV, vb_col.offset, vb_col.size, &MeshComponent::Vertex_COL::FORMAT);
			vb_col.subresource_uav = device->CreateSubresource(&generalBuffer, SubresourceType::UAV, vb_col.offset, vb_col.size, &MeshComponent::Vertex_COL::FORMAT);
			vb_col.descriptor_srv = device->GetDescriptorIndex(&generalBuffer, SubresourceType::SRV, vb_col.subresource_srv);
			vb_col.descriptor_uav = device->GetDescriptorIndex(&generalBuffer, SubresourceType::UAV, vb_col.subresource_uav);
			buffer_offset += vb_col.size;

			primitiveBuffer = wi::renderer::GetIndexBufferForQuads(MAX_PARTICLES);
		}

		if (IsSorted() && distanceBuffer.desc.size < MAX_PARTICLES * sizeof(float))
		{
			// Distance buffer:
			GPUBufferDesc bd;
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			bd.misc_flags = ResourceMiscFlag::BUFFER_STRUCTURED;
			bd.stride = sizeof(float);
			bd.size = bd.stride * MAX_PARTICLES;
			auto init_distances = [&](void* dest) {
				float* distances = (float*)dest;
				std::fill(distances, distances + MAX_PARTICLES, 0.0f);
			};
			device->CreateBuffer2(&bd, init_distances, &distanceBuffer);
			device->SetName(&distanceBuffer, "EmittedParticleSystem::distanceBuffer");
		}

		if (IsSPHEnabled())
		{
			GPUBufferDesc bd;
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			bd.misc_flags = ResourceMiscFlag::BUFFER_STRUCTURED;

			if (densityBuffer.desc.size < MAX_PARTICLES * sizeof(float))
			{
				// Density buffer (for SPH simulation):
				bd.stride = sizeof(float);
				bd.size = bd.stride * MAX_PARTICLES;
				device->CreateBuffer(&bd, nullptr, &densityBuffer);
				device->SetName(&densityBuffer, "EmittedParticleSystem::densityBuffer");
			}

			if (sphGridCells.desc.size < SPH_PARTITION_BUCKET_COUNT * sizeof(SPHGridCell))
			{
				// SPH Partitioning grid cell offsets,counts into particle index list:
				bd.stride = sizeof(SPHGridCell);
				bd.size = bd.stride * SPH_PARTITION_BUCKET_COUNT;
				device->CreateBuffer(&bd, nullptr, &sphGridCells);
				device->SetName(&sphGridCells, "EmittedParticleSystem::sphGridCells");
			}

			if (sphParticleCells.desc.size < MAX_PARTICLES * sizeof(uint))
			{
				// Compacted particle lists per grid cell  (for SPH simulation):
				bd.stride = sizeof(float);
				bd.size = bd.stride * MAX_PARTICLES;
				device->CreateBuffer(&bd, nullptr, &sphParticleCells);
				device->SetName(&sphParticleCells, "EmittedParticleSystem::sphParticleCells");
			}
		}
		else
		{
			densityBuffer = {};
			sphGridCells = {};
			sphParticleCells = {};
		}

		if (!counterBuffer.IsValid())
		{
			// Particle System statistics:
			ParticleCounters counters;
			counters.aliveCount = 0;
			counters.deadCount = MAX_PARTICLES;
			counters.realEmitCount = 0;
			counters.aliveCount_afterSimulation = 0;
			counters.cellAllocator = 0;

			GPUBufferDesc bd;
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			bd.size = sizeof(counters);
			bd.misc_flags = ResourceMiscFlag::BUFFER_RAW;
			device->CreateBuffer(&bd, &counters, &counterBuffer);
			device->SetName(&counterBuffer, "EmittedParticleSystem::counterBuffer");
		}

		if (!indirectBuffers.IsValid())
		{
			GPUBufferDesc bd;

			// Indirect Execution buffer:
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::UNORDERED_ACCESS;
			bd.misc_flags = ResourceMiscFlag::BUFFER_RAW | ResourceMiscFlag::INDIRECT_ARGS;
			bd.size =
				AlignTo(sizeof(IndirectDispatchArgs), (uint64_t)IndirectDispatchArgsAlignment) +
				AlignTo(sizeof(IndirectDispatchArgs), (uint64_t)IndirectDispatchArgsAlignment) +
				AlignTo(sizeof(IndirectDrawArgsInstanced), (uint64_t)IndirectDrawArgsAlignment);
			device->CreateBufferZeroed(&bd, &indirectBuffers);
			device->SetName(&indirectBuffers, "EmittedParticleSystem::indirectBuffers");

			// Constant buffer:
			bd.usage = Usage::DEFAULT;
			bd.size = sizeof(EmittedParticleCB);
			bd.bind_flags = BindFlag::CONSTANT_BUFFER;
			bd.misc_flags = ResourceMiscFlag::NONE;
			device->CreateBuffer(&bd, nullptr, &constantBuffer);
			device->SetName(&constantBuffer, "EmittedParticleSystem::constantBuffer");

			// Debug information CPU-readback buffer:
			{
				GPUBufferDesc debugBufDesc = counterBuffer.GetDesc();
				debugBufDesc.usage = Usage::READBACK;
				debugBufDesc.bind_flags = BindFlag::NONE;
				debugBufDesc.misc_flags = ResourceMiscFlag::NONE;
				for (int i = 0; i < arraysize(statisticsReadbackBuffer); ++i)
				{
					device->CreateBuffer(&debugBufDesc, nullptr, &statisticsReadbackBuffer[i]);
					device->SetName(&statisticsReadbackBuffer[i], "EmittedParticleSystem::statisticsReadbackBuffer");
				}
			}
		}
	}

	void EmittedParticleSystem::CreateRaytracingRenderData()
	{
		GraphicsDevice* device = wi::graphics::GetDevice();

		if (device->CheckCapability(GraphicsDeviceCapability::RAYTRACING) && !BLAS.IsValid() && primitiveBuffer.IsValid())
		{
			RaytracingAccelerationStructureDesc desc;
			desc.type = RaytracingAccelerationStructureDesc::Type::BOTTOMLEVEL;
			desc.flags |= RaytracingAccelerationStructureDesc::FLAG_ALLOW_UPDATE;
			desc.flags |= RaytracingAccelerationStructureDesc::FLAG_PREFER_FAST_BUILD;

			desc.bottom_level.geometries.emplace_back();
			auto& geometry = desc.bottom_level.geometries.back();
			geometry.type = RaytracingAccelerationStructureDesc::BottomLevel::Geometry::Type::TRIANGLES;
			geometry.triangles.vertex_buffer = generalBuffer;
			geometry.triangles.index_buffer = primitiveBuffer;
			geometry.triangles.index_format = GetIndexBufferFormat(primitiveBuffer.desc.format);
			geometry.triangles.index_count = MAX_PARTICLES * 6;
			geometry.triangles.index_offset = 0;
			geometry.triangles.vertex_count = (uint32_t)(vb_pos.size / sizeof(MeshComponent::Vertex_POS32W));
			geometry.triangles.vertex_format = Format::R32G32B32_FLOAT;
			geometry.triangles.vertex_stride = sizeof(MeshComponent::Vertex_POS32W);

			bool success = device->CreateRaytracingAccelerationStructure(&desc, &BLAS);
			assert(success);
			device->SetName(&BLAS, "EmittedParticleSystem::BLAS");
		}
	}

	uint64_t EmittedParticleSystem::GetMemorySizeInBytes() const
	{
		if (!particleBuffer.IsValid())
			return 0;

		uint64_t retVal = 0;

		retVal += particleBuffer.GetDesc().size;
		retVal += aliveList[0].GetDesc().size;
		retVal += aliveList[1].GetDesc().size;
		retVal += deadList.GetDesc().size;
		retVal += distanceBuffer.GetDesc().size;
		retVal += sphGridCells.GetDesc().size;
		retVal += sphParticleCells.GetDesc().size;
		retVal += densityBuffer.GetDesc().size;
		retVal += counterBuffer.GetDesc().size;
		retVal += indirectBuffers.GetDesc().size;
		retVal += constantBuffer.GetDesc().size;
		retVal += generalBuffer.GetDesc().size;
		retVal += culledIndirectionBuffer.GetDesc().size;
		retVal += culledIndirectionBuffer2.GetDesc().size;
		retVal += ComputeTextureMemorySizeInBytes(opacityCurveTex.GetDesc());

		return retVal;
	}

	void EmittedParticleSystem::UpdateCPU(const TransformComponent& transform, float dt)
	{
		this->dt = dt;

		// GG: a paused emitter makes its buffers when it is first resumed. A pooled effect is loaded and parked paused, and
		// every one made its buffers in the frame after the loads
		if (IsPaused())
			return;

		CreateSelfBuffers();

		if (dt == 0)
			return;

		active_frames <<= 1; // advance next frame

		emit = std::max(0.0f, emit - std::floor(emit));

		center = transform.GetPosition();

		// GGMAX 2.00: GameGuru WPE emission model, ported verbatim from the DX11 fork
		// (WickedRepo/WickedEngine/wiEmittedParticle.cpp UpdateCPU). Two behaviours upstream
		// does not have:
		//  1. spawn_random - a gust/stutter model. Emission pauses for a randomised number of
		//     frames and then decays back in, which is what stops rain and fire looking like a
		//     metronome. With spawn_random == 0 randpause is always 0, so this reduces exactly
		//     to upstream's 'emit += count * dt'.
		//  2. burst_delay (MILLISECONDS) + burst_split - a burst is released in fractions over
		//     several frames rather than all in one, which is what makes an explosion read as a
		//     sequence rather than a single pop.
		// Do NOT "simplify" these: the shipped effects were tuned against these exact curves.
		if (randpause <= 0.01f) // GG: counted in 60ths of a second below, not frames
		{
			randpause = (float)(uint32_t)((wi::random::GetRandom(0, 1000) * 0.001f) * spawn_random);
			if (randpause < (spawn_random * 0.6f))
				randpause = 0;
			else
				randpause -= (float)(uint32_t)(spawn_random * 0.5f);
			if (randpause > spawn_random)
				randpause = 0;

			emit += (float)count * count_scale * dt; // GG: the layer's count scale
			randemit = (float)count * count_scale;
		}
		else
		{
			emit += randemit * dt;
			// GG: the decay and the pause per 60th of a second, so the stutter keeps its timing at any frame rate
			randemit *= std::pow(0.05f, dt * 60.0f);
			randpause -= dt * 60.0f;
		}

		if (emit < 0)
			emit = 0;

		if (burst_delay_timer > 0)
		{
			burst_delay_timer -= 1000.0f * dt;
			if (burst_delay_timer < 0) burst_delay_timer = 0;
		}
		else
		{
			if (burst_split > 0)
			{
				// GG: a burst_split'th of what is left each 60th of a second whatever the frame rate, the count taken down by whole
				// particles, rounded up, so it always runs out (truncating the share, as before, left a burst smaller than the split
				// stuck, a fraction of it emitted every frame for as long as the emitter lived); a split of 1 or less releases it at once
				if (burst_split <= 1.0f)
				{
					emit += (float)burst / burst_split;
					burst = 0;
				}
				else if (dt > 0)
				{
					const float share = 1.0f - std::pow(1.0f - 1.0f / burst_split, dt * 60.0f);
					const float amount = (float)burst * share;
					emit += amount;
					burst -= std::max(1, (int)std::ceil(amount));
					if (burst < 0) burst = 0;
				}
			}
			else
			{
				emit += (float)burst;
				burst = 0;
			}
		}

		if (IsEmitPaused())
		{
			// let the live particles time out rather than killing them
			emit = 0;
			burst = 0;
		}

		total_emit_count += (uint32_t)emit;

		if ((uint)emit > 0)
		{
			EmitLocation& location = emit_locations.emplace_back();
			location.transform.init();
			location.count = (uint)emit;
			location.color = wi::Color::White();
			active_frames |= 1; // activate current frame
		}

		worldMatrix = transform.world;

		// Swap CURRENT alivelist with NEW alivelist
		std::swap(aliveList[0], aliveList[1]);

		// Read back statistics (with GPU delay):
		const uint32_t oldest_stat_index = wi::graphics::GetDevice()->GetBufferIndex();
		memcpy(&statistics, statisticsReadbackBuffer[oldest_stat_index].mapped_data, sizeof(statistics));

		if (statistics.aliveCount > 0 || statistics.aliveCount_afterSimulation > 0)
		{
			active_frames |= 1; // activate current frame
		}

		// GGMAX 2.00: bActive auto-deactivate (DX11 fork behaviour). Once nothing has been
		// emitted and nothing is alive for twice the worst-case particle lifetime, park the
		// emitter so it costs neither a simulate dispatch nor a draw.
		if (active_frames & 1)
		{
			inactive_timer = 0.0f;
			bActive = true;
		}
		else if (bActive)
		{
			inactive_timer += dt;
			const float worst_case_life = life * life_scale * (1.0f + std::abs(random_life) * 0.5f); // GG: the layer's life scale lengthens the wait
			if (inactive_timer > worst_case_life * 2.0f + 0.1f)
			{
				bActive = false;
			}
		}

		if (!opacityCurveTex.IsValid())
		{
			SetOpacityCurveControl(opacityCurveControlPeakStart, opacityCurveControlPeakEnd);
		}
	}
	void EmittedParticleSystem::Burst(int num)
	{
		if (IsPaused())
			return;

		// GGMAX 2.00: the DX11 GameGuru fork treated num <= 0 as "fire the emitter's own
		// configured burst_amount" and (re)armed the staged-release timer. Callers rely on
		// it: the editor preview, the LUA WParticleEffectAction(1) and the weapon/decal
		// paths all call Burst(0). With upstream's plain 'burst += num' that is a no-op, so
		// every burst-only effect (explosions, blood, impacts - count == 0) fired nothing.
		if (num <= 0)
		{
			num = (int)burst_amount;
			burst_delay_timer = burst_delay;
		}

		burst += (int)((float)num * count_scale + 0.5f); // GG: the layer's count scale

		if (num > 0)
		{
			active_frames |= 1; // activate current frame
		}
	}
	void EmittedParticleSystem::Burst(int num, const XMFLOAT3& position, const wi::Color& color)
	{
		if (IsPaused() || num <= 0)
			return;

		XMFLOAT4X4 transform;
		XMStoreFloat4x4(&transform, XMMatrixTranslation(position.x, position.y, position.z));

		EmitLocation& location = emit_locations.emplace_back();
		location.count = (uint)num;
		location.transform.Create(transform);
		location.color = color;

		if (num > 0)
		{
			active_frames |= 1; // activate current frame
		}
	}
	void EmittedParticleSystem::Burst(int num, const XMFLOAT4X4& transform, const wi::Color& color)
	{
		if (IsPaused() || num <= 0)
			return;

		EmitLocation& location = emit_locations.emplace_back();
		location.count = (uint)num;
		location.transform.Create(transform);
		location.color = color;

		if (num > 0)
		{
			active_frames |= 1; // activate current frame
		}
	}
	void EmittedParticleSystem::Restart()
	{
		SetPaused(false);
		// GG: the particles are cleared in the next UpdateGPU and the buffers kept: making the counters again left the dead list as
		// it was, with the live particles' indices missing from it, and a pooled effect's every reuse paid for it
		if (counterBuffer.IsValid()) restartPending = true;
	}

	void EmittedParticleSystem::UpdateGPU(uint32_t instanceIndex, const MeshComponent* mesh, CommandList cmd) const
	{
		if (IsInactive())
			return;
		if (!particleBuffer.IsValid())
			return;

		GraphicsDevice* device = wi::graphics::GetDevice();
		if (restartPending)
		{
			// GG: every particle dead, as in new buffers: the counters, and every index on the dead list again (before this frame's emit)
			restartPending = false;
			ParticleCounters counters;
			counters.aliveCount = 0;
			counters.deadCount = MAX_PARTICLES;
			counters.realEmitCount = 0;
			counters.aliveCount_afterSimulation = 0;
			counters.cellAllocator = 0;
			device->UpdateBuffer(&counterBuffer, &counters, cmd);
			static thread_local wi::vector<uint32_t> indices;
			indices.resize(MAX_PARTICLES);
			for (uint32_t i = 0; i < MAX_PARTICLES; ++i) indices[i] = i;
			device->UpdateBuffer(&deadList, indices.data(), cmd, sizeof(uint32_t) * MAX_PARTICLES);
		}
		device->EventBegin("UpdateEmittedParticles", cmd);

		if (!IsPaused() && dt > 0)
		{
			auto alloc = device->AllocateGPU(sizeof(EmitLocation) * emit_locations.size(), cmd);
			const XMMATRIX W = XMMatrixScaling(spread_scale, spread_scale, spread_scale) * XMLoadFloat4x4(&worldMatrix); // GG: the layer's spread widens the volume (or mesh) it emits from, as on DX11
			for (size_t i = 0; i < emit_locations.size(); ++i)
			{
				EmitLocation& location = emit_locations[i];
				XMFLOAT4X4 mat = location.transform.GetMatrix();
				XMMATRIX M = XMMatrixTranspose(XMLoadFloat4x4(&mat));
				M = W * M;
				XMStoreFloat4x4(&mat, M);
				location.transform.Create(mat);
				std::memcpy((EmitLocation*)alloc.data + i, &location, sizeof(location));
			}
			device->BindResource(&alloc.buffer, 0, cmd);

			EmittedParticleCB cb;
			cb.xEmitBufferOffset = (uint)alloc.offset;
			if (mesh == nullptr || !IsFormatUnorm(mesh->position_format) || mesh->so_pos.IsValid())
			{
				cb.xEmitterBaseMeshUnormRemap.init();
			}
			else
			{
				XMFLOAT4X4 unormRemap;
				XMStoreFloat4x4(&unormRemap, mesh->aabb.getUnormRemapMatrix());
				cb.xEmitterBaseMeshUnormRemap.Create(unormRemap);
			}
			if (mesh == nullptr)
			{
				cb.xEmitterMeshGeometryOffset = 0;
				cb.xEmitterMeshGeometryCount = 0;
			}
			else
			{
				uint32_t first_subset = 0;
				uint32_t last_subset = 0;
				mesh->GetLODSubsetRange(0, first_subset, last_subset);
				cb.xEmitterMeshGeometryOffset = mesh->geometryOffset + first_subset;
				cb.xEmitterMeshGeometryCount = last_subset - first_subset;
			}
			cb.xEmitterRandomness = wi::random::GetRandom(0.0f, 1.0f);
			cb.xParticleLifeSpan = life * life_scale; // GG: the layer's life scale
			cb.xParticleLifeSpanRandomness = random_life;
			cb.xParticleNormalFactor = normal_factor * speed_scale; // GG: the layer's speed scale
			cb.xParticleRandomFactor = random_factor;
			cb.xParticleScaling = scaleX;
			cb.xParticleSize = size * size_scale * layer_size_scale; // GG: the script's and the layer's size scales
			cb.xParticleMotionBlurAmount = motionBlurAmount;
			cb.xParticleRotation = rotation * XM_PI;
			cb.xParticleMass = mass;
			cb.xEmitterMaxParticleCount = MAX_PARTICLES;
			cb.xEmitterFixedTimestep = FIXED_TIMESTEP;
			cb.xEmitterRestitution = restitution;
			cb.xEmitterFramesXY = uint2(std::max(1u, framesX), std::max(1u, framesY));
			cb.xEmitterFrameCount = std::max(1u, frameCount);
			cb.xEmitterFrameStart = frameStart;
			cb.xEmitterTexMul = float2(1.0f / (float)cb.xEmitterFramesXY.x, 1.0f / (float)cb.xEmitterFramesXY.y);
			cb.xEmitterFrameRate = frameRate;
			cb.xParticleGravity = gravity;
			cb.xParticleDrag = drag;
			XMStoreFloat3(&cb.xParticleVelocity, XMVector3TransformNormal(XMLoadFloat3(&velocity), XMLoadFloat4x4(&worldMatrix)) * speed_scale); // GG: the layer's speed scale
			cb.xParticleRandomColorFactor = random_color;
			cb.xEmitterLayerMask = layerMask;
			cb.xEmitterInstanceIndex = instanceIndex;

			// GGMAX 2.00: GameGuru WPE parameters (see PARTICLE_SYSTEM_PLAN.md section 6).
			cb.xParticleEndColor = XMFLOAT3(endcolor_red, endcolor_green, endcolor_blue);
			cb.xEmitterFadeinTime = fadein_time;
			cb.xParticleNormalFactorXYZ = XMFLOAT3(normal_factor_x * speed_scale, normal_factor_y * speed_scale, normal_factor_z * speed_scale);
			cb.xParticleBurstFactorSpeed = burst_factor_speed;
			cb.xParticleBurstFactor = XMFLOAT3(burst_factor_x * speed_scale, burst_factor_y * speed_scale, burst_factor_z * speed_scale);
			cb.xParticleNormalRandom = normal_random;
			cb.xParticleSinPos = XMFLOAT3(startpos.x * spread_scale, startpos.y * spread_scale, startpos.z * spread_scale); // GG: the layer's spread scale
			cb.xParticleRotationRandom = rotation_random;
			cb.xParticleSizeRandom = size_random;
			cb.xParticleScalingRandom = scaling_random;
			cb.xParticleStartRotation = start_rotation;
			cb.xParticleRandomPos = random_position;
			cb.xParticleRandomPosScale = random_position_scale * spread_scale; // GG: the layer's spread scale, as on DX11
			cb.xTotalEmitCount = total_emit_count;
			// GG: the layer's options (wpe_filler_3: bit 1 the spreads turn with the emitter, 2 the trail, 4 no fog), 8 and 16 the blend's
			// fog form, 32 half resolution, 64 a wrapping volume
			cb.xParticleLocalOptions = (uint32_t)std::max(0.0f, wpe_filler_3) & 7; // masked, as on DX11: a clamp turned 8 and up into 7 (no fog, a trail)
			if (gg_blendmode == (uint32_t)wi::enums::BLENDMODE_ADDITIVE) cb.xParticleLocalOptions |= 8;
			else if (gg_blendmode == (uint32_t)wi::enums::BLENDMODE_PREMULTIPLIED) cb.xParticleLocalOptions |= 16;
			if (wi::renderer::GetParticleHalfResolutionEnabled() && IsHalfResolution()) cb.xParticleLocalOptions |= 32;
			if (IsWrapVolume()) cb.xParticleLocalOptions |= 64;
			cb.xParticleFadeOutStart = wi::math::Clamp(wpe_filler_1, 0.0f, 0.99f);
			cb.xParticleMirror = (uint32_t)wi::math::Clamp(wpe_filler_2, 0.0f, 3.0f);
			cb.xParticleWind = layer_wind;
			cb.xParticleEmissiveScale = emissive_scale;
			{
				// where the emitter was last frame, if last frame it was (a load or a teleport is no trail)
				const XMFLOAT3 emitterPos = XMFLOAT3(worldMatrix._41, worldMatrix._42, worldMatrix._43);
				const uint64_t frame = device->GetFrameCount();
				const bool bPrevValid = prev_emitter_frame + 1 == frame && wi::math::Distance(prev_emitter_pos, emitterPos) < 4000.0f;
				cb.xEmitterPrevPos = bPrevValid ? prev_emitter_pos : emitterPos;
				cb.xEmitterPrevValid = bPrevValid ? 1.0f : 0.0f;
				prev_emitter_pos = emitterPos;
				prev_emitter_frame = frame;
			}
			cb.xEmitterWrapMotion = wrap_motion;
			cb.xEmitterWrapMotionSet = bWrapMotion ? 1.0f : 0.0f;
			cb.xParticleColorTint = color_tint;
			cb.xParticleOpacityScale = opacity_scale;
			cb.xEmitterKillBoxCount = std::min(killbox_count, (uint32_t)EMITTER_KILLBOX_COUNT);
			for (uint32_t i = 0; i < cb.xEmitterKillBoxCount; ++i)
			{
				cb.xEmitterKillBoxMin[i] = killbox_min[i];
				cb.xEmitterKillBoxMax[i] = killbox_max[i];
			}
			cb.xParticleSpreadScale = spread_scale; // GG: the wrapping volume's box widened as the emit volume
			cb.xParticlePadding3 = 0;

			cb.xEmitterOptions = 0;
			if (IsSPHEnabled())
			{
				cb.xEmitterOptions |= EMITTER_OPTION_BIT_SPH_ENABLED;
			}
			if (IsFrameBlendingEnabled())
			{
				cb.xEmitterOptions |= EMITTER_OPTION_BIT_FRAME_BLENDING_ENABLED;
			}
			if (ALLOW_MESH_SHADER && device->CheckCapability(GraphicsDeviceCapability::MESH_SHADER))
			{
				cb.xEmitterOptions |= EMITTER_OPTION_BIT_MESH_SHADER_ENABLED;
			}
			if (IsCollidersDisabled())
			{
				cb.xEmitterOptions |= EMITTER_OPTION_BIT_COLLIDERS_DISABLED;
			}
			if (_flags & FLAG_USE_RAIN_BLOCKER)
			{
				cb.xEmitterOptions |= EMITTER_OPTION_BIT_USE_RAIN_BLOCKER;
			}
			if (IsTakeColorFromMesh())
			{
				cb.xEmitterOptions |= EMITTER_OPTION_BIT_TAKE_COLOR_FROM_MESH;
			}

			// SPH:
			cb.xSPH_h = SPH_h;
			cb.xSPH_h_rcp = 1.0f / SPH_h;
			cb.xSPH_h2 = SPH_h * SPH_h;
			cb.xSPH_h3 = cb.xSPH_h2 * SPH_h;
			const float h6 = cb.xSPH_h2 * cb.xSPH_h2 * cb.xSPH_h2;
			const float h9 = h6 * cb.xSPH_h3;
			cb.xSPH_poly6_constant = (315.0f / (64.0f * XM_PI * h9));
			cb.xSPH_spiky_constant = (-45.0f / (XM_PI * h6));
			cb.xSPH_visc_constant = (45.0f / (XM_PI * h6));
			cb.xSPH_K = SPH_K;
			cb.xSPH_p0 = SPH_p0;
			cb.xSPH_e = SPH_e;

			device->UpdateBuffer(&constantBuffer, &cb, cmd);
			{
				GPUBarrier barriers[] = {
					GPUBarrier::Buffer(&constantBuffer, ResourceState::COPY_DST, ResourceState::CONSTANT_BUFFER),
				};
				device->Barrier(barriers, arraysize(barriers), cmd);
			}

			device->BindConstantBuffer(&constantBuffer, CB_GETBINDSLOT(EmittedParticleCB), cmd);

			device->BindUAV(&particleBuffer, 0, cmd);
			device->BindUAV(&aliveList[0], 1, cmd);
			device->BindUAV(&aliveList[1], 2, cmd);
			device->BindUAV(&deadList, 3, cmd);
			device->BindUAV(&counterBuffer, 4, cmd);
			device->BindUAV(&indirectBuffers, 5, cmd);
			device->BindUAV(&distanceBuffer, 6, cmd);
			device->BindUAV(&generalBuffer, 7, cmd, vb_pos.subresource_uav);
			device->BindUAV(&generalBuffer, 8, cmd, vb_nor.subresource_uav);
			device->BindUAV(&generalBuffer, 9, cmd, vb_uvs.subresource_uav);
			device->BindUAV(&generalBuffer, 10, cmd, vb_col.subresource_uav);
			device->BindUAV(&culledIndirectionBuffer, 11, cmd);
			device->BindUAV(&culledIndirectionBuffer2, 12, cmd);

			GPUBarrier barrier_indirect_uav = GPUBarrier::Buffer(&indirectBuffers, ResourceState::INDIRECT_ARGUMENT, ResourceState::UNORDERED_ACCESS);
			GPUBarrier barrier_uav_indirect = GPUBarrier::Buffer(&indirectBuffers, ResourceState::UNORDERED_ACCESS, ResourceState::INDIRECT_ARGUMENT);

			device->Barrier(&barrier_indirect_uav, 1, cmd);

			// emit the required amounts
			device->EventBegin("Emit", cmd);
			device->BindComputeShader(mesh == nullptr ? (IsVolumeEnabled() ? &emitCS_VOLUME : &emitCS) : &emitCS_FROMMESH, cmd);
			uint bufferoffset = (uint)alloc.offset;
			for (auto& location : emit_locations)
			{
				device->PushConstants(&bufferoffset, sizeof(bufferoffset), cmd);
				device->Dispatch((location.count + 63u) / 64u, 1, 1, cmd);
				bufferoffset += sizeof(EmitLocation);
			}
			emit_locations.clear();
			device->Barrier(GPUBarrier::Memory(), cmd);
			device->EventEnd(cmd);

			// kick off indirect updating
			device->EventBegin("KickOff Update", cmd);
			device->BindComputeShader(&kickoffUpdateCS, cmd);
			device->Dispatch(1, 1, 1, cmd);
			device->EventEnd(cmd);

			device->Barrier(&barrier_uav_indirect, 1, cmd);

			if (IsSPHEnabled())
			{
				auto range = wi::profiler::BeginRangeGPU("SPH - Simulation", cmd);

				// Smooth Particle Hydrodynamics:
				device->EventBegin("SPH - Simulation", cmd);

#ifdef SPH_USE_ACCELERATION_GRID
				// Reset grid cell offset buffer with invalid offsets (max uint):
				device->EventBegin("Grid - Reset", cmd);
				device->ClearUAV(&sphGridCells, 0, cmd);
				device->Barrier(GPUBarrier::Memory(), cmd);
				device->EventEnd(cmd);

				// Assign particles into partitioning grid:
				device->EventBegin("Particles - Grid Partitioning", cmd);
				device->BindComputeShader(&sphpartitionCS, cmd);
				const GPUResource* res_partition[] = {
					&aliveList[0], // CURRENT alivelist
					&particleBuffer,
				};
				device->BindResources(res_partition, 0, arraysize(res_partition), cmd);
				const GPUResource* uav_partition[] = {
					&counterBuffer,
					&sphGridCells,
				};
				device->BindUAVs(uav_partition, 0, arraysize(uav_partition), cmd);
				device->DispatchIndirect(&indirectBuffers, ARGUMENTBUFFER_OFFSET_DISPATCHSIMULATION, cmd);
				{
					GPUBarrier barriers[] = {
						GPUBarrier::Memory(&sphGridCells),
					};
					device->Barrier(barriers, arraysize(barriers), cmd);
				}
				device->EventEnd(cmd);

				// Each grid cell allocates space for its own particle list:
				device->EventBegin("Grid - Cell Allocation", cmd);
				device->BindComputeShader(&sphcellallocationCS, cmd);
				const GPUResource* res_cellallocation[] = {
					&aliveList[0], // CURRENT alivelist
					&particleBuffer,
				};
				device->BindResources(res_cellallocation, 0, arraysize(res_cellallocation), cmd);
				const GPUResource* uav_cellallocation[] = {
					&counterBuffer,
					&sphGridCells,
				};
				device->BindUAVs(uav_cellallocation, 0, arraysize(uav_cellallocation), cmd);
				device->Dispatch((SPH_PARTITION_BUCKET_COUNT + THREADCOUNT_SIMULATION - 1) / THREADCOUNT_SIMULATION, 1, 1, cmd);
				{
					GPUBarrier barriers[] = {
						GPUBarrier::Memory(&sphGridCells),
					};
					device->Barrier(barriers, arraysize(barriers), cmd);
				}
				device->EventEnd(cmd);

				// Each particle bins itself to a grid cell:
				device->EventBegin("Particles - Binning", cmd);
				device->BindComputeShader(&sphbinningCS, cmd);
				const GPUResource* res_binning[] = {
					&aliveList[0], // CURRENT alivelist
					&particleBuffer,
				};
				device->BindResources(res_binning, 0, arraysize(res_binning), cmd);
				const GPUResource* uav_binning[] = {
					&counterBuffer,
					&sphGridCells,
					&sphParticleCells,
				};
				device->BindUAVs(uav_binning, 0, arraysize(uav_binning), cmd);
				device->DispatchIndirect(&indirectBuffers, ARGUMENTBUFFER_OFFSET_DISPATCHSIMULATION, cmd);
				{
					GPUBarrier barriers[] = {
						GPUBarrier::Buffer(&sphGridCells, ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE_COMPUTE),
						GPUBarrier::Buffer(&sphParticleCells, ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE_COMPUTE),
					};
					device->Barrier(barriers, arraysize(barriers), cmd);
				}
				device->EventEnd(cmd);

#endif // SPH_USE_ACCELERATION_GRID

				// 5.) Compute particle density field:
				device->EventBegin("Density Evaluation", cmd);
				device->BindComputeShader(&sphdensityCS, cmd);
				const GPUResource* res_density[] = {
					&aliveList[0], // CURRENT alivelist
					&particleBuffer,
					&sphParticleCells,
					&sphGridCells,
				};
				device->BindResources(res_density, 0, arraysize(res_density), cmd);
				const GPUResource* uav_density[] = {
					&counterBuffer,
					&densityBuffer
				};
				device->BindUAVs(uav_density, 0, arraysize(uav_density), cmd);
				device->DispatchIndirect(&indirectBuffers, ARGUMENTBUFFER_OFFSET_DISPATCHSIMULATION, cmd);
				device->Barrier(GPUBarrier::Memory(), cmd);
				device->EventEnd(cmd);

				// 6.) Compute particle pressure forces:
				device->EventBegin("Force Evaluation", cmd);
				device->BindComputeShader(&sphforceCS, cmd);
				const GPUResource* res_force[] = {
					&aliveList[0], // CURRENT alivelist
					&densityBuffer,
					&sphParticleCells,
					&sphGridCells,
				};
				device->BindResources(res_force, 0, arraysize(res_force), cmd);
				const GPUResource* uav_force[] = {
					&counterBuffer,
					&particleBuffer,
				};
				device->BindUAVs(uav_force, 0, arraysize(uav_force), cmd);
				device->DispatchIndirect(&indirectBuffers, ARGUMENTBUFFER_OFFSET_DISPATCHSIMULATION, cmd);
				device->Barrier(GPUBarrier::Memory(), cmd);
				device->EventEnd(cmd);


				device->EventEnd(cmd);

				wi::profiler::EndRange(range);
			}

			device->EventBegin("Simulate", cmd);

			device->BindResource(&opacityCurveTex, 0, cmd);

			device->BindUAV(&particleBuffer, 0, cmd);
			device->BindUAV(&aliveList[0], 1, cmd);
			device->BindUAV(&aliveList[1], 2, cmd);
			device->BindUAV(&deadList, 3, cmd);
			device->BindUAV(&counterBuffer, 4, cmd);
			device->BindUAV(&indirectBuffers, 5, cmd);
			device->BindUAV(&distanceBuffer, 6, cmd);
			device->BindUAV(&generalBuffer, 7, cmd, vb_pos.subresource_uav);
			device->BindUAV(&generalBuffer, 8, cmd, vb_nor.subresource_uav);
			device->BindUAV(&generalBuffer, 9, cmd, vb_uvs.subresource_uav);
			device->BindUAV(&generalBuffer, 10, cmd, vb_col.subresource_uav);
			device->BindUAV(&culledIndirectionBuffer, 11, cmd);
			device->BindUAV(&culledIndirectionBuffer2, 12, cmd);

			// update CURRENT alive list, write NEW alive list
			if (IsSorted())
			{
				if (IsDepthCollisionEnabled())
				{
					device->BindComputeShader(&simulateCS_SORTING_DEPTHCOLLISIONS, cmd);
				}
				else
				{
					device->BindComputeShader(&simulateCS_SORTING, cmd);
				}
			}
			else
			{
				if (IsDepthCollisionEnabled())
				{
					device->BindComputeShader(&simulateCS_DEPTHCOLLISIONS, cmd);
				}
				else
				{
					device->BindComputeShader(&simulateCS, cmd);
				}
			}
			device->DispatchIndirect(&indirectBuffers, ARGUMENTBUFFER_OFFSET_DISPATCHSIMULATION, cmd);
			{
				GPUBarrier barriers[] = {
					GPUBarrier::Memory(),
					GPUBarrier::Buffer(&counterBuffer, ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE),
					GPUBarrier::Buffer(&distanceBuffer, ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE_COMPUTE),
				};
				device->Barrier(barriers, arraysize(barriers), cmd);
			}
			device->EventEnd(cmd);

		}

		if (IsSorted())
		{
			wi::gpusortlib::Sort(MAX_PARTICLES, distanceBuffer, counterBuffer, PARTICLECOUNTER_OFFSET_CULLEDCOUNT, culledIndirectionBuffer, cmd);
		}

		if (!IsPaused() && dt > 0)
		{
			// finish updating, update draw argument buffer:
			device->EventBegin("FinishUpdate", cmd);
			device->BindComputeShader(&finishUpdateCS, cmd);

			const GPUResource* res[] = {
				&counterBuffer,
			};
			device->BindResources(res, 0, arraysize(res), cmd);

			const GPUResource* uavs[] = {
				&indirectBuffers,
			};
			device->BindUAVs(uavs, 0, arraysize(uavs), cmd);

			{
				GPUBarrier barriers[] = {
					GPUBarrier::Memory(),
					GPUBarrier::Buffer(&indirectBuffers, ResourceState::INDIRECT_ARGUMENT, ResourceState::UNORDERED_ACCESS),
				};
				device->Barrier(barriers, arraysize(barriers), cmd);
			}

			device->Dispatch(1, 1, 1, cmd);

			device->EventEnd(cmd);

		}

		{
			GPUBarrier barriers[] = {
				GPUBarrier::Memory(),
				GPUBarrier::Buffer(&counterBuffer, ResourceState::SHADER_RESOURCE, ResourceState::COPY_SRC),
			};
			device->Barrier(barriers, arraysize(barriers), cmd);
		}

		// Statistics is copied to readback:
		const uint32_t oldest_stat_index = wi::graphics::GetDevice()->GetBufferIndex();
		device->CopyResource(&statisticsReadbackBuffer[oldest_stat_index], &counterBuffer, cmd);

		{
			const GPUBarrier barriers[] = {
				GPUBarrier::Buffer(&indirectBuffers, ResourceState::UNORDERED_ACCESS, ResourceState::INDIRECT_ARGUMENT),
				GPUBarrier::Buffer(&counterBuffer, ResourceState::COPY_SRC, ResourceState::SHADER_RESOURCE),
				GPUBarrier::Buffer(&particleBuffer, ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE),
				GPUBarrier::Buffer(&aliveList[1], ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE),
				GPUBarrier::Buffer(&generalBuffer, ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE),
				GPUBarrier::Buffer(&culledIndirectionBuffer, ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE_COMPUTE),
				GPUBarrier::Buffer(&culledIndirectionBuffer2, ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE_COMPUTE),
			};
			device->Barrier(barriers, arraysize(barriers), cmd);
		}

		device->EventEnd(cmd);
	}


	void EmittedParticleSystem::Draw(const MaterialComponent& material, CommandList cmd, const PARTICLESHADERTYPE* shadertype_override, bool bHalfResolution) const
	{
		if (IsInactive())
			return;
		if (!particleBuffer.IsValid())
			return; // GG: paused since it was loaded, so no buffers yet (UpdateCPU) and nothing to draw
		GraphicsDevice* device = wi::graphics::GetDevice();
		device->EventBegin("EmittedParticle", cmd);

		if (wi::renderer::IsWireRender())
		{
			device->BindPipelineState(&PSO_wire, cmd);
		}
		else
		{
			const wi::enums::BLENDMODE blendMode = material.GetBlendMode();
			gg_blendmode = (uint32_t)blendMode;
			const PARTICLESHADERTYPE type = shadertype_override == nullptr ? shaderType : *shadertype_override;
			device->BindPipelineState(bHalfResolution ? &PSO_half[blendMode][type] : &PSO[blendMode][type], cmd);

			device->BindShadingRate(material.shadingRate, cmd);
		}

		device->BindConstantBuffer(&constantBuffer, CB_GETBINDSLOT(EmittedParticleCB), cmd);

		const GPUResource* res[] = {
			&counterBuffer,
			&particleBuffer,
			&culledIndirectionBuffer,
			&culledIndirectionBuffer2,
		};
		device->BindResources(res, 0, arraysize(res), cmd);

		if (ALLOW_MESH_SHADER && device->CheckCapability(GraphicsDeviceCapability::MESH_SHADER))
		{
			device->DispatchMeshIndirect(&indirectBuffers, ARGUMENTBUFFER_OFFSET_DRAWPARTICLES, cmd);
		}
		else
		{
			device->DrawInstancedIndirect(&indirectBuffers, ARGUMENTBUFFER_OFFSET_DRAWPARTICLES, cmd);
		}

		device->EventEnd(cmd);
	}

	void EmittedParticleSystem::SetOpacityCurveControl(float peakStart, float peakEnd)
	{
		peakStart = saturate(peakStart);
		peakEnd = saturate(peakEnd);

		peakEnd = std::max(peakStart, peakEnd);

		opacityCurveControlPeakStart = peakStart;
		opacityCurveControlPeakEnd = peakEnd;

		uint16_t data[2048];
		int startup_length = int(peakStart * float(arraysize(data) - 1));
		// Ramp up:
		for (int i = 0; i < startup_length; ++i)
		{
			float t = smoothstep(0.0f, 1.0f, float(i) / (startup_length - 1));
			data[i] = uint16_t(t * 65535);
		}
		int keep_length = int((peakEnd - peakStart) * float(arraysize(data) - 1));
		// Keep value:
		for (int i = 0; i < keep_length; ++i)
		{
			data[i + startup_length] = uint16_t(65535);
		}
		// Ramp down:
		for (int i = 0; i < (arraysize(data) - startup_length - keep_length); ++i)
		{
			float t = smoothstep(1.0f, 0.0f, float(i) / (arraysize(data) - startup_length - keep_length - 1));
			data[i + startup_length + keep_length] = uint16_t(t * 65535);
		}
		TextureDesc desc;
		desc.type = TextureDesc::Type::TEXTURE_1D;
		desc.width = arraysize(data);
		desc.format = Format::R16_UNORM;
		desc.bind_flags = BindFlag::SHADER_RESOURCE;
		desc.swizzle = SwizzleFromString("rrr1");
		SubresourceData initdata;
		initdata.data_ptr = data;
		initdata.row_pitch = sizeof(data);
		GetDevice()->CreateTexture(&desc, &initdata, &opacityCurveTex);
		GetDevice()->SetName(&opacityCurveTex, "EmittedParticleSystem::opacityCurveTex");
	}

	namespace EmittedParticleSystem_Internal
	{
		void LoadShaders()
		{
			wi::renderer::LoadShader(ShaderStage::VS, vertexShader, "emittedparticleVS.cso");

			if (ALLOW_MESH_SHADER && wi::graphics::GetDevice()->CheckCapability(GraphicsDeviceCapability::MESH_SHADER))
			{
				wi::renderer::LoadShader(ShaderStage::MS, meshShader, "emittedparticleMS.cso");
			}

			wi::renderer::LoadShader(ShaderStage::PS, pixelShader[EmittedParticleSystem::SOFT], "emittedparticlePS_soft.cso");
			wi::renderer::LoadShader(ShaderStage::PS, pixelShader[EmittedParticleSystem::SOFT_DISTORTION], "emittedparticlePS_soft_distortion.cso");
			wi::renderer::LoadShader(ShaderStage::PS, pixelShader[EmittedParticleSystem::SIMPLE], "emittedparticlePS_simple.cso");
			wi::renderer::LoadShader(ShaderStage::PS, pixelShader[EmittedParticleSystem::SOFT_LIGHTING], "emittedparticlePS_soft_lighting.cso");

			wi::renderer::LoadShader(ShaderStage::CS, kickoffUpdateCS, "emittedparticle_kickoffUpdateCS.cso");
			wi::renderer::LoadShader(ShaderStage::CS, finishUpdateCS, "emittedparticle_finishUpdateCS.cso");
			wi::renderer::LoadShader(ShaderStage::CS, emitCS, "emittedparticle_emitCS.cso");
			wi::renderer::LoadShader(ShaderStage::CS, emitCS_VOLUME, "emittedparticle_emitCS_volume.cso");
			wi::renderer::LoadShader(ShaderStage::CS, emitCS_FROMMESH, "emittedparticle_emitCS_FROMMESH.cso");
			wi::renderer::LoadShader(ShaderStage::CS, sphpartitionCS, "emittedparticle_sphpartitionCS.cso");
			wi::renderer::LoadShader(ShaderStage::CS, sphcellallocationCS, "emittedparticle_sphcellallocationCS.cso");
			wi::renderer::LoadShader(ShaderStage::CS, sphbinningCS, "emittedparticle_sphbinningCS.cso");
			wi::renderer::LoadShader(ShaderStage::CS, sphdensityCS, "emittedparticle_sphdensityCS.cso");
			wi::renderer::LoadShader(ShaderStage::CS, sphforceCS, "emittedparticle_sphforceCS.cso");
			wi::renderer::LoadShader(ShaderStage::CS, simulateCS, "emittedparticle_simulateCS.cso");
			wi::renderer::LoadShader(ShaderStage::CS, simulateCS_SORTING, "emittedparticle_simulateCS_SORTING.cso");
			wi::renderer::LoadShader(ShaderStage::CS, simulateCS_DEPTHCOLLISIONS, "emittedparticle_simulateCS_DEPTHCOLLISIONS.cso");
			wi::renderer::LoadShader(ShaderStage::CS, simulateCS_SORTING_DEPTHCOLLISIONS, "emittedparticle_simulateCS_SORTING_DEPTHCOLLISIONS.cso");


			GraphicsDevice* device = wi::graphics::GetDevice();

			for (int i = 0; i < BLENDMODE_COUNT; ++i)
			{
				PipelineStateDesc desc;
				desc.pt = PrimitiveTopology::TRIANGLESTRIP;
				if (ALLOW_MESH_SHADER && wi::graphics::GetDevice()->CheckCapability(GraphicsDeviceCapability::MESH_SHADER))
				{
					desc.ms = &meshShader;
				}
				else
				{
					desc.vs = &vertexShader;
				}
				desc.bs = &blendStates[i];
				desc.rs = &rasterizerState;
				desc.dss = &depthStencilState;

				for (int j = 0; j < EmittedParticleSystem::PARTICLESHADERTYPE_COUNT; ++j)
				{
					desc.ps = &pixelShader[j];
					device->CreatePipelineState(&desc, &PSO[i][j]);
					// GG: the same into the half resolution target, its blend and no depth test
					PipelineStateDesc half = desc;
					half.bs = &halfBlendStates[i];
					half.dss = &depthStencilStateNone;
					device->CreatePipelineState(&half, &PSO_half[i][j]);
				}
			}

			{
				PipelineStateDesc desc;
				desc.pt = PrimitiveTopology::TRIANGLESTRIP;
				if (ALLOW_MESH_SHADER && wi::graphics::GetDevice()->CheckCapability(GraphicsDeviceCapability::MESH_SHADER))
				{
					desc.ms = &meshShader;
				}
				else
				{
					desc.vs = &vertexShader;
				}
				desc.ps = &pixelShader[EmittedParticleSystem::SIMPLE];
				desc.bs = &blendStates[BLENDMODE_ALPHA];
				desc.rs = &wireFrameRS;
				desc.dss = &depthStencilState;

				device->CreatePipelineState(&desc, &PSO_wire);
			}

		}
	}

	void EmittedParticleSystem::Initialize()
	{
		wi::Timer timer;

		RasterizerState rs;
		rs.fill_mode = FillMode::SOLID;
		rs.cull_mode = CullMode::NONE;
		rs.front_counter_clockwise = true;
		rs.depth_bias = 0;
		rs.depth_bias_clamp = 0;
		rs.slope_scaled_depth_bias = 0;
		rs.depth_clip_enable = false;
		rs.multisample_enable = false;
		rs.antialiased_line_enable = false;
		rasterizerState = rs;


		rs.fill_mode = FillMode::WIREFRAME;
		rs.cull_mode = CullMode::NONE;
		rs.front_counter_clockwise = true;
		rs.depth_bias = 0;
		rs.depth_bias_clamp = 0;
		rs.slope_scaled_depth_bias = 0;
		rs.depth_clip_enable = false;
		rs.multisample_enable = false;
		rs.antialiased_line_enable = false;
		wireFrameRS = rs;


		DepthStencilState dsd;
		dsd.depth_enable = true;
		dsd.depth_write_mask = DepthWriteMask::ZERO;
		dsd.depth_func = ComparisonFunc::GREATER_EQUAL;
		dsd.stencil_enable = false;
		depthStencilState = dsd;


		BlendState bd;
		bd.render_target[0].blend_enable = true;
		bd.render_target[0].src_blend = Blend::SRC_ALPHA;
		bd.render_target[0].dest_blend = Blend::INV_SRC_ALPHA;
		bd.render_target[0].blend_op = BlendOp::ADD;
		bd.render_target[0].src_blend_alpha = Blend::ONE;
		bd.render_target[0].dest_blend_alpha = Blend::INV_SRC_ALPHA;
		bd.render_target[0].blend_op_alpha = BlendOp::ADD;
		bd.render_target[0].render_target_write_mask = ColorWrite::ENABLE_ALL;
		bd.independent_blend_enable = false;
		blendStates[BLENDMODE_ALPHA] = bd;

		bd.render_target[0].blend_enable = true;
		bd.render_target[0].src_blend = Blend::SRC_ALPHA;
		bd.render_target[0].dest_blend = Blend::ONE;
		bd.render_target[0].blend_op = BlendOp::ADD;
		bd.render_target[0].src_blend_alpha = Blend::ZERO;
		bd.render_target[0].dest_blend_alpha = Blend::ONE;
		bd.render_target[0].blend_op_alpha = BlendOp::ADD;
		bd.render_target[0].render_target_write_mask = ColorWrite::ENABLE_ALL;
		bd.independent_blend_enable = false;
		blendStates[BLENDMODE_ADDITIVE] = bd;

		bd.render_target[0].blend_enable = true;
		bd.render_target[0].src_blend = Blend::ONE;
		bd.render_target[0].dest_blend = Blend::INV_SRC_ALPHA;
		bd.render_target[0].blend_op = BlendOp::ADD;
		bd.render_target[0].src_blend_alpha = Blend::ONE;
		bd.render_target[0].dest_blend_alpha = Blend::ONE;
		bd.render_target[0].blend_op_alpha = BlendOp::ADD;
		bd.render_target[0].render_target_write_mask = ColorWrite::ENABLE_ALL;
		bd.independent_blend_enable = false;
		blendStates[BLENDMODE_PREMULTIPLIED] = bd;

		bd.render_target[0].src_blend = Blend::DEST_COLOR;
		bd.render_target[0].dest_blend = Blend::ZERO;
		bd.render_target[0].blend_op = BlendOp::ADD;
		bd.render_target[0].src_blend_alpha = Blend::DEST_ALPHA;
		bd.render_target[0].dest_blend_alpha = Blend::ZERO;
		bd.render_target[0].blend_op_alpha = BlendOp::ADD;
		bd.render_target[0].blend_enable = true;
		bd.render_target[0].render_target_write_mask = ColorWrite::ENABLE_ALL;
		bd.alpha_to_coverage_enable = false;
		bd.independent_blend_enable = false;
		blendStates[BLENDMODE_MULTIPLY] = bd;

		bd.render_target[0].src_blend = Blend::INV_DEST_COLOR;
		bd.render_target[0].dest_blend = Blend::ZERO;
		bd.render_target[0].blend_op = BlendOp::ADD;
		bd.render_target[0].src_blend_alpha = Blend::DEST_ALPHA;
		bd.render_target[0].dest_blend_alpha = Blend::ZERO;
		bd.render_target[0].blend_op_alpha = BlendOp::ADD;
		bd.render_target[0].blend_enable = true;
		bd.render_target[0].render_target_write_mask = ColorWrite::ENABLE_ALL;
		bd.alpha_to_coverage_enable = false;
		bd.independent_blend_enable = false;
		blendStates[BLENDMODE_INVERSE] = bd;

		bd.render_target[0].blend_enable = false;
		blendStates[BLENDMODE_OPAQUE] = bd;

		// GG: into the half resolution target, cleared to no colour and the whole scene showing through: over, its alpha taking
		// from what shows through; premultiplied likewise; additive adding colour and leaving it
		BlendState half;
		half.render_target[0].blend_enable = true;
		half.render_target[0].src_blend = Blend::SRC_ALPHA;
		half.render_target[0].dest_blend = Blend::INV_SRC_ALPHA;
		half.render_target[0].blend_op = BlendOp::ADD;
		half.render_target[0].src_blend_alpha = Blend::ZERO;
		half.render_target[0].dest_blend_alpha = Blend::INV_SRC_ALPHA;
		half.render_target[0].blend_op_alpha = BlendOp::ADD;
		half.render_target[0].render_target_write_mask = ColorWrite::ENABLE_ALL;
		half.independent_blend_enable = false;
		for (int i = 0; i < BLENDMODE_COUNT; ++i) halfBlendStates[i] = half;
		half.render_target[0].src_blend = Blend::ONE;
		halfBlendStates[BLENDMODE_PREMULTIPLIED] = half;
		half.render_target[0].src_blend = Blend::SRC_ALPHA;
		half.render_target[0].dest_blend = Blend::ONE;
		half.render_target[0].dest_blend_alpha = Blend::ONE;
		halfBlendStates[BLENDMODE_ADDITIVE] = half;

		DepthStencilState none;
		none.depth_enable = false;
		none.stencil_enable = false;
		depthStencilStateNone = none;

		static wi::eventhandler::Handle handle = wi::eventhandler::Subscribe(wi::eventhandler::EVENT_RELOAD_SHADERS, [](uint64_t userdata) { EmittedParticleSystem_Internal::LoadShaders(); });
		EmittedParticleSystem_Internal::LoadShaders();

		wilog("wi::EmittedParticleSystem Initialized (%d ms)", (int)std::round(timer.elapsed()));
	}


	void EmittedParticleSystem::Serialize(wi::Archive& archive, wi::ecs::EntitySerializer& seri)
	{
		if (archive.IsReadMode())
		{
			archive >> _flags;
			archive >> (uint32_t&)shaderType;
			wi::ecs::SerializeEntity(archive, meshID, seri);
			archive >> MAX_PARTICLES;
			archive >> FIXED_TIMESTEP;
			archive >> size;
			archive >> random_factor;
			archive >> normal_factor;
			archive >> count;
			archive >> life;
			archive >> random_life;
			archive >> scaleX;
			archive >> scaleY;
			archive >> rotation;
			archive >> motionBlurAmount;
			archive >> mass;
			archive >> SPH_h;
			archive >> SPH_K;
			archive >> SPH_p0;
			archive >> SPH_e;

			if (archive.GetVersion() >= 45)
			{
				archive >> framesX;
				archive >> framesY;
				archive >> frameCount;
				archive >> frameStart;
				archive >> frameRate;
			}

			if (archive.GetVersion() == 48)
			{
				uint8_t shadingRate;
				archive >> shadingRate; // no longer needed
			}

			if (archive.GetVersion() >= 64)
			{
				archive >> velocity;
				archive >> gravity;
				archive >> drag;
				archive >> random_color;
			}
			else
			{
				if (IsSPHEnabled())
				{
					gravity = XMFLOAT3(0, -9.8f * 2, 0);
					drag = 0.98f;
				}
			}

			if (archive.GetVersion() >= 74)
			{
				archive >> restitution;
			}

			if (seri.GetVersion() >= 1)
			{
				archive >> opacityCurveControlPeakStart;
			}
			else
			{
				opacityCurveControlPeakStart = 0;
			}

			if (seri.GetVersion() >= 2)
			{
				archive >> opacityCurveControlPeakEnd;
			}
			else
			{
				opacityCurveControlPeakEnd = opacityCurveControlPeakStart;
			}

			// GGMAX 2.01: the GameGuru emitter extensions added in 2.00 were NOT round-tripped
			// here, and Scene::Entity_Duplicate() duplicates entities by serializing them
			// through this very function. preload_wicked_particle_effect() (M-Entity_part5.cpp)
			// builds its 5-slot ready_decals[] clone cache with exactly that call, so clones
			// 1-4 came back with every field below at its default - burst_amount = 0 above all.
			// Burst(0) resolves num = burst_amount, so 4 of every 5 impact/blood/explosion
			// effects fired nothing at all, and overlapping repeat hits looked like "only one
			// effect can play at a time". The DX11 fork serialized this same set (see
			// D:\max\WickedRepo\WickedEngine\wiEmittedParticle.cpp:953-1003, its 5072-5077
			// version bands); this is that set, minus the three wpe_filler_* placeholders and
			// spawn_pause/spawn_pause_random which have no member in this struct.
			// Guarded on the emitters component-library version (wiScene.h), bumped 2 -> 4 (3 is skipped: upstream took it for burst_on_create), so
			// scenes written before this change still read correctly and keep the defaults.
			if (seri.GetVersion() >= 4)
			{
				archive >> fadein_time;
				archive >> burst_amount;
				archive >> burst_delay;
				archive >> normal_factor_x;
				archive >> normal_factor_y;
				archive >> normal_factor_z;
				archive >> normal_random;
				archive >> rotation_random;
				archive >> size_random;
				archive >> spawn_random;
				archive >> scaling_random;
				archive >> endcolor_red;
				archive >> endcolor_green;
				archive >> endcolor_blue;
				archive >> burst_split;
				archive >> burst_factor_x;
				archive >> burst_factor_y;
				archive >> burst_factor_z;
				archive >> startpos;
				archive >> bFindFloor;
				archive >> burst_factor_speed;
				archive >> start_rotation;
				archive >> bFollowCamera;
				archive >> random_position;
				archive >> random_position_scale;
				archive >> distance_sort_bias;
			}
		}
		else
		{
			archive << _flags;
			archive << (uint32_t)shaderType;
			wi::ecs::SerializeEntity(archive, meshID, seri);
			archive << MAX_PARTICLES;
			archive << FIXED_TIMESTEP;
			archive << size;
			archive << random_factor;
			archive << normal_factor;
			archive << count;
			archive << life;
			archive << random_life;
			archive << scaleX;
			archive << scaleY;
			archive << rotation;
			archive << motionBlurAmount;
			archive << mass;
			archive << SPH_h;
			archive << SPH_K;
			archive << SPH_p0;
			archive << SPH_e;

			if (archive.GetVersion() >= 45)
			{
				archive << framesX;
				archive << framesY;
				archive << frameCount;
				archive << frameStart;
				archive << frameRate;
			}

			if (archive.GetVersion() >= 64)
			{
				archive << velocity;
				archive << gravity;
				archive << drag;
				archive << random_color;
			}

			if (archive.GetVersion() >= 74)
			{
				archive << restitution;
			}

			if (seri.GetVersion() >= 1)
			{
				archive << opacityCurveControlPeakStart;
			}

			if (seri.GetVersion() >= 2)
			{
				archive << opacityCurveControlPeakEnd;
			}

			// GGMAX 2.01: must mirror the read block above field-for-field and in order.
			if (seri.GetVersion() >= 4)
			{
				archive << fadein_time;
				archive << burst_amount;
				archive << burst_delay;
				archive << normal_factor_x;
				archive << normal_factor_y;
				archive << normal_factor_z;
				archive << normal_random;
				archive << rotation_random;
				archive << size_random;
				archive << spawn_random;
				archive << scaling_random;
				archive << endcolor_red;
				archive << endcolor_green;
				archive << endcolor_blue;
				archive << burst_split;
				archive << burst_factor_x;
				archive << burst_factor_y;
				archive << burst_factor_z;
				archive << startpos;
				archive << bFindFloor;
				archive << burst_factor_speed;
				archive << start_rotation;
				archive << bFollowCamera;
				archive << random_position;
				archive << random_position_scale;
				archive << distance_sort_bias;
			}
		}
	}

}
