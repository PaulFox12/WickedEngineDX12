#include "globals.hlsli"
#include "ggEmitterRS.hlsli"
#include "ShaderInterop_EmittedParticle.h"

// GG: one dispatch for every emitter simulating this frame, a thread an emitter (see the kickoff)
PUSHCONSTANT(push, EmitterBatchPush);

ByteAddressBuffer slotBuffer : register(t0);
ByteAddressBuffer counterBuffer : register(t1);

RWByteAddressBuffer indirectBuffers : register(u0);

[numthreads(THREADCOUNT_EMITTERBATCH, 1, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
	if (DTid.x >= push.slotCount)
		return;
	const uint slot = slotBuffer.Load(push.slotOffset + DTid.x * 4);
	const uint counters = slot * PARTICLECOUNTER_STRIDE;
	const uint args = slot * ARGUMENTBUFFER_STRIDE;

	uint particleCount = counterBuffer.Load(counters + PARTICLECOUNTER_OFFSET_CULLEDCOUNT);

	if (push.options & EMITTER_OPTION_BIT_MESH_SHADER_ENABLED)
	{
		// Create DispatchMesh argument buffer:
		IndirectDispatchArgs args_mesh;
		args_mesh.ThreadGroupCountX = (particleCount + 31) / 32;
		args_mesh.ThreadGroupCountY = 1;
		args_mesh.ThreadGroupCountZ = 1;
#ifdef __PSSL__
		indirectBuffers.TypedStore<IndirectDispatchArgs>(args + ARGUMENTBUFFER_OFFSET_DRAWPARTICLES, args_mesh);
#else
		indirectBuffers.Store<IndirectDispatchArgs>(args + ARGUMENTBUFFER_OFFSET_DRAWPARTICLES, args_mesh);
#endif // __PSSL__
	}
	else
	{
		// Create draw argument buffer
		IndirectDrawArgsInstanced args_draw;
		args_draw.VertexCountPerInstance = 4;
		args_draw.InstanceCount = particleCount;
		args_draw.StartVertexLocation = 0;
		args_draw.StartInstanceLocation = 0;
#ifdef __PSSL__
		indirectBuffers.TypedStore<IndirectDrawArgsInstanced>(args + ARGUMENTBUFFER_OFFSET_DRAWPARTICLES, args_draw);
#else
		indirectBuffers.Store<IndirectDrawArgsInstanced>(args + ARGUMENTBUFFER_OFFSET_DRAWPARTICLES, args_draw);
#endif // __PSSL__
	}
}
