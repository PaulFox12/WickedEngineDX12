#include "globals.hlsli"
#include "ggEmitterRS.hlsli"
#include "ShaderInterop_EmittedParticle.h"

// GG: one dispatch for every emitter simulating this frame, a thread an emitter (a dispatch each was a third of the
// recording's cost with hundreds of emitters): their slots of the shared counter and argument pages come from the list
PUSHCONSTANT(push, EmitterBatchPush);

ByteAddressBuffer slotBuffer : register(t0);

RWByteAddressBuffer counterBuffer : register(u4);
RWByteAddressBuffer indirectBuffers : register(u5);

[numthreads(THREADCOUNT_EMITTERBATCH, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	if (DTid.x >= push.slotCount)
		return;
	const uint slot = slotBuffer.Load(push.slotOffset + DTid.x * 4);
	const uint counters = slot * PARTICLECOUNTER_STRIDE;
	const uint args = slot * ARGUMENTBUFFER_STRIDE;

	uint aliveCount = counterBuffer.Load(counters + PARTICLECOUNTER_OFFSET_ALIVECOUNT_AFTERSIMULATION);
	int deadCount = counterBuffer.Load<int>(counters + PARTICLECOUNTER_OFFSET_DEADCOUNT);

	indirectBuffers.Store3(args + ARGUMENTBUFFER_OFFSET_DISPATCHSIMULATION, uint3((aliveCount + THREADCOUNT_SIMULATION - 1) / THREADCOUNT_SIMULATION, 1, 1));

	counterBuffer.Store(counters + PARTICLECOUNTER_OFFSET_ALIVECOUNT, aliveCount);
	counterBuffer.Store(counters + PARTICLECOUNTER_OFFSET_ALIVECOUNT_AFTERSIMULATION, 0);

#ifdef __PSSL__
	counterBuffer.TypedStore<int>(counters + PARTICLECOUNTER_OFFSET_DEADCOUNT, max(0, deadCount));
#else
	counterBuffer.Store<int>(counters + PARTICLECOUNTER_OFFSET_DEADCOUNT, max(0, deadCount));
#endif // __PSSL__

	counterBuffer.Store(counters + PARTICLECOUNTER_OFFSET_CULLEDCOUNT, 0);

	counterBuffer.Store(counters + PARTICLECOUNTER_OFFSET_CELLALLOCATOR, 0);
}
