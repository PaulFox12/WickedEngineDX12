#include "globals.hlsli"
// Resolve MSAA depth buffer to a non MSAA texture
// GGMAX: GG_SINGLE_SAMPLE (a permutation) copies a single-sample depth buffer the same way, for the planar reflection
// drawn without MSAA (gg_reflection_samples 1)

#ifdef GG_SINGLE_SAMPLE
Texture2D<float> input : register(t0);
#else
Texture2DMS<float> input : register(t0);
#endif // GG_SINGLE_SAMPLE

RWTexture2D<float> output : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
	uint2 dim;
#ifdef GG_SINGLE_SAMPLE
	input.GetDimensions(dim.x, dim.y);
	if (dispatchThreadId.x > dim.x || dispatchThreadId.y > dim.y)
	{
		return;
	}

	output[dispatchThreadId.xy] = input.Load(uint3(dispatchThreadId.xy, 0)).r;
#else
	uint sampleCount;
	input.GetDimensions(dim.x, dim.y, sampleCount);
	if (dispatchThreadId.x > dim.x || dispatchThreadId.y > dim.y)
	{
		return;
	}

	float result = 0;
	for (uint i = 0; i < sampleCount; ++i)
	{
		result = max(result, input.Load(dispatchThreadId.xy, i).r);
	}

	output[dispatchThreadId.xy] = result;
#endif // GG_SINGLE_SAMPLE
}
