#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"
// The screen-space lens flare's ghosts and halo (ComputeScreenLensFlare), from the blurred bright pass: each bright spot
// repeats along the line through the centre of the screen, mirrored, and as a ring round it, with a colour fringe, so each
// ghost takes its source's colour. The tonemap adds the result.

PUSHCONSTANT(push, ScreenLensFlarePush);

float3 SampleFringe(Texture2D<half4> input, float2 uv, float2 fringe, float mip)
{
	return float3(
		input.SampleLevel(sampler_linear_clamp, uv + fringe, mip).r,
		input.SampleLevel(sampler_linear_clamp, uv, mip).g,
		input.SampleLevel(sampler_linear_clamp, uv - fringe, mip).b
	);
}

[numthreads(POSTPROCESS_BLOCKSIZE, POSTPROCESS_BLOCKSIZE, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	const float2 uv = (DTid.xy + 0.5) * push.resolution_rcp;

	Texture2D<half4> input = bindless_textures_half4[descriptor_index(push.texture_input)];

	const float2 flipped = 1 - uv;
	const float2 ghostVec = (0.5 - flipped) * push.spacing;
	const float ghostLength = length(ghostVec);
	const float2 dir = ghostLength > 0.0001 ? ghostVec / ghostLength : float2(0, 0);
	const float2 fringe = dir * 0.004; // the colour fringe
	const float centreDist = length(float2(0.5, 0.5));

	float3 result = 0;
	for (uint i = 0; i < 5; ++i) // the ghosts
	{
		const float2 ghostUV = frac(flipped + ghostVec * i);
		float weight = 1.0 - length(0.5 - ghostUV) / centreDist;
		weight = pow(saturate(weight), 10.0);
		result += SampleFringe(input, ghostUV, fringe, push.mip) * weight;
	}

	// the halo: a ring at the given radius round the centre
	const float2 haloUV = frac(flipped + dir * push.halo);
	float haloWeight = 1.0 - length(0.5 - haloUV) / centreDist;
	haloWeight = pow(saturate(haloWeight), 5.0);
	result += SampleFringe(input, haloUV, fringe, push.mip) * haloWeight;

	bindless_rwtextures[descriptor_index(push.texture_output)][DTid.xy] = float4(result, 1);
}
