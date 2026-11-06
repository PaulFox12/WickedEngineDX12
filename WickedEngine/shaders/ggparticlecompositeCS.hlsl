#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"

// GG: the particles drawn apart while the temporal AA is on (RenderPath3D gg_particlesAfterTAA), laid over its resolved
// picture: the full resolution ones (t1) and then the half resolution ones (t2), each texel colour premultiplied with
// alpha the scene showing through; the half resolution texels taken as particlecompositePS takes them (bilinear where
// the four around a pixel are at its depth, else the one nearest its depth). Blended into the history, small fast
// sparks averaged away and moving flames smeared brighter (2026-11-06). params0.x: 1 the full, 2 the half resolution

PUSHCONSTANT(postprocess, PostProcess);

Texture2D<float4> input : register(t0);
Texture2D<float4> input_full : register(t1);
Texture2D<float4> input_half : register(t2);

RWTexture2D<float4> output : register(u0);

[numthreads(POSTPROCESS_BLOCKSIZE, POSTPROCESS_BLOCKSIZE, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	if (DTid.x >= postprocess.resolution.x || DTid.y >= postprocess.resolution.y)
		return;

	float4 color = input[DTid.xy];
	const uint parts = (uint)postprocess.params0.x;

	[branch]
	if (parts & 1)
	{
		const float4 full = input_full[DTid.xy];
		color.rgb = color.rgb * full.a + full.rgb;
	}

	[branch]
	if (parts & 2)
	{
		uint2 halfDim;
		input_half.GetDimensions(halfDim.x, halfDim.y);
		const float depth = texture_lineardepth[DTid.xy];
		const float2 h = (DTid.xy + 0.5) * 0.5 - 0.5;
		const int2 base = (int2)floor(h);
		const float2 f = h - base;
		const int2 offsets[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };
		const float weights[4] = { (1 - f.x) * (1 - f.y), f.x * (1 - f.y), (1 - f.x) * f.y, f.x * f.y };
		float4 bilinear = 0;
		float4 nearest = float4(0, 0, 0, 1);
		float bestDiff = 1e30;
		float maxDiff = 0;
		[unroll]
		for (int k = 0; k < 4; ++k)
		{
			const int2 texel = clamp(base + offsets[k], int2(0, 0), int2(halfDim) - 1);
			const float4 c = input_half[texel];
			const float4 block = texture_lineardepth.GatherRed(sampler_point_clamp, (float2(texel) * 2 + 1) * GetCamera().internal_resolution_rcp);
			const float diff = abs(max(max(block.x, block.y), max(block.z, block.w)) - depth);
			bilinear += c * weights[k];
			maxDiff = max(maxDiff, diff);
			if (diff < bestDiff)
			{
				bestDiff = diff;
				nearest = c;
			}
		}
		const float4 halfColor = maxDiff <= depth * 0.03 ? bilinear : nearest;
		color.rgb = color.rgb * halfColor.a + halfColor.rgb;
	}

	output[DTid.xy] = color;
}
