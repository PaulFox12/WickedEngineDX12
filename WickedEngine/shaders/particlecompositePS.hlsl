#include "globals.hlsli"

// GG: the emitters drawn at half resolution (wi::renderer::DrawSoftParticles_Half) laid over the scene. Each texel holds their
// colour premultiplied and in alpha how much of the scene shows through; its depth is the farthest of the 2x2 pixels it
// covers, which the particles' depth fade read. A pixel takes the four texels around it bilinear where all are at its own
// depth, else the one nearest its depth, so smoke stops at a building's edge with no halo round it. Blended as
// scene * alpha + colour
Texture2D<float4> texture_particlehalf : register(t0);

float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
	uint2 halfDim;
	texture_particlehalf.GetDimensions(halfDim.x, halfDim.y);
	const float depth = texture_lineardepth[uint2(pos.xy)];

	// the texels around this pixel, their places between it
	const float2 h = pos.xy * 0.5 - 0.5;
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
		const float4 color = texture_particlehalf[texel];
		const float4 block = texture_lineardepth.GatherRed(sampler_point_clamp, (float2(texel) * 2 + 1) * GetCamera().internal_resolution_rcp);
		const float diff = abs(max(max(block.x, block.y), max(block.z, block.w)) - depth);
		bilinear += color * weights[k];
		maxDiff = max(maxDiff, diff);
		if (diff < bestDiff)
		{
			bestDiff = diff;
			nearest = color;
		}
	}

	// all four at this depth (within 3% of it): smooth; across an edge: the texel on this pixel's side of it
	return maxDiff <= depth * 0.03 ? bilinear : nearest;
}
