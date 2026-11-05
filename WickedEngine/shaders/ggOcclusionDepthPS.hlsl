#include "globals.hlsli"

// GGMAX: the frame's depth at half resolution for the objects' occlusion boxes (GG_OcclusionCulling_ObjectsSingleSample):
// each 2x2 block's farthest depth (reversed Z: the smallest), so a box counts as hidden only where it is behind all four
Texture2D<float> input_depth : register(t0);

float main(float4 pos : SV_Position) : SV_Depth
{
	uint width, height, levels;
	input_depth.GetDimensions(0, width, height, levels);
	const uint2 p = uint2(pos.xy) * 2;
	const uint2 q = min(p + 1, uint2(width, height) - 1);
	return min(min(input_depth[p], input_depth[uint2(q.x, p.y)]), min(input_depth[uint2(p.x, q.y)], input_depth[q]));
}
