#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"

// GG: the volumetric lights (the sun's shafts, half resolution) averaged over frames before they are laid on the scene
// (2026-11-26). Under FSR 3.1's jitter a half resolution texel on a depth edge (a tree against the bright sky by the sun)
// marched to the sky one frame and to the leaves the next, and the upscaler showed it as a shimmer: 16,276 pixels flickered
// in a still view with the shafts on, 5 with them off. Each texel takes its history from where it was last frame (the
// motion vectors), held inside the range of its 3x3 neighbourhood this frame so nothing moving leaves a trail.
// params0.x: the history's share (0.9); params0.y: 1 when the history holds a frame, 0 the first frame

PUSHCONSTANT(postprocess, PostProcess);

Texture2D<float4> input_current : register(t0);
Texture2D<float4> input_history : register(t1);
RWTexture2D<float4> output : register(u0);

[numthreads(POSTPROCESS_BLOCKSIZE, POSTPROCESS_BLOCKSIZE, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	if (DTid.x >= postprocess.resolution.x || DTid.y >= postprocess.resolution.y)
		return;

	const float4 current = input_current[DTid.xy];
	if (postprocess.params0.y < 0.5)
	{
		output[DTid.xy] = current;
		return;
	}

	float4 neighborhoodMin = current;
	float4 neighborhoodMax = current;
	float4 m1 = 0, m2 = 0;
	const int2 last = int2(postprocess.resolution) - 1;
	for (int y = -1; y <= 1; y++)
	{
		for (int x = -1; x <= 1; x++)
		{
			const float4 neighbor = input_current[clamp(int2(DTid.xy) + int2(x, y), 0, last)];
			neighborhoodMin = min(neighborhoodMin, neighbor);
			neighborhoodMax = max(neighborhoodMax, neighbor);
			m1 += neighbor;
			m2 += neighbor * neighbor;
		}
	}
	// the history held to the neighbourhood's spread (variance clipping), inside its range: an edge's 3x3 holds both sides,
	// and the min/max range alone let a stale side through (a dark trail behind the held gun over the glow, run 2032)
	m1 /= 9.0;
	const float4 sigma = sqrt(max(m2 / 9.0 - m1 * m1, 0));
	neighborhoodMin = max(neighborhoodMin, m1 - 1.25 * sigma);
	neighborhoodMax = min(neighborhoodMax, m1 + 1.25 * sigma);

	const float2 uv = (DTid.xy + 0.5) * postprocess.resolution_rcp;
	const float2 velocity = texture_velocity.SampleLevel(sampler_point_clamp, uv, 0).xy;
	const float2 prevUV = uv + velocity;
	float4 result = current;
	// no history where the surface there last frame was another one (its depth then against now): the held gun or a canopy
	// moving over the glow
	const float depthNow = compute_lineardepth(texture_depth.SampleLevel(sampler_point_clamp, uv, 0));
	const float depthThen = compute_lineardepth(texture_depth_history.SampleLevel(sampler_point_clamp, prevUV, 0));
	const bool sameSurface = abs(depthNow - depthThen) <= 2 + depthNow * 0.02;
	if (is_saturated(prevUV) && sameSurface)
	{
		float4 history = input_history.SampleLevel(sampler_linear_clamp, prevUV, 0);
		history = clamp(history, neighborhoodMin, neighborhoodMax);
		result = lerp(current, history, saturate(postprocess.params0.x));
	}
	output[DTid.xy] = max(result, 0);
}
