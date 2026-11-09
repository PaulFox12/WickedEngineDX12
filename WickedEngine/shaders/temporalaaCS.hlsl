#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"

PUSHCONSTANT(postprocess, PostProcess);

Texture2D<float3> input_current : register(t0);
Texture2D<float3> input_history : register(t1);

RWTexture2D<float3> output : register(u0);

// This hack can improve bright areas:
#define HDR_CORRECTION

static const int THREADCOUNT = POSTPROCESS_BLOCKSIZE;
static const int TILE_BORDER = 1;
static const int TILE_SIZE = TILE_BORDER + THREADCOUNT + TILE_BORDER;
groupshared uint2 tile_cache[TILE_SIZE*TILE_SIZE];

inline uint coord_to_cache(int2 coord)
{
	return flatten2D(clamp(coord, 0, TILE_SIZE - 1), TILE_SIZE);
}

// GG: the history by a Catmull-Rom filter (five taps): a bilinear sample softens it a little every frame it moves
inline half3 gg_sample_history(float2 uv)
{
	const float2 resolution = 1.0 / postprocess.resolution_rcp;
	const float2 position = uv * resolution;
	const float2 center = floor(position - 0.5) + 0.5;
	const float2 f = position - center;
	const float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
	const float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
	const float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
	const float2 w3 = f * f * (-0.5 + 0.5 * f);
	const float2 w12 = w1 + w2;
	const float2 uv0 = (center - 1.0) * postprocess.resolution_rcp;
	const float2 uv12 = (center + w2 / w12) * postprocess.resolution_rcp;
	const float2 uv3 = (center + 2.0) * postprocess.resolution_rcp;
	float3 sum = input_history.SampleLevel(sampler_linear_clamp, float2(uv12.x, uv0.y), 0).rgb * w12.x * w0.y;
	sum += input_history.SampleLevel(sampler_linear_clamp, float2(uv0.x, uv12.y), 0).rgb * w0.x * w12.y;
	sum += input_history.SampleLevel(sampler_linear_clamp, float2(uv12.x, uv12.y), 0).rgb * w12.x * w12.y;
	sum += input_history.SampleLevel(sampler_linear_clamp, float2(uv3.x, uv12.y), 0).rgb * w3.x * w12.y;
	sum += input_history.SampleLevel(sampler_linear_clamp, float2(uv12.x, uv3.y), 0).rgb * w12.x * w3.y;
	const float weight = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
	return (half3)max(sum / max(weight, 0.0001), 0);
}

[numthreads(THREADCOUNT, THREADCOUNT, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
	// preload grid cache:
	const int2 tile_upperleft = Gid.xy * THREADCOUNT - TILE_BORDER;
	for(uint y = GTid.y * 2; y < TILE_SIZE; y += THREADCOUNT * 2)
	for(uint x = GTid.x * 2; x < TILE_SIZE; x += THREADCOUNT * 2)
	{
		const int2 pixel = tile_upperleft + int2(x, y);
		const float2 uv = (pixel + 1.0) * postprocess.resolution_rcp;
		const half4 dddd = texture_lineardepth.GatherRed(sampler_linear_clamp, uv);
		const half4 rrrr = input_current.GatherRed(sampler_linear_clamp, uv);
		const half4 gggg = input_current.GatherGreen(sampler_linear_clamp, uv);
		const half4 bbbb = input_current.GatherBlue(sampler_linear_clamp, uv);
		const uint t = coord_to_cache(int2(x, y));
		tile_cache[t] = pack_half4(half4(rrrr.w, gggg.w, bbbb.w, dddd.w));
		tile_cache[t + 1] = pack_half4(half4(rrrr.z, gggg.z, bbbb.z, dddd.z));
		tile_cache[t + TILE_SIZE] = pack_half4(half4(rrrr.x, gggg.x, bbbb.x, dddd.x));
		tile_cache[t + TILE_SIZE + 1] = pack_half4(half4(rrrr.y, gggg.y, bbbb.y, dddd.y));
	}
	GroupMemoryBarrierWithGroupSync();
	
	const float2 uv = (DTid.xy + 0.5) * postprocess.resolution_rcp;
	half3 neighborhoodMin = 10000;
	half3 neighborhoodMax = -10000;
	half3 current;
	float bestDepth = 1;
	const int2 pixel_local = TILE_BORDER + GTid.xy;

	// Search for best velocity and compute color clamping range in 3x3 neighborhood:
	int2 bestOffset = 0;
	for (int y = -TILE_BORDER; y <= TILE_BORDER; ++y)
	for (int x = -TILE_BORDER; x <= TILE_BORDER; ++x)
	{
		const int2 offset = int2(x, y);
		const uint t = coord_to_cache(pixel_local + int2(x, y));

		const half4 neighbor = unpack_half4(tile_cache[t]);
		neighborhoodMin = min(neighborhoodMin, neighbor.xyz);
		neighborhoodMax = max(neighborhoodMax, neighbor.xyz);
		if (x == 0 && y == 0)
		{
			current = neighbor.xyz;
		}

		const half depth = neighbor.w;
		if (depth < bestDepth)
		{
			bestDepth = depth;
			bestOffset = offset;
		}
	}
	const float2 velocity = texture_velocity[DTid.xy + bestOffset].xy;

	const float2 prevUV = GetCamera().clamp_uv_to_scissor(uv + velocity);

#if 1
	// Disocclusion fallback:
	float depth_current = texture_lineardepth[DTid.xy] * GetCamera().z_far;
	float depth_history = compute_lineardepth(texture_depth_history.SampleLevel(sampler_point_clamp, prevUV, 0));
	// GG: 60 units, not 1: a metre in Wicked's scale but an inch in GameGuru's, so a camera moving forward had every near pixel
	// taken as uncovered and drawn raw, the ground sharp and speckled up to a line a few metres out, which moved with the camera
	if (length(velocity) > 0.01 && abs(depth_current - depth_history) > 60)
	{
		output[DTid.xy] = current;
		//output[DTid.xy] = float3(1, 0, 0);
		return;
	}
#endif

	// we cannot avoid the linear filter here because point sampling could sample irrelevant pixels but we try to correct it later:
	half3 history = gg_sample_history(prevUV);

	// simple correction of image signal incoherency (eg. moving shadows or lighting changes):
	history.rgb = clamp(history.rgb, neighborhoodMin, neighborhoodMax);

	// GG: the new frame's share grows with the motion up to one pixel and then holds. It came from the fraction of the pixels moved,
	// so on moving ground it fell back at every whole pixel a frame, a band across the ground at each (and the history is no longer
	// softened by moving, which the larger share was there to hide)
	const float motion = max(abs(velocity.x) * GetCamera().internal_resolution.x, abs(velocity.y) * GetCamera().internal_resolution.y);
	half blendfactor = saturate(lerp(0.05, 0.15, saturate(motion)));

	// if information can not be found on the screen, revert to aliased image:
	blendfactor = is_saturated(prevUV) ? blendfactor : 1.0;

#ifdef HDR_CORRECTION
	history.rgb = tonemap(history.rgb);
	current.rgb = tonemap(current.rgb);
#endif

	// do the temporal super sampling by linearly accumulating previous samples with the current one:
	half3 resolved = lerp(history.rgb, current.rgb, blendfactor);

#ifdef HDR_CORRECTION
	resolved.rgb = inverse_tonemap(resolved.rgb);
#endif

	output[DTid.xy] = saturateMediump(resolved);
}
