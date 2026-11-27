#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"

#ifndef UPSAMPLE_FORMAT
#define UPSAMPLE_FORMAT half4
#endif // UPSAMPLE_FORMAT

PUSHCONSTANT(postprocess, PostProcess);

Texture2D<UPSAMPLE_FORMAT> input : register(t0);
Texture2D<float> input_lineardepth_high : register(t1);
Texture2D<float> input_lineardepth_low : register(t2);

static const float2 offsets[] = {
	float2(-0.5, -0.5),
	float2(0.5, -0.5),
	float2(-0.5, 0.5),
	float2(0.5, 0.5),
};

// Note: this post process can be either a pixel shader or compute shader, depending on use case

#ifdef USE_PIXELSHADER
// Run this post process as pixel shader:
float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_TARGET
{
	const uint2 pixel = pos.xy;
#else
// Run this post process as compute shader:
RWTexture2D<UPSAMPLE_FORMAT> output : register(u0);
[numthreads(POSTPROCESS_BLOCKSIZE, POSTPROCESS_BLOCKSIZE, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	const uint2 pixel = DTid.xy;
	const float2 uv = (pixel + 0.5f) * postprocess.resolution_rcp;
#endif // USE_PIXELSHADER

	const float threshold_param = postprocess.params0.x;
	const uint lowres_depthchain_mip = uint(postprocess.params0.w);
	const float2 lowres_size = postprocess.params1.xy;
	const float2 lowres_texel_size = postprocess.params1.zw;
	
	const float lineardepth_highres = input_lineardepth_high[pixel] * GetCamera().z_far;

	// GG: a negative threshold is relative to the pixel's depth (its inverse a share of it). The absolute one, under a unit,
	// rejected every sample on a far ridge, so one sample won by chance and the jitter changed which: the sun shafts' staircase
	// and flicker along the mountains under FSR 3.1 (2026-11-27)
	const bool gg_relative = threshold_param < 0;
	const float threshold = gg_relative ? -threshold_param / max(lineardepth_highres, 1.0) : threshold_param;
	UPSAMPLE_FORMAT gg_min = 65504, gg_max = 0;
	float gg_zmin = 1e30, gg_zmax = 0;
	
	UPSAMPLE_FORMAT color = 0;
	half sum = 0;

	int2 lowres_pixel = int2(float2(pixel) * postprocess.params0.w);

	for(uint i = 0; i < 4; ++i)
	{
		const float2 sample_uv = uv + offsets[i] * lowres_texel_size;
		const float4 zzzz = input_lineardepth_low.GatherRed(sampler_linear_clamp, sample_uv) * GetCamera().z_far;
		const half4 wwww = max(0.001, 1 - saturate(abs(zzzz - lineardepth_highres) * threshold));
		const half4 rrrr = input.GatherRed(sampler_linear_clamp, sample_uv);
		const half4 gggg = input.GatherGreen(sampler_linear_clamp, sample_uv);
		const half4 bbbb = input.GatherBlue(sampler_linear_clamp, sample_uv);
		const half4 aaaa = input.GatherAlpha(sampler_linear_clamp, sample_uv);
		if (gg_relative)
		{
			gg_min = min(gg_min, (UPSAMPLE_FORMAT)half4(min(min(rrrr.x, rrrr.y), min(rrrr.z, rrrr.w)), min(min(gggg.x, gggg.y), min(gggg.z, gggg.w)), min(min(bbbb.x, bbbb.y), min(bbbb.z, bbbb.w)), min(min(aaaa.x, aaaa.y), min(aaaa.z, aaaa.w))));
			gg_max = max(gg_max, (UPSAMPLE_FORMAT)half4(max(max(rrrr.x, rrrr.y), max(rrrr.z, rrrr.w)), max(max(gggg.x, gggg.y), max(gggg.z, gggg.w)), max(max(bbbb.x, bbbb.y), max(bbbb.z, bbbb.w)), max(max(aaaa.x, aaaa.y), max(aaaa.z, aaaa.w))));
			gg_zmin = min(gg_zmin, min(min(zzzz.x, zzzz.y), min(zzzz.z, zzzz.w)));
			gg_zmax = max(gg_zmax, max(max(zzzz.x, zzzz.y), max(zzzz.z, zzzz.w)));
		}
		
		float2 sam_pixel = sample_uv * lowres_size + (-0.5 + 1.0 / 512.0); // (1.0 / 512.0) correction is described here: https://www.reedbeta.com/blog/texture-gathers-and-coordinate-precision/
		half2 sam_pixel_frac = frac(sam_pixel);

		color += (UPSAMPLE_FORMAT)half4(
			bilinear(rrrr * wwww, sam_pixel_frac),
			bilinear(gggg * wwww, sam_pixel_frac),
			bilinear(bbbb * wwww, sam_pixel_frac),
			bilinear(aaaa * wwww, sam_pixel_frac)
		);
		
		half weight = bilinear(wwww, sam_pixel_frac);
		sum += weight;
	}
	
	if(sum > 0)
	{
		color /= sum;
	}
	// GG: every sample rejected (a thing smaller than a half resolution texel, or the far side of an edge): a pixel nearer than
	// all of them has less of the march in front of it and takes the least light, one farther takes the most, not a blend of both
	if (gg_relative && sum < 0.0045)
	{
		if (lineardepth_highres < gg_zmin)
			color = gg_min;
		else if (lineardepth_highres > gg_zmax)
			color = gg_max;
	}

#ifdef USE_PIXELSHADER
	return color;
#else
	output[DTid.xy] = color;
#endif // USE_PIXELSHADER
}
