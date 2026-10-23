#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"
// The screen-space lens flare's bright pass (ComputeScreenLensFlare): what is over its own threshold, at a quarter of the
// resolution, each sample capped and taken on the scene's own values; the sky can be left out, as the sun has its own flare

PUSHCONSTANT(push, ScreenLensFlareBrightPush);

[numthreads(POSTPROCESS_BLOCKSIZE, POSTPROCESS_BLOCKSIZE, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	const float2 uv = DTid.xy + 0.5;

	Texture2D<half4> texture_input = bindless_textures_half4[descriptor_index(push.texture_input)];

	// the bloom the tonemap will add, back on the scene's own scale, so what blooms flares as on DX11, where the bloom was
	// added to the image before the flare
	float bloom_unexpose = 0;
	[branch]
	if (push.texture_bloom >= 0)
	{
		float exposure = push.exposure;
		[branch]
		if (push.buffer_input_luminance >= 0)
		{
			exposure *= bindless_buffers[descriptor_index(push.buffer_input_luminance)].Load<float>(LUMINANCE_BUFFER_OFFSET_EXPOSURE);
		}
		bloom_unexpose = exposure > 0 ? 1.0 / exposure : 0;
	}

	half3 color = 0;
	[unroll]
	for (uint i = 0; i < 4; ++i)
	{
		const float2 tapUV = (uv + float2((i & 1) ? 0.5 : -0.5, (i & 2) ? 0.5 : -0.5)) * push.resolution_rcp;
		[branch]
		if (push.nosky && push.texture_depth_buffer >= 0)
		{
			if (bindless_textures_float[descriptor_index(push.texture_depth_buffer)].SampleLevel(sampler_point_clamp, tapUV, 0) == 0)
				continue;
		}
		half3 tap = texture_input.SampleLevel(sampler_linear_clamp, tapUV, 0).rgb;
		[branch]
		if (push.texture_bloom >= 0)
		{
			Texture2D<half4> texture_bloom = bindless_textures_half4[descriptor_index(push.texture_bloom)];
			half3 bloom = texture_bloom.SampleLevel(sampler_linear_clamp, tapUV, 1.5).rgb;
			bloom += texture_bloom.SampleLevel(sampler_linear_clamp, tapUV, 3.5).rgb;
			bloom += texture_bloom.SampleLevel(sampler_linear_clamp, tapUV, 4).rgb;
			bloom /= 3.0;
			if (push.bloom_radius > 0)
			{
				bloom += push.bloom_radius * 0.5 * (texture_bloom.SampleLevel(sampler_linear_clamp, tapUV, 5).rgb + texture_bloom.SampleLevel(sampler_linear_clamp, tapUV, 6).rgb);
			}
			tap += bloom * bloom_unexpose;
		}
		color += min(tap, (half)push.cap);
	}
	color /= 4.0;

	color = max(color - push.threshold, 0);

	bindless_rwtextures[descriptor_index(push.texture_output)][DTid.xy] = float4(color, 1);
}
