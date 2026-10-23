#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"
// The screen-space lens flare's bright pass (ComputeScreenLensFlare): what is over its own threshold, at a quarter of the
// resolution, each sample capped and taken on the scene's own values; the sky can be left out, as the sun has its own flare

PUSHCONSTANT(push, ScreenLensFlarePush);

[numthreads(POSTPROCESS_BLOCKSIZE, POSTPROCESS_BLOCKSIZE, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	const float2 uv = DTid.xy + 0.5;

	Texture2D<half4> texture_input = bindless_textures_half4[descriptor_index(push.texture_input)];

	half3 color = 0;
	[unroll]
	for (uint i = 0; i < 4; ++i)
	{
		const float2 tapUV = (uv + float2((i & 1) ? 0.5 : -0.5, (i & 2) ? 0.5 : -0.5)) * push.resolution_rcp;
		[branch]
		if (push.nosky && push.texture_aux >= 0)
		{
			if (bindless_textures_float[descriptor_index(push.texture_aux)].SampleLevel(sampler_point_clamp, tapUV, 0) == 0)
				continue;
		}
		color += min(texture_input.SampleLevel(sampler_linear_clamp, tapUV, 0).rgb, (half)push.cap);
	}
	color /= 4.0;

	color = max(color - push.threshold, 0);

	bindless_rwtextures[descriptor_index(push.texture_output)][DTid.xy] = float4(color, 1);
}
