#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"
// The screen-space lens flare added to the HDR image before the bloom and the tone map (Postprocess_ScreenLensFlare), times
// its intensity, as the DX11 build added it

PUSHCONSTANT(push, ScreenLensFlarePush);

[numthreads(POSTPROCESS_BLOCKSIZE, POSTPROCESS_BLOCKSIZE, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	const float2 uv = (DTid.xy + 0.5) * push.resolution_rcp;

	float4 color = bindless_textures[descriptor_index(push.texture_input)][DTid.xy];
	color.rgb += bindless_textures_half4[descriptor_index(push.texture_aux)].SampleLevel(sampler_linear_clamp, uv, 0).rgb * push.intensity;

	bindless_rwtextures[descriptor_index(push.texture_output)][DTid.xy] = color;
}
