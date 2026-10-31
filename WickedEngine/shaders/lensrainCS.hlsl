#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"
#include "lensrainHF.hlsli"
// GG: rain on the lens, after the tone map (Postprocess_LensRain): the drops of lensrainHF.hlsli over the image, each
// showing the scene through it; the fog, if any, blurs the image between them

PUSHCONSTANT(postprocess, PostProcess);

Texture2D<float4> input : register(t0);

RWTexture2D<float4> output : register(u0);

[numthreads(POSTPROCESS_BLOCKSIZE, POSTPROCESS_BLOCKSIZE, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	const float2 uv = (DTid.xy + 0.5) * postprocess.resolution_rcp;
	const float aspect = postprocess.resolution.x * postprocess.resolution_rcp.y;

	LensRainParams rain;
	rain.amount = postprocess.params0.x;
	rain.size = postprocess.params0.y;
	rain.flow = postprocess.params0.zw;
	rain.slide = postprocess.params1.x;
	rain.refraction = postprocess.params1.y;
	rain.fog = postprocess.params1.z;
	rain.time = postprocess.params1.w;

	float wiped;
	const LensRainDrop drop = LensRain_Drops(float2(uv.x * aspect, uv.y), rain, wiped);

	float4 color = input[DTid.xy];
	if (rain.fog > 0)
	{
		float3 blur = color.rgb;
		const float radius = 0.004 + 0.012 * rain.fog;
		[unroll]
		for (int i = 0; i < 8; ++i)
		{
			const float a = i * 0.785398 + 0.392699;
			blur += input.SampleLevel(sampler_linear_clamp, uv + float2(cos(a) / aspect, sin(a)) * radius, 0).rgb;
		}
		color.rgb = lerp(color.rgb, blur / 9.0, rain.fog * (1 - wiped));
	}
	if (drop.mask > 0)
	{
		float3 through = input.SampleLevel(sampler_linear_clamp, uv + float2(drop.offset.x / aspect, drop.offset.y), 0).rgb;
		through = through * (1 - 0.6 * drop.rim) + drop.spec * 0.5;
		color.rgb = lerp(color.rgb, through, drop.mask);
	}

	output[DTid.xy] = color;
}
