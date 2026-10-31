#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"
#include "lensrainHF.hlsli"
// GG: rain on the lens, after the tone map (Postprocess_LensRain): the water of lensrainHF.hlsli over the image. Inside a
// drop the view is read moved by the water's lens, turned over and wide; between the drops the fog, if any, reads the
// image blurred a little and much (blurNear, blurFar), cleared along the drops' trails

PUSHCONSTANT(postprocess, PostProcess);

Texture2D<float4> input : register(t0);
Texture2D<float4> blurNear : register(t1);
Texture2D<float4> blurFar : register(t2);

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

	const LensRainWater water = LensRain_Water(float2(uv.x * aspect, uv.y), rain, postprocess.resolution_rcp.y);

	// there is a drop where the surface is above 0.3; inside it the view is read moved against each drop's slope times its
	// size, so it is turned over and wide, and toward the edge the move fades back to nothing, which draws the outline. The
	// drops are out of focus, as water on a lens is while the eye looks past it: a soft edge, the view through them blurred,
	// a dim highlight, and over the scene at partial strength
	const float t = saturate((water.height - 0.3) / 0.5);
	const float inside = smoothstep(0.0, 0.7, t) * saturate(rain.amount * 4.0);
	const float2 offset = -water.lens * (0.5 + rain.refraction * 2.5) * inside;
	const float rim = inside * (1 - inside) * 4.0;

	// the running drops' film bends the view a little along their paths
	const float2 filmOffset = -water.film * rain.refraction * rain.size * rain.size * 0.25 * saturate(rain.amount * 4.0);
	float4 color = input[DTid.xy];
	if (any(filmOffset != 0))
		color.rgb = input.SampleLevel(sampler_linear_clamp, uv + float2(filmOffset.x / aspect, filmOffset.y), 0).rgb;
	if (rain.fog > 0)
	{
		const float3 soft = blurNear.SampleLevel(sampler_linear_clamp, uv, 0).rgb;
		const float3 softer = blurFar.SampleLevel(sampler_linear_clamp, uv, 0).rgb;
		const float fog = rain.fog * (1 - water.trail);
		color.rgb = fog < 0.5 ? lerp(color.rgb, soft, fog * 2) : lerp(soft, softer, fog * 2 - 1);
	}
	if (inside > 0)
	{
		// the view through the drop, blurred: a ring of eight reads round the moved point and the point itself
		const float2 at = uv + float2(offset.x / aspect, offset.y);
		const float blur = 0.1 * rain.size;
		float3 through = input.SampleLevel(sampler_linear_clamp, at, 0).rgb;
		[unroll]
		for (int i = 0; i < 8; ++i)
		{
			const float a = i * 0.785398 + 0.392699;
			through += input.SampleLevel(sampler_linear_clamp, at + float2(cos(a) / aspect, sin(a)) * blur, 0).rgb;
		}
		through /= 9.0;
		if (rain.fog > 0)
			through = lerp(through, blurNear.SampleLevel(sampler_linear_clamp, at, 0).rgb, rain.fog * 0.35);
		// a little brighter as a drop gathers light, a little darker in the band of its outline, and a dim highlight from
		// above on the water's slope
		const float3 n = normalize(float3(-water.slope * rain.size * 0.08, 1.0));
		const float spec = pow(saturate(dot(n, float3(-0.32, -0.7, 0.64))), 12.0) * inside;
		color.rgb = lerp(color.rgb, through * (1.0 + 0.04 * inside - 0.2 * rim) + spec * 0.2, inside * 0.85);
	}

	output[DTid.xy] = color;
}
