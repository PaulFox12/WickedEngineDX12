// GG: heat shimmer, drawn into the screen distortion target (rtParticleDistortion, R16G16 offsets in UV, added to the heat
// haze particles' own) that the tone map bends the picture by. Only far pixels shimmer, in a band round the horizon line,
// most of all towards a low sun; the offset wavers mostly sideways and drifts upwards, as air over hot ground does. The
// tone map already keeps nearer things (the gun, a wall's edge) out of a far pixel's offset. Off unless a script sets a
// strength (SetHeatShimmer).
#include "globals.hlsli"

struct GGHeatShimmerPush
{
	float strength;     // the largest offset in pixels at the screen's height (scaled for other heights)
	float start;        // the distance, in units, the shimmer starts at
	float full;         // and the distance it is full at
	float sun_boost;    // how much more it bends towards a low sun (0 none)
	float band;         // the half width of the band round the horizon, as the view ray's height (sine of the angle)
	float scale;        // the size of the ripples (larger is finer)
	float speed;        // how fast they move
	float pad;
};
PUSHCONSTANT(push, GGHeatShimmerPush);

float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
	const float lin = texture_lineardepth.SampleLevel(sampler_point_clamp, uv, 0);
	const float dist = lin * GetCamera().z_far;
	const float near_fade = smoothstep(push.start, max(push.full, push.start + 1), dist);
	if (near_fade <= 0)
		return 0;

	// the view ray: its height picks the band round the horizon, its heading the sun's side
	const float3 P = reconstruct_position(uv, 0.5, GetCamera().inverse_view_projection);
	const float3 dir = normalize(P - GetCamera().position);
	const float band = 1 - smoothstep(0, max(push.band, 0.001), abs(dir.y));
	if (band <= 0)
		return 0;

	const float3 sun = GetSunDirection();
	const float low_sun = sun.y > 0 ? saturate(1 - sun.y * 3) : 0; // under about 20 degrees up
	const float2 dir_h = normalize(dir.xz + 0.0001);
	const float2 sun_h = normalize(sun.xz + 0.0001);
	const float towards = pow(saturate(dot(dir_h, sun_h)), 6);
	const float amount = near_fade * band * (1 + push.sun_boost * low_sun * towards);

	// ripples in screen space, at the screen's aspect, drifting upwards
	const float2 res = GetCamera().internal_resolution;
	const float2 p = float2(uv.x * res.x / res.y, uv.y) * push.scale;
	const float t = GetFrame().time * push.speed;
	const float w1 = sin(p.y * 31.0 + t * 2.3 + sin(p.x * 11.0 + t * 0.7) * 1.9);
	const float w2 = sin(p.y * 57.0 + t * 3.7 + p.x * 7.0 + sin(p.x * 23.0 - t * 1.3) * 1.1);
	const float w3 = sin(p.x * 41.0 - t * 1.9 + p.y * 13.0);
	const float2 offset_px = float2(w1 * 0.6 + w2 * 0.4, (w2 * 0.5 + w3 * 0.5) * 0.35) * push.strength * (res.y / 1440.0);

	// alpha 1: the additive blend scales by the source alpha
	return float4(offset_px * GetCamera().internal_resolution_rcp * amount, 0, 1);
}
