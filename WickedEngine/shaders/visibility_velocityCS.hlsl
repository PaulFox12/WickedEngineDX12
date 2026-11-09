#define SURFACE_LOAD_ENABLE_WIND
#define SURFACE_LOAD_GG_MOTION
#include "globals.hlsli"
#include "ShaderInterop_Renderer.h"
#include "surfaceHF.hlsli"
#include "raytracingHF.hlsli"

// GG: the GG draws' prepass targets: RT1 their depth (0 where none drew), RT2 how far their surface was in world x and z last
// frame from where it is now (the trees' sway; 0, none of its own)
#ifdef VISIBILITY_MSAA
Texture2DMS<float> input_customDepth : register(t0);
Texture2DMS<float2> input_customMotion : register(t1);
#else
Texture2D<float> input_customDepth : register(t0);
Texture2D<float2> input_customMotion : register(t1);
#endif // VISIBILITY_MSAA

RWTexture2D<float2> output_velocity : register(u0);

[numthreads(8, 8, 1)]
void main(uint2 DTid : SV_DispatchThreadID)
{
	uint2 pixel = DTid.xy;
	const float2 uv = ((float2)pixel + 0.5) * GetCamera().internal_resolution_rcp;

	[branch]
	if (!GetCamera().is_uv_inside_scissor(uv))
	{
		output_velocity[pixel] = 0;
		return;
	}

	const float2 clipspace = uv_to_clipspace(uv);
	RayDesc ray = CreateCameraRay(pixel);

	uint primitiveID = texture_primitiveID[pixel];

#ifdef VISIBILITY_MSAA
	const float ggDepth = input_customDepth.Load(pixel, 0);
#else
	const float ggDepth = input_customDepth[pixel];
#endif // VISIBILITY_MSAA

	// GG: GameGuru's own passes (the trees, the far terrain) write depth but no primitive. Where one is the nearest surface,
	// as the resolve takes it (visibility_resolveCS: nearer than the mesh whose primitive is there, or none is), the pixel
	// moves as that surface: before, a tree against the sky moved as the sky (no parallax: temporal AA smeared it whenever
	// the camera moved) and a tree before the ground or a building moved as the surface behind it
	bool gg = ggDepth > 0;

	// GG: the sky's unless found below (a surface that wouldn't load was left unset)
	float3 pre = ray.Origin + ray.Direction * GetCamera().z_far;
	float3 ggMotion = 0;
	bool ggExact = true;
	[branch]
	if (any(primitiveID))
	{
		PrimitiveID prim;
		prim.init();
		prim.unpack(primitiveID);

		Surface surface;
		surface.init();

		[branch]
		if (surface.load(prim, ray.Origin, ray.Direction))
		{
			pre = surface.pre;
			ggMotion = surface.gg_motion;
			ggExact = surface.gg_motion_exact;
			float4 tmp = mul(GetCamera().view_projection, float4(surface.P, 1));
			tmp.xyz /= max(0.0001, tmp.w);
			gg = gg && ggDepth > saturate(tmp.z);
		}
	}
	[branch]
	if (gg)
	{
		// GG: placed by its depth and moved by the camera as a still surface there, then by its own motion (the sway)
#ifdef VISIBILITY_MSAA
		const float2 ggSway = input_customMotion.Load(pixel, 0);
#else
		const float2 ggSway = input_customMotion[pixel];
#endif // VISIBILITY_MSAA
		pre = reconstruct_position(uv, ggDepth);
		pre.xz += ggSway;
		ggMotion = float3(ggSway.x, 0, ggSway.y);
		ggExact = true;
	}

	float2 pos2D = clipspace;
	float4 pos2DPrev;
	[branch]
	if (ggExact)
	{
		// GG: this pixel taken back by the camera's rotation and move alone (the reprojection is made from those), then by the surface's own motion. Through
		// world positions and the absolute matrices the rounding far from the origin put up to half a pixel of noise on every pixel, the sky's too: it softened
		// the world, and made a gun, which has the sharpest edges and is nearest, shake under the TAA
		const float ggPixelDepth = texture_depth.SampleLevel(sampler_point_clamp, uv, 0);
		pos2DPrev = mul(GetCamera().reprojection, float4(clipspace, ggPixelDepth, 1)) * compute_lineardepth(ggPixelDepth) + mul(GetCamera().previous_view_projection, float4(ggMotion, 0));
	}
	else
	{
		pos2DPrev = mul(GetCamera().previous_view_projection, float4(pre, 1));
	}
	if(pos2DPrev.w > 0)
	{
		pos2DPrev.xy /= pos2DPrev.w;
		float2 velocity = ((pos2DPrev.xy - GetCamera().temporalaa_jitter_prev) - (pos2D.xy - GetCamera().temporalaa_jitter)) * float2(0.5, -0.5);
		output_velocity[pixel] = clamp(velocity, -1, 1);
	}
}
