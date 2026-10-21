#define TRANSPARENT // uses transparent light lists
#define LIGHTING_SCATTER
#include "globals.hlsli"
#include "emittedparticleHF.hlsli"
#include "ShaderInterop_EmittedParticle.h"
#include "objectHF.hlsli"

#ifdef EMITTEDPARTICLE_DISTORTION
static const uint SLOT = NORMALMAP;
#else
static const uint SLOT = BASECOLORMAP;
#endif // EMITTEDPARTICLE_DISTORTION

[earlydepthstencil]
float4 main(VertextoPixel input) : SV_TARGET
{
	float3 pos3D = input.GetPos3D();
	
	// Blocker shadow map check:
	[branch]
	if ((xEmitterOptions & EMITTER_OPTION_BIT_USE_RAIN_BLOCKER) && rain_blocker_check(pos3D))
	{
		return 0;
	}

	float2 ScreenCoord = input.pos.xy * GetCamera().internal_resolution_rcp; // use pixel center!
	// GG: drawn at half resolution, so the scene's depth (full resolution) is at twice the pixel: its gather the farthest of
	// the 2x2 pixels this one covers, which the half resolution composite matches
	if (xParticleLocalOptions & 32) ScreenCoord *= 2;
	min16uint2 pixel = input.pos.xy; // no longer pixel center!
	
	write_mipmap_feedback(EmitterGetGeometry().materialIndex, ddx_coarse(input.tex.xyxy), ddy_coarse(input.tex.xyxy));
	
	ShaderMaterial material = EmitterGetMaterial();

	half4 color = 1;

	[branch]
	if (material.textures[SLOT].IsValid())
	{
		color = material.textures[SLOT].Sample(sampler_linear_clamp, input.tex.xyxy);
#ifndef EMITTEDPARTICLE_DISTORTION
		// GG: a sheet stored plain but read through an sRGB view, put back to its stored values (option bit 128), as the DX11
		// build sampled it: the effects' colours were made against them
		if (xParticleLocalOptions & 128) color.rgb = (half3)ApplySRGBCurve(color.rgb);
#endif // EMITTEDPARTICLE_DISTORTION

		[branch]
		if (xEmitterOptions & EMITTER_OPTION_BIT_FRAME_BLENDING_ENABLED)
		{
			half4 color2 = material.textures[SLOT].Sample(sampler_linear_clamp, input.tex.zwzw);
#ifndef EMITTEDPARTICLE_DISTORTION
			if (xParticleLocalOptions & 128) color2.rgb = (half3)ApplySRGBCurve(color2.rgb);
#endif // EMITTEDPARTICLE_DISTORTION
			color = lerp(color, color2, input.frameBlend);
		}
	}
	
	half4 inputColor;
	inputColor.r = ((input.color >> 0)  & 0xFF) / 255.0f;
	inputColor.g = ((input.color >> 8)  & 0xFF) / 255.0f;
	inputColor.b = ((input.color >> 16) & 0xFF) / 255.0f;
	inputColor.a = ((input.color >> 24) & 0xFF) / 255.0f;

	half opacity = color.a * inputColor.a;

	half3 normal = 0;
#ifndef EMITTEDPARTICLE_DISTORTION // the "distortion" shader is just using normal map as the color map and uses it for signed blending, this normal logic won't be used for that
	[branch]
	if (material.textures[NORMALMAP].IsValid())
	{
		normal = material.textures[NORMALMAP].Sample(sampler_linear_clamp, input.tex.xyxy).rgb;

		[branch]
		if (xEmitterOptions & EMITTER_OPTION_BIT_FRAME_BLENDING_ENABLED)
		{
			half3 normal2 = material.textures[NORMALMAP].Sample(sampler_linear_clamp, input.tex.zwzw).rgb;
			normal = lerp(normal, normal2, input.frameBlend);
		}

		normal -= 0.5;
		normal *= material.GetNormalMapStrength();
		normal *= opacity;
	}
#endif // EMITTEDPARTICLE_DISTORTION
	
	[branch]
	if (GetCamera().texture_lineardepth_index >= 0)
	{
		float4 depthScene = texture_lineardepth.GatherRed(sampler_linear_clamp, ScreenCoord) * GetCamera().z_far;
		float depthFragment = input.pos.w;
		opacity *= saturate(1.0 / input.size * (max(max(depthScene.x, depthScene.y), max(depthScene.z, depthScene.w)) - depthFragment));
	}

	opacity = saturate(opacity);

	// GG: the material's emissive map on the sprite (its colour times the emissive strength, added), else the strength as a gain;
	// the layer's emissive scale on either
	half3 ggEmissive = 0;
	[branch]
	if (material.textures[EMISSIVEMAP].IsValid())
	{
		half3 emissiveColor = material.textures[EMISSIVEMAP].Sample(sampler_linear_clamp, input.tex.xyxy).rgb;
		[branch]
		if (xEmitterOptions & EMITTER_OPTION_BIT_FRAME_BLENDING_ENABLED)
		{
			emissiveColor = lerp(emissiveColor, material.textures[EMISSIVEMAP].Sample(sampler_linear_clamp, input.tex.zwzw).rgb, input.frameBlend);
		}
		if (xParticleLocalOptions & 256) emissiveColor = (half3)ApplySRGBCurve(emissiveColor); // GG: as the colour sheet (bit 256)
		color.rgb *= inputColor.rgb;
		ggEmissive = emissiveColor * inputColor.rgb * material.GetEmissive() * xParticleEmissiveScale;
	}
	else
	{
		color.rgb *= inputColor.rgb * (1 + material.GetEmissive() * xParticleEmissiveScale);
	}
	color.a = opacity;

#ifdef EMITTEDPARTICLE_DISTORTION
	// just make normal maps blendable:
	color.rgb = color.rgb - 0.5;
#endif // EMITTEDPARTICLE_DISTORTION

#ifdef EMITTEDPARTICLE_LIGHTING

	// GG: a GameGuru layer takes the vertex shader's light, the centre's blended into the corners' towards the edges
	[branch]
	if (xEmitterFadeinTime >= 0)
	{
		const float edge = saturate(length(float2(input.unrotated_uv) - 0.5) * 2);
		color.rgb *= (half3)lerp(input.litCentre, input.litCorner, edge);
	}
	else if (color.a > 0)
	{
		float3 N;
		N.x = -cos(PI * input.unrotated_uv.x);
		N.y = cos(PI * input.unrotated_uv.y);
		N.z = -sin(PI * length(input.unrotated_uv));
		N.xz += normal.rg;
		N = mul((float3x3)GetCamera().inverse_view, N);
		N = normalize(N);
		
		float3 V = input.GetViewVector();
		float dist = length(V);
		V /= dist;

		Lighting lighting;
		lighting.create(0, 0, GetAmbient(N), 0);

		Surface surface;
		surface.init();
		surface.create(material, color, surfacemap_simple);
		surface.P = pos3D;
		surface.N = N;
		surface.V = V;
		surface.pixel = pixel;
		surface.sss = material.GetSSS();
		surface.sss_inv = material.GetSSSInverse();
		surface.extinction = 0;
		surface.update();

		TiledLighting(surface, lighting, GetFlatTileIndex(pixel));

		// the direct light divided by PI, as ApplyLighting does for objects: BRDF_GetDiffuse leaves it out, so without it a lit
		// particle took PI times the direct light an object does
		color.rgb *= lighting.direct.diffuse / PI + lighting.indirect.diffuse;
		color.rgb += lighting.indirect.specular;

		//color.rgb = float3(unrotated_uv, 0);
		//color.rgb = float3(input.tex, 0);

		color = max(0, color);
	}

#endif // EMITTEDPARTICLE_LIGHTING

	// GG: the emissive map's glow added after the light, so the light does not dim it (as on DX11)
#ifndef EMITTEDPARTICLE_DISTORTION
	color.rgb += ggEmissive;
#endif // EMITTEDPARTICLE_DISTORTION

	color.rgb = mul(saturationMatrix(material.GetSaturation()), color.rgb);

	// GG: the scene's fog on every layer, lit or not (upstream fogs only the lit ones), as the blend wants it: none for a
	// layer that skips it (option bit 4); a distortion bends less; an additive layer dims (fog colour added would brighten);
	// a premultiplied one fogs its colour by its opacity; the rest as objects have it
	[branch]
	if ((xParticleLocalOptions & 4) == 0)
	{
		float3 fogV = input.GetViewVector();
		const float fogDist = length(fogV);
		fogV /= max(fogDist, 0.0001);
		const half4 fog = GetFog(fogDist, GetCamera().position, -fogV);
#ifdef EMITTEDPARTICLE_DISTORTION
		color.rgb *= 1 - fog.a;
#else
		if (xParticleLocalOptions & 8) color.rgb *= 1 - fog.a;
		else if (xParticleLocalOptions & 16) color.rgb = lerp(color.rgb, fog.rgb * color.a, fog.a);
		else color.rgb = lerp(color.rgb, fog.rgb, fog.a);
#endif // EMITTEDPARTICLE_DISTORTION
	}

#ifdef EMITTEDPARTICLE_DISTORTION
	// GGMAX 2.47: restore the fork's positive-only distortion.
	//
	// The fork's version of this shader ended with `return max(color, 0)`. That matters only on
	// the distortion path: above, `color.rgb = color.rgb - 0.5` deliberately makes the distortion
	// vector signed, and the fork then clamped the negative half away - so a fork distortion
	// particle could only ever push the scene one way in UV. The current shader returns the
	// signed value into an R16G16_FLOAT target (wiRenderPath3D.cpp:195), where negatives survive,
	// and tonemapCS.hlsl does an unscaled `uv += distortion.rg`. Net effect: roughly double the
	// range, and bidirectional where the fork was one-directional.
	//
	// Scoped two ways rather than copying the fork's blanket clamp:
	//   * DISTORTION permutation only. Outside it every term feeding `color` is already >= 0, so
	//     the clamp is a no-op there - confining it keeps the soft / soft_lighting paths provably
	//     untouched instead of merely believed-unchanged.
	//   * Legacy .PE emitters only (xEmitterFadeinTime >= 0, same discriminator as GGMAX 2.46).
	//     wi::renderer draws the RAIN emitter with a SOFT_DISTORTION override
	//     (wiRenderer.cpp:7289), and rain is engine-created with fadein_time -1, so it keeps
	//     upstream's signed distortion.
	[branch]
	if (xEmitterFadeinTime >= 0)
	{
		color = max(color, 0);
	}
#endif // EMITTEDPARTICLE_DISTORTION

	return color;
}
