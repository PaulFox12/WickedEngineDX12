#include "globals.hlsli"
#include "objectHF.hlsli"
#include "hairparticleHF.hlsli"
#include "ShaderInterop_HairParticle.h"

#ifdef __PSSL__
#pragma PSSL_target_output_format (target 0 FMT_32_R)
#endif // __PSSL__

uint main(VertexToPixel input, out uint coverage : SV_Coverage) : SV_Target
{
	ShaderMaterial material = HairGetMaterial();

	half alpha = 1;

	// GGMAX 1.74 merged grass: alpha must come from the SAME texture the lit pass will use, or
	// the depth silhouette is cut from the wrong sprite.
	// GGMAX 1.96 flicker fix: NonUniform annotation AT the subscript, one expression (see
	// GGGetGrassTextureIndex — the out-param Texture2D version dropped it in the DXIL).
	uint ggtexidx;
	[branch]
	if (input.GGGetGrassTextureIndex(ggtexidx))
	{
		alpha = bindless_textures_half4[NonUniformResourceIndex(descriptor_index(ggtexidx))].Sample(sampler_linear_clamp, input.tex.xy).a;
	}
	else
	[branch]
	if (material.textures[BASECOLORMAP].IsValid())
	{
		alpha = material.textures[BASECOLORMAP].Sample(sampler_linear_clamp, input.tex.xyxy).a;
	}
	
	// GG grass: the blade is solid where its alpha passes the cutout, with a one-pixel edge, as DX11's grass cut its cards
	// (GGGrassPrepassPS). AlphaToCoverage takes the raw alpha as the MSAA sample mask and dithers it under TAA, so every
	// blade's soft parts drew half see-through and the water and the sky showed through the grass in front of them.
	// In full precision and clamped to 1: a magnified blade's edge is steep, and AlphaToCoverage's sample mask shift wraps
	// for an alpha past 8 (only its low five bits count), which left a close blade's middle uncovered
	const float alphaF = alpha;
	const float alphaWidth = max(fwidth(alphaF), 0.0001);
	if (input.grasstype != 0)
	{
		const float cut = material.GetAlphaTest() > 0 ? material.GetAlphaTest() : 0.5;
		coverage = AlphaToCoverage((half)saturate((alphaF - cut) / alphaWidth + 0.5), 0.5, input.GetDither(), input.pos);
	}
	else
	{
		coverage = AlphaToCoverage(alpha, material.GetAlphaTest(), input.GetDither(), input.pos);
	}

	PrimitiveID prim;
	prim.init();
	prim.primitiveIndex = input.primitiveID;
	prim.instanceIndex = xHairInstanceIndex;
	prim.subsetIndex = 0;
	return prim.pack();
}
