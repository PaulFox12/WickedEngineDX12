#ifndef GG_TERRAIN_ROCK_HF
#define GG_TERRAIN_ROCK_HF
// GGMAX: GameGuru's Steep Rock on the terrain's chunks (objectHF.hlsli, SHADERMATERIAL_OPTION_BIT_GG_TERRAIN), as its DX11
// terrain's final pass (GGTerrainRockHF.hlsli there): a terrain texture laid on steep slopes in world space from three
// sides. The terrain's virtual texture is made from above, so on a steep face each texel covers its horizontal footprint
// and the texture is stretched down the face; this samples the rock material's textures where the face is instead and
// blends them over the surface, its edge broken up by a noise and the texture at two sizes in patches. The settings are in
// ShaderTerrain (gg_rock_*), the material GameGuru's rock texture slot's

float gg_RockHash(float2 p)
{
	float3 p3 = frac(p.xyx * 0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return frac((p3.x + p3.y) * p3.z);
}

// value noise, -1 to 1, smooth at a scale of 1
float gg_RockNoise(float2 p)
{
	float2 cell = floor(p);
	float2 f = frac(p);
	f = f * f * (3 - 2 * f);
	float a = gg_RockHash(cell);
	float b = gg_RockHash(cell + float2(1, 0));
	float c = gg_RockHash(cell + float2(0, 1));
	float d = gg_RockHash(cell + float2(1, 1));
	return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y) * 2 - 1;
}

struct GGRockSurface
{
	float3 color;
	float3 normal;
	float occlusion;
	float roughness;
	float metalness;
};

// one side's sample, by its weight: its UV, and the world directions of its U and V (the normal map's X and Y)
void gg_AddRockSide(inout GGRockSurface rock, ShaderMaterial m, SamplerState samp, float weight, float2 uv, float2 uvDX, float2 uvDY, float3 dirU, float3 dirV, float3 N)
{
	float4 color = 1;
	[branch]
	if (m.textures[BASECOLORMAP].IsValid())
		color = bindless_textures[descriptor_index(m.textures[BASECOLORMAP].texture_descriptor)].SampleGrad(samp, uv, uvDX, uvDY);
	rock.color += color.rgb * weight;

	[branch]
	if (m.textures[SURFACEMAP].IsValid())
	{
		const float4 surface = bindless_textures[descriptor_index(m.textures[SURFACEMAP].texture_descriptor)].SampleGrad(samp, uv, uvDX, uvDY);
		rock.occlusion += surface.r * weight;
		rock.roughness += surface.g * weight;
		rock.metalness += surface.b * weight;
	}
	else
	{
		rock.occlusion += weight;
		rock.roughness += 0.8 * weight;
	}

	float3 tangentNormal = float3(0, 0, 1);
	[branch]
	if (m.textures[NORMALMAP].IsValid())
	{
		tangentNormal.xy = bindless_textures[descriptor_index(m.textures[NORMALMAP].texture_descriptor)].SampleGrad(samp, uv, uvDX, uvDY).rg * 2 - 1;
		tangentNormal.z = sqrt(saturate(1 - dot(tangentNormal.xy, tangentNormal.xy)));
	}

	// U and V laid along the surface, so the normal map tilts the surface's own normal
	dirU = normalize(dirU - N * dot(N, dirU));
	dirV = normalize(dirV - N * dot(N, dirV));
	rock.normal += normalize(dirU * tangentNormal.x + dirV * tangentNormal.y + N * tangentNormal.z) * weight;
}

// the rock over the surface where the terrain is steep: geometricNormal the mesh's normal, posDX and posDY the world
// position's derivatives (taken outside any branch); N, baseColor and surfaceMap (occlusion, roughness, metalness) as the
// terrain's virtual texture gave them, before the surface is made from them
void gg_terrain_rock(ShaderTerrain t, SamplerState samp, float3 worldPos, float3 geometricNormal, float3 posDX, float3 posDY, inout float3 N_surface, inout half4 baseColor, inout half4 surfaceMap)
{
	const float3 N = normalize(geometricNormal);
	float slope = 1 - abs(N.y);

	// the edge broken up: the slope moved by a noise in world space (about 8 m and 3 m across, the height mixed in so it
	// varies down a face too), so the rock runs unevenly into the ground above and below a face rather than along a line;
	// skipped where even the largest noise can't lift the slope past where the rock starts
	[branch]
	if (t.gg_rock_breakup > 0 && slope + 0.2 * t.gg_rock_breakup > t.gg_rock_start)
	{
		const float2 np = float2(worldPos.x + worldPos.y * 0.6, worldPos.z - worldPos.y * 0.4) / 315.0;
		const float n = gg_RockNoise(np) * 0.7 + gg_RockNoise(np * 2.7 + 17.3) * 0.3;
		slope += n * 0.2 * t.gg_rock_breakup;
	}
	slope = saturate((slope - t.gg_rock_start) * t.gg_rock_transition);
	slope = slope * slope * (3 - 2 * slope);

	[branch]
	if (slope <= 0)
		return;

	const ShaderMaterial m = load_material(t.gg_rock_material);

	// three sides by the normal, sharpened so a face takes mostly one; U along +X and V along -Z from above (as the
	// virtual texture), down the face on the sides so a texture stands upright
	float3 weights = pow(abs(N), 4);
	weights /= weights.x + weights.y + weights.z;

	// the texture at two sizes, in patches by a noise about two tiles across, so its repeats don't line up across a big
	// face (0 the one size); where a patch is all one size the other isn't sampled
	float variation = 0;
	if (t.gg_rock_variation > 0)
	{
		const float2 vp = float2(worldPos.x + worldPos.y * 0.5, worldPos.z - worldPos.y * 0.5) * t.gg_rock_scale * 0.45 + 41.7;
		variation = smoothstep(0.35, 0.65, gg_RockNoise(vp) * 0.5 + 0.5) * t.gg_rock_variation;
	}
	GGRockSurface rock = (GGRockSurface)0;
	float sampled = 0;
	[unroll]
	for (int size = 0; size < 2; size++)
	{
		const float share = size == 0 ? 1 - variation : variation;
		[branch]
		if (share > 0.01)
		{
			const float scale = size == 0 ? t.gg_rock_scale : t.gg_rock_scale * 0.43;
			const float2 shift = size == 0 ? float2(0, 0) : float2(0.37, 0.71);
			[branch] if (weights.x > 0.01) gg_AddRockSide(rock, m, samp, weights.x * share, float2(worldPos.z, -worldPos.y) * scale + shift, float2(posDX.z, -posDX.y) * scale, float2(posDY.z, -posDY.y) * scale, float3(0, 0, 1), float3(0, -1, 0), N);
			[branch] if (weights.y > 0.01) gg_AddRockSide(rock, m, samp, weights.y * share, float2(worldPos.x, -worldPos.z) * scale + shift, float2(posDX.x, -posDX.z) * scale, float2(posDY.x, -posDY.z) * scale, float3(1, 0, 0), float3(0, 0, -1), N);
			[branch] if (weights.z > 0.01) gg_AddRockSide(rock, m, samp, weights.z * share, float2(worldPos.x, -worldPos.y) * scale + shift, float2(posDX.x, -posDX.y) * scale, float2(posDY.x, -posDY.y) * scale, float3(1, 0, 0), float3(0, -1, 0), N);
			sampled += ((weights.x > 0.01 ? weights.x : 0) + (weights.y > 0.01 ? weights.y : 0) + (weights.z > 0.01 ? weights.z : 0)) * share;
		}
	}
	rock.color /= sampled;
	rock.occlusion /= sampled;
	rock.roughness /= sampled;
	rock.metalness /= sampled;
	float3 rockNormal = normalize(rock.normal);
	rockNormal = normalize(lerp(N, rockNormal, t.gg_bumpiness));

	// where it starts, the rock's raised parts (light in its occlusion) show first and its cracks fill last; the edge
	// breakup widens the blend, so the ground thins over the rock rather than stopping
	float rockWeight = slope + rock.occlusion * 0.5;
	float groundWeight = (1 - slope) + 0.25;
	const float top = max(rockWeight, groundWeight) - (0.2 + 0.3 * t.gg_rock_breakup);
	rockWeight = max(rockWeight - top, 0);
	groundWeight = max(groundWeight - top, 0);
	const float amount = rockWeight / (rockWeight + groundWeight) * t.gg_rock_strength;

	baseColor.rgb = (half3)lerp((float3)baseColor.rgb, rock.color, amount);
	surfaceMap.r = (half)lerp(surfaceMap.r, rock.occlusion, amount);
	surfaceMap.g = (half)lerp(surfaceMap.g, rock.roughness, amount);
	surfaceMap.b = (half)lerp(surfaceMap.b, rock.metalness, amount);
	N_surface = normalize(lerp(N_surface, rockNormal, amount));
}

#endif // GG_TERRAIN_ROCK_HF
