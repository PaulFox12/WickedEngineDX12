#ifndef GG_TERRAIN_PAINT_HF
#define GG_TERRAIN_PAINT_HF
// GGMAX: GameGuru's painted terrain textures, for the terrain tile shaders (terrainVirtualTextureUpdateCS.hlsl,
// terrainBakeChunkCS.hlsl). GameGuru's paint map holds a value a cell (0 unpainted, else the painted texture + 1), about
// 77 units a cell on an 8 km map. The per-vertex blendmap (80 units a vertex) drew each painted cell as a soft square,
// a diagonal stroke as a staircase. Here a texel blends the paint map's 4 x 4 cells round it as GameGuru's DX11 page
// shader does (GGTerrainPageGenPS.hlsl, painted edges): each cell's weight bilinear, eased into a cubic B-spline by the
// smoothing, so a stroke's edge follows a smooth line through the cells; a texture's share the sum over its cells, the
// four largest kept; moved by the breakup, for an uneven edge; sharpened by the softness. All faded out on texels a cell
// or more across, which keep the plain bilinear blend. The share of unpainted cells goes to the blendmap's layers (the
// height and slope textures), which GameGuru writes without the paint.
//
// The block GameGuru's settings arrive in follows the blendmap's material table in blendmap_buffer
// (wi::terrain::gg_WritePaintBlock), GG_TERRAIN_PAINT_BLOCK uints:
//	0 flags (bit 0 on), 1 paint map descriptor, 2 paint map resolution, 3 world to paint map scale (uv = world * scale + 0.5),
//	4-5 the tile's world x and z at uv 0.5, 6 the world size of uv 0 to 1, 7 smoothing, 8 softness, 9 breakup,
//	10-41 the material index for paint values 1-32 (0xFFFFFFFF none: the cell counts as unpainted)

struct GGTerrainPaint
{
	uint material[4];	// ShaderMaterial index of each painted texture taking part
	float weight[4];
	float auto_share;	// the blendmap layers' share
};

float gg_EdgeHash(float2 p)
{
	float3 p3 = frac(p.xyx * 0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return frac((p3.x + p3.y) * p3.z);
}

float gg_EdgeNoise(float2 p)
{
	float2 cell = floor(p);
	float2 f = frac(p);
	f = f * f * (3 - 2 * f);
	float a = gg_EdgeHash(cell);
	float b = gg_EdgeHash(cell + float2(1, 0));
	float c = gg_EdgeHash(cell + float2(0, 1));
	float d = gg_EdgeHash(cell + float2(1, 1));
	return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y) * 2 - 1;
}

// a uniform cubic B-spline's four weights at f (0-1, between the second and third of four samples)
float4 gg_BSplineWeights(float f)
{
	float f2 = f * f;
	float f3 = f2 * f;
	float g = 1 - f;
	return float4(g * g * g, 3 * f3 - 6 * f2 + 4, -3 * f3 + 3 * f2 + 3 * f + 1, f3) / 6;
}

// uv: the tile texel's uv in the chunk (0-1); texel_uv: a texel's size in uv
GGTerrainPaint gg_terrain_paint(ByteAddressBuffer buf, uint block, float2 uv, float texel_uv)
{
	GGTerrainPaint o;
	[unroll]
	for (uint z = 0; z < 4; z++)
	{
		o.material[z] = 0;
		o.weight[z] = 0;
	}
	o.auto_share = 1;
	if ((buf.Load(block) & 1u) == 0)
		return o;

	Texture2D<float> paintmap = bindless_textures_float[descriptor_index(buf.Load(block + 4))];
	const int map_res = (int)buf.Load(block + 8);
	const float map_scale = asfloat(buf.Load(block + 12));
	const float2 origin = asfloat(buf.Load2(block + 16));
	const float size = asfloat(buf.Load(block + 24));
	const float edge_smoothing = asfloat(buf.Load(block + 28));
	const float edge_softness = asfloat(buf.Load(block + 32));
	const float edge_breakup = asfloat(buf.Load(block + 36));

	const float2 world = origin + (uv - 0.5) * size;
	float2 uvMat = world * map_scale + 0.5;
	const float2 matCell = uvMat * map_res;
	const float footprint = size * texel_uv * map_scale * map_res; // the texel's size in cells
	const float edgeFade = saturate(2 - footprint * 2);
	const float breakup = edge_breakup * edgeFade;
	const float smoothing = edge_smoothing * edgeFade;
	if (breakup > 0)
	{
		float2 warp = float2(gg_EdgeNoise(matCell * 0.7), gg_EdgeNoise(matCell * 0.7 + 31.7));
		uvMat += warp * (breakup * 0.35 / map_res);
	}

	float2 p = uvMat * map_res - 0.5;
	float2 cellBase = floor(p);
	float2 f = p - cellBase;
	float4 wx = lerp(float4(0, 1 - f.x, f.x, 0), gg_BSplineWeights(f.x), smoothing);
	float4 wy = lerp(float4(0, 1 - f.y, f.y, 0), gg_BSplineWeights(f.y), smoothing);

	// each cell's paint value, 0 where unpainted, off the map or a value with no texture
	uint cellMat[16];
	float cellWeight[16];
	bool alike = true;
	[unroll]
	for (uint cy = 0; cy < 4; cy++)
	{
		[unroll]
		for (uint cx = 0; cx < 4; cx++)
		{
			int2 cellPos = int2(cellBase) + int2(cx, cy) - 1;
			uint value = 0;
			if (all(cellPos >= 0) && all(cellPos < map_res))
			{
				value = (uint)(paintmap.Load(int3(cellPos, 0)) * 255 + 0.5);
				if (value > 32 || (value > 0 && buf.Load(block + (9 + value) * 4) == 0xFFFFFFFF))
					value = 0;
			}
			cellMat[cy * 4 + cx] = value;
			cellWeight[cy * 4 + cx] = wx[cx] * wy[cy];
			alike = alike && value == cellMat[0];
		}
	}
	if (alike)
	{
		if (cellMat[0] > 0)
		{
			o.material[0] = buf.Load(block + (9 + cellMat[0]) * 4);
			o.weight[0] = 1;
			o.auto_share = 0;
		}
		return o;
	}

	float share[16];
	[loop]
	for (uint a = 0; a < 16; a++)
	{
		share[a] = 0;
		for (uint b = 0; b < 16; b++) if (cellMat[b] == cellMat[a]) share[a] += cellWeight[b];
	}
	uint blendMat[4];
	float4 pickShare = float4(0, 0, 0, 0);
	[unroll]
	for (uint k = 0; k < 4; k++)
	{
		blendMat[k] = 0;
		[loop]
		for (uint c = 0; c < 16; c++)
		{
			bool taken = false;
			for (uint q = 0; q < k; q++) if (pickShare[q] > 0 && blendMat[q] == cellMat[c]) taken = true;
			if (!taken && share[c] > pickShare[k])
			{
				pickShare[k] = share[c];
				blendMat[k] = cellMat[c];
			}
		}
	}

	// moved by the breakup, each texture by a noise of its own, and sharpened by the softness (1 the shares as they are, a
	// blend over a few metres with the smoothing; 0 a crisp line); neither smoothing nor breakup: the plain bilinear blend
	const bool shaped = smoothing > 0 || breakup > 0;
	const float sharpness = shaped ? 1 + 11 * (1 - edge_softness) * edgeFade : 1;
	float4 blendWeight = float4(0, 0, 0, 0);
	[unroll]
	for (uint m = 0; m < 4; m++)
	{
		float pushed = pickShare[m];
		if (pushed > 0 && breakup > 0)
		{
			float2 np = matCell * 1.3 + blendMat[m] * 7.31;
			float n = gg_EdgeNoise(np) * 0.7 + gg_EdgeNoise(np * 2.8 + 11.1) * 0.3;
			pushed = saturate(pushed + n * 0.3 * breakup);
		}
		blendWeight[m] = pushed > 0 ? pow(pushed, sharpness) : 0;
	}
	float total = dot(blendWeight, float4(1, 1, 1, 1));
	if (total > 0.0001) blendWeight /= total;
	else blendWeight = pickShare / max(dot(pickShare, float4(1, 1, 1, 1)), 0.0001);

	// a texture with almost nothing left isn't sampled
	[unroll]
	for (uint t = 0; t < 4; t++)
	{
		if (blendWeight[t] < 0.002)
			blendWeight[t] = 0;
	}
	blendWeight /= max(dot(blendWeight, float4(1, 1, 1, 1)), 0.0001);

	o.auto_share = 0;
	[unroll]
	for (uint r = 0; r < 4; r++)
	{
		if (blendWeight[r] == 0)
			continue;
		if (blendMat[r] == 0)
		{
			o.auto_share += blendWeight[r];
			continue;
		}
		o.material[r] = buf.Load(block + (9 + blendMat[r]) * 4);
		o.weight[r] = blendWeight[r];
	}
	return o;
}

#endif // GG_TERRAIN_PAINT_HF
