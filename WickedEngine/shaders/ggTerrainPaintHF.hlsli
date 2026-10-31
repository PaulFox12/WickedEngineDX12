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
//	10-41 the material index for paint values 1-32 (0xFFFFFFFF none: the cell counts as unpainted),
//	42 tiling flags (bit 0 on), 43 the rotation mask's descriptor, 44 mask texels per texture repeat, 45 the blendmap layers
//	turned right round (a bit per layer)
//
// Tiling (gg_terrain_sample): GameGuru's DX11 page shader turns each terrain texture sample by a 1024 x 1024 fractal mask
// (values 0-31) laid over the texture's repeats, a little (up to 9 degrees) or right round (32 steps; painted textures and
// the layers flagged to rotate), blending the four cells round the point over the middle 40% of each border, so the
// texture's repeats never line up into a visible grid. Done the same here: the mask and the turns are the level's, not the
// chunk's (each chunk's offset taken in), each cell turned round its own centre so neighbouring chunks agree, and the
// normal map's direction turned back with it.

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

	// each cell's paint value, 0 where unpainted or off the map: the 4 x 4 cells in four gathers of 2 x 2 (a gather's
	// w, z, x, y are its top left, top right, bottom left, bottom right texels), all in registers (the loops unrolled)
	const int2 base = int2(cellBase) - 1;
	uint cellMat[16];
	float cellWeight[16];
	[unroll]
	for (uint gy = 0; gy < 2; gy++)
	{
		[unroll]
		for (uint gx = 0; gx < 2; gx++)
		{
			const float4 g = paintmap.GatherRed(sampler_point_clamp, (float2(base + int2(gx * 2 + 1, gy * 2 + 1))) / map_res);
			const uint4 v = (uint4)(g * 255 + 0.5);
			const uint i = gy * 8 + gx * 2;
			cellMat[i] = v.w;
			cellMat[i + 1] = v.z;
			cellMat[i + 4] = v.x;
			cellMat[i + 5] = v.y;
		}
	}
	bool alike = true;
	[unroll]
	for (uint cy = 0; cy < 4; cy++)
	{
		[unroll]
		for (uint cx = 0; cx < 4; cx++)
		{
			const int2 cellPos = base + int2(cx, cy);
			if (any(cellPos < 0) || any(cellPos >= map_res) || cellMat[cy * 4 + cx] > 32)
				cellMat[cy * 4 + cx] = 0;
			cellWeight[cy * 4 + cx] = wx[cx] * wy[cy];
			alike = alike && cellMat[cy * 4 + cx] == cellMat[0];
		}
	}
	if (alike)
	{
		const uint mi = cellMat[0] > 0 ? buf.Load(block + (9 + cellMat[0]) * 4) : 0xFFFFFFFF;
		if (mi != 0xFFFFFFFF)
		{
			o.material[0] = mi;
			o.weight[0] = 1;
			o.auto_share = 0;
		}
		return o;
	}

	float share[16];
	[unroll]
	for (uint a = 0; a < 16; a++)
	{
		float s = 0;
		[unroll]
		for (uint b = 0; b < 16; b++) s += cellMat[b] == cellMat[a] ? cellWeight[b] : 0;
		share[a] = s;
	}
	uint blendMat[4];
	float4 pickShare = float4(0, 0, 0, 0);
	[unroll]
	for (uint k = 0; k < 4; k++)
	{
		blendMat[k] = 0;
		[unroll]
		for (uint c = 0; c < 16; c++)
		{
			bool taken = false;
			[unroll]
			for (uint q = 0; q < k; q++) taken = taken || (pickShare[q] > 0 && blendMat[q] == cellMat[c]);
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
		const uint mi = blendMat[r] > 0 ? buf.Load(block + (9 + blendMat[r]) * 4) : 0xFFFFFFFF;
		if (mi == 0xFFFFFFFF) // unpainted, or a value with no texture
		{
			o.auto_share += blendWeight[r];
			continue;
		}
		o.material[r] = mi;
		o.weight[r] = blendWeight[r];
	}
	return o;
}

struct GGTiling
{
	uint on;
	uint maskmap;
	float cells;		// mask texels per texture repeat
	uint rotate_layers;
	float2 origin;
	float size;
};

GGTiling gg_terrain_tiling(ByteAddressBuffer buf, uint block)
{
	GGTiling t;
	t.on = buf.Load(block + 42 * 4) & 1u;
	t.maskmap = buf.Load(block + 43 * 4);
	t.cells = asfloat(buf.Load(block + 44 * 4));
	t.rotate_layers = buf.Load(block + 45 * 4);
	t.origin = asfloat(buf.Load2(block + 16));
	t.size = asfloat(buf.Load(block + 24));
	return t;
}

// a terrain texture at the tile texel uv, ts = (repeats across the chunk, mip) from gg_tile_uv_scale_lod: as stock (uv2 * repeats)
// with the tiling off, else turned by the rotation mask; full: right round; normalmap: rg is a normal's direction (0.5 flat)
half4 gg_terrain_sample(Texture2D tex, GGTiling t, float2 uv, float3 ts, bool full, bool normalmap)
{
	const float2 s = float2(uv.x, 1 - uv.y) * ts.xy;
	[branch]
	if (t.on == 0 || t.cells <= 0 || t.size <= 0)
		return tex.SampleLevel(sampler_linear_wrap, s, ts.z);

	// the chunk's offset in the level's texture coordinates (u along x, v against z): s plus it is the level's coordinate;
	// its whole part is left out of the sampling (the texture wraps) and out of the mask (it wraps at 1024)
	const float2 n = float2(t.origin.x / t.size - 0.5, -t.origin.y / t.size - 0.5) * ts.xy;
	const float2 ul = s + frac(n);
	const float2 nk = n * t.cells;
	const float2 mT = nk - 1024 * floor(nk / 1024) + s * t.cells - 0.5; // mask texel coordinate, a cell's centre at its whole number
	const float2 c0 = floor(mT);
	const float2 f = smoothstep(0.3, 0.7, mT - c0);
	Texture2D<float> mask = bindless_textures_float[descriptor_index(t.maskmap)];

	half4 acc = 0;
	[unroll]
	for (uint q = 0; q < 4; q++)
	{
		const float2 d = float2(q & 1, q >> 1);
		const float w = (d.x > 0 ? f.x : 1 - f.x) * (d.y > 0 ? f.y : 1 - f.y);
		[branch]
		if (w <= 0)
			continue;
		const float m = floor(mask.Load(int3(int2(c0 + d) & 1023, 0)) * 255 + 0.5);
		const float a = full ? m * 0.19634954 : (m - 16) * 0.0098174770; // DX11's terrain_maskRotMat
		float sn, cs;
		sincos(a, sn, cs);
		const float2 rel = (mT - c0 - d) / t.cells; // from the cell's centre, in texture repeats
		const float2 r = ul - rel + float2(cs * rel.x - sn * rel.y, sn * rel.x + cs * rel.y);
		half4 v = tex.SampleLevel(sampler_linear_wrap, r, ts.z);
		[branch]
		if (normalmap)
		{
			const float2 nn = v.rg * 2 - 1;
			v.rg = half2(float2(nn.x * cs + nn.y * sn, -nn.x * sn + nn.y * cs) * 0.5 + 0.5);
		}
		acc += (half)w * v;
	}
	return acc;
}

#endif // GG_TERRAIN_PAINT_HF
