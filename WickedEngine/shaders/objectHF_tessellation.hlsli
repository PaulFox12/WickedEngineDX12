#include "objectHF.hlsli"

struct ConstantOutput
{
	float edges[3] : SV_TessFactor;
	float inside : SV_InsideTessFactor;

	float3 b210 : CONTROLPOINT0;
	float3 b120 : CONTROLPOINT1;
	float3 b021 : CONTROLPOINT2;
	float3 b012 : CONTROLPOINT3;
	float3 b102 : CONTROLPOINT4;
	float3 b201 : CONTROLPOINT5;
	float3 b111 : CENTER;
};

// GGMAX: GameGuru's terrain, tessellated (a GameGuru terrain material, ShaderTerrain gg_rings): the chunk's triangles cut
// finer near the camera (sub-triangles distance / gg_tess_k across, not below the finest ring's step) and each new vertex
// put at the height of GameGuru's height rings round the player, the same data its DX11 terrain draws and physics and
// GetTerrainHeight read; flat between the chunk's vertices otherwise, where Wicked's own path curves them (PN triangles).
// A shared edge is cut the same in both triangles (its length and midpoint), and every vertex takes one height, so no cracks
inline bool gg_terrain_tessellated()
{
	return GetMaterial().IsGGTerrain() && GetScene().terrain.gg_rings >= 0;
}

// the finest ring holding the point (a few texels in from its edge): its height there, and the normal from the heights a
// ring step either side; false outside every ring. Each ring's texel (column, row) is the vertex at x = left + column * step,
// z = far - row * step (GameGuru stores a ring's rows from its far z edge)
bool gg_ring_height(float2 p, out float height, out float3 normal)
{
	height = 0;
	normal = float3(0, 1, 0);
	const ShaderTerrain t = GetScene().terrain;
	Texture2DArray rings = bindless_textures2DArray[descriptor_index(t.gg_rings)];
	float res, res_h, slices;
	rings.GetDimensions(res, res_h, slices);
	float step = t.gg_ring_seg0;
	for (uint ring = 0; ring < t.gg_ring_count; ++ring, step *= 2)
	{
		const float4 pair = t.gg_ring_centers[ring >> 1];
		const float2 c = (ring & 1) ? pair.zw : pair.xy;
		const float half_size = res * 0.5 * step;
		const float tx = (p.x - (c.x - half_size)) / step;
		const float tz = (p.y - (c.y - half_size)) / step;
		if (tx < 2 || tx > res - 3 || tz < 3 || tz > res - 2)
			continue;
		const float2 uv = float2(tx + 0.5, res - tz + 0.5) / res;
		const float d = 1.0 / res;
		height = rings.SampleLevel(sampler_linear_clamp, float3(uv, ring), 0).r;
		const float hxm = rings.SampleLevel(sampler_linear_clamp, float3(uv - float2(d, 0), ring), 0).r;
		const float hxp = rings.SampleLevel(sampler_linear_clamp, float3(uv + float2(d, 0), ring), 0).r;
		const float hzp = rings.SampleLevel(sampler_linear_clamp, float3(uv - float2(0, d), ring), 0).r;
		const float hzm = rings.SampleLevel(sampler_linear_clamp, float3(uv + float2(0, d), ring), 0).r;
		normal = normalize(float3(hxm - hxp, 2 * step, hzm - hzp));
		return true;
	}
	return false;
}

#ifdef OBJECTSHADER_COMPILE_HS

ConstantOutput PatchConstantFunction(InputPatch<PixelInput, 3> patch)
{
	ConstantOutput output;

	// tessellation strength based on distance and orientation:
	const bool gg_terrain = gg_terrain_tessellated(); // GGMAX
	[unroll]
	for (uint ie = 0; ie < 3; ++ie)
	{
#if 0
		output.edges[ie] = GetMesh().tessellation_factor;
#else
		float3 edge = patch[(ie + 1) % 3].pos.xyz - patch[ie].pos.xyz;
		float3 vec = (patch[(ie + 1) % 3].pos.xyz + patch[ie].pos.xyz) / 2 - GetCamera().position.xyz;
		[branch]
		if (gg_terrain)
		{
			// GGMAX: sub-triangles distance / k across, not below the finest ring's step; the distance from the view's eye in
			// every pass, the shadow passes too, so they cut the chunk as the screen does
			const ShaderTerrain gg_t = GetScene().terrain;
			const float3 gg_vec = (patch[(ie + 1) % 3].pos.xyz + patch[ie].pos.xyz) / 2 - gg_t.gg_tess_eye;
			const float target = max(gg_t.gg_ring_seg0, length(gg_vec) / max(gg_t.gg_tess_k, 1.0));
			output.edges[(ie + 1) % 3] = clamp(length(edge) / target, 1, 64);
			continue;
		}
		float len = sqrt(dot(edge, edge) / dot(vec, vec));
		output.edges[(ie + 1) % 3] = max(1, len * GetMesh().tessellation_factor);
#endif
	}


#if 1
	// frustum culling:
	float rad = gg_terrain ? 200.0 : GetMaterial().GetDisplacement(); // GGMAX: the rings' heights stray from the chunk's triangle
#ifdef OBJECTSHADER_USE_CAMERAINDEX
	const ShaderCamera cull_camera = GetCameraIndexed(patch[0].GetCameraIndex()); // GGMAX: a shadow pass, the instance's cascade
#else
	const ShaderCamera cull_camera = GetCamera();
#endif // OBJECTSHADER_USE_CAMERAINDEX
	int culled[4];
	[unroll]
	for (int ip = 0; ip < 4; ++ip)
	{
		culled[ip] = 1;
		culled[ip] &= dot(cull_camera.frustum.planes[ip + 2], patch[0].pos) < -rad;
		culled[ip] &= dot(cull_camera.frustum.planes[ip + 2], patch[1].pos) < -rad;
		culled[ip] &= dot(cull_camera.frustum.planes[ip + 2], patch[2].pos) < -rad;
	}
	if (culled[0] || culled[1] || culled[2] || culled[3])
	{
		output.edges[0] = 0;
	}
#endif


	// compute the cubic geometry control points
	const float3 nor[] = {
		normalize(patch[0].nor),
		normalize(patch[1].nor),
		normalize(patch[2].nor),
	};
	output.b210 = ((2 * patch[0].pos.xyz) + patch[1].pos.xyz - (dot((patch[1].pos.xyz - patch[0].pos.xyz), nor[0]) * nor[0])) / 3.0;
	output.b120 = ((2 * patch[1].pos.xyz) + patch[0].pos.xyz - (dot((patch[0].pos.xyz - patch[1].pos.xyz), nor[1]) * nor[1])) / 3.0;
	output.b021 = ((2 * patch[1].pos.xyz) + patch[2].pos.xyz - (dot((patch[2].pos.xyz - patch[1].pos.xyz), nor[1]) * nor[1])) / 3.0;
	output.b012 = ((2 * patch[2].pos.xyz) + patch[1].pos.xyz - (dot((patch[1].pos.xyz - patch[2].pos.xyz), nor[2]) * nor[2])) / 3.0;
	output.b102 = ((2 * patch[2].pos.xyz) + patch[0].pos.xyz - (dot((patch[0].pos.xyz - patch[2].pos.xyz), nor[2]) * nor[2])) / 3.0;
	output.b201 = ((2 * patch[0].pos.xyz) + patch[2].pos.xyz - (dot((patch[2].pos.xyz - patch[0].pos.xyz), nor[0]) * nor[0])) / 3.0;
	float3 E = (output.b210 + output.b120 + output.b021 + output.b012 + output.b102 + output.b201) / 6.0;
	float3 V = (patch[0].pos.xyz + patch[1].pos.xyz + patch[2].pos.xyz) / 3.0;
	output.b111 = E + ((E - V) / 2.0);


	output.inside = (output.edges[0] + output.edges[1] + output.edges[2]) / 3.0;

	return output;
}

[domain("tri")]
[partitioning("fractional_odd")]
//[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("PatchConstantFunction")]
PixelInput main(InputPatch<PixelInput, 3> patch, uint pointID : SV_OutputControlPointID, uint patchId : SV_PrimitiveID)
{
	return patch[pointID];
}
#endif // OBJECTSHADER_COMPILE_HS



#ifdef OBJECTSHADER_COMPILE_DS

[domain("tri")]
PixelInput main(ConstantOutput input, float3 uvw : SV_DomainLocation, const OutputPatch<PixelInput, 3> patch)
{
	PixelInput output = patch[0];

	const float u = uvw.x;
	const float v = uvw.y;
	const float w = uvw.z;
	const float uu = u * u;
	const float vv = v * v;
	const float ww = w * w;
	const float uu3 = uu * 3;
	const float vv3 = vv * 3;
	const float ww3 = ww * 3;

	// compute position from cubic control points and barycentric coords:
	output.pos.xyz = patch[0].pos.xyz * ww * w + patch[1].pos.xyz * uu * u + patch[2].pos.xyz * vv * v +
		input.b210 * ww3 * u + input.b120 * w * uu3 + input.b201 * ww3 * v + input.b021 * uu3 * v +
		input.b102 * w * vv3 + input.b012 * u * vv3 + input.b111 * 6.0 * w * u * v;

#ifdef OBJECTSHADER_USE_UVSETS
	output.uvsets = w * patch[0].uvsets + u * patch[1].uvsets + v * patch[2].uvsets;
#endif // OBJECTSHADER_USE_UVSETS

#ifdef OBJECTSHADER_USE_COLOR
	output.color = half4(w * patch[0].color + u * patch[1].color + v * patch[2].color);
#endif // OBJECTSHADER_USE_COLOR

#ifdef OBJECTSHADER_USE_COMMON
	output.atl = w * patch[0].atl + u * patch[1].atl + v * patch[2].atl;
	output.ao_wet = w * patch[0].ao_wet + u * patch[1].ao_wet + v * patch[2].ao_wet;
#endif // OBJECTSHADER_USE_COMMON

#ifdef OBJECTSHADER_USE_NORMAL
	output.nor = normalize(w * patch[0].nor + u * patch[1].nor + v * patch[2].nor);
#endif // OBJECTSHADER_USE_NORMAL

#ifdef OBJECTSHADER_USE_TANGENT
	output.tan = half4(normalize(w * patch[0].tan + u * patch[1].tan + v * patch[2].tan));
#endif // OBJECTSHADER_USE_TANGENT


#ifdef OBJECTSHADER_USE_NORMAL
#ifdef OBJECTSHADER_USE_UVSETS
	// displacement mapping:
	[branch]
	if (GetMaterial().GetDisplacement() > 0 && GetMaterial().textures[DISPLACEMENTMAP].IsValid())
	{
		float displacement = GetMaterial().textures[DISPLACEMENTMAP].SampleLevel(sampler_objectshader, output.uvsets, 0).r;
		displacement *= GetMaterial().GetDisplacement();
		output.pos.xyz += normalize(float3(output.nor)) * displacement;
	}
#endif // OBJECTSHADER_USE_UVSETS
#endif // OBJECTSHADER_USE_NORMAL


	// GGMAX: GameGuru's terrain: flat across the chunk's triangle, then the height (and normal) of its height rings
	[branch]
	if (gg_terrain_tessellated())
	{
		output.pos.xyz = patch[0].pos.xyz * w + patch[1].pos.xyz * u + patch[2].pos.xyz * v;
		float gg_height;
		float3 gg_normal;
		if (gg_ring_height(output.pos.xz, gg_height, gg_normal))
		{
			output.pos.y = gg_height;
#ifdef OBJECTSHADER_USE_NORMAL
			output.nor = gg_normal;
#endif // OBJECTSHADER_USE_NORMAL
		}
	}

#ifdef OBJECTSHADER_USE_CLIPPLANE
	output.clip = dot(output.pos, GetCamera().clip_plane);
#endif // OBJECTSHADER_USE_CLIPPLANE

#ifndef OBJECTSHADER_USE_NOCAMERA
	output.pos = mul(GetCamera().view_projection, output.pos);
#endif // OBJECTSHADER_USE_NOCAMERA

	return output;
}

#endif // OBJECTSHADER_COMPILE_DS
