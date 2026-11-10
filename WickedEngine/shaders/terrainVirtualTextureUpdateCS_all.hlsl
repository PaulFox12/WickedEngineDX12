// GGMAX: the terrain's tile bake in one pass, colour, normal, surface and emissive written together. The four-pass bake
// (terrainVirtualTextureUpdateCS.hlsl, one map a dispatch) decoded the paint map, sampled every blendmap layer, loaded each
// material and decoded the road markings once per map: four times a texel. Here they are done once and only the textures
// differ; a map's output is its descriptor in the paint block's slots 48 to 50 (normal, surface, emissive; -1 none), the
// colour's push.output_texture. The blend is the four-pass one per map, splat or the stock front-to-back stack.
#define UPDATE_ALL
#include "terrainVirtualTextureUpdateCS.hlsl"

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	if (DTid.x >= push.write_size || DTid.y >= push.write_size)
		return;

	Texture2DArray blendmap = bindless_textures2DArray[descriptor_index(push.blendmap_texture)];
	ByteAddressBuffer blendmap_buffer = bindless_buffers[descriptor_index(push.blendmap_buffer)];
	const uint gg_block = push.blendmap_buffer_offset + push.blendmap_layers * sizeof(uint);
	const GGTiling gg_tiling = gg_terrain_tiling(blendmap_buffer, gg_block);
	const int out_normal = asint(blendmap_buffer.Load(gg_block + 48 * 4));
	const int out_surface = asint(blendmap_buffer.Load(gg_block + 49 * 4));
	const int out_emissive = asint(blendmap_buffer.Load(gg_block + 50 * 4));
	const bool want_emissive = out_emissive >= 0;
	const bool gg_splat = (push.gg_tile_share & 1u) != 0;

	float3 base_rgb[16];
	float normal_x[16];
	float normal_y[16];
	float3 surface_rgb[16];
	float surface_a[16];
	float3 emissive_rgb[16];

	for (uint idx = 0; idx < 16; ++idx)
	{
		const uint2 block_offset = block_offsets[idx];
		const int2 pixel = push.offset + DTid.xy * 4 + block_offset;
		const float2 uv = (pixel.xy + 0.5f) * push.resolution_rcp;

		half4 total_base = 0, total_surface = 0, total_emissive = 0;
		half2 total_normal = 0;
		half acc_base = 0, acc_normal = 0, acc_surface = 0, acc_emissive = 0;
		bool done_base = false, done_normal = false, done_surface = false, done_emissive = !want_emissive;
		float splat_weight = 0;

		const GGTerrainPaint gg_paint = gg_terrain_paint(blendmap_buffer, gg_block, uv, push.resolution_rcp);

		// front to back, as the four-pass loop; in the stock stack each map stops where its own coverage is full
		for (int blendmap_index = push.blendmap_layers + 3; blendmap_index >= 0; blendmap_index--)
		{
			float weight;
			uint materialIndex;
			bool gg_full = true;
			if (blendmap_index >= (int)push.blendmap_layers)
			{
				weight = gg_paint.weight[blendmap_index - push.blendmap_layers];
				if (weight == 0)
					continue;
				materialIndex = gg_paint.material[blendmap_index - push.blendmap_layers];
			}
			else
			{
				weight = blendmap.SampleLevel(sampler_linear_clamp, float3(uv, blendmap_index), 0).r * gg_paint.auto_share;
				if (weight == 0)
					continue;
				materialIndex = blendmap_buffer.Load(push.blendmap_buffer_offset + blendmap_index * sizeof(uint));
				gg_full = ((gg_tiling.rotate_layers >> blendmap_index) & 1u) != 0;
			}
			ShaderMaterial material = load_material(materialIndex);

			// colour
			[branch]
			if (!done_base)
			{
				float4 baseColor = material.GetBaseColor();
				[branch]
				if (material.textures[BASECOLORMAP].IsValid())
				{
					Texture2D tex = bindless_textures[descriptor_index(material.textures[BASECOLORMAP].texture_descriptor)];
					float2 dim = 0;
					tex.GetDimensions(dim.x, dim.y);
					baseColor *= gg_terrain_sample(tex, gg_tiling, uv, gg_tile_uv_scale_lod(dim), gg_full, false);
				}
				if (gg_splat)
				{
					total_base += weight * baseColor;
				}
				else
				{
					total_base = mad(1 - acc_base, weight * baseColor, total_base);
					acc_base = mad(1 - weight, acc_base, weight);
					done_base = acc_base >= 1;
				}
			}

			// normal: in a splat a layer with no normal map is flat; in the stack it takes no part
			[branch]
			if (!done_normal)
			{
				const bool has_normal = material.textures[NORMALMAP].IsValid();
				half2 normalMap = half2(0.5, 0.5);
				[branch]
				if (has_normal)
				{
					Texture2D tex = bindless_textures[descriptor_index(material.textures[NORMALMAP].texture_descriptor)];
					float2 dim = 0;
					tex.GetDimensions(dim.x, dim.y);
					normalMap = gg_terrain_sample(tex, gg_tiling, uv, gg_tile_uv_scale_lod(dim), gg_full, true).rg;
				}
				if (gg_splat)
				{
					total_normal += weight * normalMap;
				}
				else if (has_normal)
				{
					total_normal = mad(1 - acc_normal, weight * normalMap, total_normal);
					acc_normal = mad(1 - weight, acc_normal, weight);
					done_normal = acc_normal >= 1;
				}
			}

			// surface
			[branch]
			if (!done_surface)
			{
				float4 surface = float4(1, material.GetRoughness(), material.GetMetalness(), material.GetReflectance());
				[branch]
				if (material.textures[SURFACEMAP].IsValid())
				{
					Texture2D tex = bindless_textures[descriptor_index(material.textures[SURFACEMAP].texture_descriptor)];
					float2 dim = 0;
					tex.GetDimensions(dim.x, dim.y);
					surface *= gg_terrain_sample(tex, gg_tiling, uv, gg_tile_uv_scale_lod(dim), gg_full, false);
				}
				if (gg_splat)
				{
					total_surface += weight * surface;
				}
				else
				{
					total_surface = mad(1 - acc_surface, weight * surface, total_surface);
					acc_surface = mad(1 - weight, acc_surface, weight);
					done_surface = acc_surface >= 1;
				}
			}

			// emissive
			[branch]
			if (!done_emissive)
			{
				float4 emissiveColor = 0;
				[branch]
				if (material.textures[EMISSIVEMAP].IsValid())
				{
					Texture2D tex = bindless_textures[descriptor_index(material.textures[EMISSIVEMAP].texture_descriptor)];
					float2 dim = 0;
					tex.GetDimensions(dim.x, dim.y);
					half4 emissiveMap = gg_terrain_sample(tex, gg_tiling, uv, gg_tile_uv_scale_lod(dim), gg_full, false);
					emissiveColor.rgb = emissiveMap.rgb * emissiveMap.a;
					emissiveColor.a = emissiveMap.a;
				}
				if (gg_splat)
				{
					total_emissive += weight * emissiveColor;
				}
				else
				{
					total_emissive = mad(1 - acc_emissive, weight * emissiveColor, total_emissive);
					acc_emissive = mad(1 - weight, acc_emissive, weight);
					done_emissive = acc_emissive >= 1;
				}
			}

			if (gg_splat)
				splat_weight += weight;
			else if (done_base && done_normal && done_surface && done_emissive)
				break;
		}
		if (gg_splat && splat_weight > 0)
		{
			total_base /= splat_weight;
			total_normal /= splat_weight;
			total_surface /= splat_weight;
			total_emissive /= splat_weight;
		}

		// the road markings over the ground, as the four-pass bake
		const float4 gg_mark = gg_terrain_markings(blendmap_buffer, gg_block, uv, push.resolution_rcp);
		[branch]
		if (gg_mark.x > 0)
		{
			total_base.rgb = lerp(total_base.rgb, (half3)gg_mark.yzw, (half)gg_mark.x);
			total_normal = lerp(total_normal, 0.5, (half)(0.6 * gg_mark.x));
			total_surface.g = lerp(total_surface.g, 0.55, (half)gg_mark.x);
			total_surface.b = lerp(total_surface.b, 0, (half)gg_mark.x);
		}

		base_rgb[idx] = ApplySRGBCurve_Fast(total_base.rgb);
		normal_x[idx] = total_normal.r;
		normal_y[idx] = total_normal.g;
		surface_rgb[idx] = total_surface.rgb;
		surface_a[idx] = total_surface.a;
		emissive_rgb[idx] = ApplySRGBCurve_Fast(total_emissive.rgb);
	}

	const uint2 write_coord = push.write_offset + DTid.xy;
	RWTexture2D<uint2> output_base = bindless_rwtextures_uint2[descriptor_index(push.output_texture)];
	output_base[write_coord] = CompressBC1Block(base_rgb);
	[branch]
	if (out_normal >= 0)
	{
		RWTexture2D<uint4> output_normal = bindless_rwtextures_uint4[descriptor_index(out_normal)];
		output_normal[write_coord] = CompressBC5Block(normal_x, normal_y);
	}
	[branch]
	if (out_surface >= 0)
	{
		RWTexture2D<uint4> output_surface = bindless_rwtextures_uint4[descriptor_index(out_surface)];
		output_surface[write_coord] = CompressBC3Block(surface_rgb, surface_a);
	}
	[branch]
	if (want_emissive)
	{
		RWTexture2D<uint2> output_emissive = bindless_rwtextures_uint2[descriptor_index(out_emissive)];
		output_emissive[write_coord] = CompressBC1Block(emissive_rgb);
	}
}
