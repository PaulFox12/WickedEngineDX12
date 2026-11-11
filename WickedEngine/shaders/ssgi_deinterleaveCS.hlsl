#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"

PUSHCONSTANT(postprocess, PostProcess);

Texture2D<float3> texture_input : register(t0);

RWTexture2DArray<float>		atlas2x_depth : register(u0);
RWTexture2DArray<float>		atlas4x_depth : register(u1);
RWTexture2DArray<float>		atlas8x_depth : register(u2);
RWTexture2DArray<float>		atlas16x_depth : register(u3);
RWTexture2DArray<float3>	atlas2x_color : register(u4);
RWTexture2DArray<float3>	atlas4x_color : register(u5);
RWTexture2DArray<float3>	atlas8x_color : register(u6);
RWTexture2DArray<float3>	atlas16x_color : register(u7);
RWTexture2D<float>			regular2x_depth : register(u8);
RWTexture2D<float>			regular4x_depth : register(u9);
RWTexture2D<float>			regular8x_depth : register(u10);
RWTexture2D<float>			regular16x_depth : register(u11);
RWTexture2D<float2>			regular2x_normal : register(u12);
RWTexture2D<float2>			regular4x_normal : register(u13);
RWTexture2D<float2>			regular8x_normal : register(u14);
RWTexture2D<float2>			regular16x_normal : register(u15);

groupshared float shared_depths[256];
groupshared uint shared_normals[256];
groupshared uint shared_colors[256];

// GG: a normal worked out from the depth buffer (oct-encoded, world space, facing the camera), for when the surface pass
// that writes texture_normal is skipped (wi::renderer::gg_ssgi_depth_normals, push params1.w): each side's step taken from
// the neighbour nearer in depth, so an edge doesn't bend it
float2 gg_depth_normal(uint2 p, uint2 dim, float2 dim_rcp)
{
	const uint2 last = dim - 1;
	const float z = texture_depth[min(p, last)];
	const float zr = texture_depth[min(p + uint2(1, 0), last)];
	const float zl = texture_depth[uint2(p.x > 0 ? p.x - 1 : 0, min(p.y, last.y))];
	const float zd = texture_depth[min(p + uint2(0, 1), last)];
	const float zu = texture_depth[uint2(min(p.x, last.x), p.y > 0 ? p.y - 1 : 0)];
	const float2 uv = (float2(p) + 0.5) * dim_rcp;
	const float3 P = reconstruct_position(uv, z);
	const float3 dx = abs(zr - z) < abs(zl - z) ? reconstruct_position(uv + float2(dim_rcp.x, 0), zr) - P : P - reconstruct_position(uv - float2(dim_rcp.x, 0), zl);
	const float3 dy = abs(zd - z) < abs(zu - z) ? reconstruct_position(uv + float2(0, dim_rcp.y), zd) - P : P - reconstruct_position(uv - float2(0, dim_rcp.y), zu);
	float3 N = normalize(cross(dy, dx) + 1e-12);
	if (dot(N, GetCamera().position - P) < 0)
		N = -N;
	return encode_oct(N);
}

[numthreads(8, 8, 1)]
void main(uint3 Gid : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 GTid : SV_GroupThreadID, uint3 DTid : SV_DispatchThreadID)
{
	uint2 dim;
	texture_depth.GetDimensions(dim.x, dim.y);
	const float2 dim_rcp = rcp(dim);

    uint2 startST = Gid.xy << 4 | GTid.xy;
    uint destIdx = GTid.y << 4 | GTid.x;
    shared_depths[destIdx + 0] = texture_depth[min(startST | uint2(0, 0), dim - 1)];
    shared_depths[destIdx + 8] = texture_depth[min(startST | uint2(8, 0), dim - 1)];
    shared_depths[destIdx + 128] = texture_depth[min(startST | uint2(0, 8), dim - 1)];
    shared_depths[destIdx + 136] = texture_depth[min(startST | uint2(8, 8), dim - 1)];

	[branch]
	if (postprocess.params1.w > 0)
	{
		// GG: normals from the depth (the surface pass skipped)
		shared_normals[destIdx + 0] = pack_half2(gg_depth_normal(startST | uint2(0, 0), dim, dim_rcp));
		shared_normals[destIdx + 8] = pack_half2(gg_depth_normal(startST | uint2(8, 0), dim, dim_rcp));
		shared_normals[destIdx + 128] = pack_half2(gg_depth_normal(startST | uint2(0, 8), dim, dim_rcp));
		shared_normals[destIdx + 136] = pack_half2(gg_depth_normal(startST | uint2(8, 8), dim, dim_rcp));
	}
	else
	{
    shared_normals[destIdx + 0] = pack_half2(texture_normal[min(startST | uint2(0, 0), dim - 1)]);
    shared_normals[destIdx + 8] = pack_half2(texture_normal[min(startST | uint2(8, 0), dim - 1)]);
    shared_normals[destIdx + 128] = pack_half2(texture_normal[min(startST | uint2(0, 8), dim - 1)]);
    shared_normals[destIdx + 136] = pack_half2(texture_normal[min(startST | uint2(8, 8), dim - 1)]);
	}
	
	const float2 uv0 = float2(startST | uint2(0, 0)) * dim_rcp;
	const float2 uv1 = float2(startST | uint2(8, 0)) * dim_rcp;
	const float2 uv2 = float2(startST | uint2(0, 8)) * dim_rcp;
	const float2 uv3 = float2(startST | uint2(8, 8)) * dim_rcp;
	const float2 velocity0 = texture_velocity[min(startST | uint2(0, 0), dim - 1)];
	const float2 velocity1 = texture_velocity[min(startST | uint2(8, 0), dim - 1)];
	const float2 velocity2 = texture_velocity[min(startST | uint2(0, 8), dim - 1)];
	const float2 velocity3 = texture_velocity[min(startST | uint2(8, 8), dim - 1)];
	const float2 prevUV0 = uv0 + velocity0;
	const float2 prevUV1 = uv1 + velocity1;
	const float2 prevUV2 = uv2 + velocity2;
	const float2 prevUV3 = uv3 + velocity3;
    shared_colors[destIdx + 0]		= Pack_R11G11B10_FLOAT(clamp(texture_input.SampleLevel(sampler_linear_clamp, prevUV0, 0), 0, postprocess.params1.z)); // GG: the brightest light a pixel gives (stock 1000)
    shared_colors[destIdx + 8]		= Pack_R11G11B10_FLOAT(clamp(texture_input.SampleLevel(sampler_linear_clamp, prevUV1, 0), 0, postprocess.params1.z));
    shared_colors[destIdx + 128]	= Pack_R11G11B10_FLOAT(clamp(texture_input.SampleLevel(sampler_linear_clamp, prevUV2, 0), 0, postprocess.params1.z));
    shared_colors[destIdx + 136]	= Pack_R11G11B10_FLOAT(clamp(texture_input.SampleLevel(sampler_linear_clamp, prevUV3, 0), 0, postprocess.params1.z));
	
    GroupMemoryBarrierWithGroupSync();

    uint ldsIndex = (GTid.x << 1) | (GTid.y << 5);

    float depth = shared_depths[ldsIndex];
    float2 normal = unpack_half2(shared_normals[ldsIndex]);
    float3 color = Unpack_R11G11B10_FLOAT(shared_colors[ldsIndex]);

	color = all(color <= postprocess.params1.x) ? 0 : color; // cut out pixels that are not acting as lights (GG: the threshold a setting, stock 1)
	color = (postprocess.params1.y > 0 && depth == 0) ? 0 : color; // GG: the sky (reversed depth 0) gives no light when asked: its light is in the ambient already
	color *= 0.96; // accumulation energy loss
	color = max(0, color);

    uint2 st = DTid.xy;
    uint slice = flatten2D(st % 4, 4);
    atlas2x_depth[uint3(st >> 2, slice)] = depth;
    atlas2x_color[uint3(st >> 2, slice)] = color;
    regular2x_depth[st] = depth;
    regular2x_normal[st] = normal;

    if (all(GTid.xy % 2) == 0)
    {
        st = DTid.xy >> 1;
        slice = flatten2D(st % 4, 4);
        atlas4x_depth[uint3(st >> 2, slice)] = depth;
        atlas4x_color[uint3(st >> 2, slice)] = color;
		regular4x_depth[st] = depth;
		regular4x_normal[st] = normal;
		
		if (all(GTid.xy % 4) == 0)
		{
			st = DTid.xy >> 2;
			slice = flatten2D(st % 4, 4);
			atlas8x_depth[uint3(st >> 2, slice)] = depth;
			atlas8x_color[uint3(st >> 2, slice)] = color;
			regular8x_depth[st] = depth;
			regular8x_normal[st] = normal;
			
			if (groupIndex == 0)
			{
				st = DTid.xy >> 3;
				slice = flatten2D(st % 4, 4);
				atlas16x_depth[uint3(st >> 2, slice)] = depth;
				atlas16x_color[uint3(st >> 2, slice)] = color;
				regular16x_depth[st] = depth;
				regular16x_normal[st] = normal;
			}
		}
    }

}
