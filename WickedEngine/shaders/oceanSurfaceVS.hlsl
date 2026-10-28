#include "globals.hlsli"
#include "oceanSurfaceHF.hlsli"

Texture2D<float4> texture_displacementmap : register(t0);
Texture2D<float> texture_shoreheightmap : register(t2); // GG: Ocean::SetShoreHeightMap

PSIn main(uint vertexID : SV_VertexID)
{
	PSIn Out;
	
	float2 dim = xOceanScreenSpaceParams.xy;
	float2 invdim = xOceanScreenSpaceParams.zw;
	uint2 grid_coord = unflatten2D(vertexID, dim.xy);
	
	// Assemble screen space grid:
	Out.pos = float4(grid_coord / float2(dim - 1), 1, 1);
	Out.pos.xy = uv_to_clipspace(Out.pos.xy);
	float4 originalpos = Out.pos;
	Out.pos.xy *= max(1, xOceanSurfaceDisplacementTolerance); // extrude screen space grid to tolerate displacement

	// Perform ray tracing of screen grid and plane surface to unproject to world space:
	float4 unprojNEAR = mul(GetCamera().inverse_view_projection, float4(Out.pos.xy, 1, 1));
	unprojNEAR.xyz /= unprojNEAR.w;
	float4 unprojFAR = mul(GetCamera().inverse_view_projection, float4(Out.pos.xy, 0, 1));
	unprojFAR.xyz /= unprojFAR.w;
	const float3 d = normalize(unprojNEAR.xyz - unprojFAR.xyz);
	const float3 o = unprojNEAR.xyz;

	float3 worldPos = intersectPlaneClampInfinite(o, d, float3(0, 1, 0), xOceanWaterHeight);
	
	float2 uv = worldPos.xz * xOceanPatchSizeRecip;
	if (grid_coord.x > 0 && grid_coord.x < dim.x - 1 && grid_coord.y > 0 && grid_coord.y < dim.y - 1) // don't displace the side edges, to avoid holes
	{
		// Displace surface:
		float3 displacement = texture_displacementmap.SampleLevel(sampler_linear_wrap, uv, 0).xzy * OCEAN_M_TO_UNITS;
		float dist = length(worldPos - GetCamera().position);
		displacement *= saturate(1 - saturate(dist / GetCamera().z_far - 0.8) * 5.0); // fade will be on edge and inwards 20%
		// GG: and with distance in metres, as DX11's Wicked fades them (gone by 400 m): further out the projected grid is too
		// coarse for a wave's height, which lifted the far water over low ground
		displacement *= 1 - saturate(dist * 0.0025 / OCEAN_M_TO_UNITS);

		// GG: waves fade out as the water gets shallow and stay flat over land (Ocean::SetShoreHeightMap)
		[branch]
		if (xOceanShoreMap.z > 0)
		{
			float2 shoreUV = (worldPos.xz - xOceanShoreMap.xy) / xOceanShoreMap.z;
			if (all(shoreUV >= 0) && all(shoreUV <= 1))
			{
				float terrainHeight = texture_shoreheightmap.SampleLevel(sampler_linear_clamp, shoreUV, 0);
				displacement *= smoothstep(0, xOceanShoreMap.w, xOceanWaterHeight - terrainHeight);
			}
		}
		worldPos += displacement;
	}

	// Reproject displaced surface and output:
	Out.pos = mul(GetCamera().view_projection, float4(worldPos, 1));
	Out.uv = uv;

	return Out;
}
