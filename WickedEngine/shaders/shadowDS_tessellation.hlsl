// GGMAX: the shadow pass for tessellated meshes (GameGuru's terrain chunks): each new vertex where the screen's passes
// put it (flat across the chunk's triangle, then the height of GameGuru's height rings), so the ground casts from the
// surface it is drawn with; the cascade's camera and viewport from the instance, as shadowVS picks them
#define OBJECTSHADER_LAYOUT_SHADOW
#define OBJECTSHADER_USE_NOCAMERA
#define OBJECTSHADER_USE_NORMAL
#include "objectHF_tessellation.hlsli"

struct ShadowTessellationOutput
{
	precise float4 pos : SV_Position;
	uint VPIndex : SV_ViewportArrayIndex;
};

[domain("tri")]
ShadowTessellationOutput main(ConstantOutput input, float3 uvw : SV_DomainLocation, const OutputPatch<PixelInput, 3> patch)
{
	const float u = uvw.x;
	const float v = uvw.y;
	const float w = uvw.z;

	precise float4 pos = float4(patch[0].pos.xyz * w + patch[1].pos.xyz * u + patch[2].pos.xyz * v, 1);
	[branch]
	if (gg_terrain_tessellated())
	{
		float height;
		float3 normal;
		if (gg_ring_height(pos.xz, height, normal))
		{
			pos.y = height + GetScene().terrain.gg_tess_shadow_offset;
		}
	}

	const ShaderCamera camera = GetCameraIndexed(patch[0].GetCameraIndex());
	ShadowTessellationOutput output;
	output.pos = mul(camera.view_projection, pos);
	output.VPIndex = camera.output_index;
	return output;
}
