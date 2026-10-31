#ifndef WI_SHADERINTEROP_TERRAIN_H
#define WI_SHADERINTEROP_TERRAIN_H
#include "ShaderInterop.h"
#include "ShaderInterop_Renderer.h"

struct ShaderTerrainChunk
{
	int heightmap;
	uint materialID;

	void init()
	{
		heightmap = -1;
		materialID = 0;
	}
};
struct alignas(16) ShaderTerrain
{
	float3 center_chunk_pos;
	float chunk_size;

	int chunk_buffer;
	int chunk_buffer_range;
	float min_height;
	float max_height;

	// GGMAX: GameGuru's wet ground (SetTerrainWetness), on the chunks (SHADERMATERIAL_OPTION_BIT_GG_TERRAIN): its paint map,
	// world x and z to the map's uv (uv = world * scale + 0.5), how wet (0 dry), and a buffer of a float2 per paint value
	// 0-32 (how much glossier, how much darker) followed by the dry boxes (min xyz, max xyz)
	int gg_paintmap;
	float gg_map_scale;
	uint gg_map_resolution;
	float gg_wetness;

	int gg_wet_buffer;
	uint gg_dry_boxes;
	float2 gg_padding;

	// GGMAX: GameGuru's Steep Rock on the chunks (ggTerrainRockHF.hlsli): the rock material, how strong (0 off), the slope
	// where it starts and 1 / its width, 1 / the tile size, the edge breakup, the two-size variation, the level's bumpiness
	uint gg_rock_material;
	float gg_rock_strength;
	float gg_rock_start;
	float gg_rock_transition;

	float gg_rock_scale;
	float gg_rock_breakup;
	float gg_rock_variation;
	float gg_bumpiness;

	void init()
	{
		center_chunk_pos = float3(0, 0, 0);
		chunk_size = 0;
		chunk_buffer = -1;
		chunk_buffer_range = 0;
		gg_paintmap = -1;
		gg_map_scale = 0;
		gg_map_resolution = 0;
		gg_wetness = 0;
		gg_wet_buffer = -1;
		gg_dry_boxes = 0;
		gg_padding = float2(0, 0);
		gg_rock_material = 0;
		gg_rock_strength = 0;
		gg_rock_start = 0;
		gg_rock_transition = 0;
		gg_rock_scale = 0;
		gg_rock_breakup = 0;
		gg_rock_variation = 0;
		gg_bumpiness = 1;
	}
};

#endif // WI_SHADERINTEROP_TERRAIN_H
