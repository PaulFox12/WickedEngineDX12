// GGMAX: the shadow pass for tessellated meshes (GameGuru's terrain chunks): the world position goes on to the hull and
// domain shaders, which cut the chunk as the screen's passes do; the cascade's camera is picked in shadowDS_tessellation
#define OBJECTSHADER_COMPILE_VS
#define OBJECTSHADER_LAYOUT_SHADOW
#define OBJECTSHADER_USE_NOCAMERA
#define OBJECTSHADER_USE_NORMAL
#include "objectHF.hlsli"
