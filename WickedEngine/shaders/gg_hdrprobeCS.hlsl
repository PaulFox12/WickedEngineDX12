#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"

// GG: the white screen probe (2026-11-07). Counts the pixels of an HDR image that are NaN, infinite or negative, and
// takes its brightest finite value, into slot params0.x of a raw buffer (four uints a slot: NaN, Inf, negative, the
// brightest value's bits). The bits are tested directly so no compiler setting can fold the NaN test away.

PUSHCONSTANT(postprocess, PostProcess);

Texture2D<float4> input : register(t0);
RWByteAddressBuffer output : register(u0);

[numthreads(POSTPROCESS_BLOCKSIZE, POSTPROCESS_BLOCKSIZE, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	bool inside = DTid.x < (uint)postprocess.resolution.x && DTid.y < (uint)postprocess.resolution.y;
	float3 c = inside ? input[DTid.xy].rgb : 0;
	bool is_nan = false, is_inf = false, is_neg = false;
	[unroll]
	for (uint i = 0; i < 3; ++i)
	{
		const uint b = asuint(c[i]);
		const bool special = (b & 0x7f800000) == 0x7f800000;
		const bool mantissa = (b & 0x007fffff) != 0;
		is_nan = is_nan || (special && mantissa);
		is_inf = is_inf || (special && !mantissa);
		is_neg = is_neg || (!(special && mantissa) && (b & 0x80000000) != 0 && (b & 0x7fffffff) != 0);
	}
	float brightest = (is_nan || is_inf) ? 0 : max(0, max(c.r, max(c.g, c.b)));

	const uint nans = WaveActiveCountBits(inside && is_nan);
	const uint infs = WaveActiveCountBits(inside && is_inf);
	const uint negs = WaveActiveCountBits(inside && is_neg);
	const uint top = WaveActiveMax(asuint(brightest));
	if (WaveIsFirstLane())
	{
		const uint base = (uint)postprocess.params0.x * 16;
		if (nans > 0) output.InterlockedAdd(base + 0, nans);
		if (infs > 0) output.InterlockedAdd(base + 4, infs);
		if (negs > 0) output.InterlockedAdd(base + 8, negs);
		output.InterlockedMax(base + 12, top);
	}
}
