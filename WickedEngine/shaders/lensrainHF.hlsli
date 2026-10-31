#ifndef WI_LENSRAIN_HF
#define WI_LENSRAIN_HF
// GG: rain on the lens (Postprocess_LensRain), drawn procedurally after the tone map. The screen is split into cells, each
// with at most one drop at a random spot and size: two layers of beads at different scales that land and evaporate, and a
// layer of taller cells whose drops run the way the rain flows, in quick runs and pauses, wandering a little sideways and
// leaving a trail of droplets that shrink away behind them. Each drop is a small lens: the scene is sampled through it,
// turned over and drawn in, with a dark rim and a highlight near its top. Positions are in screen heights (across times
// the aspect ratio), so drops are round at any aspect.

struct LensRainParams
{
	float amount;		// 0-1.5: how much rain is on the lens, the density of the drops and how clearly they show
	float size;			// a drop's usual diameter, in screen heights
	float2 flow;		// the way drops run, in screen heights a second, +y down
	float slide;		// the share of drops that run rather than bead
	float refraction;	// how far a drop bends what is behind it
	float fog;			// the blur between the drops
	float time;			// seconds, held while the game is paused
};

struct LensRainDrop
{
	float mask;			// how much of the pixel is drop
	float2 offset;		// where the drop samples the scene, from the pixel, in screen heights
	float rim;			// near the drop's edge
	float spec;			// the highlight
};

uint LensRain_Hash(uint x)
{
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

// four random values in [0, 1) for a cell and a seed, a byte of one hash each (fine enough for a drop's place and size)
float4 LensRain_Random(int2 cell, uint seed)
{
	const uint h = LensRain_Hash(asuint(cell.x) ^ LensRain_Hash(asuint(cell.y) ^ (seed * 0x9e3779b9u)));
	return float4(h & 255u, (h >> 8) & 255u, (h >> 16) & 255u, h >> 24) * (1.0 / 256.0);
}

// d: the pixel's offset from the drop's centre in radii, in screen orientation; the drop that covers the pixel most wins it
void LensRain_AddDrop(float2 d, float r, float visible, float bend, inout LensRainDrop drop)
{
	const float l = length(d);
	if (l >= 1 || visible <= 0)
		return;
	const float m = smoothstep(1.0, 0.8, l) * visible;
	if (m <= drop.mask)
		return;
	drop.mask = m;
	drop.offset = -d * r * bend; // the scene at p - (p - centre) * bend: turned over past a bend of 1
	drop.rim = smoothstep(0.55, 1.0, l);
	const float s = saturate(1 - length(d - float2(-0.25, -0.45)) / 0.3);
	drop.spec = s * s;
}

// beads: a cell's drop lands at a random spot, sits, and evaporates over its second half, shrinking; then the next lands
// somewhere else in the cell. A drop's reach stays inside its cell
void LensRain_Beads(float2 p, LensRainParams rain, float scale, float density, float2 shift, uint seed, float bend, inout LensRainDrop drop)
{
	const float cell = 2.0 * rain.size * scale;
	const float2 g = p / cell + shift;
	const int2 ci = (int2)floor(g);
	const float2 f = g - ci;
	const float4 h = LensRain_Random(ci, seed);
	const float t = rain.time / lerp(3.0, 9.0, h.x) + h.y;
	const float cycle = floor(t);
	const float age = t - cycle;
	const float4 k = LensRain_Random(ci, seed + (uint)cycle * 7919u);
	const float present = saturate((density - k.x) * 12.0);
	const float life = smoothstep(0.0, 0.02, age) * (1.0 - smoothstep(0.5, 1.0, age));
	const float r = 0.5 * rain.size * scale * lerp(0.45, 1.0, k.y) * lerp(0.5, 1.0, life);
	const float2 c = lerp(0.25, 0.75, k.zw);
	LensRain_AddDrop((f - c) * cell / r, r, present * saturate(life * 4.0), bend, drop);
}

// the sideways wander of a running drop at a point along its cell, in cell widths, so its trail follows the same path
float LensRain_Wander(float y, float seed)
{
	return 0.06 * sin(y * 23.0 + seed * 6.283) + 0.035 * sin(y * 57.0 + seed * 12.566);
}

// running drops, in cells three drops wide and ten long laid along the flow: a drop beads at the top for a fifth of its
// time, runs to the bottom by three quarters of it in three to five quick runs with pauses between, then sits and
// evaporates; its runs go at the flow's speed. Behind it, a droplet every third of a drop, smaller further back
void LensRain_Runs(float2 p, LensRainParams rain, float bend, inout LensRainDrop drop, inout float wiped)
{
	const float speed = length(rain.flow);
	if (speed < 0.0001 || rain.slide <= 0)
		return;
	const float2 along = rain.flow / speed;
	const float2 across = float2(-along.y, along.x);
	const float2 cellSize = float2(3.0, 10.0) * rain.size;
	const float2 g = float2(dot(p, across), dot(p, along)) / cellSize;
	const int2 ci = (int2)floor(g);
	const float2 f = g - ci;
	const float4 h = LensRain_Random(ci, 501u);
	const float period = 0.73 * cellSize.y / (speed * 0.19 * lerp(0.7, 1.3, h.z));
	const float t = rain.time / period + h.x;
	const float cycle = floor(t);
	const float age = t - cycle;
	const float4 k = LensRain_Random(ci, 701u + (uint)cycle * 7919u);
	const float present = saturate((rain.amount * rain.slide * 1.2 - k.x) * 12.0);
	if (present <= 0)
		return;
	const float life = smoothstep(0.0, 0.02, age) * (1.0 - smoothstep(0.85, 1.0, age));
	const float visible = present * saturate(life * 4.0);
	const float steps = 3.0 + floor(k.w * 3.0);
	const float u = saturate((age - 0.2) / 0.55) * steps;
	const float run = min(1.0, (floor(u) + smoothstep(0.0, 0.35, frac(u))) / steps);
	const float top = 0.12;
	const float y = lerp(top, 0.85, run);
	const float x0 = lerp(0.35, 0.65, k.z);
	const float r = 0.5 * rain.size * lerp(0.7, 1.2, k.y);

	// the drop, a little longer behind than ahead
	float2 dq = (f - float2(x0 + LensRain_Wander(y, k.y), y)) * cellSize / r;
	if (dq.y < 0)
		dq.y *= 0.7;
	LensRain_AddDrop(dq.x * across + dq.y * along, r, visible, bend, drop);

	// the trail and its wet streak, which wipes the fog
	if (f.y > top && f.y < y)
	{
		const float path = abs(f.x - x0 - LensRain_Wander(f.y, k.y)) * cellSize.x;
		wiped = max(wiped, visible * saturate(1 - path / r));

		const float spacing = 0.35 * rain.size / cellSize.y;
		const float n = floor(f.y / spacing);
		const float ty = (n + 0.5) * spacing;
		if (ty > top && ty < y - 1.3 * r / cellSize.y)
		{
			const float4 j = LensRain_Random(int2(ci.x, ci.y * 64 + (int)n), 901u + (uint)cycle * 7919u);
			const float back = saturate((y - ty) / max(y - top, 0.001));
			const float tr = r * 0.28 * lerp(0.6, 1.0, j.y) * (1.0 - 0.7 * back);
			const float2 dt = (f - float2(x0 + LensRain_Wander(ty, k.y) + (j.z - 0.5) * 0.03, ty)) * cellSize / tr;
			LensRain_AddDrop(dt.x * across + dt.y * along, tr, visible * step(j.x, 0.8), bend, drop);
		}
	}
}

// every drop over the point p (screen heights); wiped is how far the fog is cleared there
LensRainDrop LensRain_Drops(float2 p, LensRainParams rain, out float wiped)
{
	LensRainDrop drop;
	drop.mask = 0;
	drop.offset = 0;
	drop.rim = 0;
	drop.spec = 0;
	wiped = 0;
	const float bend = rain.refraction * 8.0;
	const float beads = rain.amount * (1.0 - 0.5 * rain.slide);
	LensRain_Beads(p, rain, 1.0, beads * 0.3, float2(0, 0), 101u, bend, drop);
	LensRain_Beads(p, rain, 0.55, beads * 0.45, float2(0.37, 0.61), 301u, bend, drop);
	LensRain_Runs(p, rain, bend, drop, wiped);
	drop.mask *= saturate(rain.amount * 4.0);
	wiped = max(wiped, drop.mask);
	return drop;
}

#endif // WI_LENSRAIN_HF
