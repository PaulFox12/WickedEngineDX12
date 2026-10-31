#ifndef WI_LENSRAIN_HF
#define WI_LENSRAIN_HF
// GG: rain on the lens (Postprocess_LensRain), drawn procedurally after the tone map. Every drop adds a smooth bump to one
// water surface, so drops that touch merge into one with an uneven outline; there is water where the surface is high
// enough, and through it the view is read moved by each drop's slope times its size, so every drop, large or small,
// shows a wide view of what is behind it turned over, and the move fading out toward its edge draws the outline.
// Droplets of two sizes land, sit and dry; drops in columns laid along the flow stick and slip down the glass, leaving
// droplets that stay where they fell, a thin film along their path and a trail through the fog. Positions are in screen
// heights (across times the aspect ratio), so drops are round at any aspect.

struct LensRainParams
{
	float amount;		// 0-1.5: how much rain is on the lens, the density of the drops and how clearly they show
	float size;			// a running drop's width, in screen heights
	float2 flow;		// the way drops run, in screen heights a second, +y down
	float slide;		// the share of drops that run rather than bead
	float refraction;	// how far a drop bends what is behind it
	float fog;			// the blur between the drops
	float time;			// seconds, held while the game is paused
};

struct LensRainWater
{
	float height;		// the summed bumps of every drop over the pixel
	float2 slope;		// the surface's gradient, per screen height
	float2 lens;		// each drop's slope times its size, in screen heights: how far it moves the view, the same for a
						// drop of any size at the same place in it, so every drop shows the same wide view turned over
	float trail;		// how clear of fog the glass is where drops have run, 0-1
	float2 film;		// the slope of the thin film a running drop leaves on its path, which bends the view a little
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

// one drop's bump: (1 - l^2)^2 out to its radii, l the distance in radii in the drop's own axes (x across, y along), v
// from the drop's centre to the pixel in screen heights; height its peak
void LensRain_Bump(float2 v, float2 xAxis, float2 yAxis, float2 radius, float height, inout LensRainWater water)
{
	const float2 q = float2(dot(v, xAxis), dot(v, yAxis)) / radius;
	const float l2 = dot(q, q);
	if (l2 >= 1 || height <= 0)
		return;
	const float s = 1 - l2;
	water.height += height * s * s;
	const float2 g = -4.0 * s * q / radius;
	water.slope += height * (g.x * xAxis + g.y * yAxis);
	water.lens += height * -4.0 * s * (q.x * xAxis + q.y * yAxis) * (radius.x + radius.y) * 0.5;
}

// droplets that land at a random spot in their cell, sit, and dry over most of their life, shrinking; then the next lands
// somewhere else in the cell
void LensRain_Droplets(float2 p, LensRainParams rain, float scale, float density, uint seed, inout LensRainWater water)
{
	const float cell = rain.size * scale;
	const float2 g = p / cell;
	const int2 ci = (int2)floor(g);
	const float2 f = g - ci;
	const float4 h = LensRain_Random(ci, seed);
	const float t = rain.time / lerp(3.0, 9.0, h.x) + h.y;
	const float cycle = floor(t);
	const float age = t - cycle;
	const float4 k = LensRain_Random(ci, seed + (uint)cycle * 7919u);
	const float present = saturate((density - k.x) * 10.0);
	if (present <= 0)
		return;
	// it keeps its depth while it dries, shrinking, and goes at the very end; stretched a little at a random angle, and
	// half of them have a smaller one merged in at one side, so outlines are uneven, as water's are
	const float life = 1.0 - smoothstep(0.3, 1.0, age);
	const float r = cell * 0.25 * lerp(0.5, 1.0, k.y) * lerp(0.35, 1.0, life);
	const float depth = present * smoothstep(0.0, 0.02, age) * smoothstep(0.0, 0.08, life) * lerp(1.0, 1.4, k.y);
	const float2 xAxis = float2(cos(k.w * 3.1416), sin(k.w * 3.1416));
	const float2 yAxis = float2(-xAxis.y, xAxis.x);
	const float2 v = (f - lerp(0.3, 0.7, k.zw)) * cell;
	LensRain_Bump(v, xAxis, yAxis, r * float2(1.0 + (k.y - 0.5) * 0.4, 1.0), depth, water);
	if (h.z > 0.5)
		LensRain_Bump(v - xAxis * r * 0.5, xAxis, yAxis, r * float2(0.55, 0.5), depth * 0.8, water);
}

// drops that run: columns laid along the flow, cut into cells six columns long that drift along it slowly; in its cell a
// drop creeps back against the drift for most of its time, so it nearly holds still on the glass, then slips forward the
// length of the cell, at about the flow's speed. Behind it, droplets that stay where they fell on the glass, a film along
// its path, and a trail through the fog that closes again further back
void LensRain_Runs(float2 p, LensRainParams rain, float scale, float density, uint seed, inout LensRainWater water)
{
	const float speed = length(rain.flow);
	if (speed < 0.0001 || density <= 0)
		return;
	const float2 along = rain.flow / speed;
	const float2 across = float2(-along.y, along.x);
	const float width = 1.3 * rain.size * scale;
	const float cellLength = 6.0 * width;
	const float a = dot(p, across) / width;
	const float column = floor(a);
	const float4 cr = LensRain_Random(int2((int)column, 0), seed);
	const float b = (dot(p, along) - rain.time * speed * 0.2) / cellLength + cr.x;
	const int2 ci = int2((int)column, (int)floor(b));
	const float2 f = float2(a - column, b - floor(b));
	const float4 h = LensRain_Random(ci, seed + 17u);
	const float present = saturate((density - h.x) * 10.0);
	if (present <= 0)
		return;

	// creep back 0.9 to 0.1 over 85% of the cycle, slip forward over the rest
	const float cycleTime = 5.3 * cellLength / speed * lerp(0.75, 1.25, h.y);
	const float ti = frac(rain.time / cycleTime + h.z);
	const float y = ti < 0.85 ? lerp(0.9, 0.1, smoothstep(0.0, 0.85, ti)) : lerp(0.1, 0.9, smoothstep(0.85, 1.0, ti));
	const float wander = (h.w - 0.5) * 0.25 + 0.06 * sin(dot(p, along) / width * 2.1 + h.w * 6.283);
	const float x = 0.5 + wander * (0.5 - abs(h.w - 0.5));
	const float r = width * 0.36 * lerp(0.75, 1.0, h.y);
	const float2 v = (f - float2(x, y)) * float2(width, cellLength);
	const float2 drop = v.x * across + v.y * along;
	// a rounded blob a little longer than wide, with a smaller one merged in at a random side: an uneven outline, not a
	// drawn drop
	LensRain_Bump(drop, across, along, r * float2(1.0, 1.1), present * 1.4, water);
	const float2 side = float2(cos(h.w * 6.283), sin(h.w * 6.283));
	LensRain_Bump(drop - (side.x * across + side.y * along) * r * 0.55, across, along, r * float2(0.6, 0.55), present * 0.8, water);

	// the trail behind it, to the cell's start: clear glass along its path, fading further back
	if (f.y < y)
	{
		const float behind = (y - f.y) * cellLength;
		const float fade = present * saturate(1.0 - behind / (y * cellLength + 0.0001));
		const float path = abs(v.x) / (r * 0.7);
		water.trail = max(water.trail, fade * saturate(1.2 - path));
		// the film: a low ridge along the path, (1 - s^2)^2 across it, s its distance in half widths
		const float s = v.x / (r * 0.6);
		if (abs(s) < 1)
			water.film += across * (-4.0 * s * (1 - s * s) / (r * 0.6)) * fade * 0.12;

		// droplets left on the glass, at fixed places along the column (they don't drift with the cells)
		const float spacing = 0.28 * width;
		const float along0 = dot(p, along) / spacing;
		const float n = floor(along0);
		const float4 j = LensRain_Random(int2((int)column, (int)n), seed + 31u);
		const float2 dv = float2((f.x - x - (j.x - 0.5) * 0.2) * width, (along0 - n - 0.5) * spacing);
		const float dr = r * lerp(0.15, 0.32, j.y) * (0.4 + 0.6 * fade);
		LensRain_Bump(dv.x * across + dv.y * along, across, along, float2(dr, dr), step(j.z, 0.7) * fade * 1.2, water);
	}
}

// all the water over the point p (screen heights)
LensRainWater LensRain_Water(float2 p, LensRainParams rain)
{
	LensRainWater water;
	water.height = 0;
	water.slope = 0;
	water.lens = 0;
	water.trail = 0;
	water.film = 0;
	const float beads = rain.amount * (1.0 - 0.4 * rain.slide);
	const float runs = rain.amount * rain.slide;
	LensRain_Droplets(p, rain, 0.35, beads * 0.9, 101u, water);
	LensRain_Droplets(p, rain, 0.8, beads * 0.45, 211u, water);
	LensRain_Runs(p, rain, 1.0, runs * 1.6, 307u, water);
	LensRain_Runs(p + float2(0.37, 0.11) * rain.size, rain, 0.6, runs * 1.2, 401u, water);
	return water;
}

#endif // WI_LENSRAIN_HF
