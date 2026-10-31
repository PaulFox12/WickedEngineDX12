#ifndef GG_DECALHF
#define GG_DECALHF

// GameGuru's projected decals (AddProjectedDecal, AddBlastDecal), applied by TiledDecals (shadingHF.hlsli) to every
// object and the terrain's chunks; a decal without a GameGuru flag is Wicked's own, unchanged. The DX11 branch's
// GGDecalHF.hlsli, on this build's entity packing (wiRenderer.cpp):
// - ENTITY_FLAG_DECAL_GG_FACING: paints only surfaces whose geometric normal faces along the decal's Z by more than the
//   cosine in the cone slot, fading in (smoothstep) over the width in the radius (0.1 unless set), so the inside face of a
//   thin wall and the ground under an upright box stay clean
// - ENTITY_FLAG_DECAL_GG_BLAST: a blast round the decal's centre. Its box is a cube of half-size radius, and it paints the
//   sphere inside it, fading out over the outer quarter, on every surface facing the centre by more than the cone slot's
//   cosine. Each surface takes the texture from the box's planes it faces most (triplanar), the weights sharpened by the
//   power in the direction slot's x, so a flat face takes one unstretched sample. remap holds how far the first surface
//   lies along each of the box's six axes (5 bits each from bit 0: +x, -x, +y, -y, +z, -z, in 30ths of the radius, 31 for
//   none), and a point further along the axis nearest its direction from the centre lies behind that surface and stays
//   clean. When the surface hit is a flat face across one of the box's axes, the direction slot's y is the direction into
//   it (0 +x to 5 -z; below 0 for none) and its z how far along it a point may lie: a room behind a wall stays clean
// Under them the surface is a matte, non-metal coat (TiledDecals, after the loop: roughness 1, metalness 0, reflectance 0.04),
// or a dark one would not show on metal, whose albedo is black and whose colour is all in its reflection

// how a GameGuru decal fades in past its facing cutoff: 0 at the cutoff, full at the cutoff plus the width in the radius
inline half GGDecalFacingBlend(in ShaderEntity decal, in half facing)
{
	const half cutoff = decal.GetConeAngleCos();
	const half fade = max(decal.GetRadius(), (half)0.001);
	return smoothstep(cutoff, cutoff + fade, facing);
}

// the direction slot as written (GetDirection normalises it): a blast's sharpness, the face hit and its clip
inline float3 GGDecalBlastParams(in ShaderEntity decal)
{
	return float3(
		f16tof32(decal.direction16_coneAngleCos16.x),
		f16tof32(decal.direction16_coneAngleCos16.x >> 16u),
		f16tof32(decal.direction16_coneAngleCos16.y)
	);
}

// one of a blast decal's three planes: the position and its gradients in the box's units (-1 to 1 across it)
inline half4 GGBlastDecalSample(in int texture, in SamplerState sam, in float2 planePos, in float2 planeDX, in float2 planeDY, in float4 texMulAdd)
{
	const float2 uvScale = float2(0.5, -0.5) * texMulAdd.xy;
	return bindless_textures_half4[descriptor_index(texture)].SampleGrad(sam, (planePos * float2(0.5, -0.5) + 0.5) * texMulAdd.xy + texMulAdd.zw, planeDX * uvScale, planeDY * uvScale);
}

// a decal's normal map sample turned into a world space normal: its red along T (the texture's u in the world), its green
// along B (its v; DirectX normal maps, as the models'), both laid on the surface round N; strength scales the tilt.
// Built from the decal's own axes, so it needs no tangents on the surface (the terrain has none) and stays right on every
// face a blast's triplanar planes cover
inline float3 GGDecalNormalWorld(in half4 normalSample, in float3 T, in float3 B, in float3 N, in float strength)
{
	const float2 tilt = ((float2)normalSample.rg * 2 - 1) * strength;
	float3 Ts = T - N * dot(T, N);
	float3 Bs = B - N * dot(B, N);
	Ts /= max(length(Ts), 0.0001);
	Bs /= max(length(Bs), 0.0001);
	return normalize(tilt.x * Ts + tilt.y * Bs + N);
}

// a blast decal's colour at P, boxPos being P in its box (-1 to 1 across it); with a normal map, decalN its normal there
// (0 without one), each plane's sample laid on that plane's axes and weighted as the colour
inline half4 GGBlastDecalColor(in ShaderEntity decal, in float4x4 decalProjection, in int texture, in int normalTexture, in float3 boxPos, in float3 P, in half3 faceN, in float3 N, in float3 P_dx, in float3 P_dy, in SamplerState sam, out float3 decalN)
{
	half4 decalColor = 0;
	decalN = 0;

	// the sphere, fading out over its outer quarter (none from its edge out), on surfaces facing the centre, fading in past
	// the cutoff
	const float3 toCentre = decal.position - P;
	const float lenToCentre = length(toCentre);
	const half facing = lenToCentre > 0.001 ? (half)(dot((float3)faceN, toCentre) / lenToCentre) : (half)1;
	half blend = (half)(1 - smoothstep(0.75, 1, length(boxPos))) * GGDecalFacingBlend(decal, facing);

	// nothing behind the first surface along the box's axis nearest the point's direction from the centre
	const float3 absPos = abs(boxPos);
	const uint axis = absPos.x >= absPos.y ? (absPos.x >= absPos.z ? 0 : 2) : (absPos.y >= absPos.z ? 1 : 2);
	const float along = axis == 0 ? boxPos.x : (axis == 1 ? boxPos.y : boxPos.z);
	const uint firstSurface = (decal.remap >> ((axis * 2 + (along < 0 ? 1 : 0)) * 5)) & 31;
	if (firstSurface < 31 && abs(along) > firstSurface / 30.0 + 0.06)
		blend = 0;

	// nothing behind the surface hit, when it is a flat face across one of the box's axes
	const float3 blastParams = GGDecalBlastParams(decal);
	if (blastParams.y >= 0)
	{
		const uint hitDirection = (uint)(blastParams.y + 0.5);
		const uint hitAxis = hitDirection / 2;
		const float alongHit = (hitAxis == 0 ? boxPos.x : (hitAxis == 1 ? boxPos.y : boxPos.z)) * ((hitDirection & 1) ? -1 : 1);
		if (alongHit > blastParams.z)
			blend = 0;
	}

	[branch]
	if (blend > 0 && texture >= 0)
	{
		// triplanar: each plane weighted by how squarely the surface faces it, sharpened, and planes under 0.01 skipped
		const float4 texMulAdd = decal.shadowAtlasMulAdd;
		const float3 boxN = normalize(mul((float3x3)decalProjection, (float3)faceN));
		const float sharpness = max(1, blastParams.x);
		float3 weights = pow(abs(boxN), float3(sharpness, sharpness, sharpness));
		weights /= weights.x + weights.y + weights.z;
		const float3 boxDX = mul((float3x3)decalProjection, P_dx);
		const float3 boxDY = mul((float3x3)decalProjection, P_dy);
		// the box's axes in the world (its rows are the gradients of its x, y and z), for the normal map: a plane's u runs
		// along its first axis and its v against its second, as GGBlastDecalSample reads it
		const float3 axisX = normalize(decalProjection[0].xyz);
		const float3 axisY = normalize(decalProjection[1].xyz);
		const float3 axisZ = normalize(decalProjection[2].xyz);
		const float strength = decal.GetLength(); // the normal map's strength (remap, Wicked's slot for it, holds the occlusion)
		float weightSum = 0;
		[branch]
		if (weights.x > 0.01)
		{
			decalColor += (half)weights.x * GGBlastDecalSample(texture, sam, boxPos.zy, boxDX.zy, boxDY.zy, texMulAdd);
			if (normalTexture >= 0)
				decalN += weights.x * GGDecalNormalWorld(GGBlastDecalSample(normalTexture, sam, boxPos.zy, boxDX.zy, boxDY.zy, texMulAdd), axisZ, -axisY, N, strength);
			weightSum += weights.x;
		}
		[branch]
		if (weights.y > 0.01)
		{
			decalColor += (half)weights.y * GGBlastDecalSample(texture, sam, boxPos.xz, boxDX.xz, boxDY.xz, texMulAdd);
			if (normalTexture >= 0)
				decalN += weights.y * GGDecalNormalWorld(GGBlastDecalSample(normalTexture, sam, boxPos.xz, boxDX.xz, boxDY.xz, texMulAdd), axisX, -axisZ, N, strength);
			weightSum += weights.y;
		}
		[branch]
		if (weights.z > 0.01)
		{
			decalColor += (half)weights.z * GGBlastDecalSample(texture, sam, boxPos.xy, boxDX.xy, boxDY.xy, texMulAdd);
			if (normalTexture >= 0)
				decalN += weights.z * GGDecalNormalWorld(GGBlastDecalSample(normalTexture, sam, boxPos.xy, boxDX.xy, boxDY.xy, texMulAdd), axisX, -axisY, N, strength);
			weightSum += weights.z;
		}
		decalColor /= (half)max(weightSum, 0.0001);
		if (normalTexture >= 0) decalN = normalize(decalN);
		decalColor *= decal.GetColor();
		decalColor.a *= blend;
	}
	return decalColor;
}

// a GameGuru decal's colour at P (its alpha how much of it applies there, 0 where it paints nothing); clipSpacePos is P in
// its box, decalProjection with its texture row cleared; with a normal map (normalTexture), decalN the decal's normal at P
// in the world, from N, the surface's (0 without one)
inline half4 GGDecalColor(in ShaderEntity decal, in float4x4 decalProjection, in int texture, in int normalTexture, in float3 clipSpacePos, in float3 P, in half3 faceN, in float3 N, in float3 P_dx, in float3 P_dy, in SamplerState sam, out float3 decalN)
{
	[branch]
	if (decal.GetFlags() & ENTITY_FLAG_DECAL_GG_BLAST)
		return GGBlastDecalColor(decal, decalProjection, texture, normalTexture, clipSpacePos, P, faceN, N, P_dx, P_dy, sam, decalN);

	half4 decalColor = 0;
	decalN = 0;
	const float3 uvw = box_to_uv(clipSpacePos);
	// the decal's Z in world space is the gradient of its box's z
	const half facingBlend = GGDecalFacingBlend(decal, (half)dot((float3)faceN, normalize(decalProjection[2].xyz)));
	[branch]
	if (is_saturated(uvw) && facingBlend > 0 && texture >= 0)
	{
		const float4 texMulAdd = decal.shadowAtlasMulAdd;
		// mipmapping needs to be performed by hand:
		const float2 decalDX = mul(P_dx, (float3x3)decalProjection).xy * texMulAdd.xy;
		const float2 decalDY = mul(P_dy, (float3x3)decalProjection).xy * texMulAdd.xy;
		decalColor = bindless_textures_half4[descriptor_index(texture)].SampleGrad(sam, mad(uvw.xy, texMulAdd.xy, texMulAdd.zw), decalDX, decalDY);
		[branch]
		if (normalTexture >= 0)
		{
			// u along the box's x, v against its y (box_to_uv)
			const half4 normalSample = bindless_textures_half4[descriptor_index(normalTexture)].SampleGrad(sam, mad(uvw.xy, texMulAdd.xy, texMulAdd.zw), decalDX, decalDY);
			decalN = GGDecalNormalWorld(normalSample, normalize(decalProjection[0].xyz), -normalize(decalProjection[1].xyz), N, decal.GetLength());
		}
		decalColor *= decal.GetColor();
		// blend out if close to cube Z:
		decalColor.a *= (1 - pow8(saturate(abs((half)clipSpacePos.z)))) * facingBlend;
	}
	return decalColor;
}

#endif // GG_DECALHF
