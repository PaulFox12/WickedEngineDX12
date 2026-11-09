#pragma once
#include "CommonInclude.h"
#include "wiGraphicsDevice.h"

namespace wi::gpusortlib
{
	// Perform bitonic sort on a GPU dataset
	//	maxCount				-	Maximum size of the dataset. GPU count can be smaller (see: counterBuffer_read param)
	//	comparisonBuffer_read	-	Buffer containing values to compare by (Read Only)
	//	counterBuffer_read		-	Buffer containing count of values to sort (Read Only)
	//	counterReadOffset		-	Byte offset into the counter buffer to read the count value (Read Only)
	//	indexBuffer_write		-	The index list which to sort. Contains index values which can index the sortBase_read buffer. This will be modified (Read + Write)
	void Sort(
		uint32_t maxCount, 
		const wi::graphics::GPUBuffer& comparisonBuffer_read, 
		const wi::graphics::GPUBuffer& counterBuffer_read, 
		uint32_t counterReadOffset, 
		const wi::graphics::GPUBuffer& indexBuffer_write,
		wi::graphics::CommandList cmd
	);

	// GG: the same for a dataset of at most 512, which needs one group of the first pass and no arguments worked out on the GPU. No
	// barriers: a caller with many of these puts one after them all
	void SortSmall(
		const wi::graphics::GPUBuffer& comparisonBuffer_read,
		const wi::graphics::GPUBuffer& counterBuffer_read,
		uint32_t counterReadOffset,
		const wi::graphics::GPUBuffer& indexBuffer_write,
		wi::graphics::CommandList cmd
	);

	void Initialize();
};
