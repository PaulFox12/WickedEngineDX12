#pragma once

// If this is defined, job system will use the custom wi::function with disabled allocation
//	otherwise it will use standard std::function that might allocate, but more flexible in how much you can store in lambdas
#define JOB_SYSTEM_FIXED_SIZE_FUNCTION

#ifdef JOB_SYSTEM_FIXED_SIZE_FUNCTION
#include "wiFunction.h"
#else
#include <functional>
#endif // JOB_SYSTEM_FIXED_SIZE_FUNCTION

#include <atomic>

namespace wi::jobsystem
{
	void Initialize(uint32_t maxThreadCount = ~0u);
	void ShutDown();

	// GG: how the workers are tied to cores, set before Initialize: 1 each worker's core as a preference only
	// (SetThreadIdealProcessor, the default), 0 each pinned to its own core (Wicked's way), 2 not tied. A pinned worker cannot
	// move when its core is slow to run it (an efficiency or low power core on a hybrid CPU), and a job it holds waits with it
	void SetAffinityMode(int mode);

	// Returns true if the job system is shutting down
	//	Long-running (multi-frame) jobs should ideally check this and exit themselves if true
	bool IsShuttingDown();

	struct JobArgs
	{
		uint32_t jobIndex;		// job index relative to dispatch (like SV_DispatchThreadID in HLSL)
		uint32_t groupID;		// group index relative to dispatch (like SV_GroupID in HLSL)
		uint32_t groupIndex;	// job index relative to group (like SV_GroupIndex in HLSL)
		bool isFirstJobInGroup;	// is the current job the first one in the group?
		bool isLastJobInGroup;	// is the current job the last one in the group?
		void* sharedmemory;		// stack memory shared within the current group (jobs within a group execute serially)
	};

#ifdef JOB_SYSTEM_FIXED_SIZE_FUNCTION
	using job_function_type = wi::function<void(JobArgs), 96>; // 96 is chosen to fit the whole job storage with parameters in 128 bytes
#else
	using job_function_type = std::function<void(JobArgs)>;
#endif // JOB_SYSTEM_FIXED_SIZE_FUNCTION

	enum class Priority
	{
		High,		// Default
		Low,		// Pool of low priority threads, useful for generic tasks that shouldn't interfere with high priority tasks
		Streaming,	// Single low priority thread, for streaming resources
		Count
	};

	// Defines a state of execution, can be waited on
	struct context
	{
		std::atomic<uint32_t> counter{ 0 };
		Priority priority = Priority::High;
	};

	uint32_t GetThreadCount(Priority priority = Priority::High);

	// Add a task to execute asynchronously. Any idle thread will execute this.
	void Execute(context& ctx, const job_function_type& task);

	// Divide a task onto multiple jobs and execute in parallel.
	//	jobCount	: how many jobs to generate for this task.
	//	groupSize	: how many jobs to execute per thread. Jobs inside a group execute serially. It might be worth to increase for small jobs
	//	task		: receives a JobArgs as parameter
	//	sharedmemory_size	: size of shared memory allocation for every thread group. The pointer will be 64-byte aligned
	void Dispatch(context& ctx, uint32_t jobCount, uint32_t groupSize, const job_function_type& task, size_t sharedmemory_size = 0);

	// Returns the amount of job groups that will be created for a set number of jobs and group size
	uint32_t DispatchGroupCount(uint32_t jobCount, uint32_t groupSize);

	// Check if any threads are working currently or not
	bool IsBusy(const context& ctx);

	// Wait until all threads become idle
	//	Current thread will become a worker thread, executing jobs
	void Wait(const context& ctx);

	// Returns the number of remaining jobs
	uint32_t GetRemainingJobCount(const context& ctx);
}

// GG: a high priority job that ran for g_dWickedJobStallMilliseconds or more, told to g_pfnWickedJobStall by the thread that
// ran it: what queued it (the job's function, for a lambda its type name, which names the function it was written in), how
// long it ran, and whether that thread was waiting on a context (Wait runs queued jobs of any context). The time it queued
// is not measured here: a job is kept in 128 bytes, with no room for it
struct WickedJobStallInfo
{
	const char* name;
	const void* context;			// the job's own
	const void* waitingFor;			// the context the running thread was waiting on, or null for a worker's own loop
	double runMilliseconds;
	uint32_t jobs;					// jobs in its group, run one after the other
	uint32_t coreStart, coreEnd;	// GetCurrentProcessorNumber when it started and ended
};
extern void (*g_pfnWickedJobStall)(const WickedJobStallInfo& info);
extern double g_dWickedJobStallMilliseconds; // 20 by default
