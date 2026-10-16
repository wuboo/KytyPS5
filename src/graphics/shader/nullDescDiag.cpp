#include "graphics/shader/nullDescDiag.h"

#include "common/logging/log.h"

#include <atomic>
#include <mutex>

namespace Libs::Graphics::NullDesc {
namespace {

std::atomic<uint64_t> g_standin[5] {};
std::mutex            g_standin_lock;
TopList               g_standin_top;
bool                  g_top_dumped = false;

uint64_t g_last_standin[5] {};
uint64_t g_last_walker_nulltable   = 0;
uint64_t g_last_walker_unresolved  = 0;
uint64_t g_last_minlod_remap       = 0;
uint64_t g_last_minlod_dropped     = 0;
bool     g_announced               = false;

} // namespace

void NoteStandin(uint64_t shader, uint32_t slot, uint64_t addr, Reason reason) {
	if (!Enabled()) {
		return;
	}
	const auto index = static_cast<uint32_t>(reason);
	if (index < 5) {
		g_standin[index].fetch_add(1, std::memory_order_relaxed);
	}
	if (!InWindow(Window(), DrawingFrame().load(std::memory_order_relaxed))) {
		return;
	}
	std::lock_guard lock {g_standin_lock};
	g_standin_top.Add(shader, slot, addr, reason);
}

void OnPresented(uint32_t frame, uint32_t guest_flips, uint64_t minlod_remap_total, uint64_t minlod_dropped_total) {
	if (!Enabled()) {
		return;
	}
	if (!g_announced) {
		g_announced = true;
		const auto window = Window();
		LOGF("[nulldesc] on frame_window=%u-%u (frames.csv frame / frame_flip_N, not guest flips)\n", window.lo,
		     window.hi);
	}
	uint64_t standin[5];
	for (uint32_t i = 0; i < 5; i++) {
		standin[i] = g_standin[i].load(std::memory_order_relaxed);
	}
	const auto walker_null = WalkerNullTotal().load(std::memory_order_relaxed);
	const auto walker_bad  = WalkerUnresolvedTotal().load(std::memory_order_relaxed);
	const auto empty       = static_cast<uint32_t>(Reason::EmptyT);
	const auto nulltable   = static_cast<uint32_t>(Reason::NullTable);
	const auto unresolved  = static_cast<uint32_t>(Reason::Unresolved);
	const auto other       = static_cast<uint32_t>(Reason::Other);
	LOGF("[nulldesc] frame=%u guest_flips=%u standin_empty=%llu standin_nulltable=%llu "
	     "standin_unresolved=%llu standin_other=%llu walker_nulltable=%llu walker_unresolved=%llu "
	     "minlod_remap=%llu minlod_dropped=%llu\n",
	     frame, guest_flips, static_cast<unsigned long long>(standin[empty] - g_last_standin[empty]),
	     static_cast<unsigned long long>(standin[nulltable] - g_last_standin[nulltable]),
	     static_cast<unsigned long long>(standin[unresolved] - g_last_standin[unresolved]),
	     static_cast<unsigned long long>(standin[other] - g_last_standin[other]),
	     static_cast<unsigned long long>(walker_null - g_last_walker_nulltable),
	     static_cast<unsigned long long>(walker_bad - g_last_walker_unresolved),
	     static_cast<unsigned long long>(minlod_remap_total - g_last_minlod_remap),
	     static_cast<unsigned long long>(minlod_dropped_total - g_last_minlod_dropped));
	for (uint32_t i = 0; i < 5; i++) {
		g_last_standin[i] = standin[i];
	}
	g_last_walker_nulltable  = walker_null;
	g_last_walker_unresolved = walker_bad;
	g_last_minlod_remap      = minlod_remap_total;
	g_last_minlod_dropped    = minlod_dropped_total;
	DrawingFrame().store(frame + 1, std::memory_order_relaxed);

	const auto window = Window();
	if (g_top_dumped || frame != window.hi) {
		return;
	}
	g_top_dumped = true;
	std::vector<TopEntry> standin_top;
	std::vector<TopEntry> walker_top;
	{
		std::lock_guard lock {g_standin_lock};
		standin_top = g_standin_top.Top(20);
	}
	{
		std::lock_guard lock {WalkerTopLock()};
		walker_top = WalkerTop().Top(20);
	}
	LOGF("[nulldesc] top frame=%u-%u standin=%zu walker=%zu\n", window.lo, window.hi, standin_top.size(),
	     walker_top.size());
	for (uint32_t rank = 0; rank < standin_top.size(); rank++) {
		const auto& hit = standin_top[rank];
		LOGF("[nulldesc] top rank=%u n=%u reason=%s shader=0x%016llx slot=%u addr=0x%llx\n", rank + 1, hit.n,
		     ReasonName(hit.reason), static_cast<unsigned long long>(hit.shader), hit.slot,
		     static_cast<unsigned long long>(hit.addr));
	}
	for (uint32_t rank = 0; rank < walker_top.size(); rank++) {
		const auto& hit = walker_top[rank];
		LOGF("[nulldesc] walker-top rank=%u n=%u table=0x%llx\n", rank + 1, hit.n,
		     static_cast<unsigned long long>(hit.addr));
	}
}

} // namespace Libs::Graphics::NullDesc
