#include "graphics/host_gpu/renderer/gpuTimestamps.h"

#include <algorithm>
#include <array>
#include <cstdarg>
#include <string>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

#include "common/assert.h"
#include "common/diagOut.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/presentation/frameTiming.h"

namespace Libs::Graphics::GpuTs {

namespace {

template <typename... A>
void DiagPrintf(const char* fmt, A... args) {
	char line[700];
	std::snprintf(line, sizeof(line), fmt, args...);
	Common::DiagOut::Write(line);
}

struct Pending {
	int32_t  slot;
	uint64_t tick;
};

struct State {
	std::mutex                    mutex;
	bool                          on = false, inited = false, ready = false, broken = false;
	vk::QueryPool                 pool = nullptr;
	double                        period = 1.0;
	uint64_t                      mask = ~0ull;
	std::array<bool, kSlots>      busy {};
	uint32_t                      next = 0;
	std::deque<Pending>           pending;
	Acc                           acc;
	std::array<uint64_t, 6>       hist_n {}, hist_ns {};
	uint64_t                      window_n = 0, window_busy = 0, window_idle = 0;
	uint64_t                      window_begin = 0;
	std::vector<uint64_t>         bounds;
	size_t                        next_bound = 0;
	uint32_t                      poll_tick = 0, raw_logged = 0;
	// S544 DIAG segments
	bool                                                     seg_on = false;
	vk::QueryPool                                            seg_pool = nullptr;
	std::array<uint32_t, kSlots>                             seg_n {};
	std::vector<uint8_t>                                     seg_cls;  // kSlots * kSegQ
	std::vector<std::array<char, 48>>                        seg_key;  // kSlots * kSegQ
	std::unordered_map<VkCommandBuffer, int32_t>             seg_open;
	std::array<uint64_t, 14>                                 cls_ns {}, cls_n {};
	std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> key_acc;
	uint64_t                                                 seg_overflow = 0, seg_unread = 0;
};

constexpr uint32_t kSegQ = 128;

State& S() {
	static State s;
	return s;
}

void InitEnv(State& s) {
	s.inited      = true;
	const char* v = std::getenv("KYTY_GPU_TIMESTAMPS");
	s.on          = v != nullptr && v[0] == '1' && v[1] == '\0';
	const char* at = std::getenv("KYTY_GPU_TIMESTAMPS_AT");
	const char* list = at != nullptr && at[0] != '\0' ? at : "6300,6800,6990";
	char*       end  = nullptr;
	for (const char* p = list; *p != '\0';) {
		const auto b = std::strtoull(p, &end, 10);
		if (end == p) break;
		s.bounds.push_back(b);
		p = (*end == ',') ? end + 1 : end;
	}
	const char* seg = std::getenv("KYTY_GPU_TS_SEG");
	s.seg_on        = s.on && seg != nullptr && seg[0] == '1' && seg[1] == '\0';
}

void EnsurePool(State& s, GraphicContext& graphics) {
	if (s.ready || s.broken) return;
	const auto family = graphics.physical_device.getQueueFamilyProperties();
	const uint32_t bits = graphics.queue_family < family.size() ? family[graphics.queue_family].timestampValidBits : 0;
	if (bits == 0) {
		DiagPrintf("[gpu-ts] queue family has no timestamp support: disabled\n");
		s.broken = true;
		return;
	}
	s.mask   = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
	s.period = static_cast<double>(graphics.physical_device_properties.limits.timestampPeriod);
	vk::QueryPoolCreateInfo info {};
	info.queryType  = vk::QueryType::eTimestamp;
	info.queryCount = kSlots * 2;
	auto result     = graphics.device.createQueryPool(&info, nullptr, &s.pool);
	if (result != vk::Result::eSuccess) {
		DiagPrintf("[gpu-ts] createQueryPool failed: disabled\n");
		s.broken = true;
		return;
	}
	if (s.seg_on) {
		vk::QueryPoolCreateInfo seg_info {};
		seg_info.queryType  = vk::QueryType::eTimestamp;
		seg_info.queryCount = kSlots * kSegQ;
		if (graphics.device.createQueryPool(&seg_info, nullptr, &s.seg_pool) != vk::Result::eSuccess) {
			DiagPrintf("[gpu-ts-seg] createQueryPool failed: segments off\n");
			s.seg_on = false;
		} else {
			s.seg_cls.assign(static_cast<size_t>(kSlots) * kSegQ, 0);
			s.seg_key.assign(static_cast<size_t>(kSlots) * kSegQ, {});
			DiagPrintf("[gpu-ts-seg] enabled: %u marks per buffer\n", kSegQ);
		}
	}
	s.ready = true;
	DiagPrintf("[gpu-ts] enabled: period=%.3f ns valid_bits=%u slots=%u\n", s.period, bits, kSlots);
}

void Dump(State& s, uint64_t flip) {
	static const char* const names[] = {"<100us", "<300us", "<1ms", "<3ms", "<10ms", ">=10ms"};
	const double             fl      = static_cast<double>(flip > s.window_begin ? flip - s.window_begin : 1);
	char                     line[600];
	int                      len = std::snprintf(line, sizeof(line), "[gpu-ts] window flips %llu-%llu submits/flip=%.2f busy_ms/flip=%.3f idle_ms/flip=%.3f hist(n/flip|ms/flip):",
	                                             static_cast<unsigned long long>(s.window_begin), static_cast<unsigned long long>(flip), static_cast<double>(s.window_n) / fl,
	                                             static_cast<double>(s.window_busy) / fl / 1e6, static_cast<double>(s.window_idle) / fl / 1e6);
	for (size_t i = 0; i < 6 && len > 0 && static_cast<size_t>(len) < sizeof(line); i++) {
		len += std::snprintf(line + len, sizeof(line) - static_cast<size_t>(len), " %s=%.2f|%.3f", names[i], static_cast<double>(s.hist_n[i]) / fl,
		                     static_cast<double>(s.hist_ns[i]) / fl / 1e6);
	}
	DiagPrintf("%s\n", line);
	if (s.seg_on) {
		static const char* const cls_names[14] = {"G.cs", "G.rp", "I.tiler", "I.up", "I.fault", "I.pvind", "I.vsstore", "I.rb", "G.fill", "G.clr", "I.tex", "I.present", "head", "rp.end"};
		uint64_t total = 0;
		for (auto v : s.cls_ns) total += v;
		DiagPrintf("[gpu-ts-seg] window flips %llu-%llu total_ms/flip=%.3f overflow=%llu unread=%llu\n", static_cast<unsigned long long>(s.window_begin),
		           static_cast<unsigned long long>(flip), static_cast<double>(total) / fl / 1e6, static_cast<unsigned long long>(s.seg_overflow),
		           static_cast<unsigned long long>(s.seg_unread));
		for (size_t i = 0; i < 14; i++) {
			if (s.cls_n[i] == 0) continue;
			DiagPrintf("[gpu-ts-seg] class %-9s ms/flip=%.3f n/flip=%.2f us/seg=%.1f\n", cls_names[i], static_cast<double>(s.cls_ns[i]) / fl / 1e6,
			           static_cast<double>(s.cls_n[i]) / fl, static_cast<double>(s.cls_ns[i]) / static_cast<double>(s.cls_n[i]) / 1e3);
		}
		std::vector<std::pair<uint64_t, const std::string*>> top;
		top.reserve(s.key_acc.size());
		for (const auto& [k, v] : s.key_acc) top.emplace_back(v.first, &k);
		std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
		for (size_t i = 0; i < top.size() && i < 60; i++) {
			const auto& v = s.key_acc[*top[i].second];
			DiagPrintf("[gpu-ts-seg] key %-40s ms/flip=%.3f n/flip=%.2f us/seg=%.1f\n", top[i].second->c_str(), static_cast<double>(v.first) / fl / 1e6,
			           static_cast<double>(v.second) / fl, static_cast<double>(v.first) / static_cast<double>(v.second) / 1e3);
		}
		DiagPrintf("[gpu-ts-seg] keys=%zu\n", s.key_acc.size());
		s.cls_ns.fill(0);
		s.cls_n.fill(0);
		s.key_acc.clear();
		s.seg_overflow = s.seg_unread = 0;
	}
	s.hist_n.fill(0);
	s.hist_ns.fill(0);
	s.window_n = s.window_busy = s.window_idle = 0;
	s.window_begin = flip;
}

} // namespace

bool Enabled() {
	auto& s = S();
	if (!s.inited) InitEnv(s);
	return s.on && !s.broken;
}

int32_t OnBegin(GraphicContext& graphics, vk::CommandBuffer buffer) {
	if (!Enabled()) return -1;
	auto&            s = S();
	std::lock_guard  lock(s.mutex);
	EnsurePool(s, graphics);
	if (!s.ready) return -1;
	const auto slot = static_cast<int32_t>(s.next % kSlots);
	if (s.busy[static_cast<size_t>(slot)]) {
		FrameTiming::Add(FrameTiming::Counter::GpuTsSkipN);
		return -1;
	}
	s.busy[static_cast<size_t>(slot)] = true;
	s.next++;
	buffer.resetQueryPool(s.pool, static_cast<uint32_t>(slot) * 2, 2);
	buffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, s.pool, static_cast<uint32_t>(slot) * 2);
	if (s.seg_on) {
		buffer.resetQueryPool(s.seg_pool, static_cast<uint32_t>(slot) * kSegQ, kSegQ);
		s.seg_n[static_cast<size_t>(slot)] = 0;
		s.seg_open[static_cast<VkCommandBuffer>(buffer)] = slot;
	}
	return slot;
}

void OnEnd(GraphicContext&, vk::CommandBuffer buffer, int32_t slot) {
	if (slot < 0) return;
	auto& s = S();
	if (s.seg_on) {
		std::lock_guard lock(s.mutex);
		s.seg_open.erase(static_cast<VkCommandBuffer>(buffer));
	}
	buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, s.pool, static_cast<uint32_t>(slot) * 2 + 1);
}

void OnSubmit(int32_t slot, uint64_t tick) {
	if (slot < 0) return;
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	auto            it = s.pending.end();
	while (it != s.pending.begin() && (it - 1)->tick > tick) --it;
	s.pending.insert(it, Pending {slot, tick});
}

void Poll(GraphicContext& graphics, MasterSemaphore& master) {
	if (!Enabled()) return;
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	if (!s.ready) return;
	if ((s.poll_tick++ & 7u) == 0) master.Refresh();
	while (!s.pending.empty()) {
		const auto p = s.pending.front();
		if (!master.IsFree(p.tick)) break;
		uint64_t data[2] = {};
		const auto result = graphics.device.getQueryPoolResults(s.pool, static_cast<uint32_t>(p.slot) * 2, 2, sizeof(data), data, sizeof(uint64_t),
		                                                       vk::QueryResultFlagBits::e64);
		if (result == vk::Result::eNotReady) break;
		s.pending.pop_front();
		s.busy[static_cast<size_t>(p.slot)] = false;
		if (result != vk::Result::eSuccess) continue;
		const auto     start = data[0] & s.mask, end = data[1] & s.mask;
		if (s.raw_logged < 60 || end < start || (end - start) * s.period > 1e9) { // anomalies and the first pairs, raw
			if (s.raw_logged < 400) {
				s.raw_logged++;
				DiagPrintf("[gpu-ts] raw slot=%d tick=%llu start=%llu end=%llu dur_us=%.1f gap_to_prev_end_us=%.1f\n", p.slot, static_cast<unsigned long long>(p.tick),
				           static_cast<unsigned long long>(start), static_cast<unsigned long long>(end), static_cast<double>(end - start) * s.period / 1e3,
				           s.acc.have ? (static_cast<double>(start) - static_cast<double>(s.acc.prev_end)) * s.period / 1e3 : 0.0);
			}
		}
		if (end < start || static_cast<double>(end - start) * s.period > 2e9) { // a pair that cannot be a submit: counted, not accumulated
			FrameTiming::Add(FrameTiming::Counter::GpuTsSkipN);
			continue;
		}
		if (s.seg_on) {
			const uint32_t n    = s.seg_n[static_cast<size_t>(p.slot)];
			uint64_t       m[kSegQ] = {};
			bool           ok   = true;
			if (n > 0) {
				ok = graphics.device.getQueryPoolResults(s.seg_pool, static_cast<uint32_t>(p.slot) * kSegQ, n, sizeof(uint64_t) * n, m, sizeof(uint64_t),
				                                         vk::QueryResultFlagBits::e64) == vk::Result::eSuccess;
			}
			if (!ok) {
				s.seg_unread++;
			} else {
				const size_t base = static_cast<size_t>(p.slot) * kSegQ;
				uint64_t     prev = start;
				auto         give = [&](uint8_t cls, const char* key, uint64_t from, uint64_t to) {
                    const uint64_t ns = to > from ? static_cast<uint64_t>(static_cast<double>(to - from) * s.period) : 0;
                    s.cls_ns[cls] += ns;
                    s.cls_n[cls]++;
                    auto& a = s.key_acc[key];
                    a.first += ns;
                    a.second++;
				};
				uint8_t     cls = Seg::kHead;
				const char* key = "head";
				for (uint32_t i = 0; i < n; i++) {
					const uint64_t t = std::max(m[i] & s.mask, prev);
					give(cls, key, prev, std::min(t, end));
					prev = std::min(t, end);
					cls  = s.seg_cls[base + i];
					key  = s.seg_key[base + i].data();
				}
				give(cls, key, prev, end);
			}
		}
		const auto     before_busy = s.acc.busy_ns, before_idle = s.acc.idle_ns, before_sum = s.acc.sum_ns, before_ov = s.acc.overlap_n;
		const uint64_t busy = Accumulate(s.acc, start, end, s.period);
		using C = FrameTiming::Counter;
		FrameTiming::Add(C::GpuTsN);
		FrameTiming::Add(C::GpuTsBusyNs, s.acc.busy_ns - before_busy);
		FrameTiming::Add(C::GpuTsIdleNs, s.acc.idle_ns - before_idle);
		FrameTiming::Add(C::GpuTsSumNs, s.acc.sum_ns - before_sum);
		FrameTiming::Add(C::GpuTsOverlapN, s.acc.overlap_n - before_ov);
		const auto b = Bucket(s.acc.sum_ns - before_sum);
		s.hist_n[b]++;
		s.hist_ns[b] += busy;
		s.window_n++;
		s.window_busy += s.acc.busy_ns - before_busy;
		s.window_idle += s.acc.idle_ns - before_idle;
	}
	const auto flip = FrameTiming::GuestFlips();
	if (s.next_bound < s.bounds.size() && flip >= s.bounds[s.next_bound]) {
		s.next_bound++;
		Dump(s, flip);
	}
}

namespace Seg {

bool Enabled() {
	static const bool on = [] {
		auto& s = S();
		if (!s.inited) InitEnv(s);
		return s.seg_on;
	}();
	return on;
}

void MarkV(vk::CommandBuffer buffer, uint8_t cls, const char* format, ...) {
	if (!Enabled()) return;
	char    text[64];
	va_list args;
	va_start(args, format);
	std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	// fold the varying parts of the key: pass id (last field of G.rp), digits of our own work (sizes, counts); keep the shader hash of G.cs
	if (cls == 1) {
		if (char* dot = std::strrchr(text, '.'); dot != nullptr) *dot = '\0';
	} else if (cls != 0) {
		char* w = text;
		for (const char* r = text; *r != '\0'; r++) {
			if (*r >= '0' && *r <= '9') {
				if (w == text || w[-1] != '#') *w++ = '#';
			} else {
				*w++ = *r;
			}
		}
		*w = '\0';
	}
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	if (!s.seg_on || !s.ready) return;
	const auto it = s.seg_open.find(static_cast<VkCommandBuffer>(buffer));
	if (it == s.seg_open.end()) return;
	const auto slot = static_cast<size_t>(it->second);
	auto&      n    = s.seg_n[slot];
	if (n >= kSegQ) {
		s.seg_overflow++;
		return;
	}
	const size_t idx = slot * kSegQ + n;
	buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, s.seg_pool, static_cast<uint32_t>(idx));
	s.seg_cls[idx] = cls < 14 ? cls : Seg::kHead;
	std::snprintf(s.seg_key[idx].data(), 48, "%s", text);
	n++;
}

} // namespace Seg

} // namespace Libs::Graphics::GpuTs
