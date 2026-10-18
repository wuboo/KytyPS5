#ifndef KYTY_GRAPHICS_PRESENTATION_CPSAMPLER_H_
#define KYTY_GRAPHICS_PRESENTATION_CPSAMPLER_H_

// S406 (KYTY_DIAG_CP_SAMPLE=1): breadcrumb sampler of the command-processor thread, without signals.
// The CP thread only stores two relaxed atomics: the site it is in (g_cp_site, set by CpPhases::Scope and by
// KYTY_CP_SITE(id), restored on leaving) and the PM4 opcode of the handler running (g_cp_opcode). A separate
// thread reads both at 10 kHz and counts (site, opcode) pairs. No clock on the CP thread, no SIGPROF (which does
// not symbolize Rosetta code and is risky under heavy SIGSEGV traffic).

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <thread>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

namespace Libs::Graphics::CpSampler {

// Sites 0..16 are the CpPhases::Phase values (same numbers); the rest are the sub-sites of the draw / dispatch
// handlers. Names (SiteName) are the strings of the [cp-sample] log lines.
enum class Site : uint8_t {
	// 0..16 = CpPhases::Phase (Parse .. Cmds, Wait); kept as raw numbers, see PhaseSite().
	None = 17,
	// DrawShaders / Dispatch get_program
	ShStaticKeyFind,
	ShRefreshCallSource,
	ShWalk,
	ShMaterializeRest,
	ShPermSearch,
	ShProgramsRest,
	ShSpecRead, // S430: ReadShaderGuestMemory (the strict reader), a child of ShWalk / ShPermSearch
	// DrawBind
	BindResolveTexture,
	BindSamplers,
	BindFindBuffers,
	BindRebindImages,
	BindRebindBuffers,
	BindNativeUpload,
	// S430: children of BindResolveTexture (ResolveTexture)
	BindResolveDescCopy,
	BindResolveFindImage,
	BindResolveValidate,
	// PvStore
	PvIndexHash,
	PvRangeScan,
	PvObtainTail,
	PvPlace,
	// Dispatch
	DispBind,
	DispDesc,
	DispFillResolve,
	DispRecord,
	// Rest of the draw handler
	DrawVertexBuffers,
	DrawIndexBuffer,
	DrawRenderState,
	DrawFlushProto,
	DrawEagerSplit,
	Count,
};
constexpr size_t kPhaseSites = 17;
constexpr size_t kSiteCount  = static_cast<size_t>(Site::Count);
constexpr uint32_t kNoOpcode = 0xFFFFFFFFu;

constexpr uint32_t PhaseSite(uint32_t phase) { return phase; }
constexpr uint32_t SiteId(Site site) { return static_cast<uint32_t>(site); }

inline const char* SiteName(uint32_t id) {
	static constexpr const char* kNames[] = {
	    "parse", "handler", "draw_shaders", "draw_bind", "draw_pipeline", "draw_targets", "draw_desc", "draw_vkcmd", "pvstore",
	    "dispatch", "eop", "barrier", "readback_cpu", "gc", "submit_cpu", "cmds", "wait",
	    "none",
	    "sh.static_key_find", "sh.refresh_call_source", "sh.walk", "sh.materialize_rest", "sh.perm_search", "sh.programs_rest", "sh.spec_read",
	    "bind.resolve_texture", "bind.samplers", "bind.find_buffers", "bind.rebind_images", "bind.rebind_buffers", "bind.native_upload", "bind.resolve.desc_copy", "bind.resolve.find_image", "bind.resolve.validate",
	    "pv.index_hash", "pv.range_scan", "pv.obtain_tail", "pv.place",
	    "disp.bind", "disp.desc", "disp.fill_resolve", "disp.record",
	    "draw.vertex_buffers", "draw.index_buffer", "draw.render_state", "draw.flush_proto", "draw.eager_split",
	};
	static_assert(sizeof(kNames) / sizeof(kNames[0]) == kSiteCount);
	return id < kSiteCount ? kNames[id] : "?";
}

constexpr uint32_t kWaitSite = 16;

// False unless KYTY_DIAG_CP_SAMPLE=1 (StartProduction) or a test: then SiteScope costs one relaxed load.
inline std::atomic<bool>& Active() {
	static std::atomic<bool> on {false};
	return on;
}

// Breadcrumbs written by the sampled thread. One pair per process (there is a single CP thread).
inline std::atomic<uint32_t>& CpSite() {
	static std::atomic<uint32_t> site {SiteId(Site::None)};
	return site;
}
// S413: the site that was current when the CP thread entered the Wait site (its parent), so a wait is attributed to what asked for it.
inline std::atomic<uint32_t>& WaitParent() {
	static std::atomic<uint32_t> parent {SiteId(Site::None)};
	return parent;
}
inline std::atomic<uint32_t>& CpOpcode() {
	static std::atomic<uint32_t> opcode {kNoOpcode};
	return opcode;
}

// Histogram [site][opcode byte or 256 = none]. Plain counters: the sampler thread is the only writer.
class Histogram {
public:
	static constexpr size_t kOps = 257;
	// wait_parent: the site the CP thread was in when it entered the Wait site; only read when site == kWaitSite.
	void Add(uint32_t site, uint32_t opcode, uint32_t wait_parent = SiteId(Site::None)) {
		if (site >= kSiteCount) site = SiteId(Site::None);
		const size_t op = opcode == kNoOpcode ? kOps - 1 : (opcode & 0xFFu);
		m_cells[site * kOps + op]++;
		m_total++;
		if (site == kWaitSite) {
			if (wait_parent >= kSiteCount || wait_parent == kWaitSite) wait_parent = SiteId(Site::None);
			m_wait_cells[wait_parent * kOps + op]++;
			m_wait++;
		}
	}
	uint64_t Cell(uint32_t site, size_t op) const { return m_cells[site * kOps + op]; }
	// Samples in the Wait site whose parent site was `parent`.
	uint64_t WaitCell(uint32_t parent, size_t op) const { return m_wait_cells[parent * kOps + op]; }
	uint64_t SiteTotal(uint32_t site) const {
		uint64_t sum = 0;
		for (size_t op = 0; op < kOps; op++) sum += Cell(site, op);
		return sum;
	}
	uint64_t Total() const { return m_total; }
	uint64_t Wait() const { return m_wait; }
	void Clear() {
		m_cells.fill(0);
		m_wait_cells.fill(0);
		m_total = 0;
		m_wait  = 0;
	}

private:
	std::array<uint64_t, kSiteCount * kOps> m_cells {};
	std::array<uint64_t, kSiteCount * kOps> m_wait_cells {};
	uint64_t                                m_total = 0;
	uint64_t                                m_wait  = 0;
};

// Setting / restoring the breadcrumb. Restore is the point of the RAII: a nested site returns to its parent.
class SiteScope {
public:
	explicit SiteScope(uint32_t site) {
		if (!Active().load(std::memory_order_relaxed)) {
			m_on = false;
			return;
		}
		auto& cell = CpSite();
		m_prev     = cell.load(std::memory_order_relaxed);
		if (site == kWaitSite) WaitParent().store(m_prev, std::memory_order_relaxed);
		cell.store(site, std::memory_order_relaxed);
	}
	~SiteScope() {
		if (!m_on) return;
#if !defined(KYTY_CPSAMPLER_MUTANT_NO_RESTORE)
		CpSite().store(m_prev, std::memory_order_relaxed);
#endif
	}
	SiteScope(const SiteScope&)            = delete;
	SiteScope& operator=(const SiteScope&) = delete;

private:
	uint32_t m_prev = 0;
	bool     m_on   = true;
};

// Flip window of the diagnostics (KYTY_DIAG_CP_SAMPLE_WINDOW=from:to, default 6300:6800); also used by KYTY_DIAG_MATMEMO.
struct FlipWindow {
	uint64_t from = 6300;
	uint64_t to   = 6800;
};
inline FlipWindow Window() {
	static const FlipWindow window = [] {
		FlipWindow  result;
		const char* text = std::getenv("KYTY_DIAG_CP_SAMPLE_WINDOW");
		if (text != nullptr && *text != '\0') {
			unsigned long long a = 0, b = 0;
			if (std::sscanf(text, "%llu:%llu", &a, &b) == 2 && b >= a) result = {a, b};
		}
		return result;
	}();
	return window;
}

// The 10 kHz sampler thread. `on_sample` (optional) runs on the sampler thread before every sample.
class Sampler {
public:
	// Runs before every sample; returns whether to count it (production: only inside the flip window, and logs/clears).
	using Hook = bool (*)(Histogram&);
	~Sampler() { Stop(); }
	void Start(uint32_t period_us, Hook on_sample = nullptr) {
		m_stop.store(false);
		m_thread = std::thread([this, period_us, on_sample] { Run(period_us, on_sample); });
	}
	void Stop() {
		m_stop.store(true);
		if (m_thread.joinable()) m_thread.join();
	}
	Histogram&       Result() { return m_hist; }
	const Histogram& Result() const { return m_hist; }

private:
	void Run(uint32_t period_us, Hook on_sample) {
#if defined(__APPLE__)
		mach_timebase_info_data_t tb {};
		mach_timebase_info(&tb);
		const uint64_t period_ticks = static_cast<uint64_t>(period_us) * 1000ull * tb.denom / tb.numer;
		uint64_t       next         = mach_absolute_time() + period_ticks;
#else
		auto next = std::chrono::steady_clock::now() + std::chrono::microseconds(period_us);
#endif
		while (!m_stop.load(std::memory_order_relaxed)) {
#if defined(__APPLE__)
			mach_wait_until(next);
			next += period_ticks;
#else
			std::this_thread::sleep_until(next);
			next += std::chrono::microseconds(period_us);
#endif
			if (on_sample != nullptr && !on_sample(m_hist)) continue;
			const auto site = CpSite().load(std::memory_order_relaxed);
			m_hist.Add(site, CpOpcode().load(std::memory_order_relaxed), WaitParent().load(std::memory_order_relaxed));
		}
	}

	Histogram         m_hist;
	std::atomic<bool> m_stop {false};
	std::thread       m_thread;
};

// Process-wide production sampler (frameTiming.cpp); no-op unless KYTY_DIAG_CP_SAMPLE=1.
bool Enabled();
void StartProduction();

} // namespace Libs::Graphics::CpSampler

#define KYTY_CP_SITE(id) \
	::Libs::Graphics::CpSampler::SiteScope kyty_cp_site_scope_##id(::Libs::Graphics::CpSampler::SiteId(::Libs::Graphics::CpSampler::Site::id))

#endif
