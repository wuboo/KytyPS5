// S607 (KYTY_WORKER_CLEAN_WALK): a pipeline worker serves a clean walker load of a hot page from the backing; the recorder checks the served bytes
// again when it joins the job (cpPipeCleanWalk.h). Every check ends in exit(1) when it fails (no crash), so each mutant (CMake
// cp_pipe_clean_walk_mutant_*, WILL_FAIL) is caught by a failed check:
//   BUSY_SERVES          - a load whose tracker query gave up is served (the decision table);
//   JOIN_NO_BYTES        - the join does not check the served bytes (the oracle: a dword that became GPU-dirty before the join is used);
//   JOIN_NO_INFLIGHT     - the join's byte question forgets the copies in flight (the oracle: a dword whose download is not published is used);
//   JOIN_NO_WRITTEN      - the pages of served loads are not asked for backing writes (the oracle: a dword rewritten by a publish is used);
//   SPLIT_DROPS_READ     - a plain read on a served page is checked neither by page nor by bytes (split cases, the oracle);
//   PAGE_CHECKS_SERVED   - the served pages keep the page check (never wrong, nothing gained: the split cases and the oracle's count of jobs
//                          used although their served page held other GPU-dirty bytes);
//   SPLIT_IGNORES_BACKING - a served page the worker also read through the backing is checked by bytes only (the oracle: that read is stale).
// The oracle runs the real tracker pieces (GpuDirtyRanges, InflightRanges, BackingWriteLog): a CP thread that marks GPU writes, records downloads
// and joins jobs, a publish thread that writes the downloaded bytes into the backing, and three workers that walk jobs as RunGfpJob does.

#include "graphics/guest_gpu/command_processor/cpPipeCleanWalk.h"
#include "graphics/host_gpu/renderer/cache/gpuDirtyRanges.h"
#include "graphics/host_gpu/renderer/cache/inflightRanges.h"
#include "kernel/backingWriteLog.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics::CpPipe::WorkerCleanWalk;
using Range = std::pair<uint64_t, uint64_t>;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "CpPipeCleanWalkTests: failed: %s\n", text);
		std::exit(1);
	}
}

void ModeTests() {
	Check(Parse(nullptr) == Mode::Off && Parse("0") == Mode::Off && Parse("yes") == Mode::Off, "default off");
	Check(Parse("1") == Mode::On && Parse("shadow") == Mode::Shadow, "1 and shadow");
	Check(std::string_view(ModeName(Mode::Shadow)) == "shadow" && std::string_view(ModeName(Mode::On)) == "1" && std::string_view(ModeName(Mode::Off)) == "0",
	      "names");
	Check(ParseSpin(nullptr) == kDefaultSpin && ParseSpin("") == kDefaultSpin && ParseSpin("x") == kDefaultSpin && ParseSpin("0") == kDefaultSpin &&
	          ParseSpin("128") == 128 && ParseSpin("99999999") == kDefaultSpin,
	      "spin: default, junk, value");
	Check(ActiveFor(Mode::On, true, true, true) && ActiveFor(Mode::Shadow, true, true, true), "active with the pipe, clean walk and nofault");
	Check(!ActiveFor(Mode::Off, true, true, true) && !ActiveFor(Mode::On, false, true, true) && !ActiveFor(Mode::On, true, false, true) &&
	          !ActiveFor(Mode::On, true, true, false),
	      "inactive without any of them");
}

void DecideTests() {
	Check(Decide(true, true, false, false) == Answer::Serve, "clean, answered: serve");
	Check(Decide(true, true, true, false) == Answer::Dirty && Decide(true, true, false, true) == Answer::Dirty && Decide(true, true, true, true) == Answer::Dirty,
	      "dirty or in flight: refuse");
	Check(Decide(true, false, false, false) == Answer::Busy, "the query gave up: busy, not served");
	Check(Decide(false, true, false, false) == Answer::Dirty, "the CP's clean walk is off: never served");
	Check(BytesBusy(true, false) && BytesBusy(false, true) && !BytesBusy(false, false), "the join's byte question: dirty or in flight");
}

bool Has(const std::vector<uint64_t>& v, uint64_t x) {
	return std::find(v.begin(), v.end(), x) != v.end();
}
bool HasRange(const std::vector<Range>& v, uint64_t a, uint64_t n) {
	return std::find(v.begin(), v.end(), Range {a, n}) != v.end();
}

void SplitTests() {
	constexpr uint64_t P = 0x100000;
	// A served load on a page nobody read through the backing: its dword goes to the byte check, its page only to the backing-write check.
	{
		std::vector<uint64_t> pages {P + 0x3000};
		const std::vector<uint64_t> served {P + 0x1010};
		const std::vector<Range>    reads {{P + 0x1010, 4}, {P + 0x1020, 4}, {P + 0x2000, 8}};
		std::vector<Range>          bytes;
		std::vector<uint64_t>       cw;
		Check(SplitReads(pages, served, reads, bytes, cw, 4096), "split ok");
		Check(!Has(pages, P + 0x1000), "a served page is not checked as a page");
		Check(Has(cw, P + 0x1000) && cw.size() == 1, "the served page is asked for backing writes");
		Check(HasRange(bytes, P + 0x1010, 4), "the served dword is checked by bytes");
		Check(HasRange(bytes, P + 0x1020, 4), "a plain read on the served page is checked by bytes");
		Check(Has(pages, P + 0x2000) && Has(pages, P + 0x3000) && pages.size() == 2, "plain pages and the backing log keep the page check");
	}
	// A served page the worker also read through the backing keeps the page check (that read has no byte range).
	{
		std::vector<uint64_t> pages {P + 0x1000};
		const std::vector<uint64_t> served {P + 0x1010};
		const std::vector<Range>    reads {{P + 0x1010, 4}};
		std::vector<Range>          bytes;
		std::vector<uint64_t>       cw;
		Check(SplitReads(pages, served, reads, bytes, cw, 4096), "split ok (backing)");
		Check(Has(pages, P + 0x1000) && bytes.empty() && cw.empty(), "a served page read through the backing stays a page check");
	}
	// A read across a served page and a plain page: checked by bytes, both pages asked for writes.
	{
		std::vector<uint64_t> pages;
		const std::vector<uint64_t> served {P + 0x1ff0};
		const std::vector<Range>    reads {{P + 0x1ffc, 8}};
		std::vector<Range>          bytes;
		std::vector<uint64_t>       cw;
		Check(SplitReads(pages, served, reads, bytes, cw, 4096), "split ok (across)");
		Check(HasRange(bytes, P + 0x1ffc, 8) && Has(cw, P + 0x1000) && Has(cw, P + 0x2000), "a read across a served page: bytes, both pages");
	}
	// Too many pages: the job is left to the CP.
	{
		std::vector<uint64_t> pages;
		const std::vector<uint64_t> served {P};
		const std::vector<Range>    reads {{P + 0x4000, 0x10000}};
		std::vector<Range>          bytes;
		std::vector<uint64_t>       cw;
		Check(!SplitReads(pages, served, reads, bytes, cw, 8), "a range of more pages than the limit");
	}
	// Random cases against a reference written from the rule, not from the code.
	std::mt19937_64 rng(607);
	for (int round = 0; round < 4000; round++) {
		std::vector<uint64_t> backing;
		std::vector<uint64_t> served;
		std::vector<Range>    reads;
		const auto            page = [&] { return P + (rng() % 8) * kPage; };
		for (int i = rng() % 3; i > 0; i--) backing.push_back(page());
		for (int i = rng() % 4; i > 0; i--) served.push_back(page() + (rng() % 1024) * 4);
		for (int i = rng() % 6; i > 0; i--) reads.emplace_back(page() + (rng() % 1024) * 4, (rng() % 4 == 0) ? 4 + (rng() % 3) * 4096 : 4);
		for (const uint64_t a: served) reads.emplace_back(a, 4);
		std::set<uint64_t> byte_pages;
		for (const uint64_t a: served) {
			if (std::find(backing.begin(), backing.end(), PageOf(a)) == backing.end()) byte_pages.insert(PageOf(a));
		}
		std::set<uint64_t> want_pages(backing.begin(), backing.end()), want_cw;
		std::set<Range>    want_bytes;
		for (const auto& [a, n]: reads) {
			bool touches = false;
			for (uint64_t p = PageOf(a); p <= PageOf(a + n - 1); p += kPage) touches = touches || byte_pages.count(p) != 0;
			for (uint64_t p = PageOf(a); p <= PageOf(a + n - 1); p += kPage) (touches ? want_cw : want_pages).insert(p);
			if (touches) want_bytes.insert({a, n});
		}
		std::vector<uint64_t> pages = backing;
		std::vector<Range>    bytes;
		std::vector<uint64_t> cw;
		Check(SplitReads(pages, served, reads, bytes, cw, 4096), "random split ok");
		Check(std::set<uint64_t>(pages.begin(), pages.end()) == want_pages && pages.size() == want_pages.size(), "random split: the page checks");
		Check(std::set<uint64_t>(cw.begin(), cw.end()) == want_cw && cw.size() == want_cw.size(), "random split: the pages asked for writes");
		Check(std::set<Range>(bytes.begin(), bytes.end()) == want_bytes && bytes.size() == want_bytes.size(), "random split: the byte checks");
		Check(std::is_sorted(pages.begin(), pages.end()) && std::is_sorted(cw.begin(), cw.end()), "sorted");
	}
}

void JoinTests() {
	const std::vector<uint64_t> pages {0x1000};
	const std::vector<Range>    bytes {{0x2010, 4}};
	const std::vector<uint64_t> cw {0x2000};
	int                         asked_pages = 0;
	const auto                  none_dirty  = [&](uint64_t) {
        asked_pages++;
        return false;
	};
	const auto no_bytes  = [](uint64_t, uint64_t) { return false; };
	const auto unwritten = [](const std::vector<uint64_t>&) { return false; };
	Check(Join(pages, bytes, cw, none_dirty, no_bytes, unwritten) == Verdict::Ok && asked_pages == 1, "clean: ok, the page check runs on the plain page only");
	Check(Join(pages, bytes, cw, [](uint64_t) { return true; }, no_bytes, unwritten) == Verdict::PageDirty, "a dirty plain page");
	Check(Join(pages, bytes, cw, none_dirty, [](uint64_t a, uint64_t) { return a == 0x2010; }, unwritten) == Verdict::BytesDirty,
	      "served bytes dirty now: refused");
	Check(Join(pages, bytes, cw, none_dirty, no_bytes, [](const std::vector<uint64_t>& p) { return p.size() == 1 && p[0] == 0x2000; }) == Verdict::Written,
	      "a backing write on a served page: refused");
	Check(Join(pages, {}, {}, none_dirty, [](uint64_t, uint64_t) { return true; }, unwritten) == Verdict::Ok, "without served loads: S490's join");
	// The served page holds other GPU-dirty bytes (the reason it is protected): the job is still used.
	std::vector<uint64_t> split_pages;
	std::vector<Range>    split_bytes;
	std::vector<uint64_t> split_cw;
	Check(SplitReads(split_pages, std::vector<uint64_t> {0x2010}, std::vector<Range> {{0x2010, 4}}, split_bytes, split_cw, 4096), "split for the join");
	Check(Join(split_pages, split_bytes, split_cw, [](uint64_t page) { return page == 0x2000; },
	           [](uint64_t a, uint64_t n) { return a < 0x2104 && a + n > 0x2100; }, unwritten) == Verdict::Ok,
	      "a served page with other dirty bytes: used");
}

// ---------------------------------------------------------------- the oracle ----------------------------------------------------------------

constexpr uint64_t kBase   = 0x200000;
constexpr uint64_t kPages  = 6;
constexpr uint64_t kDwords = kPages * kPage / 4;

enum class Kind : uint8_t { Served, Plain, Backing };

struct Load {
	Kind     kind  = Kind::Served;
	uint64_t addr  = 0;
	uint32_t value = 0;
	bool     served = false;
};

struct Job {
	std::vector<Load>     loads;
	std::atomic<int>      state {0}; // 0 queued, 1 done
	bool                  answered = false;
	uint64_t              write_seq = 0;
	std::vector<uint64_t> pages;
	std::vector<Range>    bytes;
	std::vector<uint64_t> cw_pages;
	std::vector<uint64_t> served;
};

struct World {
	Libs::Graphics::GpuDirtyRanges   dirty;
	Libs::Graphics::InflightRanges   inflight;
	Libs::LibKernel::BackingWriteLog log;
	std::unique_ptr<std::atomic<uint32_t>[]> backing {new std::atomic<uint32_t>[kDwords]};
	std::vector<uint32_t>            gpu = std::vector<uint32_t>(kDwords, 0); // CP thread only: what the GPU wrote last
	// The publish thread.
	struct Publish {
		uint64_t              addr = 0;
		std::vector<uint32_t> values;
	};
	std::mutex              pub_mutex;
	std::condition_variable pub_cv;
	std::deque<Publish>     pub_queue;
	std::atomic<int>        pub_pending {0};
	bool                    pub_stop = false;
	// The workers.
	std::mutex              job_mutex;
	std::condition_variable job_cv;
	std::deque<Job*>        job_queue;
	bool                    job_stop = false;

	World() {
		dirty.SetMemoModes(Libs::Graphics::DirtyMemo::Mode::Off, Libs::Graphics::DirtyMemo::Mode::Off);
		for (uint64_t i = 0; i < kDwords; i++) backing[i].store(static_cast<uint32_t>(i * 2654435761u), std::memory_order_relaxed);
	}
	static uint64_t Index(uint64_t addr) { return (addr - kBase) / 4; }
};

// The worker's walk of one job, as RunGfpJob with the reader of renderDraw.cpp (WorkerCleanRead).
void Walk(World& w, Job& job) {
	auto& cw = WorkerLog();
	cw.Begin();
	job.pages.clear();
	job.write_seq = w.log.Seq();
	std::vector<Range> reads;
	bool               answered = true;
	for (auto& load: job.loads) {
		const uint64_t i = World::Index(load.addr);
		if (load.kind == Kind::Backing) {
			load.value = w.backing[i].load(std::memory_order_relaxed);
			job.pages.push_back(PageOf(load.addr)); // TryReadBackingWorker logs the page, no range
			continue;
		}
		if (load.kind == Kind::Served) {
			bool       dirty = true, inflight = true;
			// The oracle's workers do nothing but query (production: ~6 queries in a ~9 us job), so they spin as long as a guest thread does.
			const bool queried = w.dirty.TryIntersectsSpin(load.addr, 4, &dirty, Libs::Graphics::GuestWait::kSpinTries) &&
			                     w.inflight.TryIntersectsSpin(load.addr, 4, &inflight, nullptr, Libs::Graphics::GuestWait::kSpinTries);
			const auto answer = Decide(true, queried, dirty, inflight);
			if (answer == Answer::Serve) {
				load.value  = w.backing[i].load(std::memory_order_relaxed);
				load.served = true;
				cw.served.push_back(load.addr);
				reads.emplace_back(load.addr, 4);
				continue;
			}
			answer == Answer::Busy ? cw.busy++ : cw.dirty++;
		}
		// The ordinary load: a page with GPU-dirty bytes is protected, the load faults and the job is the CP's (fail7).
		bool page_dirty = true;
		if (!w.dirty.TryIntersectsSpin(PageOf(load.addr), kPage, &page_dirty, Libs::Graphics::GuestWait::kSpinTries) || page_dirty) {
			answered = false;
			continue;
		}
		load.value = w.backing[i].load(std::memory_order_relaxed);
		reads.emplace_back(load.addr, 4);
	}
	cw.End();
	job.served = cw.served;
	job.bytes.clear();
	job.cw_pages.clear();
	if (answered && !job.served.empty()) {
		answered = SplitReads(job.pages, job.served, reads, job.bytes, job.cw_pages, 4096);
	} else if (answered) {
		for (const auto& [a, n]: reads) job.pages.push_back(PageOf(a));
		SortUnique(job.pages);
	}
	job.answered = answered;
}

struct Stats {
	uint64_t joined = 0, used = 0, used_served = 0, used_dirty_page = 0, bytes_dirty = 0, page_dirty = 0, written = 0, bails = 0, served = 0;
};

void Oracle(uint64_t seed, int jobs_total, Stats& st) {
	World w;
	std::vector<std::thread> threads;
	for (int t = 0; t < 3; t++) {
		threads.emplace_back([&w] {
			for (;;) {
				Job* job = nullptr;
				{
					std::unique_lock lock(w.job_mutex);
					w.job_cv.wait(lock, [&w] { return w.job_stop || !w.job_queue.empty(); });
					if (w.job_queue.empty()) return;
					job = w.job_queue.front();
					w.job_queue.pop_front();
				}
				Walk(w, *job);
				job->state.store(1, std::memory_order_release);
			}
		});
	}
	std::thread publisher([&w] {
		for (;;) {
			World::Publish p;
			{
				std::unique_lock lock(w.pub_mutex);
				w.pub_cv.wait(lock, [&w] { return w.pub_stop || !w.pub_queue.empty(); });
				if (w.pub_queue.empty()) return;
				p = std::move(w.pub_queue.front());
				w.pub_queue.pop_front();
			}
			const auto token = w.log.Begin(p.addr, p.values.size() * 4);
			for (size_t k = 0; k < p.values.size(); k++) w.backing[World::Index(p.addr) + k].store(p.values[k], std::memory_order_relaxed);
			w.log.End(token);
			w.inflight.Remove(p.addr, p.values.size() * 4);
			w.pub_pending.fetch_sub(1, std::memory_order_acq_rel);
		}
	});

	std::mt19937_64              rng(seed);
	std::deque<std::unique_ptr<Job>> pending;
	std::deque<World::Publish>   held; // downloads recorded, their publish not handed over yet (a copy the GPU has not finished)
	const auto                   hot_page = [&] { return kBase + (rng() % 3) * kPage; };         // pages 0..2: walker tables, often GPU-written around
	const auto                   cold_page = [&] { return kBase + (3 + rng() % (kPages - 3)) * kPage; }; // pages 3..5: never GPU-written
	const auto                   dword_in = [&](uint64_t page) { return page + (rng() % 256) * 4; }; // a small hot area: collisions are frequent
	int                          submitted = 0;
	const auto                   join      = [&](Job& job) {
        while (job.state.load(std::memory_order_acquire) != 1) std::this_thread::yield();
        st.joined++;
        if (!job.answered) {
            st.bails++;
            return;
        }
        // The CP thread: its own reads of the dirty set need no lock (it is the only thread that changes it).
        const auto verdict = Join(
            job.pages, job.bytes, job.cw_pages, [&](uint64_t page) { return w.dirty.Intersects(page, kPage) || w.inflight.Intersects(page, kPage); },
            [&](uint64_t a, uint64_t n) { return BytesBusy(w.dirty.Intersects(a, n), w.inflight.Intersects(a, n)); },
            [&](const std::vector<uint64_t>& p) { return w.log.WrittenSince(job.write_seq, p.data(), p.size()); });
        if (verdict == Verdict::BytesDirty) st.bytes_dirty++;
        if (verdict == Verdict::PageDirty) st.page_dirty++;
        if (verdict == Verdict::Written) st.written++;
        if (verdict != Verdict::Ok) return;
        st.used++;
        if (!job.served.empty()) st.used_served++;
        st.served += job.served.size();
        // Every value the job used is the one the CP's own walk reads now: the bytes are neither GPU-dirty nor in flight, and the backing holds them.
        bool dirty_page = false;
        for (const auto& load: job.loads) {
            const bool now_dirty    = w.dirty.Intersects(load.addr, 4);
            const bool now_inflight = w.inflight.Intersects(load.addr, 4);
            const uint32_t now      = w.backing[World::Index(load.addr)].load(std::memory_order_relaxed);
            Check(!now_dirty, "a used value is GPU-dirty at the join");
            Check(!now_inflight, "a used value has a copy in flight at the join");
            Check(load.value == now, "a used value differs from the backing at the join");
            if (load.served && w.dirty.Intersects(PageOf(load.addr), kPage)) dirty_page = true;
        }
        if (dirty_page) st.used_dirty_page++;
	};
	while (submitted < jobs_total || !pending.empty()) {
		const int op = static_cast<int>(rng() % 100);
		if (op < 15) {
			// A GPU write: the bytes become dirty (the page protected), the GPU's value is the truth. Mostly the GPU's own area of the page (dwords
			// 512..1023), sometimes the walker's tables (0..255): a served dword that the GPU rewrites before the join.
			const uint64_t page = hot_page();
			const uint64_t a    = rng() % 30 == 0 ? dword_in(page) : page + (512 + rng() % 500) * 4;
			const uint64_t n    = 4 * (1 + rng() % 4);
			w.dirty.Add(a, n);
			for (uint64_t k = 0; k < n / 4; k++) w.gpu[World::Index(a) + k] = static_cast<uint32_t>(rng());
		} else if (op < 17) {
			// A download: dirty -> in flight; the publish comes later.
			const uint64_t page = hot_page();
			w.dirty.BeginDownloadCopy(w.inflight, page, kPage, [&](uint64_t s, uint64_t e) {
				World::Publish p;
				p.addr = s;
				for (uint64_t x = s; x < e; x += 4) p.values.push_back(w.gpu[World::Index(x)]);
				held.push_back(std::move(p));
			});
		} else if (op < 20 && !held.empty()) {
			// The GPU finished a copy: hand its publish to the publish thread.
			{
				std::lock_guard lock(w.pub_mutex);
				w.pub_queue.push_back(std::move(held.front()));
				w.pub_pending.fetch_add(1, std::memory_order_acq_rel);
			}
			held.pop_front();
			w.pub_cv.notify_one();
		} else if (op < 60 && submitted < jobs_total) {
			auto job = std::make_unique<Job>();
			const uint64_t hp = hot_page();
			for (int k = 0; k < 4; k++) job->loads.push_back({Kind::Served, dword_in(hp)});
			// A walker load on a page that is not hot (the ordinary load), and a read through the backing (a specialization or fetch read, logged as a page).
			if (rng() % 2 == 0) job->loads.push_back({Kind::Plain, dword_in(cold_page())});
			if (rng() % 4 == 0) job->loads.push_back({Kind::Backing, dword_in(rng() % 2 == 0 ? hp : cold_page())});
			{
				std::lock_guard lock(w.job_mutex);
				w.job_queue.push_back(job.get());
			}
			w.job_cv.notify_one();
			pending.push_back(std::move(job));
			submitted++;
		} else if (!pending.empty() && (pending.size() > 6 || submitted >= jobs_total || rng() % 2 == 0)) {
			join(*pending.front());
			pending.pop_front();
		}
	}
	{
		std::lock_guard lock(w.job_mutex);
		w.job_stop = true;
	}
	w.job_cv.notify_all();
	for (auto& t: threads) t.join();
	{
		std::lock_guard lock(w.pub_mutex);
		for (auto& p: held) w.pub_queue.push_back(std::move(p));
		w.pub_stop = true;
	}
	w.pub_cv.notify_all();
	publisher.join();
}

void OracleTests() {
	for (const uint64_t seed: {1u, 2u, 3u}) {
		Stats st;
		Oracle(seed, 20000, st);
		std::printf("oracle seed %llu: joined %llu, used %llu (with served loads %llu, %llu of them on a page with other GPU-dirty bytes, %llu served dwords); "
		            "refused: bytes %llu, page %llu, written %llu; not answered %llu\n",
		            static_cast<unsigned long long>(seed), static_cast<unsigned long long>(st.joined), static_cast<unsigned long long>(st.used),
		            static_cast<unsigned long long>(st.used_served), static_cast<unsigned long long>(st.used_dirty_page),
		            static_cast<unsigned long long>(st.served), static_cast<unsigned long long>(st.bytes_dirty), static_cast<unsigned long long>(st.page_dirty),
		            static_cast<unsigned long long>(st.written), static_cast<unsigned long long>(st.bails));
		Check(st.joined == 20000, "every job joined");
		Check(st.used_served > 1000, "the oracle uses many jobs with served loads");
		Check(st.used_dirty_page > 100, "jobs with a served load on a page with other GPU-dirty bytes are used (the gain)");
		Check(st.bytes_dirty > 0 && st.written > 0, "the join refuses both ways (bytes dirty now, a backing write since)");
	}
}

} // namespace

int main() {
	// KYTY_WCW_ORACLE_ONLY (a check of the test itself): only the oracle, which must catch JOIN_NO_BYTES, JOIN_NO_INFLIGHT, JOIN_NO_WRITTEN,
	// SPLIT_IGNORES_BACKING and PAGE_CHECKS_SERVED on its own.
#if !defined(KYTY_WCW_ORACLE_ONLY)
	ModeTests();
	DecideTests();
	SplitTests();
	JoinTests();
#endif
	OracleTests();
	std::printf("CpPipeCleanWalkTests: ok\n");
	return 0;
}
