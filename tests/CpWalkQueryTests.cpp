// S608 (KYTY_CP_WALK_QUERY): the CP's clean-walk byte query from its own view (cpWalkQuery.h). Every check ends in exit(1) when it fails
// (no crash, no hang: a watchdog thread exits with 1 when a query does not return), so each mutant (CMake cp_walk_query_mutant_*, WILL_FAIL)
// is caught by a failed check:
//   MODE_IGNORED    - the CP view without the switch (the gate table);
//   ANY_THREAD      - the CP view off the GPU thread (the gate table);
//   TRY_LOCK        - the dirty part takes one try_lock as before (a reader holds the dirty lock: the CP view must still answer, and the
//                     oracle's count of busy answers under reader contention must be 0);
//   NO_INFLIGHT     - copies in flight are not asked (the oracle: a dword whose download is not published yet is taken as clean);
//   NO_DIRTY        - the GPU-dirty set is not asked (the oracle: a dword the GPU rewrote is taken as clean);
//   INFLIGHT_BLOCKS - the in-flight lock is taken without a bound (a holder that keeps it: the query must give up, the watchdog catches the hang).
// The oracle runs the real GpuDirtyRanges and InflightRanges: a CP thread that marks GPU writes, records downloads (BeginDownloadCopy) and asks
// the CP view on random dwords, a publish thread that writes the downloaded bytes into the backing and removes the in-flight entries (as the
// deferred publish does), and three reader threads that hammer both locks as the pipeline workers of S607 do. Each answer "neither dirty nor in
// flight" must find the backing equal to the GPU's last write: that is the dword a drain would read.
// -DKYTY_CPWQ_ORACLE_ONLY builds the oracle alone (a control of what it catches without the deterministic tests).

#include "graphics/host_gpu/renderer/cache/cpWalkQuery.h"
#include "graphics/host_gpu/renderer/cache/gpuDirtyRanges.h"
#include "graphics/host_gpu/renderer/cache/inflightRanges.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace Q = Libs::Graphics::CpWalkQuery;
using Libs::Graphics::GpuDirtyRanges;
using Libs::Graphics::InflightRanges;
namespace DirtyMemo = Libs::Graphics::DirtyMemo;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "CpWalkQueryTests: failed: %s\n", text);
		std::exit(1);
	}
}

// A query that does not return within the limit is a failed check (exit 1), not a hung test.
class Watchdog {
public:
	explicit Watchdog(const char* what, std::chrono::milliseconds limit): m_what(what) {
		m_thread = std::thread([this, limit] {
			std::unique_lock lock(m_mutex);
			if (!m_cv.wait_for(lock, limit, [this] { return m_done; })) {
				std::fprintf(stderr, "CpWalkQueryTests: failed: %s did not return (blocked on a lock)\n", m_what);
				std::fflush(stderr);
				std::_Exit(1);
			}
		});
	}
	~Watchdog() {
		{
			std::lock_guard lock(m_mutex);
			m_done = true;
		}
		m_cv.notify_all();
		m_thread.join();
	}

private:
	const char*             m_what;
	std::mutex              m_mutex;
	std::condition_variable m_cv;
	bool                    m_done = false;
	std::thread             m_thread;
};

void GateTests() {
	Check(Q::Parse(nullptr) == Q::Mode::Off && Q::Parse("0") == Q::Mode::Off && Q::Parse("yes") == Q::Mode::Off && Q::Parse("") == Q::Mode::Off,
	      "default off");
	Check(Q::Parse("1") == Q::Mode::On, "1 = on");
	Check(std::string_view(Q::ModeName(Q::Mode::On)) == "1" && std::string_view(Q::ModeName(Q::Mode::Off)) == "0", "names");
	Check(Q::UseCpView(Q::Mode::On, true), "on, GPU thread: the CP view");
	Check(!Q::UseCpView(Q::Mode::Off, true), "off: the old try-lock path, also on the GPU thread");
	Check(!Q::UseCpView(Q::Mode::On, false), "another thread never takes the CP view");
	Check(!Q::UseCpView(Q::Mode::Off, false), "off, another thread");
}

// Single-thread answers: the CP view says what the sets hold.
void AnswerTests() {
	GpuDirtyRanges d;
	InflightRanges in;
	bool           dirty = true, inflight = true;
	Check(Q::Query(d, in, 0x1000, 4, &dirty, &inflight) && !dirty && !inflight, "empty: clean");
	d.Add(0x1000, 0x100);
	Check(Q::Query(d, in, 0x1010, 4, &dirty, &inflight) && dirty && !inflight, "GPU-dirty bytes: dirty");
	Check(Q::Query(d, in, 0x1100, 4, &dirty, &inflight) && !dirty && !inflight, "the next dword after the range: clean");
	std::vector<std::pair<uint64_t, uint64_t>> pieces;
	d.BeginDownloadCopy(in, 0x1000, 0x100, [&](uint64_t s, uint64_t e) { pieces.emplace_back(s, e); });
	Check(pieces.size() == 1 && pieces[0].first == 0x1000 && pieces[0].second == 0x1100, "download piece");
	Check(Q::Query(d, in, 0x1010, 4, &dirty, &inflight) && !dirty && inflight, "recorded download: in flight, not dirty");
	in.Remove(0x1000, 0x100);
	Check(Q::Query(d, in, 0x1010, 4, &dirty, &inflight) && !dirty && !inflight, "published: clean");
	d.SetMemoModes(DirtyMemo::Mode::On, DirtyMemo::Mode::On);
	Check(Q::Query(d, in, 0x2000, 4, &dirty, &inflight) && !dirty, "memo: clean before");
	d.Add(0x2000, 8);
	Check(Q::Query(d, in, 0x2000, 4, &dirty, &inflight) && dirty, "memo: an Add is seen");
}

// A reader holds a lock: the CP view still answers on the dirty lock (it does not take it), and gives up, bounded, on an in-flight lock that is
// never released.
void HoldTests() {
	GpuDirtyRanges d;
	InflightRanges in;
	d.Add(0x5000, 0x10);
	{
		std::atomic<bool> held {false}, release {false};
		std::thread       holder([&] {
            std::lock_guard lock(d.MutexForTest());
            held = true;
            while (!release) std::this_thread::yield();
        });
		while (!held) std::this_thread::yield();
		bool old_dirty = false;
		Check(!d.TryIntersects(0x5000, 4, &old_dirty), "control: the old one-try path is busy while a reader holds the dirty lock");
		bool dirty = false, inflight = true;
		bool answered = false;
		{
			Watchdog wd("the CP view with the dirty lock held", std::chrono::milliseconds(5000));
			answered = Q::Query(d, in, 0x5000, 4, &dirty, &inflight);
		}
		Check(answered, "the dirty lock is held by a reader: the CP view answers (not busy)");
		Check(dirty && !inflight, "the dirty lock is held by a reader: the answer is the set's");
		release = true;
		holder.join();
	}
	{
		std::atomic<bool> held {false}, release {false};
		std::thread       holder([&] {
            std::lock_guard lock(in.MutexForTest());
            held = true;
            while (!release) std::this_thread::yield();
        });
		while (!held) std::this_thread::yield();
		bool dirty = false, inflight = false;
		bool answered = true;
		{
			Watchdog wd("the CP view with the in-flight lock never released", std::chrono::milliseconds(5000));
			answered = Q::Query(d, in, 0x9000, 4, &dirty, &inflight, 256);
		}
		Check(!answered, "the in-flight lock is never released: the bounded spin gives up (busy)");
		release = true;
		holder.join();
	}
	{
		// A holder that releases soon: the spin waits it out.
		std::atomic<bool> held {false};
		std::thread       holder([&] {
            std::lock_guard lock(in.MutexForTest());
            held = true;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        });
		while (!held) std::this_thread::yield();
		bool dirty = true, inflight = true;
		bool answered = false;
		{
			Watchdog wd("the CP view with a short in-flight hold", std::chrono::milliseconds(5000));
			for (int i = 0; i < 1000 && !answered; i++) answered = Q::Query(d, in, 0x9000, 4, &dirty, &inflight);
		}
		holder.join();
		Check(answered && !dirty && !inflight, "a short in-flight hold: answered after the spin");
	}
}

// The oracle.
constexpr uint64_t kBase   = 0x40000000;
constexpr uint64_t kPages  = 8;
constexpr uint64_t kDwords = kPages * 1024;

struct Copy {
	std::vector<std::pair<uint64_t, uint64_t>> pieces; // [start, end)
	std::vector<uint32_t>                      values; // per dword of the pieces, in order
};

struct Result {
	uint64_t queries = 0, clean = 0, served_ok = 0, busy = 0, dirty = 0, inflight = 0, reader_old_busy = 0;
};

Result Oracle(uint32_t seed, bool memo, int ops) {
	GpuDirtyRanges d;
	if (memo) d.SetMemoModes(DirtyMemo::Mode::On, DirtyMemo::Mode::On);
	InflightRanges                          in;
	std::vector<uint32_t>                   gpu(kDwords, 0); // the GPU's last write (CP thread only)
	std::unique_ptr<std::atomic<uint32_t>[]> backing(new std::atomic<uint32_t>[kDwords]);
	for (uint64_t i = 0; i < kDwords; i++) backing[i].store(0, std::memory_order_relaxed);

	std::mutex              qm;
	std::condition_variable qcv;
	std::deque<Copy>        queue;
	bool                    stop = false;
	std::thread             publisher([&] {
        std::mt19937 rng(seed * 7 + 1);
        for (;;) {
            Copy c;
            {
                std::unique_lock lock(qm);
                qcv.wait(lock, [&] { return stop || !queue.empty(); });
                if (queue.empty()) return;
                c = std::move(queue.front());
                queue.pop_front();
            }
            // The GPU is not done yet: the publish lags the record.
            if (rng() % 4 == 0) std::this_thread::sleep_for(std::chrono::microseconds(rng() % 20));
            size_t k = 0;
            for (const auto& [s, e]: c.pieces) {
                for (uint64_t a = s; a < e; a += 4) backing[(a - kBase) / 4].store(c.values[k++], std::memory_order_release);
                in.Remove(s, e - s);
            }
        }
    });
	std::atomic<bool>     readers_stop {false};
	std::atomic<uint64_t> reader_old_busy {0};
	std::vector<std::thread> readers;
	for (int r = 0; r < 3; r++) {
		readers.emplace_back([&, r] {
			std::mt19937 rng(seed * 13 + r);
			while (!readers_stop.load(std::memory_order_relaxed)) {
				const uint64_t a = kBase + (rng() % kDwords) * 4;
				bool           x = false;
				(void)d.TryIntersectsSpin(a, 4, &x, 256);
				(void)in.TryIntersectsSpin(a, 4, &x, nullptr, 256);
				if (!d.TryIntersects(a, 4, &x)) reader_old_busy.fetch_add(1, std::memory_order_relaxed);
			}
		});
	}

	Result       res;
	std::mt19937 rng(seed);
	uint32_t     next_value = 1;
	for (int op = 0; op < ops; op++) {
		const uint32_t kind = rng() % 10;
		if (kind < 3) {
			// A GPU write of a small range (dwords).
			const uint64_t first = rng() % kDwords;
			const uint64_t n     = 1 + rng() % 64;
			const uint64_t last  = std::min<uint64_t>(kDwords, first + n);
			for (uint64_t i = first; i < last; i++) gpu[i] = next_value++;
			d.Add(kBase + first * 4, (last - first) * 4);
		} else if (kind < 4) {
			// A recorded download of a page (the GPU-dirty pieces of it), published later.
			const uint64_t page = rng() % kPages;
			Copy           c;
			d.BeginDownloadCopy(in, kBase + page * 4096, 4096, [&](uint64_t s, uint64_t e) {
				c.pieces.emplace_back(s, e);
				for (uint64_t a = s; a < e; a += 4) c.values.push_back(gpu[(a - kBase) / 4]);
			});
			if (!c.pieces.empty()) {
				std::unique_lock lock(qm);
				queue.push_back(std::move(c));
				qcv.notify_one();
				// The GPU is at most 16 downloads behind (the CP waits for it, as a full ring does).
				while (queue.size() > 16) {
					lock.unlock();
					std::this_thread::yield();
					lock.lock();
				}
			}
		} else {
			// The walker's question on a dword.
			const uint64_t i     = rng() % kDwords;
			bool           dirty = true, inflight = true;
			res.queries++;
			if (!Q::Query(d, in, kBase + i * 4, 4, &dirty, &inflight)) {
				res.busy++;
				continue;
			}
			if (dirty) {
				res.dirty++;
				continue;
			}
			if (inflight) {
				res.inflight++;
				continue;
			}
			res.clean++;
			const uint32_t b = backing[i].load(std::memory_order_acquire);
			if (b != gpu[i]) {
				std::fprintf(stderr, "CpWalkQueryTests: oracle seed=%u op=%d dword=%llu backing=%u gpu=%u\n", seed, op,
				             static_cast<unsigned long long>(i), b, gpu[i]);
				Check(false, "a dword the CP view called clean is not the GPU's last write in the backing");
			}
			res.served_ok++;
		}
	}
	readers_stop = true;
	for (auto& t: readers) t.join();
	{
		std::lock_guard lock(qm);
		stop = true;
	}
	qcv.notify_all();
	publisher.join();
	res.reader_old_busy = reader_old_busy.load();
	return res;
}

void OracleTests() {
	uint64_t clean = 0, dirty = 0, inflight = 0, busy = 0, old_busy = 0;
	for (uint32_t seed = 1; seed <= 3; seed++) {
		for (bool memo: {false, true}) {
			const auto r = Oracle(seed, memo, 60000);
			std::printf("oracle seed=%u memo=%d: queries %llu, clean %llu (backing = GPU), dirty %llu, in flight %llu, busy %llu; readers' one-try busy %llu\n",
			            seed, memo ? 1 : 0, static_cast<unsigned long long>(r.queries), static_cast<unsigned long long>(r.clean),
			            static_cast<unsigned long long>(r.dirty), static_cast<unsigned long long>(r.inflight), static_cast<unsigned long long>(r.busy),
			            static_cast<unsigned long long>(r.reader_old_busy));
			clean += r.clean;
			dirty += r.dirty;
			inflight += r.inflight;
			busy += r.busy;
			old_busy += r.reader_old_busy;
		}
	}
	Check(clean > 1000 && dirty > 1000 && inflight > 50, "the oracle reaches every answer");
	// The readers contend with each other on the dirty lock (their own one-try queries fail); the CP view never takes it. Busy is left only when
	// the in-flight spin gave up (a lock held for the whole spin): rare, never the CP's one try against a reader.
	Check(old_busy > 0, "control: the readers' one-try queries meet a held dirty lock");
	Check(busy * 200 <= clean + dirty + inflight, "under reader contention the CP view is (almost) never busy");
	std::printf("oracle total: clean %llu, dirty %llu, in flight %llu, busy %llu, readers' one-try busy %llu\n", static_cast<unsigned long long>(clean),
	            static_cast<unsigned long long>(dirty), static_cast<unsigned long long>(inflight), static_cast<unsigned long long>(busy),
	            static_cast<unsigned long long>(old_busy));
}

} // namespace

int main() {
	// KYTY_CPWQ_ORACLE_ONLY (a check of the test itself, compile-time): only the oracle, which must catch TRY_LOCK, NO_INFLIGHT and NO_DIRTY by itself
	// (the gate mutants and INFLIGHT_BLOCKS are caught by the deterministic tests).
#if !defined(KYTY_CPWQ_ORACLE_ONLY)
	GateTests();
	AnswerTests();
	HoldTests();
#endif
	OracleTests();
	std::printf("CpWalkQueryTests: ok\n");
	return 0;
}
