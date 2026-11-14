// S587 (KYTY_GW_PAGE, guestWait.h): the guest read emulator keeps the side effects of the ordinary path, per guest page and guest flip.
//  - DecidePage / EffectsOf / ParsePage, PageFlipCounts (per page, reset by the flip, threads), PageMailbox (no post lost or doubled, threads);
//  - a model of one guest page holding a K-rf v3 deferred dword (the GPU rewrites it every flip, the walker defers 3 times per flip, the real
//    RfGpuPatch::GuestPageGuard decides bans, a hot page gets an eager download at the next submit, its publish releases the page), with the
//    guest reading in three profiles:
//      garden        375 reads per flip after the eager download landed (S586 p2k: 375 emulated loads/flip);
//      garden-light  8 reads per flip after the eager download landed (below the limit: only the post to the CP makes it hot);
//      desert        6 reads per flip right after the GPU write (S581: the guest reads the page of the deferred dword in its protected window);
//    and four policies: ordinary (KRF3 alone), emu (S585 KYTY_GUEST_WAIT=1: served, nothing else), care (S587), care without eager
//    downloads (a page nothing else releases).
//    Checked: the model reproduces S586 for emu (garden: every read faults forever, no ban; desert: no ban) and S582 for ordinary (garden:
//    no late faults, no ban; desert: ban); with care the garden is as ordinary (no late faults, no ban), the light reader gets hot, the desert
//    is banned, and a page nothing releases is emulated at most `limit` times per flip.
// Mutants (WILL_FAIL): KYTY_GWPAGE_MUTANT_NO_LIMIT, _NO_FEED, _NO_GUARD_HIT, _HIT_EVERY.

#include "graphics/host_gpu/renderer/cache/guestWait.h"
#include "graphics/shader/recompiler/ir/passes/rfGpuPatch.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <thread>
#include <vector>

using namespace Libs::Graphics;
using GuestWait::DecidePage;
using GuestWait::EffectsOf;
using GuestWait::PageAction;
namespace RfP = Libs::Graphics::ShaderRecompiler::IR::RfGpuPatch;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

static void TestDecide() {
	Check(DecidePage(0, 32) == PageAction::Emulate, "untracked: emulate");
	Check(DecidePage(1, 32) == PageAction::EmulateFirst, "first load of the flip: emulate + side effects");
	Check(DecidePage(2, 32) == PageAction::Emulate, "second: emulate");
	Check(DecidePage(32, 32) == PageAction::Emulate, "at the limit: emulate");
	Check(DecidePage(33, 32) == PageAction::Ordinary, "over the limit: ordinary path");
	Check(DecidePage(2, 1) == PageAction::Ordinary, "limit 1: second is ordinary");
	const auto first = EffectsOf(PageAction::EmulateFirst);
	Check(first.feed && first.guard_hit, "first: feed + guard hit");
	const auto other = EffectsOf(PageAction::Emulate);
	Check(!other.feed && !other.guard_hit, "later loads: no side effect");
	const auto ord = EffectsOf(PageAction::Ordinary);
	Check(!ord.feed && !ord.guard_hit, "ordinary: the ordinary path does it itself");
}

static void TestParse() {
	Check(!GuestWait::ParsePage(nullptr).on, "unset: off");
	Check(!GuestWait::ParsePage("").on, "empty: off");
	Check(!GuestWait::ParsePage("0").on, "0: off");
	const auto one = GuestWait::ParsePage("1");
	Check(one.on && one.limit == 32, "1: on, limit 32");
	const auto n = GuestWait::ParsePage("8");
	Check(n.on && n.limit == 8, "8: limit 8");
	Check(!GuestWait::ParsePage("x").on, "x: off");
	Check(!GuestWait::ParsePage("8x").on, "8x: off");
	Check(!GuestWait::ParsePage("70000").on, "too big: off");
}

static void TestCounts() {
	GuestWait::PageFlipCounts<16> c;
	Check(c.Note(0x1000, 5) == 1, "page A first");
	Check(c.Note(0x1ffc, 5) == 2, "page A same page other byte");
	Check(c.Note(0x2000, 5) == 1, "page B first");
	Check(c.Note(0x1004, 6) == 1, "new flip resets");
	for (uint64_t i = 0; i < 15; ++i) {
		(void)c.Note(0x100000 + i * 0x1000, 6);
	}
	Check(c.Note(0x900000, 6) == 0, "table full: 0 (untracked)");
	Check(c.Note(0x900000, 7) == 1, "next flip: room again");
	// Threads on one flip: the counts of one page add up.
	GuestWait::PageFlipCounts<> t;
	(void)t.Note(0x5000, 1);
	std::vector<std::thread> threads;
	std::atomic<uint32_t>    max_seen {0};
	for (int k = 0; k < 4; ++k) {
		threads.emplace_back([&] {
			for (int i = 0; i < 10000; ++i) {
				const auto v = t.Note(0x5000 + (i & 0xfff), 1);
				uint32_t   m = max_seen.load();
				while (v > m && !max_seen.compare_exchange_weak(m, v)) {
				}
			}
		});
	}
	for (auto& th: threads) th.join();
	Check(max_seen.load() == 40001, "threads: one page, 40001 loads counted");
}

static void TestMailbox() {
	GuestWait::PageMailbox<4> m;
	Check(m.Push(0x1000) && m.Push(0x1000), "post, again = already posted");
	Check(m.Push(0x2000) && m.Push(0x3000) && m.Push(0x4000), "four slots");
	Check(!m.Push(0x5000), "full: dropped");
	std::set<uint64_t> got;
	Check(m.Drain([&](uint64_t p) { got.insert(p); }) == 4 && got.size() == 4, "drain gives each once");
	Check(m.Drain([&](uint64_t) {}) == 0, "empty after drain");
	Check(!m.Push(0), "page 0 never posted");
	// Threads: 4 producers x 1000 distinct pages, each retried until posted; one consumer: every page drained exactly once.
	GuestWait::PageMailbox<64> mt;
	std::atomic<bool>           done {false};
	std::vector<uint64_t>       seen;
	std::thread                 consumer([&] {
        while (!done.load() || seen.size() < 4000) {
            mt.Drain([&](uint64_t p) { seen.push_back(p); });
            if (seen.size() >= 4000) break;
            std::this_thread::yield();
        }
    });
	std::vector<std::thread> producers;
	for (uint64_t k = 0; k < 4; ++k) {
		producers.emplace_back([&mt, k] {
			for (uint64_t i = 0; i < 1000; ++i) {
				const uint64_t page = (k * 1000 + i + 1) << 12;
				while (!mt.Push(page)) std::this_thread::yield();
			}
		});
	}
	for (auto& p: producers) p.join();
	done.store(true);
	consumer.join();
	std::sort(seen.begin(), seen.end());
	Check(seen.size() == 4000 && std::adjacent_find(seen.begin(), seen.end()) == seen.end(), "threads: 4000 posts drained once each");
}

// ---- the page model ----
enum class Policy { Ordinary, Emu, Care, CareNoEager };
enum class Profile { Garden, GardenLight, Desert };

struct ModelResult {
	double   late_faults_per_flip = 0; // guest faults per flip, flips 60..259
	uint32_t max_emu_per_flip     = 0; // emulated loads of the page in one flip (any flip)
	uint32_t bans                 = 0;
	uint32_t first_ban_flip       = 0;
	uint32_t late_bans            = 0; // bans from flip 60 on
	bool     hot                  = false;
};

static ModelResult RunModel(Profile profile, Policy policy, uint32_t limit = 32) {
	constexpr uint64_t kPage  = 0x579540000ull; // the garden page of S574/S581
	constexpr uint64_t kDword = kPage + 0x27c;
	constexpr int      kSlots = 100;
	constexpr int      kFlips = 260;
	const bool         care   = policy == Policy::Care || policy == Policy::CareNoEager;
	const bool         eager  = policy != Policy::CareNoEager;

	RfP::GuestPageGuard           guard;
	GuestWait::PageFlipCounts<16> counts;
	GuestWait::PageMailbox<8>     mailbox;
	bool                          prot = false, dirty = false, hot = false, pending = false;
	uint32_t                      read_faults = 0;   // NoteReadbackFault count (hot at 4, bufferCache.cpp EagerHotFaults)
	int64_t                       land_at     = -1;  // eager download publish time
	bool                          land_valid  = false; // no GPU write since it was recorded (seq)
	ModelResult                   r;
	uint64_t                      late_faults = 0;

	const auto note_fault = [&] {
		if (++read_faults >= 4) hot = true;
	};
	const auto drain = [&] { // ordinary path: SendCommandSync + DownloadBufferMemory<false> + unmark
		dirty = false;
		prot  = false;
		note_fault();
	};

	int reads = 0, read_from = 0, read_to = 0;
	switch (profile) {
		case Profile::Garden: reads = 375, read_from = 40, read_to = 95; break;
		case Profile::GardenLight: reads = 8, read_from = 40, read_to = 47; break;
		case Profile::Desert: reads = 6, read_from = 24, read_to = 34; break;
	}
	for (int flip = 0; flip < kFlips; ++flip) {
		uint32_t emu_this_flip = 0;
		int      read_done     = 0;
		for (int slot = 0; slot < kSlots; ++slot) {
			const int64_t t = static_cast<int64_t>(flip) * kSlots + slot;
			// eager publish (priority op): unmark when no GPU write since (seq)
			if (land_at == t) {
				if (land_valid) {
					dirty = false;
					prot  = false;
				}
				land_at = -1;
			}
			// the GPU write of the deferred dword (recorded by the CP; NoteGpuWrite queues a hot page)
			if (slot == 20) {
				dirty      = true;
				prot       = true;
				land_valid = false;
				if (hot && eager) pending = true;
			}
			// submits every 10 slots: pre-submit hook (S587 drain, then FlushEagerReadbacks)
			if (slot % 10 == 9) {
				if (care) {
					mailbox.Drain([&](uint64_t) {
						note_fault();
						if (hot && eager && dirty && land_at < 0) pending = true;
					});
				}
				if (pending && dirty) {
					land_at    = t + 5;
					land_valid = true;
				}
				pending = false;
			}
			// the walker's loads of the dword: 3 defers per flip (garden 6 on 2 pages, desert 3 on the page)
			if (slot == 22 || slot == 50 || slot == 75) {
				if (prot && dirty) {
					const auto d = guard.OnDefer(kDword, static_cast<uint64_t>(flip), true);
					if (d.banned_now) {
						++r.bans;
						if (r.first_ban_flip == 0) r.first_ban_flip = static_cast<uint32_t>(flip);
						if (flip >= 60) ++r.late_bans;
					}
					if (d.drain) drain();
				}
			}
			// guest reads (clean bytes of the page: 0x...264 next to the dword)
			while (read_done < reads) {
				const int at = read_from + (read_to - read_from) * read_done / std::max(1, reads - 1);
				if (at != slot) break;
				++read_done;
				if (!prot) continue;
				if (flip >= 60) ++late_faults;
				const uint64_t addr = kPage + 0x264;
				if (policy == Policy::Ordinary) {
					(void)guard.NoteGuestRead(addr); // RenderContext::HandleFault
					drain();
					continue;
				}
				if (policy == Policy::Emu) {
					++emu_this_flip;
					continue;
				}
				const auto action = DecidePage(counts.Note(addr, static_cast<uint64_t>(flip)), limit);
				if (action == PageAction::Ordinary) {
					(void)guard.NoteGuestRead(addr);
					drain();
					continue;
				}
				++emu_this_flip;
				const auto e = EffectsOf(action);
				if (e.feed) (void)mailbox.Push(addr & ~uint64_t {4095});
				if (e.guard_hit) (void)guard.NoteGuestRead(addr);
			}
		}
		r.max_emu_per_flip = std::max(r.max_emu_per_flip, emu_this_flip);
	}
	r.late_faults_per_flip = static_cast<double>(late_faults) / (kFlips - 60);
	r.hot                  = hot;
	return r;
}

static void Report(const char* name, const ModelResult& r) {
	std::printf("  %-26s late faults/flip %7.2f  max emu/flip %4u  bans %u (first %u, late %u)  hot %d\n", name, r.late_faults_per_flip,
	            r.max_emu_per_flip, r.bans, r.first_ban_flip, r.late_bans, r.hot ? 1 : 0);
}

static void TestModel() {
	std::printf("model (flips 60..259 = late):\n");
	const auto g_ord   = RunModel(Profile::Garden, Policy::Ordinary);
	const auto g_emu   = RunModel(Profile::Garden, Policy::Emu);
	const auto g_care  = RunModel(Profile::Garden, Policy::Care);
	const auto l_ord   = RunModel(Profile::GardenLight, Policy::Ordinary);
	const auto l_emu   = RunModel(Profile::GardenLight, Policy::Emu);
	const auto l_care  = RunModel(Profile::GardenLight, Policy::Care);
	const auto d_ord   = RunModel(Profile::Desert, Policy::Ordinary);
	const auto d_emu   = RunModel(Profile::Desert, Policy::Emu);
	const auto d_care  = RunModel(Profile::Desert, Policy::Care);
	const auto g_stuck = RunModel(Profile::Garden, Policy::CareNoEager);
	Report("garden ordinary (KRF3)", g_ord);
	Report("garden emu (S585)", g_emu);
	Report("garden care (S587)", g_care);
	Report("garden-light ordinary", l_ord);
	Report("garden-light emu", l_emu);
	Report("garden-light care", l_care);
	Report("desert ordinary (KRF3)", d_ord);
	Report("desert emu (S585)", d_emu);
	Report("desert care (S587)", d_care);
	Report("garden care, no eager", g_stuck);

	// The model reproduces the measurements it stands for.
	Check(g_ord.late_faults_per_flip < 0.2 && g_ord.bans == 0, "calibration: garden KRF3 alone has no late faults and no ban (S582)");
	Check(d_ord.bans >= 1, "calibration: desert KRF3 alone is banned (S586 d1x)");
	Check(g_emu.late_faults_per_flip > 300 && g_emu.bans == 0, "calibration: garden S585 emu faults on every read, no ban (S586 p2k)");
	Check(d_emu.bans == 0, "calibration: desert S585 emu is never banned (S586 d1k krf3_guest_page_n = 0)");
	Check(l_emu.late_faults_per_flip > 7, "calibration: light reader S585 emu faults on every read");
	// S587.
	Check(g_care.late_faults_per_flip < 0.2, "care: garden page released like KRF3 alone (no late faults)");
	Check(g_care.bans == 0, "care: garden never banned");
	Check(g_care.max_emu_per_flip <= 32, "care: at most `limit` emulated loads per page and flip");
	Check(l_care.hot && l_care.late_faults_per_flip < 0.2, "care: a light reader (below the limit) makes the page hot, no late faults");
	Check(l_care.bans == 0, "care: light reader never banned");
	Check(d_care.bans >= 1 && d_care.first_ban_flip <= 40, "care: desert banned within 40 flips (the guard sees the served loads)");
	Check(g_stuck.max_emu_per_flip <= 32, "care, nothing releases the page: still at most `limit` emulations per flip");
}

int main() {
	TestDecide();
	TestParse();
	TestCounts();
	TestMailbox();
	TestModel();
	if (g_failures != 0) {
		std::fprintf(stderr, "GuestWaitPageTests: %d failures\n", g_failures);
		return 1;
	}
	std::printf("GuestWaitPageTests: all passed\n");
	return 0;
}
