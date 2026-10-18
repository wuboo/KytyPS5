// S432: KYTY_PATCH_CACHE. A module with an EXTRQ and a 256-bit store is patched by the real patchers (AMD CPU patcher + wide store split, as the
// loader does), the effect is captured, and applying it to a pristine copy must give the same memory image, byte for byte (all 16 pages). A changed
// source byte, a changed key input, a corrupt file and a wrong key must all be refused. Mutants: see src/loader/patchCache.h.

#include "loader/guestInstructionPatcher.h"
#include "loader/patchCache.h"
#include "loader/x64InstructionEmulator.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		g_failures++;
	}
}

constexpr size_t kPageSize = 4096;
constexpr size_t kAreaSize = 16 * kPageSize;
constexpr size_t kStoreAt  = 64;

constexpr uint8_t kExtrq[] = {
    0x66, 0x0f, 0x78, 0xc2, 0x28, 0x00, // extrq xmm2, 40, 0
    0xf3, 0x0f, 0x7f, 0x17,             // movdqu %xmm2, (%rdi)
    0xc3,                               // ret
};

constexpr uint8_t kWideCopy[] = {
    0xc5, 0xfc, 0x10, 0x06,       // vmovups ymm0, [rsi]
    0xc5, 0xfc, 0x11, 0x47, 0x40, // vmovups [rdi + 0x40], ymm0
    0xc5, 0xf8, 0x77,             // vzeroupper
    0xc3,                         // ret
};

constexpr size_t kCodeSize = kStoreAt + sizeof(kWideCopy);

uint8_t* MapArea() {
	auto* area = static_cast<uint8_t*>(
	    ::mmap(nullptr, kAreaSize, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0));
	Check(area != MAP_FAILED, "map RWX area");
	return area;
}

void Fill(uint8_t* area, uint8_t extrq_length_byte) {
	std::memset(area, 0xcc, kPageSize);
	std::memset(area + kPageSize, 0, kAreaSize - kPageSize);
	std::memcpy(area, kExtrq, sizeof(kExtrq));
	area[4] = extrq_length_byte;
	std::memcpy(area + kStoreAt, kWideCopy, sizeof(kWideCopy));
}

struct Patched {
	uint8_t*                      area = nullptr;
	Loader::PatchCache::Entry     entry;
	std::vector<uint8_t>          image; // the whole area after patching
	Loader::PatchCache::Key       key;
};

Loader::PatchCache::Key MakeKey(const uint8_t* area) {
	Loader::PatchCache::KeyBuilder key;
	key.AddU64(Loader::PatchCache::PatchVersion);
	key.AddU64(reinterpret_cast<uint64_t>(area));
	key.Add(area, kCodeSize);
	return key.Finish();
}

// The same sequence as RuntimeLinker::LoadProgram: snapshot, AMD patcher, split behind it, capture.
Patched PatchArea(uint8_t* area) {
	Patched out;
	out.area = area;
	out.key  = MakeKey(area);
	const std::vector<std::pair<uint64_t, uint64_t>> segments {{reinterpret_cast<uint64_t>(area), kCodeSize}};
	Loader::PatchCache::Snapshot                     snapshot(segments);
	uint8_t*                                         trampoline = area + kPageSize;
	const uint64_t                                   tramp_size = kAreaSize - kPageSize;
	Loader::RegisterGuestInstructionPatchModule(area, kPageSize, trampoline, tramp_size);
	const std::array<uintptr_t, 2> starts {reinterpret_cast<uintptr_t>(area), reinterpret_cast<uintptr_t>(area) + kStoreAt};
	const Loader::GuestInstructionHostFeatures host {false, false, false};
	const auto result = Loader::PatchGuestInstructions(reinterpret_cast<uintptr_t>(area), kCodeSize, starts, false, true, host);
	Check(result.extrq.found == 1 && result.extrq.native == 1, "EXTRQ patched natively");
	const uint64_t patcher_bytes = Loader::GetGuestInstructionTrampolineBytesUsed(area);
	uint64_t       cursor        = reinterpret_cast<uint64_t>(trampoline) + patcher_bytes;
	auto           split         = Loader::X64InstructionEmulator::SplitWideStores(reinterpret_cast<uint64_t>(area) + kStoreAt,
	                                                                  sizeof(kWideCopy), &cursor,
	                                                                  reinterpret_cast<uint64_t>(area) + kAreaSize);
	Check(split.patched == 1, "256-bit store split");
	Loader::UnregisterGuestInstructionPatchModule(area);
	const uint64_t used = std::max<uint64_t>(patcher_bytes, cursor - reinterpret_cast<uint64_t>(trampoline));
	out.entry           = snapshot.Capture(reinterpret_cast<uint64_t>(trampoline), used);
	out.entry.patcher_bytes = patcher_bytes;
	out.entry.wide_traps    = split.traps;
	out.entry.stats         = {1, 2, 3};
	out.image.assign(area, area + kAreaSize);
	return out;
}

void TestApplyReproducesImage() {
	auto* reference_area = MapArea();
	Fill(reference_area, 0x28);
	const auto patched = PatchArea(reference_area);
	Check(patched.entry.ranges.size() >= 2, "EXTRQ site and the store site are two ranges");
	Check(!patched.entry.trampoline.empty(), "trampolines captured");
	Check(patched.entry.patcher_bytes != 0 && patched.entry.patcher_bytes < patched.entry.trampoline.size(), "the split follows the patcher bytes");
	// Restore the module to its pristine bytes (and a zeroed trampoline area), then apply the cached effect.
	Fill(reference_area, 0x28);
	std::memset(reference_area + kPageSize, 0, kAreaSize - kPageSize);
	Check(Loader::PatchCache::OriginalsMatch(patched.entry), "originals match on the pristine module");
	Check(Loader::PatchCache::Apply(patched.entry), "apply succeeds");
	Check(std::memcmp(reference_area, patched.image.data(), kAreaSize) == 0, "applied image equals the patched image, byte for byte");
	::munmap(reference_area, kAreaSize);
}

void TestChangedSourceRejected() {
	auto* area = MapArea();
	Fill(area, 0x28);
	const auto patched = PatchArea(area);

	// The cache was built for EXTRQ length 0x28; the module now holds another length under the same address.
	Fill(area, 0x30);
	std::memset(area + kPageSize, 0, kAreaSize - kPageSize);
	const std::vector<uint8_t> before(area, area + kAreaSize);
	Check(!Loader::PatchCache::OriginalsMatch(patched.entry), "a changed source byte fails the original check");
	Check(!Loader::PatchCache::Apply(patched.entry), "apply refuses a changed source byte");
	Check(std::memcmp(area, before.data(), kAreaSize) == 0, "a refused apply writes nothing");

	Check(!(MakeKey(area) == patched.key), "a changed source byte changes the key");
	::munmap(area, kAreaSize);
}

void TestSerialization() {
	auto* area = MapArea();
	Fill(area, 0x28);
	auto patched = PatchArea(area);
	patched.entry.wide_traps.emplace_back(0x1000, 0x2000);
	patched.entry.stats = {7, 8, 9};
	const auto data     = Loader::PatchCache::Serialize(patched.key, patched.entry);

	Loader::PatchCache::Entry back;
	Check(Loader::PatchCache::Deserialize(data, patched.key, &back), "roundtrip");
	Check(back == patched.entry, "deserialized entry equals the original (ranges, trampoline, traps, stats)");
	Check(back.wide_traps.size() == patched.entry.wide_traps.size() && !back.wide_traps.empty(), "wide-store traps survive");

	auto other = patched.key;
	other.lo ^= 1;
	Loader::PatchCache::Entry rejected;
	Check(!Loader::PatchCache::Deserialize(data, other, &rejected), "another key is refused");

	auto corrupt = data;
	corrupt[corrupt.size() - 16] ^= 0x40;
	Check(!Loader::PatchCache::Deserialize(corrupt, patched.key, &rejected), "a corrupt file is refused");
	auto truncated = data;
	truncated.resize(data.size() / 2);
	Check(!Loader::PatchCache::Deserialize(truncated, patched.key, &rejected), "a truncated file is refused");

	// Through the disk.
	char dir_template[] = "/tmp/patchcachetestXXXXXX";
	const char* dir     = ::mkdtemp(dir_template);
	Check(dir != nullptr, "temporary directory");
	if (dir != nullptr) {
		Check(Loader::PatchCache::Save(dir, patched.key, patched.entry), "save");
		Loader::PatchCache::Entry loaded;
		Check(Loader::PatchCache::Load(dir, patched.key, &loaded) && loaded == patched.entry, "load returns the saved entry");
		Check(!Loader::PatchCache::Load(dir, other, &loaded), "load of another key misses");
		std::filesystem::remove_all(dir);
	}
	::munmap(area, kAreaSize);
}

void TestPatchIsDeterministic() {
	// What verify relies on: patching the same module twice captures the same entry.
	auto* a = MapArea();
	Fill(a, 0x28);
	const auto first = PatchArea(a);
	Fill(a, 0x28);
	std::memset(a + kPageSize, 0, kAreaSize - kPageSize);
	const auto second = PatchArea(a);
	Check(first.entry == second.entry, "two patch runs capture equal entries");
	::munmap(a, kAreaSize);
}

void TestModeParsing() {
	using Loader::PatchCache::Mode;
	Check(Loader::PatchCache::ParseMode(nullptr) == Mode::On, "unset -> on");
	Check(Loader::PatchCache::ParseMode("") == Mode::On, "empty -> on");
	Check(Loader::PatchCache::ParseMode("1") == Mode::On, "1 -> on");
	Check(Loader::PatchCache::ParseMode("0") == Mode::Off, "0 -> off");
	Check(Loader::PatchCache::ParseMode("verify") == Mode::Verify, "verify");
	Check(Loader::PatchCache::ParseMode("bogus") == Mode::Off, "unknown -> off");
}

} // namespace

int main() {
	TestApplyReproducesImage();
	TestChangedSourceRejected();
	TestSerialization();
	TestPatchIsDeterministic();
	TestModeParsing();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failures\n", g_failures);
		return 1;
	}
	std::puts("patch cache tests passed");
	return 0;
}
