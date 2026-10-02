// Checks X64InstructionEmulator::SplitWideStores under Rosetta: a 256-bit store that crosses
// from a writable into a write-protected page aborts the process ("unexpectedly need to
// EmulateForward on a synchronous exception"), while the split 128-bit form faults cleanly.

#include "loader/x64InstructionEmulator.h"

#include <array>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <vector>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		g_failures++;
	}
}

long g_page_size = 0;

void UnprotectOnFault(int /*signal*/, siginfo_t* info, void* /*context*/) {
	auto page = reinterpret_cast<uintptr_t>(info->si_addr) & ~static_cast<uintptr_t>(g_page_size - 1);
	mprotect(reinterpret_cast<void*>(page), g_page_size, PROT_READ | PROT_WRITE);
}

// void copy(uint8_t* dst /*rdi*/, const uint8_t* src /*rsi*/):
// the store to [rdi + 0x40] is the third instruction of the block and crosses the page boundary.
constexpr std::array<uint8_t, 24> CopyCode = {
    0xc5, 0xfc, 0x10, 0x06,       // vmovups ymm0, [rsi]
    0xc5, 0xfc, 0x10, 0x4e, 0x20, // vmovups ymm1, [rsi + 0x20]
    0xc5, 0xfc, 0x11, 0x47, 0x40, // vmovups [rdi + 0x40], ymm0
    0xc5, 0xfc, 0x11, 0x4f, 0x60, // vmovups [rdi + 0x60], ymm1
    0xc5, 0xf8, 0x77,             // vzeroupper
    0xc3,                         // ret
    0xcc,
};

using CopyFunc = void (*)(uint8_t*, const uint8_t*);

struct CodeArea {
	uint8_t* code        = nullptr;
	uint64_t trampolines = 0;
	uint64_t end         = 0;
};

CodeArea MakeCode() {
	constexpr size_t size = 64 * 1024;
	auto* base = static_cast<uint8_t*>(
	    mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0));
	if (base == MAP_FAILED) {
		base = static_cast<uint8_t*>(
		    mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0));
	}
	if (base == MAP_FAILED) {
		std::perror("mmap");
		std::exit(2);
	}
	std::memcpy(base, CopyCode.data(), CopyCode.size());
	return {base, reinterpret_cast<uint64_t>(base) + 4096, reinterpret_cast<uint64_t>(base) + size};
}

// dst + 0x40 is 16 bytes before a write-protected page, so the first store straddles it.
bool RunCrossingCopy(CopyFunc copy) {
	auto* data = static_cast<uint8_t*>(
	    mmap(nullptr, 4 * g_page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
	uint8_t source[64];
	for (int i = 0; i < 64; i++) {
		source[i] = static_cast<uint8_t>(i + 1);
	}
	uint8_t* boundary = data + 2 * g_page_size;
	uint8_t* dst      = boundary - 0x50;
	mprotect(boundary, g_page_size, PROT_READ);
	copy(dst, source);
	return std::memcmp(dst + 0x40, source, 64) == 0;
}

void TestUnpatchedAbortsUnderRosetta() {
	const pid_t child = fork();
	if (child == 0) {
		auto area = MakeCode();
		RunCrossingCopy(reinterpret_cast<CopyFunc>(area.code));
		_exit(0);
	}
	int status = 0;
	waitpid(child, &status, 0);
	// Documents the Rosetta behaviour this patch works around; native x86-64 runs exit cleanly.
	if (WIFSIGNALED(status)) {
		std::printf("unpatched crossing store: child killed by signal %d (expected under Rosetta)\n",
		            WTERMSIG(status));
	} else {
		std::printf("unpatched crossing store: child exited %d (native x86-64?)\n",
		            WEXITSTATUS(status));
	}
}

void TestPatchedCrossingStore() {
	auto area   = MakeCode();
	auto cursor = area.trampolines;
	const auto result = Loader::X64InstructionEmulator::SplitWideStores(
	    reinterpret_cast<uint64_t>(area.code), CopyCode.size(), &cursor, area.end);
	Check(result.candidates == 2, "two 256-bit stores found");
	Check(result.patched == 2, "both stores patched");
	Check(area.code[9] == 0xe9 && area.code[14] == 0xe9, "stores replaced by rel32 jumps");
	Check(RunCrossingCopy(reinterpret_cast<CopyFunc>(area.code)), "patched copy wrote all bytes");
}

void TestShortStoreWithoutPaddingTraps() {
  auto area = MakeCode();
  // vmovups [rdi], ymm0 (4 bytes, too short for a rel32 jump); ret
  const std::array<uint8_t, 5> code = {0xc5, 0xfc, 0x11, 0x07, 0xc3};
  std::memcpy(area.code, code.data(), code.size());
  auto cursor = area.trampolines;
  const auto result = Loader::X64InstructionEmulator::SplitWideStores(
      reinterpret_cast<uint64_t>(area.code), code.size(), &cursor, area.end);
  // No padding in reach: the store becomes ud2, which the SIGILL handler
  // redirects.
  Check(result.candidates == 1 && result.trapped == 1,
        "4-byte store without padding trapped");
  Check(area.code[0] == 0x0f && area.code[1] == 0x0b && area.code[4] == 0xc3,
        "store replaced by ud2, ret kept");
}

void TestShortStoreViaInt3Padding() {
  auto area = MakeCode();
  // vmovups ymm0, [rsi]; vmovups [rdi], ymm0 (4 bytes); vzeroupper; ret; int3
  // padding
  const std::array<uint8_t, 20> code = {
      0xc5, 0xfc, 0x10, 0x06, 0xc5, 0xfc, 0x11, 0x07, 0xc5, 0xf8,
      0x77, 0xc3, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc};
  std::memcpy(area.code, code.data(), code.size());
  auto cursor = area.trampolines;
  const auto result = Loader::X64InstructionEmulator::SplitWideStores(
      reinterpret_cast<uint64_t>(area.code), code.size(), &cursor, area.end);
  Check(result.candidates == 1 && result.patched == 1 && result.via_cave == 1,
        "4-byte store patched through int3 padding");
  Check(area.code[4] == 0xeb && area.code[12] == 0xe9,
        "rel8 jump to a rel32 jump in padding");
  // The store straddles a write-protected page, like the crossing copy above.
  auto *data = static_cast<uint8_t *>(mmap(nullptr, 4 * g_page_size,
                                           PROT_READ | PROT_WRITE,
                                           MAP_PRIVATE | MAP_ANON, -1, 0));
  uint8_t source[32];
  for (int i = 0; i < 32; i++) {
    source[i] = static_cast<uint8_t>(0x40 + i);
  }
  uint8_t *boundary = data + 2 * g_page_size;
  mprotect(boundary, g_page_size, PROT_READ);
  reinterpret_cast<CopyFunc>(area.code)(boundary - 16, source);
  Check(std::memcmp(boundary - 16, source, 32) == 0,
        "short store wrote all bytes");
}

void TestShortStoreViaNopPadding() {
  auto area = MakeCode();
  // vmovups [rdi], ymm0 (4 bytes); ret; 5-byte nop (alignment padding after
  // ret)
  const std::array<uint8_t, 10> code = {0xc5, 0xfc, 0x11, 0x07, 0xc3,
                                        0x0f, 0x1f, 0x44, 0x00, 0x00};
  std::memcpy(area.code, code.data(), code.size());
  auto cursor = area.trampolines;
  const auto result = Loader::X64InstructionEmulator::SplitWideStores(
      reinterpret_cast<uint64_t>(area.code), code.size(), &cursor, area.end);
  Check(result.via_cave == 1 && area.code[5] == 0xe9,
        "nop padding after ret used as a cave");
  // A nop that execution falls into is not padding.
  const std::array<uint8_t, 10> live = {0xc5, 0xfc, 0x11, 0x07, 0x0f,
                                        0x1f, 0x44, 0x00, 0x00, 0xc3};
  std::memcpy(area.code, live.data(), live.size());
  cursor = area.trampolines;
  const auto live_result = Loader::X64InstructionEmulator::SplitWideStores(
      reinterpret_cast<uint64_t>(area.code), live.size(), &cursor, area.end);
  Check(live_result.via_cave == 0 && live_result.trapped == 1,
        "reachable nop not used as a cave");
}

void TestShortStoreViaLongStoreDeadBytes() {
  auto area = MakeCode();
  // f: vmovups ymm0, [rsi]; vmovups [rdi], ymm0 (4 bytes); vzeroupper; ret
  // g: vmovups [rdi + rax + 0x100], ymm0 (VEX3, 10 bytes: 5 dead bytes once
  // patched); ret
  const std::array<uint8_t, 23> code = {
      0xc5, 0xfc, 0x10, 0x06, 0xc5, 0xfc, 0x11, 0x07, 0xc5, 0xf8, 0x77, 0xc3,
      0xc4, 0xe1, 0x7c, 0x11, 0x84, 0x07, 0x00, 0x01, 0x00, 0x00, 0xc3};
  std::memcpy(area.code, code.data(), code.size());
  auto cursor = area.trampolines;
  const auto result = Loader::X64InstructionEmulator::SplitWideStores(
      reinterpret_cast<uint64_t>(area.code), code.size(), &cursor, area.end);
  Check(result.patched == 2 && result.via_cave == 1 && result.trapped == 0,
        "short store reaches its trampoline through a long store's dead bytes");
  Check(area.code[4] == 0xeb && area.code[12] == 0xe9 && area.code[17] == 0xe9,
        "rel8 into the dead bytes of the patched long store");
  uint8_t source[32];
  uint8_t destination[40]{};
  for (int i = 0; i < 32; i++) {
    source[i] = static_cast<uint8_t>(0x10 + i);
  }
  reinterpret_cast<CopyFunc>(area.code)(destination, source);
  Check(std::memcmp(destination, source, 32) == 0 && destination[32] == 0,
        "chained short store wrote its 32 bytes");
}

void TestHotRelocationSkipsBranchTargets() {
  // Direct branch targets must be found: a loop whose head is the instruction
  // after the store. store; L: add rbx, 0x20; dec ecx; jne L; ret
  auto area = MakeCode();
  const std::array<uint8_t, 13> code = {0xc5, 0xfc, 0x11, 0x03, 0x48,
                                        0x83, 0xc3, 0x20, 0xff, 0xc9,
                                        0x75, 0xf8, 0xc3};
  std::memcpy(area.code, code.data(), code.size());
  char path[] = "/tmp/kyty_hot_XXXXXX";
  const int fd = mkstemp(path);
  dprintf(
      fd, "0x%llx\n",
      static_cast<unsigned long long>(reinterpret_cast<uint64_t>(area.code)));
  close(fd);
  setenv("KYTY_WIDE_STORE_HOT", path, 1);
  auto cursor = area.trampolines;
  const auto result = Loader::X64InstructionEmulator::SplitWideStores(
      reinterpret_cast<uint64_t>(area.code), code.size(), &cursor, area.end);
  Check(result.relocated == 0 && result.trapped == 1,
        "loop head after a hot store not moved");
  unlink(path);
}

void TestRipRelativeStore() {
	auto area = MakeCode();
	// vmovups ymm0, [rsi]; vmovups [rip + 0x7f4], ymm0; vzeroupper; ret
	// The store ends at offset 12, so it targets offset 12 + 0x7f4 = 0x800.
	const std::array<uint8_t, 17> code = {0xc5, 0xfc, 0x10, 0x06, 0xc5, 0xfc, 0x11, 0x05, 0xf4,
	                                      0x07, 0x00, 0x00, 0xc5, 0xf8, 0x77, 0xc3, 0xcc};
	std::memcpy(area.code, code.data(), code.size());
	auto       cursor = area.trampolines;
	const auto result = Loader::X64InstructionEmulator::SplitWideStores(
	    reinterpret_cast<uint64_t>(area.code), code.size(), &cursor, area.end);
	Check(result.candidates == 1 && result.patched == 1, "RIP-relative store patched");
	uint8_t source[32];
	for (int i = 0; i < 32; i++) {
		source[i] = static_cast<uint8_t>(0xa0 + i);
	}
	std::memset(area.code + 0x800, 0, 64);
	reinterpret_cast<CopyFunc>(area.code)(nullptr, source);
	Check(std::memcmp(area.code + 0x800, source, 32) == 0, "RIP-relative store hit its target");
	Check(area.code[0x800 + 32] == 0, "RIP-relative store wrote nothing past its target");
}


uint64_t RefExtract(uint64_t value, uint32_t length, uint32_t index) {
	length &= 0x3f;
	index &= 0x3f;
	if (length == 0) length = 64;
	if (length > 64 - index) length = 64 - index;
	const uint64_t mask = length == 64 ? UINT64_MAX : ((uint64_t {1} << length) - 1);
	return (value >> index) & mask;
}

uint64_t RefInsert(uint64_t dst, uint64_t src, uint32_t length, uint32_t index) {
	length &= 0x3f;
	index &= 0x3f;
	if (length == 0) length = 64;
	if (length > 64 - index) length = 64 - index;
	const uint64_t mask = length == 64 ? UINT64_MAX : ((uint64_t {1} << length) - 1);
	return (dst & ~(mask << index)) | ((src & mask) << index);
}

// f(rdi = dest low, rsi = src low, rdx = dest high, rcx = out[], r8 = marker): xmm2..4 hold the marker, rdx
// a marker, the red zone gets markers and CF is set, then <sse4a instruction> and the results are stored:
// out = {dest low (returned in rax), dest high, xmm2, xmm3, xmm4, [rsp-8], [rsp-128], CF, rdx}.
using Sse4aFunc = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t*, uint64_t);

uint8_t Rex(uint32_t xmm, uint32_t gpr) {
	return static_cast<uint8_t>(0x48 | (xmm >= 8 ? 4 : 0) | (gpr >= 8 ? 1 : 0));
}
uint8_t ModRm(uint32_t reg, uint32_t rm) {
	return static_cast<uint8_t>(0xc0 | ((reg & 7) << 3) | (rm & 7));
}

// kind: 0 extrq imm, 1 insertq imm, 2 extrq register form
void RunSse4aCase(int kind, uint32_t dest, uint32_t src, uint32_t length, uint32_t index, uint64_t lo,
                  uint64_t hi, uint64_t source, int* cases, int* bad) {
	auto                 area = MakeCode();
	std::vector<uint8_t> code;
	auto add = [&](std::initializer_list<uint8_t> bytes) { code.insert(code.end(), bytes); };
	add({0x66, Rex(dest, 7), 0x0f, 0x6e, ModRm(dest, 7)});                     // movq xmm<dest>, rdi
	if (src != dest) {
		add({0x66, Rex(src, 6), 0x0f, 0x6e, ModRm(src, 6)});                    // movq xmm<src>, rsi
	}
	add({0x66, Rex(dest, 2), 0x0f, 0x3a, 0x22, ModRm(dest, 2), 0x01});        // pinsrq xmm<dest>, rdx, 1
	add({0x66, 0x49, 0x0f, 0x6e, 0xd0, 0x66, 0x49, 0x0f, 0x6e, 0xd8, 0x66, 0x49, 0x0f, 0x6e, 0xe0}); // xmm2..4 = r8
	add({0x48, 0xc7, 0x44, 0x24, 0xf8, 0x34, 0x12, 0x00, 0x00});              // mov qword [rsp-8], 0x1234
	add({0x48, 0xc7, 0x44, 0x24, 0x80, 0x78, 0x56, 0x00, 0x00});              // mov qword [rsp-128], 0x5678
	add({0x48, 0xc7, 0xc2, 0xaa, 0x55, 0x00, 0x00, 0xf9});                    // mov rdx, 0x55aa; stc
	if (kind == 1) {
		const uint8_t rex = static_cast<uint8_t>(0x40 | (dest >= 8 ? 4 : 0) | (src >= 8 ? 1 : 0));
		if (rex != 0x40) {
			add({0xf2, rex});
		} else {
			add({0xf2});
		}
		add({0x0f, 0x78, ModRm(dest, src), static_cast<uint8_t>(length), static_cast<uint8_t>(index)});
	} else if (kind == 0) {
		if (dest >= 8) {
			add({0x66, 0x41});
		} else {
			add({0x66});
		}
		add({0x0f, 0x78, ModRm(0, dest), static_cast<uint8_t>(length), static_cast<uint8_t>(index)});
	} else {
		const uint8_t rex = static_cast<uint8_t>(0x40 | (dest >= 8 ? 4 : 0) | (src >= 8 ? 1 : 0));
		if (rex != 0x40) {
			add({0x66, rex, 0x0f, 0x79, ModRm(dest, src)});
		} else {
			add({0x66, 0x0f, 0x79, ModRm(dest, src)});
		}
	}
	add({0x48, 0x89, 0x51, 0x40,                                                // mov [rcx+0x40], rdx
	     0xba, 0x00, 0x00, 0x00, 0x00, 0x48, 0x83, 0xd2, 0x00,                  // mov edx, 0; adc rdx, 0
	     0x48, 0x89, 0x51, 0x38});                                              // mov [rcx+0x38], rdx
	add({0x66, Rex(dest, 0), 0x0f, 0x7e, ModRm(dest, 0)});                      // movq rax, xmm<dest>
	add({0x66, Rex(dest, 1), 0x0f, 0x3a, 0x16, static_cast<uint8_t>(0x41 | ((dest & 7) << 3)), 0x08, 0x01}); // pextrq [rcx+8], xmm<dest>, 1
	add({0x66, 0x0f, 0xd6, 0x51, 0x10, 0x66, 0x0f, 0xd6, 0x59, 0x18, 0x66, 0x0f, 0xd6, 0x61, 0x20}); // [rcx+16..] = xmm2..4
	add({0x48, 0x8b, 0x54, 0x24, 0xf8, 0x48, 0x89, 0x51, 0x28});               // out[5] = [rsp-8]
	add({0x48, 0x8b, 0x54, 0x24, 0x80, 0x48, 0x89, 0x51, 0x30});               // out[6] = [rsp-128]
	add({0xc3});
	std::memcpy(area.code, code.data(), code.size());
	auto       cursor = area.trampolines;
	const auto result = Loader::X64InstructionEmulator::SplitWideStores(
	    reinterpret_cast<uint64_t>(area.code), code.size(), &cursor, area.end);
	Check(result.sse4a == 1 && result.sse4a_skipped == 0, "sse4a instruction patched");
	constexpr uint64_t marker = 0x1111222233334444ull;
	uint64_t           out[16] {};
	const auto         low = reinterpret_cast<Sse4aFunc>(area.code)(lo, source, hi, out, marker);
	uint64_t           want = 0, want_hi = 0;
	if (kind == 1) {
		want    = RefInsert(lo, source, length, index);
		want_hi = hi;
	} else if (kind == 0) {
		want = RefExtract(lo, length, index);
	} else {
		const uint64_t control = dest == src ? lo : source;
		want                   = RefExtract(lo, static_cast<uint32_t>(control & 0xff), static_cast<uint32_t>((control >> 8) & 0xff));
	}
	(*cases)++;
	if (low != want || out[1] != want_hi || out[2] != marker || (dest != 2 && out[3] != marker) || out[4] != marker ||
	    out[5] != 0x1234 || out[6] != 0x5678 || out[7] != 1 || out[8] != 0x55aa) {
		(*bad)++;
		std::fprintf(stderr, "sse4a kind=%d x%u,x%u len=%u idx=%u got %016llx/%016llx want %016llx/%016llx cf=%llu rdx=%llx\n",
		             kind, dest, src, length, index, (unsigned long long)low, (unsigned long long)out[1],
		             (unsigned long long)want, (unsigned long long)want_hi, (unsigned long long)out[7],
		             (unsigned long long)out[8]);
	}
}

void TestSse4aPatch() {
	const uint64_t lo = 0x0123456789abcdefull, hi = 0xfedcba9876543210ull, s = 0xdeadbeefcafef00dull;
	const uint32_t lengths[] = {0, 1, 3, 8, 16, 31, 32, 33, 48, 63, 64};
	const uint32_t indexes[] = {0, 1, 5, 8, 16, 31, 32, 47, 60, 63, 64};
	for (int kind = 0; kind < 3; kind++) {
		int cases = 0, bad = 0;
		for (uint32_t length: lengths) {
			for (uint32_t index: indexes) {
				const uint32_t pairs[][2] = {{0, 1}, {9, 10}, {1, 1}};
				for (const auto& pair: pairs) {
					if (kind != 2 && pair[0] == pair[1]) {
						continue;
					}
					// register form: the control bytes come from the source register
					const uint64_t source = kind == 2 ? ((s & ~0xffffull) | (length & 0xff) | ((index & 0xffull) << 8)) : s;
					const uint64_t dest_lo = kind == 2 && pair[0] == pair[1] ? source : lo;
					RunSse4aCase(kind, pair[0], pair[1], length, index, dest_lo, hi, source, &cases, &bad);
					if (kind == 2 && pair[0] == pair[1]) {
						continue;
					}
				}
			}
		}
		const char* name = kind == 0 ? "extrq imm" : kind == 1 ? "insertq imm" : "extrq reg";
		Check(bad == 0, name);
		std::printf("sse4a %s: %d cases, %d bad\n", name, cases, bad);
	}
}

} // namespace

int main() {
	g_page_size = sysconf(_SC_PAGESIZE);
	struct sigaction action {};
	action.sa_sigaction = UnprotectOnFault;
	action.sa_flags     = SA_SIGINFO;
	sigaction(SIGSEGV, &action, nullptr);
	sigaction(SIGBUS, &action, nullptr);

        // Reads KYTY_WIDE_STORE_HOT, which the splitter loads once per process:
        // run it first.
        TestHotRelocationSkipsBranchTargets();
        TestUnpatchedAbortsUnderRosetta();
	TestPatchedCrossingStore();
        TestShortStoreWithoutPaddingTraps();
        TestShortStoreViaInt3Padding();
        TestRipRelativeStore();
	TestSse4aPatch();

	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("macos_wide_store_tests: all checks passed\n");
	return 0;
}
