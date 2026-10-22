// S476: (1) the fingerprint of KYTY_PROGRAM_DISK_CACHE reads the real environment of the process on every platform (it was nullptr off macOS, so a
// KYTY_* switch that changes the SPIR-V, e.g. KYTY_FIX_PS_WAVE32_MASK, did not move the key on Windows); (2) KYTY_PROGRAM_DISK_PREFETCH: the read-ahead
// store answers Load() with the bytes of the file, once, under its byte limit, and the bytes still go through the checks of the file format.
// Mutants (WILL_FAIL): PROCESS_ENV_NULL, PREFETCH_UNUSED, PREFETCH_KEEPS_BYTES, PREFETCH_UNBOUNDED, LIST_IGNORES_FINGERPRINT.

#include "common/knownEnv.h"
#include "graphics/shader/programDiskCache.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace Libs::Graphics::ProgramDisk;

namespace {

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		g_failures++;
	}
}

void SetEnv(const char* name, const char* value) {
#ifdef _WIN32
	_putenv_s(name, value);
#else
	setenv(name, value, 1);
#endif
}

FingerprintInputs Base(const char* const* env) {
	FingerprintInputs inputs;
	inputs.recompiler_signature            = "sig-aaaa";
	inputs.environment                     = env;
	inputs.device.vendor_id                = 0x10de;
	inputs.device.device_id                = 0x2684;
	inputs.device.driver_version           = 9;
	inputs.device.subgroup_size            = 32;
	inputs.config.shader_optimization_type = 2;
	return inputs;
}

void TestProcessEnvironment() {
	SetEnv("KYTY_FIX_PS_WAVE32_MASK", "0");
	SetEnv("KYTY_FIX_WIN_READ_EFAULT", "0");
	SetEnv("KYTY_PROGRAM_DISK_PREFETCH", "0");
	const auto key_off = MakeFingerprint(Base(ProcessEnvironment())).key;
	Check(ProcessEnvironment() != nullptr, "the process environment exists");
	const auto entries = CodegenEnvironment(ProcessEnvironment());
	bool       seen    = false;
	for (const auto& entry: entries) seen = seen || entry == "KYTY_FIX_PS_WAVE32_MASK=0";
	Check(seen, "a KYTY_ codegen variable set in this process is in the fingerprint environment");

	SetEnv("KYTY_FIX_PS_WAVE32_MASK", "1");
	const auto key_on = MakeFingerprint(Base(ProcessEnvironment())).key;
	Check(!(key_on == key_off), "KYTY_FIX_PS_WAVE32_MASK=1 moves the key of the real environment");

	SetEnv("KYTY_FIX_WIN_READ_EFAULT", "1");
	SetEnv("KYTY_PROGRAM_DISK_PREFETCH", "1");
	Check(MakeFingerprint(Base(ProcessEnvironment())).key == key_on, "KYTY_FIX_WIN_READ_EFAULT and KYTY_PROGRAM_DISK_PREFETCH do not move the key");
	Check(!IsNonCodegenEnv("KYTY_FIX_PS_WAVE32_MASK"), "KYTY_FIX_PS_WAVE32_MASK is codegen");
	Check(IsNonCodegenEnv("KYTY_PROGRAM_DISK_PREFETCH") && IsNonCodegenEnv("KYTY_FIX_WIN_READ_EFAULT"), "prefetch and the read fix are not codegen");
}

std::vector<uint8_t> Payload(uint8_t seed, size_t size) {
	std::vector<uint8_t> payload(size);
	for (size_t i = 0; i < size; i++) payload[i] = static_cast<uint8_t>(seed + i * 7);
	return payload;
}

void TestPrefetch() {
	namespace fs = std::filesystem;
	const auto dir = fs::temp_directory_path() / ("kyty_prefetch_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())) / "prog_v1";
	fs::create_directories(dir);
	const Key program {1, 2};
	const Key permutation {3, 4};
	const auto payload_a = Payload(1, 400);
	const auto payload_b = Payload(2, 400);
	const auto path_a    = FilePath(dir.parent_path(), 7, program, permutation);
	const auto path_b    = FilePath(dir.parent_path(), 7, program, Key {5, 6});
	Check(path_a.parent_path() == dir, "test path layout");
	Check(Save(path_a, SerializeFile(program, permutation, payload_a)), "save a");
	Check(Save(path_b, SerializeFile(program, Key {5, 6}, payload_b)), "save b");

	{
		Prefetch prefetch({path_a, path_b}, 1ull << 20);
		prefetch.WaitDone();
		SetPrefetch(&prefetch);
		bool rejected = true;
		const auto loaded = Load(path_a, program, permutation, &rejected);
		Check(loaded && *loaded == payload_a && !rejected, "Load returns the payload from the read-ahead bytes");
		auto stats = prefetch.GetStats();
		Check(stats.files == 2 && stats.bytes > 800, "the worker read both files");
		Check(stats.hit == 1, "the first Load was answered from the store");
		const auto again = Load(path_a, program, permutation, &rejected);
		Check(again && *again == payload_a, "the second Load reads the file");
		stats = prefetch.GetStats();
		Check(stats.hit == 1 && stats.miss == 1, "the bytes are handed out once");
		// The same bytes under another key are refused by the format checks, prefetched or not.
		const auto wrong = Load(path_b, program, permutation, &rejected);
		Check(!wrong && rejected, "bytes of another permutation are rejected");
		Check(!Load(dir / "none.kpp", program, permutation, &rejected) && !rejected, "a file that does not exist is a plain miss");
		SetPrefetch(nullptr);
	}
	{
		// A damaged file is refused when it comes from the store.
		auto bytes = SerializeFile(program, permutation, payload_a);
		bytes[bytes.size() / 2] ^= 0xff;
		const auto path_bad = FilePath(dir.parent_path(), 9, program, permutation);
		Check(Save(path_bad, bytes), "save damaged");
		Prefetch prefetch({path_bad}, 1ull << 20);
		prefetch.WaitDone();
		SetPrefetch(&prefetch);
		bool rejected = false;
		Check(!Load(path_bad, program, permutation, &rejected) && rejected, "a damaged file from the store is rejected");
		SetPrefetch(nullptr);
	}
	{
		// The limit: 3 files of about 450 bytes, room for one.
		Prefetch prefetch({path_a, path_b, FilePath(dir.parent_path(), 9, program, permutation)}, 600);
		prefetch.WaitDone();
		const auto stats = prefetch.GetStats();
		Check(stats.files == 1 && stats.bytes <= 600, "the worker stops at the byte limit");
	}
	{
		// Only the listed files are read: a file that is in the directory but not in the list is not prefetched.
		Prefetch prefetch({path_a}, 1ull << 20);
		prefetch.WaitDone();
		Check(prefetch.GetStats().files == 1, "only the listed file is read");
		Check(!prefetch.Take(path_b), "a file that is not in the list is not in the store");
	}
	std::error_code error;
	fs::remove_all(dir.parent_path(), error);
}

void TestUsedList() {
	const Key fp {0x1111, 0x2222};
	const Key other {0x1111, 0x2223};
	const std::vector<std::string> names = {"1_00000000000000aa_00000000000000bb.kpp", "7_00000000000000cc_00000000000000dd.kpp"};
	const auto text = SerializeUsedList(fp, names);
	Check(ParseUsedList(text, fp) == names, "the used list round-trips");
	Check(ParseUsedList(text, other).empty(), "a list of another fingerprint gives no names");
	Check(ParseUsedList("", fp).empty(), "an empty list gives no names");
	const auto hostile = fp.Hex() + "\n../x.kpp\n/etc/passwd\na\\b.kpp\n1_00000000000000aa_00000000000000bb.kpp\r\nnot a name\n..kpp\n";
	const auto parsed  = ParseUsedList(hostile, fp);
	Check(parsed.size() == 1 && parsed[0] == names[0], "lines that are not file names are skipped (no path separators, no ..)");
	Check(UsedListPath("d").filename() == "prog_used_v1.lst", "list path");
}

} // namespace

int main() {
	TestProcessEnvironment();
	TestPrefetch();
	TestUsedList();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("ProgramDiskPrefetchTests: OK\n");
	return 0;
}
