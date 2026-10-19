// S434 (S442): KYTY_PROGRAM_DISK_CACHE format and key. The fingerprint must move with every code-generating input (signature, KYTY_* variable that is
// not in kNonCodegenKytyEnv - also unknown ones -, config, device) and stay put for the others; the program key with every field of the program; a
// file must be refused when short, long, damaged or of another key; the writer must write on its own thread and never show a half written file.
// Mutants: see src/graphics/shader/programDiskCache.h.

#include "common/knownEnv.h"
#include "graphics/shader/programDiskCache.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <xxhash.h>

using namespace Libs::Graphics::ProgramDisk;

namespace {

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		g_failures++;
	}
}

FingerprintInputs Base(const char* const* env) {
	FingerprintInputs inputs;
	inputs.recompiler_signature = "sig-aaaa";
	inputs.environment          = env;
	inputs.device.vendor_id     = 0x106b;
	inputs.device.device_id     = 0x1234;
	inputs.device.driver_version = 7;
	inputs.device.subgroup_size = 32;
	inputs.device.maintenance8  = true;
	inputs.device.mesh_shader   = false;
	inputs.config.shader_optimization_type = 2;
	return inputs;
}

void TestMode() {
	Check(ParseMode(nullptr) == Mode::Off, "null -> off");
	Check(ParseMode("") == Mode::Off, "empty -> off");
	Check(ParseMode("0") == Mode::Off, "0 -> off");
	Check(ParseMode("1") == Mode::On, "1 -> on");
	Check(ParseMode("verify") == Mode::Verify, "verify -> verify");
	Check(ParseMode("yes") == Mode::Off, "other -> off");
}

void TestFingerprint() {
	const char* const env0[] = {"PATH=/usr/bin", "KYTY_OPT_ON=a,b", "KYTY_AUDIO_GAIN=0.05", nullptr};
	const auto        base   = MakeFingerprint(Base(env0));
	Check(base.enabled, "base fingerprint enabled");
	Check(MakeFingerprint(Base(env0)).key == base.key, "deterministic");

	// Not codegen: the key must not move.
	const char* const env_noncodegen[] = {"PATH=/other", "KYTY_OPT_ON=a,b", "KYTY_AUDIO_GAIN=0.9", "KYTY_PROGRAM_DISK_CACHE=verify",
	                                      "KYTY_PATCH_CACHE=1", "KYTY_DIAG_CP_PHASES=1", "KYTY_FRAME_LOG=1", "KYTY_PAD_SCRIPT=/x", nullptr};
	Check(MakeFingerprint(Base(env_noncodegen)).key == base.key, "non-codegen variables and non-KYTY variables do not move the key");

	// Codegen: every one of the listed names moves the key when it appears.
	for (const auto name: Common::kCodegenKytyEnv) {
		const std::string entry = std::string(name) + "=1";
		const char* const env[] = {"PATH=/usr/bin", "KYTY_OPT_ON=a,b", "KYTY_AUDIO_GAIN=0.05", entry.c_str(), nullptr};
		const auto        fp    = MakeFingerprint(Base(env));
		if (std::string_view(name) == "KYTY_OPT_ON") continue; // already in the base environment (checked below by value)
		if (fp.key == base.key) {
			std::fprintf(stderr, "FAIL: codegen variable %s does not move the key\n", std::string(name).c_str());
			g_failures++;
		}
	}
	const char* const env_value[] = {"KYTY_OPT_ON=a", "KYTY_AUDIO_GAIN=0.05", nullptr};
	Check(MakeFingerprint(Base(env_value)).key != base.key, "the value of a codegen variable moves the key");
	const char* const env_unknown[] = {"KYTY_OPT_ON=a,b", "KYTY_BRAND_NEW_SWITCH=1", nullptr};
	Check(MakeFingerprint(Base(env_unknown)).key != base.key, "an unknown KYTY_ name is in the key");
	const char* const env_order[] = {"KYTY_AUDIO_GAIN=0.05", "KYTY_OPT_ON=a,b", "PATH=/usr/bin", nullptr};
	Check(MakeFingerprint(Base(env_order)).key == base.key, "order of the environment does not matter");

	{
		auto inputs = Base(env0);
		inputs.recompiler_signature = "sig-bbbb";
		Check(MakeFingerprint(inputs).key != base.key, "recompiler signature moves the key");
	}
	{
		auto inputs = Base(env0);
		inputs.config.shader_optimization_type = 1;
		Check(MakeFingerprint(inputs).key != base.key, "shader optimization moves the key");
	}
	{
		auto inputs = Base(env0);
		inputs.device.vendor_id ^= 1;
		Check(MakeFingerprint(inputs).key != base.key, "vendor moves the key");
		inputs = Base(env0);
		inputs.device.device_id ^= 1;
		Check(MakeFingerprint(inputs).key != base.key, "device moves the key");
		inputs = Base(env0);
		inputs.device.driver_version ^= 1;
		Check(MakeFingerprint(inputs).key != base.key, "driver version moves the key");
		inputs = Base(env0);
		inputs.device.cache_uuid[15] ^= 1;
		Check(MakeFingerprint(inputs).key != base.key, "pipeline cache uuid moves the key");
		inputs = Base(env0);
		inputs.device.subgroup_size = 64;
		Check(MakeFingerprint(inputs).key != base.key, "subgroup size moves the key");
		inputs = Base(env0);
		inputs.device.maintenance8 = false;
		Check(MakeFingerprint(inputs).key != base.key, "maintenance8 moves the key");
		inputs = Base(env0);
		inputs.device.mesh_shader = true;
		Check(MakeFingerprint(inputs).key != base.key, "mesh shaders move the key");
	}
	{
		auto inputs = Base(env0);
		inputs.config.debug_dump = true;
		const auto fp = MakeFingerprint(inputs);
		Check(!fp.enabled && !fp.reason.empty(), "debug dump disables the cache");
		inputs = Base(env0);
		inputs.config.shader_log = true;
		Check(!MakeFingerprint(inputs).enabled, "shader log disables the cache");
	}
	Check(CodegenEnvironment(nullptr).empty(), "null environment is empty");
	const auto listed = CodegenEnvironment(env_noncodegen);
	Check(listed.size() == 1 && listed[0] == "KYTY_OPT_ON=a,b", "only the codegen entry of that environment is listed");
}

void TestProgramKey() {
	const char* const env[]  = {nullptr};
	const auto        fp     = MakeFingerprint(Base(env)).key;
	const std::vector<uint32_t> state = {1, 2, 3};
	const std::vector<uint32_t> code  = {0xdeadbeef, 0x12345678, 0};
	const std::vector<uint32_t> back  = {7, 8};
	ProgramIdentity id;
	id.stage = 5, id.params_hash = 0x1122334455667788ull, id.user_data_count = 16, id.code_size = 12;
	id.static_state = state, id.user_data_base = 8, id.wave_size = 64, id.code = code, id.back_code = back;
	const auto base = MakeProgramKey(fp, id);
	Check(MakeProgramKey(fp, id) == base, "program key deterministic");
	{
		auto other = id;
		other.stage = 6;
		Check(MakeProgramKey(fp, other) != base, "stage moves the program key");
		other = id;
		other.params_hash ^= 1;
		Check(MakeProgramKey(fp, other) != base, "params hash moves the program key");
		other = id;
		other.user_data_count++;
		Check(MakeProgramKey(fp, other) != base, "user data count moves the program key");
		other = id;
		other.code_size++;
		Check(MakeProgramKey(fp, other) != base, "code size moves the program key");
		other = id;
		other.user_data_base++;
		Check(MakeProgramKey(fp, other) != base, "user data base moves the program key");
		other = id;
		other.wave_size = 32;
		Check(MakeProgramKey(fp, other) != base, "wave size moves the program key");
	}
	{
		auto s2 = state;
		s2[1] ^= 0x80000000u;
		auto other = id;
		other.static_state = s2;
		Check(MakeProgramKey(fp, other) != base, "static state moves the program key");
		auto s3 = state;
		s3.push_back(0);
		other.static_state = s3;
		Check(MakeProgramKey(fp, other) != base, "static state length moves the program key");
	}
	{
		auto c2 = code;
		c2[2] ^= 1; // the same params.hash: a 64-bit collision of the guest code must not alias
		auto other = id;
		other.code = c2;
		Check(MakeProgramKey(fp, other) != base, "code bytes move the program key");
		auto b2 = back;
		b2[1] ^= 1;
		other = id;
		other.back_code = b2;
		Check(MakeProgramKey(fp, other) != base, "back code bytes move the program key");
	}
	Check(MakeProgramKey(Key {fp.lo ^ 1, fp.hi}, id) != base, "fingerprint moves the program key");

	const std::vector<uint8_t> perm = {1, 2, 3, 4};
	const auto pk = MakePermutationKey(base, perm);
	Check(MakePermutationKey(base, perm) == pk, "permutation key deterministic");
	auto perm2 = perm;
	perm2[3] ^= 1;
	Check(MakePermutationKey(base, perm2) != pk, "permutation bytes move the key");
	Check(MakePermutationKey(Key {base.lo, base.hi ^ 1}, perm) != pk, "program key moves the permutation key");
	Check(pk.Hex().size() == 32, "hex of a key has 32 digits");
}

void TestFile() {
	const Key program {0x1111, 0x2222};
	const Key perm {0x3333, 0x4444};
	std::vector<uint8_t> payload(1000);
	for (size_t i = 0; i < payload.size(); i++) payload[i] = static_cast<uint8_t>(i * 7 + 3);
	const auto file = SerializeFile(program, perm, payload);
	std::vector<uint8_t> out;
	Check(DeserializeFile(file, program, perm, &out) && out == payload, "round trip");
	Check(!DeserializeFile(file, Key {0x1111, 0x2223}, perm, &out), "another program key refused");
	Check(!DeserializeFile(file, program, Key {0x3334, 0x4444}, &out), "another permutation key refused");

	// Truncated at any length.
	bool any_accepted = false;
	for (size_t size = 0; size < file.size(); size += 13) {
		if (DeserializeFile(std::span(file).first(size), program, perm, &out)) any_accepted = true;
	}
	if (DeserializeFile(std::span(file).first(file.size() - 1), program, perm, &out)) any_accepted = true;
	Check(!any_accepted, "a truncated file is refused");

	// A flipped byte in the payload (checksum), in the magic, in the format.
	auto damaged = file;
	damaged[file.size() / 2] ^= 0x10;
	Check(!DeserializeFile(damaged, program, perm, &out), "a damaged payload is refused");
	damaged = file;
	damaged[0] ^= 1;
	Check(!DeserializeFile(damaged, program, perm, &out), "a damaged magic is refused");

	// A well-formed checksum over a file that is longer than its size field says.
	auto longer = file;
	longer.resize(longer.size() - 8);
	longer.push_back(0xAB);
	longer.push_back(0xCD);
	const uint64_t sum = XXH3_64bits(longer.data(), longer.size());
	const auto*    sum_bytes = reinterpret_cast<const uint8_t*>(&sum);
	longer.insert(longer.end(), sum_bytes, sum_bytes + 8);
	Check(!DeserializeFile(longer, program, perm, &out), "trailing bytes under a valid checksum are refused");

	// Another format version.
	auto format = file;
	format.resize(format.size() - 8);
	format[8] ^= 0x7f;
	const uint64_t sum2 = XXH3_64bits(format.data(), format.size());
	const auto*    sum2_bytes = reinterpret_cast<const uint8_t*>(&sum2);
	format.insert(format.end(), sum2_bytes, sum2_bytes + 8);
	Check(!DeserializeFile(format, program, perm, &out), "another format version is refused");

	const std::vector<uint8_t> empty_payload;
	const auto empty_file = SerializeFile(program, perm, empty_payload);
	Check(DeserializeFile(empty_file, program, perm, &out) && out.empty(), "an empty payload round trips");
}

std::filesystem::path TempDir(const char* name) {
	auto dir = std::filesystem::temp_directory_path() / (std::string("kyty_progdisk_") + name + "_" + std::to_string(getpid()));
	std::filesystem::remove_all(dir);
	std::filesystem::create_directories(dir);
	return dir;
}

size_t CountTmp(const std::filesystem::path& dir) {
	size_t n = 0;
	for (const auto& entry: std::filesystem::recursive_directory_iterator(dir)) {
		if (entry.path().string().find(".tmp.") != std::string::npos) n++;
	}
	return n;
}

void TestLoadSave() {
	const auto dir = TempDir("loadsave");
	const Key program {0xAA, 0xBB};
	const Key perm {0xCC, 0xDD};
	const auto path = FilePath(dir, 5, program, perm);
	Check(path.parent_path().filename() == "prog_v1", "files live in prog_v1");
	bool rejected = true;
	Check(!Load(path, program, perm, &rejected).has_value() && !rejected, "a missing file is a miss, not a rejection");

	const std::vector<uint8_t> payload = {9, 8, 7, 6, 5};
	Check(Save(path, SerializeFile(program, perm, payload)), "save");
	const auto loaded = Load(path, program, perm, &rejected);
	Check(loaded.has_value() && *loaded == payload && !rejected, "load");
	Check(!Load(path, Key {1, 1}, perm, &rejected).has_value() && rejected, "another key is a rejection");
	Check(CountTmp(dir) == 0, "no tmp file left");

	{
		std::ofstream file(path, std::ios::binary | std::ios::trunc);
		file << "garbage";
	}
	Check(!Load(path, program, perm, &rejected).has_value() && rejected, "garbage is a rejection");
	{
		std::ofstream file(path, std::ios::binary | std::ios::trunc);
	}
	Check(!Load(path, program, perm, &rejected).has_value() && rejected, "an empty file is a rejection");
	std::filesystem::remove_all(dir);
}

void TestWriter() {
	const auto dir = TempDir("writer");
	const Key program {1, 2};
	std::vector<std::filesystem::path> paths;
	{
		AsyncWriter writer;
		bool final_visible_during_write = false;
		writer.before_rename = [&](const std::filesystem::path& final_path) {
			if (std::filesystem::exists(final_path)) final_visible_during_write = true;
		};
		for (uint64_t i = 0; i < 20; i++) {
			const Key perm {i, i + 100};
			const std::vector<uint8_t> payload(4096 + i, static_cast<uint8_t>(i));
			paths.push_back(FilePath(dir, 1, program, perm));
			Check(writer.Enqueue(paths.back(), SerializeFile(program, perm, payload)), "enqueue");
		}
		writer.Flush();
		Check(writer.Written() == 20 && writer.Failed() == 0 && writer.Dropped() == 0, "20 files written");
		Check(writer.Bytes() > 20 * 4096, "bytes counted");
		Check(writer.LastWriter() != std::this_thread::get_id(), "files are written on the worker thread");
		Check(!final_visible_during_write, "the final file is not visible while it is being written");
		Check(CountTmp(dir) == 0, "no tmp file left after the flush");
		for (uint64_t i = 0; i < paths.size(); i++) {
			const auto loaded = Load(paths[i], program, Key {i, i + 100});
			Check(loaded.has_value() && loaded->size() == 4096 + i, "every file loads");
		}
		// The destructor flushes what is still queued.
		const Key perm {999, 1};
		const std::vector<uint8_t> payload(100, 1);
		paths.push_back(FilePath(dir, 1, program, perm));
		Check(writer.Enqueue(paths.back(), SerializeFile(program, perm, payload)), "enqueue before destruction");
	}
	Check(Load(paths.back(), program, Key {999, 1}).has_value(), "the destructor flushed the last file");

	// A failing write is counted and leaves nothing.
	{
		AsyncWriter writer;
		const auto  blocked = dir / "blocker";
		{
			std::ofstream file(blocked);
			file << "x";
		}
		Check(writer.Enqueue(blocked / "sub" / "f.kpp", std::vector<uint8_t>(10, 1)), "enqueue to an impossible path");
		writer.Flush();
		Check(writer.Failed() == 1 && writer.Written() == 0, "failure counted");
	}
	std::filesystem::remove_all(dir);
}

} // namespace

int main() {
	TestMode();
	TestFingerprint();
	TestProgramKey();
	TestFile();
	TestLoadSave();
	TestWriter();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::puts("ProgramDiskCacheTests: OK");
	return 0;
}
