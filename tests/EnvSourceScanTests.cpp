#include "common/knownEnv.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <set>
#include <sstream>
#include <string>

// S370: every getenv("KYTY_...") literal in src/ is listed in Common::kKnownKytyEnv (KYTY_ENV_STRICT=1 aborts on a name that is not).
// Built normally and with KYTY_KNOWNENV_MUTANT_NO_EAGER (KYTY_EAGER_SUBMIT missing from the list): that build must FAIL (WILL_FAIL).
// KYTY_SOURCE_SCAN_DIR = the src/ directory, set by CMake.

#ifndef KYTY_SOURCE_SCAN_DIR
#error "KYTY_SOURCE_SCAN_DIR must be defined"
#endif

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "EnvSourceScanTests: failed: %s\n", text);
		std::exit(1); // not abort: ctest WILL_FAIL does not invert a crash
	}
}

} // namespace

int main() {
	namespace fs = std::filesystem;
	const std::regex     literal(R"re(getenv\(\s*"(KYTY_[A-Z0-9_]+)")re");
	std::set<std::string> used;
	size_t               files = 0;
	for (const auto& entry: fs::recursive_directory_iterator(KYTY_SOURCE_SCAN_DIR)) {
		if (!entry.is_regular_file()) continue;
		const auto ext = entry.path().extension().string();
		if (ext != ".cpp" && ext != ".h" && ext != ".hpp" && ext != ".mm" && ext != ".cc") continue;
		++files;
		std::ifstream      in(entry.path());
		std::ostringstream text;
		text << in.rdbuf();
		const auto body = text.str();
		for (std::sregex_iterator it(body.begin(), body.end(), literal), end; it != end; ++it) {
			used.insert((*it)[1].str());
		}
	}
	Check(files > 100, "source scan found too few files (wrong KYTY_SOURCE_SCAN_DIR?)");
	Check(used.size() > 30, "source scan found too few getenv(\"KYTY_...\") names");
	int missing = 0;
	for (const auto& name: used) {
		const bool known = std::ranges::find(Common::kKnownKytyEnv, std::string_view(name)) != std::end(Common::kKnownKytyEnv);
		if (!known) {
			std::fprintf(stderr, "EnvSourceScanTests: %s is read by getenv() but missing from knownEnv.h\n", name.c_str());
			++missing;
		}
	}
	Check(missing == 0, "a getenv(\"KYTY_...\") name is missing from knownEnv.h");
	return 0;
}
