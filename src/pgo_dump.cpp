// Only compiled with -DKYTY_EMULATOR_PGO=gen (instrumented build for profile collection).
// Benchmark runs are ended with a hard kill, so the profile runtime never gets its atexit write.
// A detached thread therefore dumps the counters every few seconds, alternating between two files
// ("<KYTY_PGO_DUMP>-a.profraw" / "-b.profraw"), so one of them is always complete.
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

extern "C" int  __llvm_profile_write_file(void);
extern "C" void __llvm_profile_set_filename(const char* name);

namespace {
struct PgoDumper
{
	PgoDumper()
	{
		const char* prefix = std::getenv("KYTY_PGO_DUMP");
		if (prefix == nullptr || *prefix == 0)
		{
			return;
		}
		std::thread(
		    [p = std::string(prefix)]
		    {
			    bool b = false;
			    for (;;)
			    {
				    std::this_thread::sleep_for(std::chrono::seconds(10));
				    std::string f = p + (b ? "-b.profraw" : "-a.profraw");
				    __llvm_profile_set_filename(f.c_str());
				    __llvm_profile_write_file();
				    b = !b;
			    }
		    })
		    .detach();
	}
} g_pgo_dumper;
} // namespace
