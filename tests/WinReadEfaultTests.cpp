// S462: on Windows a file read into a page the GPU resource tracker has made PAGE_READONLY fails inside ReadFile
// (ERROR_NOACCESS) and delivers 0 bytes. With KYTY_FIX_WIN_READ_EFAULT=1 SysFileReadEx must report EFAULT so the
// caller's bounce-buffer retry runs; on a writable page the read must be unaffected.
#include "common/platform/sysFileIO.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "WinReadEfaultTests: failed: %s\n", message);
		std::exit(1);
	}
}

} // namespace

int main() {
#if defined(_WIN32)
#ifndef KYTY_S462_MUTANT_FIX_OFF
	_putenv_s("KYTY_FIX_WIN_READ_EFAULT", "1");
#endif
	const std::filesystem::path path = "s462_win_read_efault.tmp";
	{
		auto* w = SysFileCreate(path);
		Check(w != nullptr && !SysFileIsError(*w), "create temp file");
		uint8_t block[4096];
		for (size_t i = 0; i < sizeof(block); i++) {
			block[i] = static_cast<uint8_t>(i * 7u + 3u);
		}
		uint32_t written = 0;
		SysFileWrite(block, sizeof(block), *w, &written);
		Check(written == sizeof(block), "write temp file");
		SysFileClose(w);
	}
	auto* f = SysFileOpenR(path);
	Check(f != nullptr && !SysFileIsError(*f), "open temp file");

	auto* page = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
	Check(page != nullptr, "alloc page");
	DWORD old = 0;
	Check(VirtualProtect(page, 4096, PAGE_READONLY, &old) != 0, "protect read-only");

	uint32_t got = 123;
	int      err = 0;
	SysFileReadEx(page, 4096, *f, &got, &err);
	Check(got == 0, "read into a read-only page delivers 0 bytes");
	Check(err == EFAULT, "read into a read-only page is reported as EFAULT");

	Check(VirtualProtect(page, 4096, PAGE_READWRITE, &old) != 0, "unprotect");
	Check(SysFileSeek(*f, 0), "seek to start");
	got = 0;
	err = 1;
	SysFileReadEx(page, 4096, *f, &got, &err);
	Check(err == 0 && got == 4096, "read into a writable page succeeds without an error");
	Check(page[0] == 3 && page[1] == 10 && page[4095] == static_cast<uint8_t>(4095u * 7u + 3u), "read data matches");

	VirtualFree(page, 0, MEM_RELEASE);
	SysFileClose(f);
	SysFileDeleteFile(path);
#else
	std::printf("WinReadEfaultTests: Windows only, skipped\n");
#endif
	std::printf("WinReadEfaultTests: ok\n");
	return 0;
}
