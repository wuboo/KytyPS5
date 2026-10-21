#ifndef KYTY_GRAPHICS_PRESENTATION_THREADCPUWIN_H_
#define KYTY_GRAPHICS_PRESENTATION_THREADCPUWIN_H_

// S467 (Q3): CPU time of every thread of this process on Windows, with the thread name (SetThreadDescription / GetThreadDescription).
// Cycles from QueryThreadCycleTime (exact, unlike GetThreadTimes, which is quantum-sampled) converted to ns with a TSC calibration.

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <tlhelp32.h>
#include <intrin.h>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "graphics/presentation/cpFrameStats.h"

namespace Libs::Graphics::ThreadCpuWin {

inline std::string Utf8(const wchar_t* w) {
	if (w == nullptr || w[0] == 0) return {};
	const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
	if (n <= 1) return {};
	std::string s(static_cast<size_t>(n - 1), '\0');
	WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
	return s;
}

// ns per cycle counted by QueryThreadCycleTime (reference/TSC cycles), calibrated once against QueryPerformanceCounter.
inline double NsPerCycle() {
	static const double v = [] {
		LARGE_INTEGER f, a, b;
		QueryPerformanceFrequency(&f);
		QueryPerformanceCounter(&a);
		const unsigned long long c0 = __rdtsc();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		QueryPerformanceCounter(&b);
		const unsigned long long c1 = __rdtsc();
		const double ns = static_cast<double>(b.QuadPart - a.QuadPart) * 1e9 / static_cast<double>(f.QuadPart);
		return c1 > c0 ? ns / static_cast<double>(c1 - c0) : 0.0;
	}();
	return v;
}

inline std::vector<CpFrameStats::ThreadSample> SnapshotThreads() {
	using GetDesc = HRESULT(WINAPI*)(HANDLE, PWSTR*);
	static const GetDesc get_desc = reinterpret_cast<GetDesc>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription")));
	std::vector<CpFrameStats::ThreadSample> out;
	const double                            ns_per_cycle = NsPerCycle();
	const HANDLE                            snap         = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snap == INVALID_HANDLE_VALUE) return out;
	const DWORD pid = GetCurrentProcessId();
	THREADENTRY32 te {};
	te.dwSize = sizeof(te);
	for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
		if (te.th32OwnerProcessID != pid) continue;
		const HANDLE h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, te.th32ThreadID);
		if (h == nullptr) continue;
		ULONG64 cycles = 0;
		if (QueryThreadCycleTime(h, &cycles)) {
			CpFrameStats::ThreadSample s;
			s.id     = te.th32ThreadID;
			s.cpu_ns = static_cast<uint64_t>(static_cast<double>(cycles) * ns_per_cycle);
			if (get_desc != nullptr) {
				PWSTR d = nullptr;
				if (SUCCEEDED(get_desc(h, &d)) && d != nullptr) {
					s.name = Utf8(d);
					LocalFree(d);
				}
			}
			if (s.name.empty()) s.name = "unnamed";
			out.push_back(std::move(s));
		}
		CloseHandle(h);
	}
	CloseHandle(snap);
	return out;
}

} // namespace Libs::Graphics::ThreadCpuWin

#endif

#endif /* KYTY_GRAPHICS_PRESENTATION_THREADCPUWIN_H_ */
