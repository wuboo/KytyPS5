#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_GFDIAG_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_GFDIAG_H_

// S421 (DIAG, default off): does a read fault of a GUEST thread (not the command processor) hit a page the command processor had just
// read from the backing (CP_READ_BACKING tables, PV_IDX_BACKING indices)? Hypothesis (verify-421): such a page stays protected, the dirty
// bytes are fetched later by the guest's own fault, and the wait moves from the CP to the guest.
//   KYTY_DIAG_GUEST_FAULT=1   one [gf] line per read fault on a non-CP thread (<= 400 lines, plus every line of KYTY_DIAG_GF_WINDOW=a:b, flips),
//                             frames.csv columns gf_n, gf_wait_ns (time of ReadMemory), gf_after_backing_n (the page was read from the backing
//                             by the CP in the same guest flip).
// This header is the pure part (tests/GfDiagTests.cpp pins it). Nothing here feeds a shader or pipeline cache key.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace Libs::Graphics::GuestFault {

inline bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_GUEST_FAULT");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return on;
}

// "a:b" -> [a, b]; false (and 0:0) when absent or malformed.
inline bool ParseWindow(const char* text, uint64_t* from, uint64_t* to) {
	*from = 0;
	*to   = 0;
	if (text == nullptr || text[0] == '\0') {
		return false;
	}
	char*      end = nullptr;
	const auto a   = std::strtoull(text, &end, 10);
	if (end == text || *end != ':') {
		return false;
	}
	const char* rest = end + 1;
	const auto  b    = std::strtoull(rest, &end, 10);
	if (end == rest || *end != '\0' || b < a) {
		return false;
	}
	*from = a;
	*to   = b;
	return true;
}

inline bool InWindow(uint64_t flip) {
	static const auto window = [] {
		struct W {
			uint64_t a = 0, b = 0;
			bool     on = false;
		} w;
		w.on = ParseWindow(std::getenv("KYTY_DIAG_GF_WINDOW"), &w.a, &w.b);
		return w;
	}();
	return window.on && flip >= window.a && flip <= window.b;
}

// At most 400 lines, plus every line inside the window.
inline bool ShouldLog(uint64_t index, bool in_window) {
	return index < 400 || in_window;
}

// Where the command processor read the bytes from the backing.
enum class Site : uint8_t { None, VertexAttribTable, VertexBufferTable, IndexBytes, Count };

constexpr const char* SiteName(Site site) {
	switch (site) {
		case Site::VertexAttribTable: return "vertex_attrib_table";
		case Site::VertexBufferTable: return "vertex_buffer_table";
		case Site::IndexBytes: return "index_bytes";
		default: return "-";
	}
}

constexpr uint64_t kPageSize = 4096;

// The last kReads backing reads of the command processor (range, guest flip, sequence number). Written on the CP thread, searched on the
// faulting guest thread, hence the lock.
class BackingReads {
public:
	static constexpr size_t kReads = 256;

	struct Hit {
		bool     found = false;
		uint64_t flip  = 0;
		uint64_t seq   = 0;
		Site     site  = Site::None;
	};

	void Note(uint64_t address, uint64_t size, Site site, uint64_t flip) {
		if (size == 0) {
			return;
		}
		std::lock_guard lock(m_lock);
		m_reads[m_next % kReads] = {address, size, flip, ++m_seq, site, true};
		m_next++;
	}

	// The newest read that overlaps the 4 KiB page of `address`.
	[[nodiscard]] Hit Find(uint64_t address) const {
		const uint64_t begin = address & ~(kPageSize - 1);
		const uint64_t end   = begin + kPageSize;
		std::lock_guard lock(m_lock);
		Hit             best;
		for (const auto& read: m_reads) {
			if (!read.used || read.address >= end || read.address + read.size <= begin) {
				continue;
			}
			if (!best.found || read.seq > best.seq) {
				best = {true, read.flip, read.seq, read.site};
			}
		}
		return best;
	}

	void Clear() {
		std::lock_guard lock(m_lock);
		m_reads = {};
		m_next  = 0;
	}

private:
	struct Read {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint64_t flip    = 0;
		uint64_t seq     = 0;
		Site     site    = Site::None;
		bool     used    = false;
	};
	mutable std::mutex       m_lock;
	std::array<Read, kReads> m_reads {};
	size_t                   m_next = 0;
	uint64_t                 m_seq  = 0;
};

inline BackingReads& Reads() {
	static BackingReads reads;
	return reads;
}

// True when the page was read from the backing by the CP in the same guest flip as the fault.
inline bool AfterBacking(const BackingReads::Hit& hit, uint64_t fault_flip) {
#if defined(KYTY_GFDIAG_MUTANT_NO_AFTER)
	(void)hit;
	(void)fault_flip;
	return false;
#else
	return hit.found && hit.flip == fault_flip;
#endif
}

struct Event {
	uint64_t    flip = 0;
	uint64_t    addr = 0;
	uint64_t    wait_us = 0;
	bool        eager = false;
	bool        dirty = false; // the bytes at the fault address were GPU-dirty before the drain
	bool        page_dirty = false; // the 4 KiB page was GPU-modified before the drain
	bool        after_backing = false;
	Site        backing_site = Site::None;
	std::string bt_text = "-";
};

inline std::string FormatLine(const Event& event) {
	char text[2048];
	std::snprintf(text, sizeof(text),
	              "[gf] flip=%llu addr=0x%016llx page=0x%016llx wait_us=%llu eager=%d dirty=%d page_dirty=%d after_backing=%d backing_site=%s bt=%s",
	              static_cast<unsigned long long>(event.flip), static_cast<unsigned long long>(event.addr),
	              static_cast<unsigned long long>(event.addr & ~(kPageSize - 1)), static_cast<unsigned long long>(event.wait_us),
	              event.eager ? 1 : 0, event.dirty ? 1 : 0, event.page_dirty ? 1 : 0, event.after_backing ? 1 : 0, SiteName(event.backing_site),
	              event.bt_text.c_str());
	return text;
}

} // namespace Libs::Graphics::GuestFault

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_GFDIAG_H_ */
