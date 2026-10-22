#ifndef GRAPHICS_SHADER_SCOUTTABLELOG_H
#define GRAPHICS_SHADER_SCOUTTABLELOG_H

// S471: the guest tables ShaderGetVertexInputInfo copies (attribute table, buffer table) are read outside the SRT walk. While the CP scout worker computes a
// draw's programs, every such read is logged here (address, size, site, bytes) so the CP can issue the same reads before it uses the result.

#include <cstdint>
#include <vector>

namespace Libs::Graphics::ScoutTableLog {

struct Read {
	uint64_t address = 0;
	uint32_t dwords  = 0;
	uint32_t site    = 0; // CpReadBacking::Site
	uint32_t first   = 0; // into Log::words
};

struct Log {
	std::vector<Read>     reads;
	std::vector<uint32_t> words;
	bool                  unsafe = false; // a table was read in a way the log cannot replay
};

inline Log*& Active() {
	static thread_local Log* log = nullptr;
	return log;
}

inline void Note(uint64_t address, uint32_t dwords, uint32_t site, const uint32_t* data) {
	auto* log = Active();
	if (log == nullptr) return;
	log->reads.push_back({address, dwords, site, static_cast<uint32_t>(log->words.size())});
	log->words.insert(log->words.end(), data, data + dwords);
}

inline void NoteUnsafe() {
	if (auto* log = Active()) log->unsafe = true;
}

} // namespace Libs::Graphics::ScoutTableLog

#endif
