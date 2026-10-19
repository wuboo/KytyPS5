#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMACHUNKS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMACHUNKS_H_

#include <cstdint>

namespace Libs::Graphics::UmaChunks {

// Lazy chunked import (UMA): Apple GPUs cannot page on demand, so whatever is imported is wired in
// full. The arena is cut into fixed chunks and a chunk is imported when the GPU first uses it.
// These are the decisions that need no GPU, kept apart so CPU tests can pin them.

// The import alignment is enforced here, never taken from the driver (it reports 4 KB under
// Rosetta, and MoltenVK accepts an unaligned import without complaint).
inline constexpr uint64_t MinImportAlignment = 16 * 1024;
inline constexpr uint64_t DefaultChunkMB     = 256;
inline constexpr uint64_t MinChunkMB         = 16;
inline constexpr uint64_t MaxChunkMB         = 4096;

[[nodiscard]] constexpr uint64_t ImportAlignment(uint64_t reported) {
#ifdef KYTY_UMA_MUTANT_TRUST_REPORTED_ALIGNMENT
	return reported != 0 ? reported : 1; // wrong: believes the 4 KB the driver reports under Rosetta
#else
	return reported > MinImportAlignment ? reported : MinImportAlignment;
#endif
}

// KYTY_UMA_CHUNK_MB (0 or unparsable: the default); clamped and rounded down to a power of two so
// a chunk boundary is always a page boundary.
[[nodiscard]] constexpr uint64_t ChunkBytes(uint64_t requested_mb) {
	uint64_t mb = requested_mb == 0 ? DefaultChunkMB : requested_mb;
	mb          = mb < MinChunkMB ? MinChunkMB : (mb > MaxChunkMB ? MaxChunkMB : mb);
	uint64_t pow2 = 1;
	while (pow2 * 2 <= mb) {
		pow2 *= 2;
	}
	return pow2 * 1024 * 1024;
}

[[nodiscard]] constexpr bool IsImportable(uint64_t host_address, uint64_t size, uint64_t reported) {
	const auto alignment = ImportAlignment(reported);
	return size != 0 && host_address % alignment == 0 && size % alignment == 0;
}

// Overlap between neighbouring chunks (KYTY_UMA_OVERLAP_MB). Chunk i covers
// [i * stride, i * stride + chunk) with stride = chunk - overlap, so a window of at most `overlap`
// bytes that starts in the cell [i * stride, (i + 1) * stride) always fits chunk i and the arena
// never needs a window over a boundary (chunk_straddles = 0). The overlap is imported twice as an
// alias of the same host memory. `set` = the variable exists (0 then switches the overlap off);
// unset: the default, min(64 MB, chunk / 4). Power of two, at most chunk / 2, at least the import
// alignment so the stride stays a multiple of it.
inline constexpr uint64_t DefaultOverlapMB = 64;

[[nodiscard]] constexpr uint64_t OverlapBytes(uint64_t requested_mb, bool set, uint64_t chunk_bytes) {
#ifdef KYTY_UMA_MUTANT_NO_OVERLAP
	(void)requested_mb;
	(void)set;
	(void)chunk_bytes;
	return 0; // wrong: neighbouring chunks do not overlap
#else
	const uint64_t quarter = chunk_bytes / 4;
	uint64_t       bytes   = set ? requested_mb * 1024 * 1024 : DefaultOverlapMB * 1024 * 1024;
	if (!set && bytes > quarter) {
		bytes = quarter;
	}
	if (bytes == 0) {
		return 0;
	}
	if (bytes > chunk_bytes / 2) {
		bytes = chunk_bytes / 2;
	}
	uint64_t pow2 = MinImportAlignment;
	while (pow2 * 2 <= bytes) {
		pow2 *= 2;
	}
	return bytes < MinImportAlignment ? 0 : pow2;
#endif
}

// Where a byte range of the arena lies.
struct Place {
	bool     ok    = false; // inside exactly one chunk
	uint32_t index = 0;
	uint64_t begin = 0; // arena offset of the chunk start
	uint64_t end   = 0; // arena offset past the chunk end
};

class Layout {
public:
	constexpr Layout() = default;
	constexpr Layout(uint64_t arena_size, uint64_t chunk_bytes, uint64_t overlap_bytes = 0)
	    : m_arena(arena_size), m_chunk(chunk_bytes),
	      m_stride(overlap_bytes < chunk_bytes ? chunk_bytes - overlap_bytes : chunk_bytes) {}

	[[nodiscard]] constexpr uint64_t ArenaSize() const { return m_arena; }
	[[nodiscard]] constexpr uint64_t ChunkSize() const { return m_chunk; }
	[[nodiscard]] constexpr uint64_t Stride() const { return m_stride; }
	[[nodiscard]] constexpr uint64_t Overlap() const { return m_chunk - m_stride; }
	// Chunks needed so that every cell has its chunk; the last one reaches the arena end.
	[[nodiscard]] constexpr uint32_t Count() const {
		if (m_chunk == 0 || m_arena == 0) {
			return 0;
		}
		if (m_arena <= m_chunk) {
			return 1;
		}
		return static_cast<uint32_t>((m_arena - m_chunk + m_stride - 1) / m_stride + 1);
	}
	[[nodiscard]] constexpr uint64_t Begin(uint32_t index) const { return uint64_t {index} * m_stride; }
	// The last chunk may be shorter than the others.
	[[nodiscard]] constexpr uint64_t End(uint32_t index) const {
		const auto end = uint64_t {index} * m_stride + m_chunk;
		return end < m_arena ? end : m_arena;
	}
	// The chunk whose cell holds `offset` (the one a window starting there is cut from).
	[[nodiscard]] constexpr uint32_t Cell(uint64_t offset) const {
		const auto cell = static_cast<uint32_t>(offset / m_stride);
		return cell < Count() ? cell : Count() - 1;
	}

	// The chunk that holds the whole of [offset, offset + size); ok = false when the range leaves
	// the arena or crosses the end of its chunk (a window cannot span two imports: such a range
	// takes the copy path). With an overlap, any range of at most Overlap() bytes is ok.
	[[nodiscard]] constexpr Place Find(uint64_t offset, uint64_t size) const {
		Place place;
		if (m_chunk == 0 || size == 0 || offset + size < offset || offset + size > m_arena) {
			return place;
		}
		const auto index = Cell(offset);
#ifdef KYTY_UMA_MUTANT_STRADDLE_OK
		place.index = index; // wrong: the tail is in the next chunk
		place.begin = Begin(place.index);
		place.end   = End(place.index);
		place.ok    = true;
		return place;
#else
		place.index = index;
		place.begin = Begin(index);
		place.end   = End(index);
		place.ok    = offset + size <= place.end;
		return place;
#endif
	}

	// The chunk before the cell's own one when it also holds the range (the overlap zone): an
	// alias that may already be imported while the cell's chunk is not.
	[[nodiscard]] constexpr Place FindPrevious(uint64_t offset, uint64_t size) const {
		Place place;
		const auto cell = Find(offset, size);
		if (cell.index == 0 || m_stride == m_chunk) {
			return place;
		}
		place.index = cell.index - 1;
		place.begin = Begin(place.index);
		place.end   = End(place.index);
		place.ok    = offset >= place.begin && offset + size <= place.end && offset + size > offset;
		return place;
	}

	// First cell whose window can reach [offset, offset + size): the cells to import when a range
	// becomes GPU-visible. Cells are [Cell(offset), Cell(offset + size - 1)].
	[[nodiscard]] constexpr uint32_t LastCell(uint64_t offset, uint64_t size) const {
		return Cell(offset + (size == 0 ? 0 : size - 1));
	}

private:
	uint64_t m_arena  = 0;
	uint64_t m_chunk  = 0;
	uint64_t m_stride = 0;
};

enum class State : uint8_t { NotImported = 0, Imported = 1, Failed = 2 };

// What to do with a chunk that a request needs.
enum class Action { Use, Import, Legacy };

[[nodiscard]] constexpr Action Decide(State state) {
	switch (state) {
		case State::Imported: return Action::Use;
		case State::NotImported: return Action::Import;
		case State::Failed: break;
	}
#ifdef KYTY_UMA_MUTANT_RETRY_FAILED
	return Action::Import; // wrong: a chunk whose import failed is retried on every request
#else
	return Action::Legacy;
#endif
}

// Device address of an arena byte, 0 when no imported chunk holds the whole range. The cell's own
// chunk is preferred; in the overlap zone the previous chunk serves when only it is imported.
// chunk_gpu(index) returns the device address of the chunk start (0 = not imported).
template <typename ChunkGpu>
[[nodiscard]] uint64_t DeviceAddress(const Layout& layout, uint64_t offset, uint64_t size,
                                     ChunkGpu&& chunk_gpu) {
	const auto place = layout.Find(offset, size);
	if (!place.ok) {
		return 0;
	}
	if (const auto base = chunk_gpu(place.index); base != 0) {
		return base + (offset - place.begin);
	}
	const auto previous = layout.FindPrevious(offset, size);
	if (previous.ok) {
		if (const auto base = chunk_gpu(previous.index); base != 0) {
			return base + (offset - previous.begin);
		}
	}
	return 0;
}

} // namespace Libs::Graphics::UmaChunks

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMACHUNKS_H_
