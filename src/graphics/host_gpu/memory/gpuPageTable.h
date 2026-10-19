#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_GPUPAGETABLE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_GPUPAGETABLE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace Libs::Graphics {

// Two-level page table: guest 4 KB page -> device address of that page, for the guest memory the
// GPU can reach in place (UMA). A directory of leaf pointers; a leaf exists only while one of its
// pages is mapped, so the table costs memory in proportion to the mapped regions (the flat
// 8-bytes-per-16 KB table the shaders read is 768 MB). An unmapped page reads as the sentinel.
// Holds the state already published to the GPU: the owner calls Clear only once the submits that
// could still use the old entry have retired. Not thread safe: the owner serializes access.
template <uint32_t AddressBits = 44, uint32_t LeafBits = 14>
class GpuPageTable {
public:
	static constexpr uint32_t PageBits    = 12;
	static constexpr uint64_t PageSize    = uint64_t {1} << PageBits;
	static constexpr uint64_t LeafEntries = uint64_t {1} << LeafBits;
	static constexpr uint64_t LeafSpan    = LeafEntries << PageBits;
	static constexpr uint64_t DirEntries  = uint64_t {1} << (AddressBits - PageBits - LeafBits);

	explicit GpuPageTable(uint64_t sentinel): m_sentinel(sentinel), m_dir(DirEntries) {}

	[[nodiscard]] uint64_t Sentinel() const { return m_sentinel; }

	// Maps the 4 KB page at page_vaddr to device_address (must be non-zero).
	void Set(uint64_t page_vaddr, uint64_t device_address) {
		if (page_vaddr >> AddressBits != 0 || device_address == 0) {
			return;
		}
		auto& leaf = m_dir[page_vaddr / LeafSpan];
		if (!leaf) {
			leaf = std::make_unique<Leaf>();
		}
		auto& entry = leaf->entries[(page_vaddr / PageSize) % LeafEntries];
		if (entry == 0) {
			++leaf->used;
			++m_mapped;
		}
		entry = device_address;
	}

	// Maps [vaddr, vaddr + size) (4 KB aligned) onto consecutive device pages from device_base.
	void SetRange(uint64_t vaddr, uint64_t size, uint64_t device_base) {
		for (uint64_t offset = 0; offset < size; offset += PageSize) {
			Set(vaddr + offset, device_base + offset);
		}
	}

	void Clear(uint64_t page_vaddr) {
		if (page_vaddr >> AddressBits != 0) {
			return;
		}
		auto& leaf = m_dir[page_vaddr / LeafSpan];
		if (!leaf) {
			return;
		}
		auto& entry = leaf->entries[(page_vaddr / PageSize) % LeafEntries];
		if (entry != 0) {
			entry = 0;
			--m_mapped;
			if (--leaf->used == 0) {
				leaf.reset();
			}
		}
	}

	void ClearRange(uint64_t vaddr, uint64_t size) {
		for (uint64_t offset = 0; offset < size; offset += PageSize) {
			Clear(vaddr + offset);
		}
	}

	// Device address of the page, or the sentinel when it is not mapped.
	[[nodiscard]] uint64_t Lookup(uint64_t page_vaddr) const {
		const auto entry = Entry(page_vaddr);
		return entry != 0 ? entry : m_sentinel;
	}

	[[nodiscard]] bool IsMapped(uint64_t page_vaddr) const { return Entry(page_vaddr) != 0; }

	// Entry for the 16 KB page the flat shader table holds: the device address of its first byte
	// when all four 4 KB pages are mapped and consecutive on the device. A page with mixed
	// mappings cannot be expressed by one entry ("split"): returns false.
	[[nodiscard]] bool LookupWide(uint64_t wide_vaddr, uint32_t wide_bits, uint64_t* device_address) const {
		const auto wide_size = uint64_t {1} << wide_bits;
		const auto base      = wide_vaddr & ~(wide_size - 1);
		const auto first     = Entry(base);
		if (first == 0) {
			return false;
		}
#ifndef KYTY_UMA_MUTANT_SPLIT_AS_WHOLE
		for (uint64_t offset = PageSize; offset < wide_size; offset += PageSize) {
			if (Entry(base + offset) != first + offset) {
				return false;
			}
		}
#endif
		if (device_address != nullptr) {
			*device_address = first;
		}
		return true;
	}

	// Calls fn(page_vaddr, device_address) for every mapped page, in address order.
	template <typename Fn>
	void ForEachMapped(Fn&& fn) const {
		for (uint64_t d = 0; d < DirEntries; ++d) {
			const auto& leaf = m_dir[d];
			if (!leaf) {
				continue;
			}
			for (uint64_t i = 0; i < LeafEntries; ++i) {
				if (leaf->entries[i] != 0) {
					fn(d * LeafSpan + i * PageSize, leaf->entries[i]);
				}
			}
		}
	}

	[[nodiscard]] size_t MappedPages() const { return m_mapped; }
	[[nodiscard]] size_t LeafCount() const {
		size_t count = 0;
		for (const auto& leaf: m_dir) {
			count += leaf ? 1 : 0;
		}
		return count;
	}
	// Bytes the table itself occupies (directory plus live leaves).
	[[nodiscard]] size_t MemoryBytes() const {
		return DirEntries * sizeof(std::unique_ptr<Leaf>) + LeafCount() * sizeof(Leaf);
	}

private:
	struct Leaf {
		std::vector<uint64_t> entries = std::vector<uint64_t>(LeafEntries, 0);
		uint64_t              used    = 0;
	};

	[[nodiscard]] uint64_t Entry(uint64_t page_vaddr) const {
		if (page_vaddr >> AddressBits != 0) {
			return 0;
		}
		const auto& leaf = m_dir[page_vaddr / LeafSpan];
		return leaf ? leaf->entries[(page_vaddr / PageSize) % LeafEntries] : 0;
	}

	uint64_t                          m_sentinel;
	std::vector<std::unique_ptr<Leaf>> m_dir;
	size_t                            m_mapped = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_GPUPAGETABLE_H_
