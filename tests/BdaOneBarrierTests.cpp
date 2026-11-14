// S603 (KYTY_BDA_ONE_BARRIER, port of KytyPS5 upstream b510053e8 by nmzik): the copy grouping of bdaOneBarrier.h, replayed on host memory
// (guest memory -> staging -> buffers) the way BufferCache::SynchronizeBda records it.
// Mutants (WILL_FAIL): KYTY_BDAONEBARRIER_MUTANT_NO_SPLIT, KYTY_BDAONEBARRIER_MUTANT_FIRST_FROM_ZERO, KYTY_BDAONEBARRIER_MUTANT_NO_BASE_OFFSET.

#include "graphics/host_gpu/renderer/cache/bdaOneBarrier.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <random>
#include <span>
#include <vector>

namespace {

namespace BdaOneBarrier = Libs::Graphics::BdaOneBarrier;

constexpr uint64_t kPage       = 0x1000;
constexpr uint64_t kGuestBase  = 0x100000;
constexpr uint64_t kGuestPages = 16;
constexpr uint8_t  kSentinel   = 0xee;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "BdaOneBarrierTests: failed: %s\n", message);
		std::_Exit(1);
	}
}

// vk::BufferCopy's shape: (srcOffset, dstOffset, size).
struct Copy {
	uint64_t srcOffset;
	uint64_t dstOffset;
	uint64_t size;
	Copy(uint64_t src, uint64_t dst, uint64_t bytes): srcOffset(src), dstOffset(dst), size(bytes) {}
};

struct FakeBuffer {
	uint64_t             cpu;
	std::vector<uint8_t> storage;
	FakeBuffer(uint64_t address, uint64_t size): cpu(address), storage(size, kSentinel) {}
	[[nodiscard]] uint64_t CpuAddress() const { return cpu; }
	[[nodiscard]] uint64_t Offset(uint64_t address) const { return address - cpu; }
};

struct Piece {
	FakeBuffer* buffer;
	uint64_t    address;
	uint64_t    bytes;
};

struct World {
	std::vector<uint8_t> guest = std::vector<uint8_t>(kGuestPages * kPage);
	FakeBuffer           a {kGuestBase, 4 * kPage};
	FakeBuffer           b {kGuestBase + 4 * kPage, 2 * kPage};
	FakeBuffer           c {kGuestBase + 8 * kPage, 4 * kPage};
	World() {
		for (size_t i = 0; i < guest.size(); i++) {
			guest[i] = static_cast<uint8_t>((i * 131u + 7u) & 0xffu);
		}
	}
	uint8_t* Guest(uint64_t address) { return guest.data() + (address - kGuestBase); }
};

// SynchronizeBda: Append per range, Fill inside UploadMemory, then one copy command per upload. Returns the part sizes the copy commands saw.
std::vector<size_t> Run(World& world, const std::vector<Piece>& pieces, uint64_t base_offset,
                        std::vector<BdaOneBarrier::Upload<FakeBuffer>>* uploads_out = nullptr) {
	std::vector<Copy>                              copies;
	std::vector<BdaOneBarrier::Upload<FakeBuffer>> uploads;
	uint64_t                                       total = 0;
	for (const auto& piece: pieces) {
		BdaOneBarrier::Append(uploads, copies, total, piece.buffer, piece.buffer->Offset(piece.address), piece.bytes);
	}
	std::vector<uint8_t> staging(base_offset + total, 0);
	BdaOneBarrier::Fill<FakeBuffer, Copy>(uploads, copies, staging.data() + base_offset, base_offset,
	                                      [&](uint64_t address, uint8_t* destination, uint64_t bytes) {
		                                      // Checked before the write: a mutant that replays copies must fail with an exit code, not corrupt the heap.
		                                      Check(destination >= staging.data() + base_offset &&
		                                                destination + bytes <= staging.data() + staging.size(),
		                                            "the staging fill writes outside its reservation");
		                                      Check(address >= kGuestBase && address + bytes <= kGuestBase + world.guest.size(),
		                                            "the staging fill reads outside guest memory");
		                                      std::memcpy(destination, world.Guest(address), bytes);
	                                      });
	std::vector<size_t> parts;
	BdaOneBarrier::ForEachUpload<FakeBuffer, Copy>(uploads, copies, [&](FakeBuffer& buffer, std::span<Copy> part) {
		parts.push_back(part.size());
		for (const auto& copy: part) {
			Check(copy.srcOffset + copy.size <= staging.size(), "a copy reads past the staging allocation");
			Check(copy.dstOffset + copy.size <= buffer.storage.size(), "a copy writes past its buffer");
			std::memcpy(buffer.storage.data() + copy.dstOffset, staging.data() + copy.srcOffset, copy.size);
		}
	});
	if (uploads_out != nullptr) {
		*uploads_out = uploads;
	}
	return parts;
}

// Every byte a piece covers equals the guest byte, every other byte of the buffer is the sentinel.
void CheckContents(World& world, const std::vector<Piece>& pieces, FakeBuffer& buffer, const char* message) {
	std::vector<bool> covered(buffer.storage.size(), false);
	for (const auto& piece: pieces) {
		if (piece.buffer != &buffer) {
			continue;
		}
		for (uint64_t i = 0; i < piece.bytes; i++) {
			covered[piece.address - buffer.cpu + i] = true;
		}
	}
	for (size_t i = 0; i < buffer.storage.size(); i++) {
		const auto want = covered[i] ? *world.Guest(buffer.cpu + i) : kSentinel;
		Check(buffer.storage[i] == want, message);
	}
}

void TestParse() {
	Check(BdaOneBarrier::Parse("1"), "\"1\" must enable");
	Check(!BdaOneBarrier::Parse(nullptr), "unset must disable");
	Check(!BdaOneBarrier::Parse(""), "empty must disable");
	Check(!BdaOneBarrier::Parse("0"), "\"0\" must disable");
	Check(!BdaOneBarrier::Parse("11"), "\"11\" must disable");
	Check(!BdaOneBarrier::Parse("1 "), "\"1 \" must disable");
}

// Two ranges of A, then B, C and A again (a later queued range): A's first three copies are one copy command (upstream's
// "two ranges of one buffer did not upload in one copy"), every buffer change starts a new one.
void TestGrouping() {
	World                                          world;
	std::vector<BdaOneBarrier::Upload<FakeBuffer>> uploads;
	const std::vector<Piece>                       pieces = {
        {&world.a, kGuestBase + 0x10, 0x100},
        {&world.a, kGuestBase + kPage, 0x800},
        {&world.a, kGuestBase + 3 * kPage + 0x40, 0x40},
        {&world.b, kGuestBase + 4 * kPage + 0x200, kPage},
        {&world.c, kGuestBase + 9 * kPage, 2 * kPage},
        {&world.a, kGuestBase + 2 * kPage, 0x10},
    };
	const auto parts = Run(world, pieces, 0x300, &uploads);
	Check(uploads.size() == 4, "A, A, A, B, C, A must give 4 copy commands");
	Check(parts == std::vector<size_t>({3, 1, 1, 1}), "copy commands must own 3, 1, 1, 1 copies");
	Check(uploads[0].buffer == &world.a && uploads[1].buffer == &world.b && uploads[2].buffer == &world.c && uploads[3].buffer == &world.a,
	      "copy commands must go to A, B, C, A");
	CheckContents(world, pieces, world.a, "buffer A has a wrong byte");
	CheckContents(world, pieces, world.b, "buffer B has a wrong byte");
	CheckContents(world, pieces, world.c, "buffer C has a wrong byte");
}

// One buffer, several copies: one copy command (the common case).
void TestSingleBuffer() {
	World                    world;
	const std::vector<Piece> pieces = {{&world.b, kGuestBase + 4 * kPage, 0x80}, {&world.b, kGuestBase + 5 * kPage + 0x100, 0x100}};
	const auto               parts  = Run(world, pieces, 0x40);
	Check(parts == std::vector<size_t>({2}), "two copies of one buffer must be one copy command");
	CheckContents(world, pieces, world.b, "single buffer has a wrong byte");
}

// Random pieces over the three buffers (non-overlapping within a buffer, as the tracker hands them), random staging base.
void TestRandom() {
	std::mt19937_64 rng(603);
	for (int round = 0; round < 200; round++) {
		World                    world;
		FakeBuffer*              buffers[] = {&world.a, &world.b, &world.c};
		std::vector<Piece>       pieces;
		std::vector<std::vector<bool>> used = {std::vector<bool>(world.a.storage.size()), std::vector<bool>(world.b.storage.size()),
		                                       std::vector<bool>(world.c.storage.size())};
		const int n = 1 + static_cast<int>(rng() % 12);
		for (int i = 0; i < n; i++) {
			const auto  k      = static_cast<size_t>(rng() % 3);
			FakeBuffer* buffer = buffers[k];
			const auto  size   = buffer->storage.size();
			const auto  offset = (rng() % size) & ~uint64_t {3};
			auto        bytes  = 4 + (rng() % 0x400);
			bytes              = std::min<uint64_t>(bytes, size - offset);
			bool free          = true;
			for (uint64_t j = 0; j < bytes; j++) {
				free = free && !used[k][offset + j];
			}
			if (!free || bytes == 0) {
				continue;
			}
			for (uint64_t j = 0; j < bytes; j++) {
				used[k][offset + j] = true;
			}
			pieces.push_back({buffer, buffer->cpu + offset, bytes});
		}
		if (pieces.empty()) {
			continue;
		}
		size_t changes = 1;
		for (size_t i = 1; i < pieces.size(); i++) {
			changes += pieces[i].buffer != pieces[i - 1].buffer ? 1 : 0;
		}
		const auto base  = 4 * (1 + rng() % 0x100);
		const auto parts = Run(world, pieces, base);
		Check(parts.size() == changes, "one copy command per run of copies to the same buffer");
		size_t sum = 0;
		for (auto p: parts) {
			sum += p;
		}
		Check(sum == pieces.size(), "copy commands must own every copy exactly once");
		CheckContents(world, pieces, world.a, "random: buffer A has a wrong byte");
		CheckContents(world, pieces, world.b, "random: buffer B has a wrong byte");
		CheckContents(world, pieces, world.c, "random: buffer C has a wrong byte");
	}
}

} // namespace

int main() {
	TestParse();
	TestGrouping();
	TestSingleBuffer();
	TestRandom();
	std::printf("BdaOneBarrierTests: ok\n");
	return 0;
}
