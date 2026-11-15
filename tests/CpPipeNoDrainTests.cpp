// S604 (KYTY_PIPE_NODRAIN, cpPipeNoDrain.h): the precise view of the CP pipeline's pending writes. Every check below fails under one of the KYTY_PIPEND_MUTANT_*
// defines (CMake: cp_pipe_nodrain_mutant_*, WILL_FAIL); a failure ends the process with exit code 1, never with a crash (ctest on Windows does not count a crash
// as the expected failure).
//
// The oracle: random PM4 streams run against a byte model of guest memory whose writes follow the handlers (pm4Handlers.cpp / graphicsRun.cpp), written here
// independently of the decoders. Whenever a view says "nothing pending" the bytes now must be the bytes after every pending item has run; whenever it gives a
// value, that value must be what memory holds then. The S479 view is checked against a copy of the pre-S604 switch of cpPipe.cpp.

#include "graphics/guest_gpu/command_processor/cpPipeNoDrain.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <vector>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::CpPipe;
using Verdict = PendingWrites::Verdict;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "CpPipeNoDrainTests: failed: %s\n", text);
		std::exit(1);
	}
}

// --- packets -----------------------------------------------------------------------------------------------------------------------------------------------

using Packet = std::vector<uint32_t>;

uint32_t Header(uint32_t opcode, uint32_t packet_dw, uint32_t r = 0) {
	return KYTY_PM4(packet_dw, opcode, r);
}

Packet ReleaseMem(uint64_t address, uint32_t data_sel, uint64_t value) {
	return {Header(Pm4::IT_RELEASE_MEM, 8), 0, data_sel << 29u, static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32u), static_cast<uint32_t>(value),
	        static_cast<uint32_t>(value >> 32u), 0};
}

Packet EventWrite(uint32_t type, uint64_t address) {
	if (type == kEventOcclusionDump) return {Header(Pm4::IT_EVENT_WRITE, 4), type | (1u << 8u), static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32u)};
	return {Header(Pm4::IT_EVENT_WRITE, 2), type};
}

Packet WriteData(uint64_t address, bool one, const std::vector<uint32_t>& data) {
	const auto packet_dw = static_cast<uint32_t>(4u + data.size());
	Packet     p         = {Header(Pm4::IT_WRITE_DATA, packet_dw), (one ? 1u << 16u : 0u) | (5u << 8u), static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32u)};
	p.insert(p.end(), data.begin(), data.end());
	return p;
}

// dst_sel 0 = memory, 1 = GDS, 2 = nowhere (control bits 20-21).
Packet DmaData(uint64_t dst, uint32_t dst_sel, uint32_t bytes) {
	return {Header(Pm4::IT_DMA_DATA, 7), (dst_sel & 0x3u) << 20u, 0x1000, 0, static_cast<uint32_t>(dst), static_cast<uint32_t>(dst >> 32u), bytes & 0x03ffffffu};
}

// dst field 2 = memory (TC L2), 3 = GDS (the handler's dst_sel 6).
Packet CopyData(uint64_t dst, uint32_t dst_field, bool wide) {
	return {Header(Pm4::IT_COPY_DATA, 6), (dst_field << 8u) | (wide ? 1u << 16u : 0u) | 2u, 0x2000, 0, static_cast<uint32_t>(dst), static_cast<uint32_t>(dst >> 32u)};
}

Packet Flip() {
	return {Header(Pm4::IT_NOP, 6, Pm4::R_FLIP), 0, 0, 0, 0, 0};
}

void Note(PendingWrites& pending, const Packet& p, uint64_t seq, bool precise) {
	NoteItemWrites(pending, (p[0] >> 8u) & 0xffu, p.data(), static_cast<uint32_t>(p.size()), false, seq, precise);
}

// The switch of cpPipe.cpp before S604 (PipeFrontEnqueue), unchanged: the S479 view must answer exactly as it did.
void OldNote(PendingWrites& pending, const Packet& packet_v, uint64_t seq) {
	const uint32_t* packet      = packet_v.data();
	const auto      packet_dw   = static_cast<uint32_t>(packet_v.size());
	const uint32_t  opcode      = (packet[0] >> 8u) & 0xffu;
	const auto*     payload     = packet + 1;
	const auto      r           = opcode == Pm4::IT_NOP ? KYTY_PM4_R(packet[0]) : 0u;
	const bool      async       = false;
	uint64_t        address     = 0;
	switch (opcode) {
		case Pm4::IT_NOP:
			if (r == Pm4::R_FLIP || r == Pm4::R_ZERO) pending.Note(0, kEverything, false, 0, seq);
			break;
		case Pm4::IT_RELEASE_MEM:
			if (packet_dw >= 8u) {
				const auto w = DecodeReleaseMem(payload, async, &address);
				if (w.writes) pending.Note(address, w.size, w.known, w.value, seq);
			}
			break;
		case Pm4::IT_EVENT_WRITE:
			if (packet_dw > 3u) pending.Note(0, kEverything, false, 0, seq);
			break;
		case Pm4::IT_WRITE_DATA:
			if (packet_dw >= 5u) {
				const auto w = DecodeWriteData(payload, packet_dw);
				pending.Note(w.address, w.size, false, 0, seq);
			}
			break;
		case Pm4::IT_DMA_DATA:
			if (packet_dw >= 7u) {
				const auto w = DecodeDmaData(payload);
				pending.Note(w.address, w.size, false, 0, seq);
			}
			break;
		case Pm4::IT_COPY_DATA:
			if (packet_dw >= 6u) {
				const auto w = DecodeCopyData(payload);
				pending.Note(w.address, w.size, false, 0, seq);
			}
			break;
		default: break;
	}
}

// --- the memory model ----------------------------------------------------------------------------------------------------------------------------------------

constexpr uint64_t kBase = 0x7'0000'0000ull;
constexpr uint64_t kSize = 0x4000;

struct Rng {
	uint64_t s = 0x9e3779b97f4a7c15ull;
	uint64_t Next() {
		s ^= s << 13u;
		s ^= s >> 7u;
		s ^= s << 17u;
		return s;
	}
	uint32_t Below(uint32_t n) { return static_cast<uint32_t>(Next() % n); }
};

struct Memory {
	std::vector<uint8_t> bytes = std::vector<uint8_t>(kSize, 0);

	void Write(uint64_t address, const void* data, uint64_t size) {
		Check(address >= kBase && address + size <= kBase + kSize, "the model writes inside its memory");
		std::memcpy(bytes.data() + (address - kBase), data, size);
	}
	void Write64(uint64_t address, uint64_t v) { Write(address, &v, 8); }
	void Write32(uint64_t address, uint32_t v) { Write(address, &v, 4); }
	[[nodiscard]] uint64_t Read(uint64_t address, uint64_t size) const {
		uint64_t v = 0;
		std::memcpy(&v, bytes.data() + (address - kBase), size);
		return v;
	}
	[[nodiscard]] bool Same(const Memory& other, uint64_t address, uint64_t size) const {
		return std::memcmp(bytes.data() + (address - kBase), other.bytes.data() + (address - kBase), size) == 0;
	}
};

// What the handler of the packet writes (graphicsRun.cpp ReleaseMem / TriggerEvent / WriteData / DmaData, occlusionQueries WriteZPassCounters, a flip's labels).
struct Item {
	Packet   packet;
	uint64_t seq       = 0;
	uint64_t random    = 0; // timestamp / DMA bytes / flip label / dump counter
	bool     host_dump = false;
};

uint64_t FlipLabel(uint64_t random) {
	return kBase + ((random >> 8u) % (kSize / 8u)) * 8u;
}

void Apply(Memory& m, const Item& it) {
	const auto& p      = it.packet;
	const auto  opcode = (p[0] >> 8u) & 0xffu;
	switch (opcode) {
		case Pm4::IT_RELEASE_MEM: {
			const uint32_t data_sel = (p[2] >> 29u) & 0x7u;
			const uint64_t address  = (p[3] & ~0x3u) | (static_cast<uint64_t>(p[4]) << 32u);
			const uint64_t value    = p[5] | (static_cast<uint64_t>(p[6]) << 32u);
			if (data_sel == 1) m.Write32(address, static_cast<uint32_t>(value));
			if (data_sel == 2) m.Write64(address, value);
			if (data_sel == 3) m.Write64(address, it.random); // the timestamp
			break;
		}
		case Pm4::IT_EVENT_WRITE: {
			if ((p[1] & 0x3fu) != kEventOcclusionDump) break;
			const uint64_t address = p[2] | (static_cast<uint64_t>(p[3]) << 32u);
			for (uint32_t db = 0; db < 16; db++) {
				m.Write64(address + db * 16u, (1ull << 63u) | (it.random + db));
				if (it.host_dump && (address & 8u) == 0) m.Write64(address + db * 16u + 8u, 0);
			}
			break;
		}
		case Pm4::IT_WRITE_DATA: {
			const bool     one     = ((p[1] >> 16u) & 0x1u) != 0;
			const uint64_t address = p[2] | (static_cast<uint64_t>(p[3]) << 32u);
			const auto     count   = static_cast<uint32_t>(p.size() - 4u);
			for (uint32_t i = 0; i < count; i++) m.Write32(address + (one ? 0u : i * 4u), p[4 + i]);
			break;
		}
		case Pm4::IT_DMA_DATA: {
			const uint32_t dst_sel = (p[1] >> 20u) & 0x3u;
			const uint64_t dst     = p[4] | (static_cast<uint64_t>(p[5]) << 32u);
			const uint32_t bytes   = p[6] & 0x03ffffffu;
			if (dst_sel != 0 || bytes == 0 || static_cast<uint32_t>(dst) == 0x3022cu) break;
			for (uint32_t i = 0; i < bytes; i++) {
				const auto b = static_cast<uint8_t>(it.random >> ((i % 8u) * 8u)) ^ static_cast<uint8_t>(i);
				m.Write(dst + i, &b, 1);
			}
			break;
		}
		case Pm4::IT_COPY_DATA: {
			const uint32_t dst_field = ((p[1] >> 8u) & 0xfu) << 1u;
			const uint64_t dst       = p[4] | (static_cast<uint64_t>(p[5]) << 32u);
			if (dst_field == 6u) break; // GDS
			if (((p[1] >> 16u) & 0x1u) != 0) {
				m.Write64(dst, it.random);
			} else {
				m.Write32(dst, static_cast<uint32_t>(it.random));
			}
			break;
		}
		case Pm4::IT_NOP: m.Write64(FlipLabel(it.random), it.random); break;
		default: break;
	}
}

// --- tests -----------------------------------------------------------------------------------------------------------------------------------------------------

void ModeTests() {
	Check(ParseNoDrainMode(nullptr) == NoDrainMode::Off && ParseNoDrainMode("0") == NoDrainMode::Off && ParseNoDrainMode("1") == NoDrainMode::On &&
	          ParseNoDrainMode("shadow") == NoDrainMode::Shadow && ParseNoDrainMode("on") == NoDrainMode::Off,
	      "mode parse");
	Check(std::strcmp(NoDrainModeName(NoDrainMode::Shadow), "shadow") == 0 && std::strcmp(NoDrainModeName(NoDrainMode::On), "1") == 0, "mode names");
}

void DecoderTests() {
	const uint64_t a = kBase + 0x100;
	{ // the occlusion dump: the S479 view all memory, the precise one its 256 bytes
		PendingWrites pending;
		Note(pending, EventWrite(kEventOcclusionDump, a), 1, true);
		uint64_t v = 0;
		Check(pending.Resolve(kBase + 0x3000, 4, &v) == Verdict::Unknown, "S479: the dump overlaps everything");
		Check(pending.ResolvePrecise(kBase + 0x3000, 4, &v) == Verdict::NoPending, "precise: a label far from the dump is not pending");
		Check(pending.ResolvePrecise(a + 0x10, 8, &v) == Verdict::Unknown, "precise: the second DB of the dump is pending");
		Check(pending.ResolvePrecise(a + 0xf8, 8, &v) == Verdict::Unknown, "precise: the last 8 bytes of the dump are pending");
		Check(pending.ResolvePrecise(a + 0x100, 8, &v) == Verdict::NoPending, "precise: the bytes after the dump are not");
		Check(!pending.OverlapsPrecise(a - 0x40, 0x40) && pending.OverlapsPrecise(a - 0x40, 0x48), "precise: a table that ends at the dump does not overlap it, one into it does");
		Culprit c;
		Check(pending.OverlapsS479(kBase, 4, &c) && c.src == SrcTag(Pm4::IT_EVENT_WRITE, kEventOcclusionDump), "the culprit is the dump");
	}
	{ // other event types write nothing, a partial flush has 2 dwords anyway
		PendingWrites pending;
		Packet        p = EventWrite(kEventOcclusionDump, a);
		p[1]            = 0x38u | (1u << 8u); // a 4-dword event of another type (ignored by TriggerEvent)
		Note(pending, p, 1, true);
		uint64_t v = 0;
		Check(pending.Resolve(a, 4, &v) == Verdict::Unknown && pending.ResolvePrecise(a, 4, &v) == Verdict::NoPending, "a 4-dword event of another type: no precise write");
		Note(pending, EventWrite(0x07, 0), 2, true);
		Check(pending.Size() == 1, "a 2-dword event notes nothing");
	}
	{ // DMA: nowhere, GDS, no bytes, the ignored register: S479 only; memory: both
		PendingWrites pending;
		Note(pending, DmaData(a, 2, 0x40), 1, true);
		Note(pending, DmaData(a + 0x40, 1, 0x40), 2, true);
		Note(pending, DmaData((kBase & ~0xffffffffull) | 0x3022cu, 0, 4), 3, true);
		uint64_t v = 0;
		Check(pending.Resolve(a, 4, &v) == Verdict::Unknown && pending.Resolve(a + 0x40, 4, &v) == Verdict::Unknown, "S479: a prefetch and a GDS DMA are writes");
		Check(pending.ResolvePrecise(a, 4, &v) == Verdict::NoPending && pending.ResolvePrecise(a + 0x40, 4, &v) == Verdict::NoPending, "precise: no guest write");
		Check(pending.ResolvePrecise((kBase & ~0xffffffffull) | 0x3022cu, 4, &v) == Verdict::NoPending, "precise: 0x3022c is not written");
		Note(pending, DmaData(a + 0x200, 0, 0x10), 4, true);
		Check(pending.ResolvePrecise(a + 0x208, 4, &v) == Verdict::Unknown, "precise: a DMA to memory is a write");
	}
	{ // COPY_DATA to GDS
		PendingWrites pending;
		Note(pending, CopyData(a, 3, false), 1, true);
		uint64_t v = 0;
		Check(pending.Resolve(a, 4, &v) == Verdict::Unknown && pending.ResolvePrecise(a, 4, &v) == Verdict::NoPending, "COPY_DATA to GDS: S479 only");
		Note(pending, CopyData(a, 2, true), 2, true);
		Check(pending.ResolvePrecise(a + 4, 4, &v) == Verdict::Unknown, "COPY_DATA 64-bit to memory: both");
	}
	{ // WRITE_DATA immediates
		PendingWrites pending;
		uint64_t      v = 0;
		Note(pending, WriteData(a, false, {0x11111111u}), 1, true);
		Check(pending.Resolve(a, 4, &v) == Verdict::Unknown, "S479: WRITE_DATA is not known");
		Check(pending.ResolvePrecise(a, 4, &v) == Verdict::Value && v == 0x11111111u, "precise: one dword is known");
		Note(pending, WriteData(a + 8, false, {0x22222222u, 0x33333333u}), 2, true);
		Check(pending.ResolvePrecise(a + 8, 8, &v) == Verdict::Value && v == 0x3333333322222222ull, "precise: two dwords are a 64-bit value");
		Check(pending.ResolvePrecise(a + 12, 4, &v) == Verdict::Value && v == 0x33333333u, "precise: the high dword of two");
		Note(pending, WriteData(a + 0x20, true, {1, 2, 3}), 3, true);
		Check(pending.ResolvePrecise(a + 0x20, 4, &v) == Verdict::Value && v == 3u, "precise: one address keeps the last dword");
		Note(pending, WriteData(a + 0x40, false, {1, 2, 3}), 4, true);
		Check(pending.ResolvePrecise(a + 0x40, 4, &v) == Verdict::Unknown, "precise: three dwords are not a value");
		Note(pending, WriteData(a + 0x80, true, {}), 5, true);
		Check(pending.Resolve(a + 0x80, 4, &v) == Verdict::NoPending && pending.ResolvePrecise(a + 0x80, 4, &v) == Verdict::NoPending, "no dwords: no write in either view");
	}
	{ // sub-ranges of a known 64-bit label
		PendingWrites pending;
		uint64_t      v = 0;
		Note(pending, ReleaseMem(a, 2, 0x1122334455667788ull), 1, true);
		Check(pending.Resolve(a, 4, &v) == Verdict::Unknown, "S479: a 32-bit wait on a 64-bit label is unknown");
		Check(pending.ResolvePrecise(a, 4, &v) == Verdict::Value && v == 0x55667788u, "precise: the low dword");
		Check(pending.ResolvePrecise(a + 4, 4, &v) == Verdict::Value && v == 0x11223344u, "precise: the high dword");
		Check(pending.ResolvePrecise(a + 4, 8, &v) == Verdict::Unknown, "precise: a read past the label is unknown");
		Note(pending, ReleaseMem(a + 0x10, 1, 0xaabbccddu), 2, true);
		Check(pending.ResolvePrecise(a + 0x10, 8, &v) == Verdict::Unknown, "precise: a 64-bit wait on a 32-bit label is unknown");
		Culprit c;
		Check(pending.ResolvePrecise(a + 0x10, 8, &v, &c) == Verdict::Unknown && c.reason == Reason::Partial && c.src == SrcTag(Pm4::IT_RELEASE_MEM, 1), "culprit: partial");
		Note(pending, ReleaseMem(a + 0x20, 3, 0), 3, true);
		Check(pending.ResolvePrecise(a + 0x20, 4, &v, &c) == Verdict::Unknown && c.reason == Reason::Unknown, "culprit: a timestamp is unknown");
	}
}

void DecisionTests() {
	Check(!WaitNeedsDrain(Verdict::NoPending, 0, 1, ~0ull, 3), "nothing pending: no drain");
	Check(!WaitNeedsDrain(Verdict::Value, 5, 5, ~0ull, 3), "a value that passes: no drain");
	Check(WaitNeedsDrain(Verdict::Value, 4, 5, ~0ull, 3), "a value that fails: drain");
	Check(WaitNeedsDrain(Verdict::Unknown, 5, 5, ~0ull, 3), "unknown: drain");
	Check(ShadowWaitAgrees(Verdict::NoPending, 0, 7, 7, 7, ~0ull, 3), "shadow: same memory agrees");
	Check(!ShadowWaitAgrees(Verdict::NoPending, 0, 6, 7, 7, ~0ull, 3), "shadow: memory that passes only after the drain disagrees");
	Check(!ShadowWaitAgrees(Verdict::Value, 7, 0, 6, 7, ~0ull, 3), "shadow: a value the memory does not reach disagrees");
	Check(ShadowWaitAgrees(Verdict::Value, 7, 0, 7, 7, ~0ull, 3), "shadow: a value the memory reaches agrees");
	const uint64_t key = NoDrainHistKey(0x9f, {0x4639, Reason::Unknown}, false, {0x4902, Reason::Partial});
	Check((key & 0xffffu) == 0x4639u && ((key >> 16u) & 3u) == 1u && ((key >> 18u) & 0xffffu) == 0x4902u && ((key >> 34u) & 3u) == 2u && ((key >> 36u) & 1u) == 0 &&
	          ((key >> 40u) & 0x1ffu) == 0x9fu,
	      "histogram key");
	Check(((NoDrainHistKey(0x1f1, {}, true, {}) >> 36u) & 1u) == 1u && NoDrainHistKey(0x1f1, {}, true, {}) != 0, "histogram key: avoided, never 0");
}

Item RandomItem(Rng& rng, const std::vector<uint64_t>& labels, uint64_t seq) {
	Item it;
	it.seq      = seq;
	it.random   = rng.Next();
	const auto label = labels[rng.Below(static_cast<uint32_t>(labels.size()))];
	switch (rng.Below(11)) {
		case 0:
		case 1: it.packet = ReleaseMem(label, 1, rng.Next()); break;
		case 2: it.packet = ReleaseMem(label & ~7ull, 2, rng.Below(4) == 0 ? (rng.Next() & 0xffffffffull) * 0x100000001ull : rng.Next()); break;
		case 3: it.packet = ReleaseMem(label & ~7ull, 3, 0); break;
		case 4: {
			const uint64_t dump = kBase + 0x2000 + rng.Below(8) * 0x100 + (rng.Below(2) != 0 ? 8u : 0u);
			it.packet    = EventWrite(kEventOcclusionDump, dump);
			it.host_dump = rng.Below(2) != 0;
			break;
		}
		case 5: it.packet = EventWrite(0x07, 0); break;
		case 6: {
			std::vector<uint32_t> data(1 + rng.Below(4));
			for (auto& d: data) d = static_cast<uint32_t>(rng.Next());
			it.packet = WriteData(label, rng.Below(2) != 0, data);
			break;
		}
		case 7: it.packet = DmaData(kBase + 0x1000 + rng.Below(64) * 8, rng.Below(3), 4 + rng.Below(0x40)); break;
		case 8: it.packet = CopyData(label & ~7ull, rng.Below(2) != 0 ? 2 : 3, rng.Below(2) != 0); break;
		case 9: it.packet = Flip(); break;
		default: it.packet = WriteData(label, false, {static_cast<uint32_t>(rng.Next()), static_cast<uint32_t>(rng.Next())}); break;
	}
	return it;
}

// Random streams against the memory model: every answer of both views is checked against the memory after all pending items have run, the S479 view against
// the pre-S604 switch, and the precise view is never more careful than the S479 one.
void OracleTests() {
	Rng                   rng;
	std::vector<uint64_t> labels;
	for (uint32_t i = 0; i < 24; i++) labels.push_back(kBase + 0x400 + i * 8u + (i % 3 == 0 ? 4u : 0u));
	for (uint32_t i = 0; i < 6; i++) labels.push_back(kBase + 0x2000 + i * 0x100 + 0x10 * i);
	uint64_t checks_value = 0, checks_none = 0, checks_table = 0, avoided = 0;
	for (uint32_t round = 0; round < 1200; round++) {
		Memory m;
		for (auto& b: m.bytes) b = static_cast<uint8_t>(rng.Next());
		PendingWrites    precise, s479_new, s479_old;
		std::deque<Item> fifo;
		uint64_t         seq = 1;
		for (uint32_t step = 0; step < 200; step++) {
			const uint32_t action = rng.Below(8);
			if (action <= 2) { // enqueue
				Item it = RandomItem(rng, labels, seq++);
				Note(precise, it.packet, it.seq, true);
				Note(s479_new, it.packet, it.seq, false);
				OldNote(s479_old, it.packet, it.seq);
				fifo.push_back(it);
			} else if (action == 3 && !fifo.empty()) { // the recorder runs some items
				const uint32_t n = 1 + rng.Below(static_cast<uint32_t>(fifo.size()));
				for (uint32_t i = 0; i < n; i++) {
					Apply(m, fifo.front());
					const uint64_t done = fifo.front().seq;
					fifo.pop_front();
					precise.RetireUpTo(done);
					s479_new.RetireUpTo(done);
					s479_old.RetireUpTo(done);
				}
			} else if (action == 4 && rng.Below(16) == 0) { // a drain
				while (!fifo.empty()) {
					Apply(m, fifo.front());
					fifo.pop_front();
				}
				precise.Clear();
				s479_new.Clear();
				s479_old.Clear();
			} else { // a read: a wait (4 or 8 bytes) or a table
				Memory future = m;
				for (const auto& it: fifo) Apply(future, it);
				const bool     table = rng.Below(4) == 0;
				const uint32_t where = rng.Below(6); // the dumps, the DMA targets, the labels
				const uint64_t at    = where == 0 ? kBase + 0x2000 + rng.Below(0x900) / 8u * 8u
				                       : where == 1 ? kBase + 0x1000 + rng.Below(0x280) / 4u * 4u
				                                    : labels[rng.Below(static_cast<uint32_t>(labels.size()))] + (rng.Below(4) == 0 ? 4u : 0u);
				const uint64_t size  = table ? 8u * (1u + rng.Below(8)) : (rng.Below(2) != 0 ? 8u : 4u);
				if (at + size > kBase + kSize) continue;
				if (table) {
					const bool p = precise.OverlapsPrecise(at, size);
					const bool o = s479_old.Overlaps(at, size);
					Check(o == s479_new.Overlaps(at, size) && o == precise.Overlaps(at, size), "tables: the S479 view answers as before S604");
					Culprit c;
					Check(o == precise.OverlapsS479(at, size, &c), "tables: OverlapsS479 is Overlaps");
					Check(!p || o, "tables: the precise view is never wider than the S479 one");
					if (!p) {
						Check(m.Same(future, at, size), "tables: no precise overlap, but the pending items change the table");
						checks_table++;
					}
					if (o && !p) avoided++;
					continue;
				}
				uint64_t   vo = 0, vn = 0, vp = 0, vq = 0;
				const auto old_v = s479_old.Resolve(at, size, &vo);
				Check(old_v == s479_new.Resolve(at, size, &vn) && (old_v != Verdict::Value || vo == vn), "waits: the S479 view answers as before S604 (S479 notes)");
				Check(old_v == precise.Resolve(at, size, &vq) && (old_v != Verdict::Value || vo == vq), "waits: the S479 view answers as before S604 (precise notes)");
				const auto pv = precise.ResolvePrecise(at, size, &vp);
				if (old_v == Verdict::NoPending) Check(pv == Verdict::NoPending, "waits: nothing pending in S479 is nothing pending in the precise view");
				if (old_v == Verdict::Value) Check(pv == Verdict::Value && vp == vo, "waits: an S479 value is the precise value");
				if (old_v == Verdict::NoPending) Check(m.Same(future, at, size), "waits: S479 nothing pending, but the pending items change the bytes");
				if (old_v == Verdict::Value) Check(future.Read(at, size) == vo, "waits: the S479 value is not what memory will hold");
				if (pv == Verdict::NoPending) {
					Check(m.Same(future, at, size), "waits: precise nothing pending, but the pending items change the bytes");
					checks_none++;
				}
				if (pv == Verdict::Value) {
					Check(future.Read(at, size) == vp, "waits: the precise value is not what memory will hold");
					checks_value++;
				}
				if (old_v != Verdict::NoPending && pv != Verdict::Unknown) avoided++;
			}
		}
	}
	std::printf("CpPipeNoDrainTests: oracle checks value=%llu none=%llu table=%llu avoided=%llu\n", static_cast<unsigned long long>(checks_value),
	            static_cast<unsigned long long>(checks_none), static_cast<unsigned long long>(checks_table), static_cast<unsigned long long>(avoided));
	Check(checks_value > 1000 && checks_none > 1000 && checks_table > 1000 && avoided > 1000, "the oracle reached every case");
}

} // namespace

int main() {
	ModeTests();
	DecoderTests();
	DecisionTests();
	OracleTests();
	std::puts("CpPipeNoDrainTests: ok");
	return 0;
}
