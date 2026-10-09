#ifndef KYTY_COMMON_CONTROLFILE_H_
#define KYTY_COMMON_CONTROLFILE_H_

// KYTY_CONTROL_FILE=<path>: diagnostic switches that can change inside one run. The file is polled once
// per guest flip; when its mtime or size changes it is parsed again, and the lines that apply to the
// current flip are published as an immutable Snapshot (generation + 1). Without the variable nothing is
// read and every query answers "no".
//
//   skip_cs=<hex hash>[,...]                   compute dispatches of these guest CS hashes are not run
//   dump_rt=<ps hex>[:<flip dec>][,...]        colour targets + depth after the last draw of that PS
//   force_tex=<ps hex>:<slot dec>:<rgba8 hex>  the T# in that slot reads as a 1x1 image of one colour,
//                                              0xAABBGGRR (r in the low byte)
//   label=<name>                               arm name used in the file names of dumps
//   # starts a comment
//
// The lines before the first block are the base. `at=<flip>` (or `@<flip>`) opens a block whose lines
// apply from that guest flip on; `until=<flip>` inside a block closes the window after that flip, and
// the keys then fall back to the earlier blocks / the base. A key absent from every active block keeps
// the base value; `key=` with an empty value clears it. Parsing is strict: a bad line is reported and
// ignored (a bad `at` line swallows its block).

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace Common::ControlFile {

struct ForceTex {
	uint64_t ps   = 0;
	uint32_t slot = 0;
	uint32_t rgba = 0;
	bool     operator==(const ForceTex& o) const { return ps == o.ps && slot == o.slot && rgba == o.rgba; }
};

struct DumpRt {
	uint64_t ps   = 0;
	uint64_t flip = 0; // 0 = the first flip that draws the shader after the snapshot is published
	bool     operator==(const DumpRt& o) const { return ps == o.ps && flip == o.flip; }
};

struct Snapshot {
	uint64_t              gen = 0;
	std::vector<uint64_t> skip_cs;
	std::vector<DumpRt>   dump_rt;
	std::vector<ForceTex> force_tex;
	std::string           label;
	std::string           block; // blocks in force ("at=6600-6700+at=6800"), empty = base only

	[[nodiscard]] bool SkipCs(uint64_t hash) const {
		return std::find(skip_cs.begin(), skip_cs.end(), hash) != skip_cs.end();
	}
	// force_tex colour for slot `slot` of pixel shader `ps` (0xAABBGGRR), false when not listed.
	[[nodiscard]] bool ForceTexFor(uint64_t ps, uint32_t slot, uint32_t* rgba) const {
		for (const auto& entry: force_tex) {
#ifdef KYTY_CONTROL_MUTANT_FORCE_ANY_SLOT
			if (entry.ps == ps) {
#else
			if (entry.ps == ps && entry.slot == slot) {
#endif
				*rgba = entry.rgba;
				return true;
			}
		}
		return false;
	}
	// Same settings (the generation and the block name do not count).
	[[nodiscard]] bool SameSettings(const Snapshot& o) const {
		return skip_cs == o.skip_cs && dump_rt == o.dump_rt && force_tex == o.force_tex && label == o.label;
	}
};

inline std::string_view Trim(std::string_view s) {
	while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r')) s.remove_prefix(1);
	while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
	return s;
}

// Strict number: only digits of the base, 1..16 characters.
inline bool ParseNumber(std::string_view text, int base, uint64_t& out) {
	if (text.empty() || text.size() > 16) return false;
#ifdef KYTY_CONTROL_MUTANT_LAX_NUMBERS
	out = std::strtoull(std::string(text).c_str(), nullptr, base);
	return true;
#else
	for (const char c: text) {
		const bool ok = base == 16 ? ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))
		                           : (c >= '0' && c <= '9');
		if (!ok) return false;
	}
	out = std::strtoull(std::string(text).c_str(), nullptr, base);
	return true;
#endif
}

// Calls fn(item) for every non-empty comma separated item; stops and returns false when fn does.
template <typename Fn>
inline bool ForEachItem(std::string_view text, Fn fn) {
	size_t pos = 0;
	while (pos <= text.size()) {
		const auto end  = text.find(',', pos);
		const auto item = Trim(text.substr(pos, end == std::string_view::npos ? text.size() - pos : end - pos));
		if (!item.empty() && !fn(item)) return false;
		if (end == std::string_view::npos) break;
		pos = end + 1;
	}
	return true;
}

inline bool ParseHexList(std::string_view text, std::vector<uint64_t>& out) {
	out.clear();
	return ForEachItem(text, [&](std::string_view item) {
		uint64_t value = 0;
		if (!ParseNumber(item, 16, value) || value == 0) return false;
		out.push_back(value);
		return true;
	});
}

inline bool ParseDumpRt(std::string_view text, std::vector<DumpRt>& out) {
	out.clear();
	return ForEachItem(text, [&](std::string_view item) {
		const auto colon = item.find(':');
		uint64_t   ps = 0, flip = 0;
		if (!ParseNumber(item.substr(0, colon), 16, ps) || ps == 0) return false;
		if (colon != std::string_view::npos && (!ParseNumber(item.substr(colon + 1), 10, flip) || flip == 0))
			return false;
		out.push_back({ps, flip});
		return true;
	});
}

inline bool ParseForceTex(std::string_view text, std::vector<ForceTex>& out) {
	out.clear();
	return ForEachItem(text, [&](std::string_view item) {
		const auto c1 = item.find(':');
		const auto c2 = c1 == std::string_view::npos ? c1 : item.find(':', c1 + 1);
		if (c2 == std::string_view::npos) return false;
		uint64_t   ps = 0, slot = 0, rgba = 0;
		const auto rgba_text = item.substr(c2 + 1);
		if (!ParseNumber(item.substr(0, c1), 16, ps) || ps == 0 ||
		    !ParseNumber(item.substr(c1 + 1, c2 - c1 - 1), 10, slot) || slot > 127 ||
		    rgba_text.size() != 8 || !ParseNumber(rgba_text, 16, rgba))
			return false;
		out.push_back({ps, static_cast<uint32_t>(slot), static_cast<uint32_t>(rgba)});
		return true;
	});
}

inline bool ValidLabel(std::string_view label) {
	if (label.size() > 48) return false;
	for (const char c: label) {
		const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
		                c == '_' || c == '-' || c == '.';
		if (!ok) return false;
	}
	return true;
}

struct Item {
	std::string key;
	std::string value;
};

struct Block {
	uint64_t          at    = 0;
	uint64_t          until = 0; // 0 = open
	std::vector<Item> items;
};

struct File {
	std::vector<Item>        base;
	std::vector<Block>       blocks;
	std::vector<std::string> errors; // "line N: reason"
};

inline bool ValidateItem(const Item& item, std::string& why) {
	std::vector<uint64_t> hashes;
	std::vector<DumpRt>   dumps;
	std::vector<ForceTex> forces;
	if (item.key == "skip_cs") {
		if (ParseHexList(item.value, hashes)) return true;
		why = "skip_cs is a list of hex hashes";
	} else if (item.key == "dump_rt") {
		if (ParseDumpRt(item.value, dumps)) return true;
		why = "dump_rt is <ps hex>[:<flip decimal>]";
	} else if (item.key == "force_tex") {
		if (ParseForceTex(item.value, forces)) return true;
		why = "force_tex is <ps hex>:<slot 0..127>:<rgba8 hex>";
	} else if (item.key == "label") {
		if (ValidLabel(item.value)) return true;
		why = "label is up to 48 characters of [A-Za-z0-9_.-]";
	} else {
		why = "unknown key '" + item.key + "'";
	}
	return false;
}

inline File Parse(std::string_view text) {
	File     file;
	Block*   current  = nullptr;
	bool     swallow  = false;
	uint32_t line_no  = 0;
	size_t   pos      = 0;
	while (pos <= text.size()) {
		const auto end  = text.find('\n', pos);
		auto       line = text.substr(pos, end == std::string_view::npos ? text.size() - pos : end - pos);
		pos             = end == std::string_view::npos ? text.size() + 1 : end + 1;
		++line_no;
		if (const auto hash = line.find('#'); hash != std::string_view::npos) line = line.substr(0, hash);
		line = Trim(line);
		if (line.empty()) continue;
		const auto fail = [&](const std::string& why) {
			file.errors.push_back("line " + std::to_string(line_no) + ": " + why);
		};
		std::string_view key, value;
		if (line.front() == '@') {
			key   = "at";
			value = Trim(line.substr(1));
		} else {
			const auto eq = line.find('=');
			if (eq == std::string_view::npos) {
				fail("no '='");
				continue;
			}
			key   = Trim(line.substr(0, eq));
			value = Trim(line.substr(eq + 1));
		}
		if (key == "at") {
			uint64_t flip = 0;
			if (!ParseNumber(value, 10, flip) || flip == 0) {
				fail("at is a decimal flip >= 1");
				swallow = true;
				current = nullptr;
				continue;
			}
			swallow = false;
			file.blocks.push_back({flip, 0, {}});
			current = &file.blocks.back();
			continue;
		}
		if (swallow) continue;
		if (key == "until") {
			uint64_t flip = 0;
			if (current == nullptr) {
				fail("until outside a block");
			} else if (!ParseNumber(value, 10, flip) || flip < current->at) {
				fail("until is a decimal flip >= the block's at");
			} else {
				current->until = flip;
			}
			continue;
		}
		Item        item {std::string(key), std::string(value)};
		std::string why;
		if (!ValidateItem(item, why)) {
			fail(why);
			continue;
		}
		(current != nullptr ? current->items : file.base).push_back(std::move(item));
	}
	return file;
}

inline void Apply(Snapshot& out, const Item& item) {
	if (item.key == "skip_cs") {
		ParseHexList(item.value, out.skip_cs);
	} else if (item.key == "dump_rt") {
		ParseDumpRt(item.value, out.dump_rt);
	} else if (item.key == "force_tex") {
		ParseForceTex(item.value, out.force_tex);
	} else if (item.key == "label") {
		out.label = item.value;
	}
}

// The settings in force at guest flip `flip` (1-based): the base, then every block whose window
// contains the flip, in file order.
inline Snapshot Resolve(const File& file, uint64_t flip) {
	Snapshot out;
	for (const auto& item: file.base) Apply(out, item);
	for (const auto& block: file.blocks) {
#ifdef KYTY_CONTROL_MUTANT_NO_UNTIL
		const bool active = flip >= block.at;
#else
		const bool active = flip >= block.at && (block.until == 0 || flip <= block.until);
#endif
		if (!active) continue;
		for (const auto& item: block.items) Apply(out, item);
		out.block += (out.block.empty() ? "" : "+") + std::string("at=") + std::to_string(block.at) +
		             (block.until != 0 ? "-" + std::to_string(block.until) : std::string());
	}
	return out;
}

// ---- process-wide state ------------------------------------------------------------------------

namespace Detail {

inline std::atomic<const Snapshot*>& Slot() {
	static const Snapshot empty {};
	static std::atomic<const Snapshot*> slot {&empty};
	return slot;
}

} // namespace Detail

// The snapshot readers use. Old snapshots are never freed (a run publishes a handful).
inline const Snapshot& Current() {
	return *Detail::Slot().load(std::memory_order_acquire);
}

// Publishes `next` when its settings differ from the current ones. Returns true when published.
inline bool Publish(Snapshot next) {
	const Snapshot& now = Current();
	if (now.SameSettings(next) && now.block == next.block) return false;
	next.gen = now.gen + 1;
	Detail::Slot().store(new Snapshot(std::move(next)), std::memory_order_release);
	return true;
}

struct Poller {
	std::filesystem::path path;
	std::filesystem::file_time_type mtime {};
	uintmax_t                       size = static_cast<uintmax_t>(-1);
	File                            file;
	bool                            loaded = false;

	// Re-reads the file when its mtime or size changed. Returns the errors of a fresh parse.
	std::vector<std::string> Refresh() {
		std::error_code ec;
		const auto      m = std::filesystem::last_write_time(path, ec);
		const auto      s = ec ? 0 : std::filesystem::file_size(path, ec);
		if (ec) return {};
		if (loaded && m == mtime && s == size) return {};
		std::ifstream     in(path);
		std::stringstream buffer;
		buffer << in.rdbuf();
		file   = Parse(buffer.str());
		mtime  = m;
		size   = s;
		loaded = true;
		return file.errors;
	}
};

} // namespace Common::ControlFile

#endif /* KYTY_COMMON_CONTROLFILE_H_ */
