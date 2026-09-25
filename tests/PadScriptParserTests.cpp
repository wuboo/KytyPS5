#include "libs/padScriptParser.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using namespace Libs::Controller::PadScript;

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "PadScriptParserTests: failed: %s\n", message);
    std::abort();
  }
}

std::vector<Entry> ParseOk(const char *script) {
  std::vector<Entry> entries;
  std::string error;
  Check(Parse(script, Defaults{}, entries, error), script);
  return entries;
}

bool ParseFails(const char *script) {
  std::vector<Entry> entries;
  std::string error;
  return !Parse(script, Defaults{}, entries, error) && !error.empty();
}

void TestSecondsPointUsesDefaultHold() {
  const auto entries = ParseOk("3:cross");
  Check(entries.size() == 1, "one entry");
  Check(entries[0].buttons == 0x00004000u, "cross bit");
  Check(IsActive(entries[0], 3.0, 0.0), "active at start");
  Check(IsActive(entries[0], 3.29, 0.0), "active inside 300 ms hold");
  Check(!IsActive(entries[0], 3.3, 0.0), "inactive at hold end");
  Check(!IsActive(entries[0], 2.99, 0.0), "inactive before start");
}

void TestFrameRangeIsExclusive() {
  const auto entries = ParseOk("f300-340:options");
  Check(IsActive(entries[0], 0.0, 300.0), "active at first frame");
  Check(IsActive(entries[0], 0.0, 339.0), "active at last frame");
  Check(!IsActive(entries[0], 0.0, 340.0), "end is exclusive");
  Check(!IsActive(entries[0], 1000.0, 10.0), "frame entry ignores seconds");
}

void TestHoldAndCombinedActions() {
  const auto entries = ParseOk("f100+f12:left-stick-left+cross");
  Check(entries[0].buttons == 0x00004000u, "cross with stick");
  Check(entries[0].sticks[0] == -1 && entries[0].sticks[1] == 0, "left stick left");
  Check(IsActive(entries[0], 0.0, 111.0), "inside frame hold");
  Check(!IsActive(entries[0], 0.0, 112.0), "hold end is exclusive");
}

void TestSeparatorsAndComments() {
  const auto entries = ParseOk("# route\n1:cross ; 2-4:circle # confirm\n\nf10:up");
  Check(entries.size() == 3, "three entries across ';' and newlines");
  Check(entries[1].buttons == 0x00002000u, "circle");
  Check(IsActive(entries[1], 3.9, 0.0) && !IsActive(entries[1], 4.0, 0.0), "range");
}

void TestRejectsBadInput() {
  Check(ParseFails("cross"), "missing colon");
  Check(ParseFails("1:jump"), "unknown action");
  Check(ParseFails("5-5:cross"), "empty range");
  Check(ParseFails("f10-2:cross"), "empty frame range");
  Check(ParseFails("10-f20:cross"), "range mixes units");
  Check(ParseFails("f100+12:cross"), "hold without frame prefix");
  Check(ParseFails("f10+0.5:cross"), "hold mixes units");
  Check(ParseFails("-1:cross"), "negative start");
  Check(ParseFails("1:"), "no action");
}

} // namespace

int main() {
  TestSecondsPointUsesDefaultHold();
  TestFrameRangeIsExclusive();
  TestHoldAndCombinedActions();
  TestSeparatorsAndComments();
  TestRejectsBadInput();
  std::printf("PadScriptParserTests: ok\n");
  return 0;
}
