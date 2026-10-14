#!/bin/sh
# Wrap a built variant in a game-category app bundle so macOS can apply Game Mode to it.
#   tools/bench/make_app_bundle.sh <variant-label | variant-dir>
# Creates <variant>/KytyGame.app next to kyty_emulator. Local ad-hoc signature only, no network.
#
# Run the binary from the bundle: set `emulator` in the kbench config to
#   <variant>/KytyGame.app/Contents/MacOS/kyty_emulator
# kbench runs the emulator with the binary's directory as cwd, so the bundle's MacOS directory holds symlinks to the
# variant's _ShaderCache and _PipelineCache (shared with the plain variant) and its own _SaveData/_TempData/_DownloadData.
# The binary is a copy, not a symlink: the system identifies the process by the real path of its executable.
#
# Game Mode check, while the emulator window is full screen (KYTY_BENCH_FULLSCREEN=1):
#   DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun gamepolicyctl game-mode status
#   log show --last 2m --predicate 'subsystem contains "gamepolicy"' | tail -20
set -eu
arg=$1
root=$(git rev-parse --show-toplevel)
if [ -d "$arg" ]; then
	variant=$arg
else
	variant=$root/_Build/variants/$arg
fi
[ -x "$variant/kyty_emulator" ] || { echo "no kyty_emulator in $variant" >&2; exit 1; }

app=$variant/KytyGame.app
macos=$app/Contents/MacOS
rm -rf "$app"
mkdir -p "$macos"
cp "$variant/kyty_emulator" "$macos/kyty_emulator"
[ -f "$variant/libMoltenVK.dylib" ] && cp "$variant/libMoltenVK.dylib" "$macos/libMoltenVK.dylib"
for cache in _ShaderCache _PipelineCache; do
	mkdir -p "$variant/$cache"
	ln -s "$variant/$cache" "$macos/$cache"
done

cat >"$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleIdentifier</key><string>local.kyty.game</string>
	<key>CFBundleName</key><string>KytyGame</string>
	<key>CFBundleExecutable</key><string>kyty_emulator</string>
	<key>CFBundlePackageType</key><string>APPL</string>
	<key>CFBundleVersion</key><string>1</string>
	<key>CFBundleShortVersionString</key><string>1.0</string>
	<key>LSApplicationCategoryType</key><string>public.app-category.games</string>
	<key>LSSupportsGameMode</key><true/>
	<key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
plutil -lint "$app/Contents/Info.plist" >/dev/null

codesign --force --sign - --timestamp=none "$macos/libMoltenVK.dylib" 2>/dev/null || true
codesign --force --deep --sign - --timestamp=none "$app"
echo "$app ($(cat "$variant/REVISION" 2>/dev/null || echo unknown))"
