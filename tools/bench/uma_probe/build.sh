#!/bin/sh
# Builds uma_probe for arm64 and x86_64 (the emulator runs x86_64 under Rosetta). Output: ./uma_probe.arm64, ./uma_probe.x86_64
# Regenerate the shader header after editing uma_probe.comp:
#   glslangValidator -V --target-env vulkan1.2 uma_probe.comp -o uma_probe.spv && xxd -i uma_probe.spv > uma_probe_spv.h && rm uma_probe.spv
set -e
here=$(cd "$(dirname "$0")" && pwd)
vk=${VULKAN_HEADERS:-$here/../../../3rdparty/Vulkan-Headers/include}
for a in arm64 x86_64; do
	nice -n 19 clang++ -std=c++17 -O2 -Wall -arch $a -I"$vk" -I"$here" "$here/uma_probe.cpp" -o "$here/uma_probe.$a"
done
