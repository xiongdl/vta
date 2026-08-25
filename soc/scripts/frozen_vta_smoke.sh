#!/bin/sh
set -eu

soc_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
variant=${VTA_IP_VARIANT:-apb32-axi64}
rtl="$soc_dir/../generated/vta-ip/$variant/rtl/vta.v"
build_dir="$soc_dir/build/frozen-vta-smoke"
sdk_cxx=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1

verilator --cc --exe --top-module VTAShellAPB \
  --Mdir "$build_dir" \
  -Wno-fatal -Wno-WIDTH -Wno-UNOPTFLAT \
  "$rtl" "$soc_dir/sim/frozen_vta_smoke.cpp"
make -C "$build_dir" -f VVTAShellAPB.mk \
  CXX=/usr/bin/clang++ LINK=/usr/bin/clang++ AR=/usr/bin/ar \
  "VM_USER_CFLAGS=-nostdinc++ -isystem $sdk_cxx"
"$build_dir/VVTAShellAPB"
