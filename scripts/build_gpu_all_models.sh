#!/usr/bin/env bash
# Use the dated CLI configuration without overwriting the original XML/CMake files.
# The destination CLP header is generated build code, not a source configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
build_dir=$(realpath "${1:-../build-gpu}")
clp_generator=$(sed -n 's/^GENERATECLP_EXE:FILEPATH=//p' "$build_dir/CMakeCache.txt")
for component in ukf UKFTractography; do
  "$clp_generator" --InputXML "$PWD/UKFTractography/UKFTractography-2026-10-10.xml" \
    --OutputCxx "$build_dir/$component/UKFTractographyCLP.h"
done
"${CMAKE:-cmake}" --build "$build_dir" -j "${BUILD_JOBS:-8}" --target UKFTractography
