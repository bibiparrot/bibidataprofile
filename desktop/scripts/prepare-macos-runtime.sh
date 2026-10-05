#!/usr/bin/env bash
set -euo pipefail
brew list libomp >/dev/null 2>&1 || brew install libomp
mkdir -p src-tauri/resources/runtime
cp "$(brew --prefix libomp)/lib/libomp.dylib" src-tauri/resources/runtime/libomp.dylib
curl --fail --location --retry 3 https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-21.1.8/openmp/LICENSE.TXT -o src-tauri/resources/runtime/libomp-LICENSE.txt
otool -L src-tauri/resources/runtime/libomp.dylib
