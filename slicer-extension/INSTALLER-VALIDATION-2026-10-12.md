# Portable installer validation — 2026-10-12

The offline installer generates a native Extensions Manager archive for the running Slicer 5.x version. It installs CUDA and C++ runtime dependencies privately with the extension. It does not install an NVIDIA system driver, replace glibc, create a Conda environment, or require SlicerDMRI.

## Delivered runtime

Linux x86_64; CUDA Toolkit 12.1 build, glibc 2.29 minimum derived from the binaries. Native CUDA targets: sm_75, sm_80, sm_86, sm_89 and sm_90, plus compute_75 PTX. Packaged dependencies include libcudart.so.12, libstdc++.so.6 and libgcc_s.so.1, with redistribution notices. CUDA runtime naming uses the actual SONAME rather than assuming version 12.

The CLI and GPU library were rebuilt together from the current source. Host compilation preserves the previously validated AVX target and no-strict-overflow behavior. A generic host build changed small numerical values and some degenerate trajectories, so it was rejected. An initial attempt pairing the old CLI with a new GPU library was also rejected by the loader check. Neither rejected build is delivered. Future builds must be validated with the same care when changing compiler flags.

For reproducibility, configure `platforms/standalone-2026-10-11` with a new build directory, the existing dependency directories, and:

```text
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_CUDA_ARCHITECTURES=75-real;80-real;86-real;89-real;90-real;75-virtual
-DCMAKE_CXX_FLAGS=-mtune=native -march=corei7-avx -fno-strict-overflow -DTEEM_STATIC=1
-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/path/to/cuda/lib
```

Then stage it with `scripts/package_slicer_extension.py` and build the offline ZIP with `scripts/package_portable_installer.py`. The actual toolkit/dependency paths and build logs are retained in the local validation reports.

## Checks passed

- Four installer unit tests: Linux/Windows runtime selection, Slicer 5.2/5.6/5.12 archive layouts, exclusive output creation, unsupported versions/platforms, insufficient glibc, missing native backends, checksum/payload failures and generated Python caches.
- Three existing platform/runtime packaging tests; Windows test files are structural PE fixtures, not executable native builds.
- Two model-control/preset tests.
- Real offline installation through Slicer 5.2.2 Extensions Manager, with isolated test settings. Native loader and GPU checks passed; CUDA, C++ and GCC runtime libraries were present in the installed extension.
- Running the installer again generated an update archive without replacing the existing installation.
- Twelve models × FP64/mixed precision: geometry, connectivity and every recorded array were identical to the prior GPU reference outputs on RTX 4090 (24 cases).
- Forced compute_75 PTX JIT on RTX 4090: Simple2T mixed precision matched the prior GPU reference arrays exactly.
- The installed extension passed GPU discovery, model controls, presets, custom/duplicate naming, three-model tracking, FiberBundle display, Save/reload, cancellation and failed-process cleanup with SlicerDMRI.

Detailed logs and small result data: `/data/yijie/ukf-GPU/validation/installer-2026-10-12/`. Regression references: `validation/reference-gpu/`.

Only Slicer 5.2.2 was executed for this installer revision. Other 5.x layouts were tested structurally; historical application tests do not constitute a new native test of this installer on those versions. Other GPU architectures and operating systems remain unverified.

Windows x86_64 installer selection and native build/packaging scripts are provided, but this delivery contains no native Windows runtime. The installer explicitly rejects a platform absent from the bundle. macOS and ARM are unsupported. Existing strict CPU/GPU accuracy failures remain unchanged; GPU-reference regression is not CPU equivalence.
