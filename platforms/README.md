# Standalone build entries

- `standalone-2026-10-14-r2/`: current Linux/Windows CUDA backend build, for environments with dependencies already available.
- `superbuild-2026-10-14-r4/`: current dependency superbuild; configures the standalone entry above.
- `standalone-2026-10-11/main.cc` and `UKFGPU-2026-10-11.manifest`: shared executable entry point and Windows UTF-8 manifest used by the current standalone build.

See the [repository build instructions](../README.md#build). The root CMake/SuperBuild files support the original upstream build route. Superseded standalone/superbuild configurations have been removed; their versions remain available in Git history.
