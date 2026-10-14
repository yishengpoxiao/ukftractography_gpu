# GPU discovery and manual selection — 2026-10-14

The Slicer widget now defaults to listing all accessible NVIDIA CUDA GPUs. Each entry shows the CUDA ordinal, model, total memory and PCI bus address, allowing identical cards to be distinguished. Choose one card from the GPU device dropdown for tracking.

**Show all accessible CUDA GPUs** clears `CUDA_VISIBLE_DEVICES` only in discovery and tracking child processes. Unchecking it uses Slicer's startup visibility and ordering. The parent environment, `CUDA_DEVICE_ORDER` and container-level `NVIDIA_VISIBLE_DEVICES` are preserved. Driver/container access restrictions remain in effect. The default programmatic `UKFGPULogic.start()` behavior continues to inherit startup visibility; the widget explicitly supplies its selected mode.

Refresh and visibility changes preserve the physical GPU by PCI address when it remains accessible, even if CUDA ordinals change. Successful discovery also preserves a manually entered CUDA index when it is valid. Discovery failure retains manual index entry; confirmed zero devices disables tracking. GPU selection and visibility controls are locked while discovery or tracking runs. Result nodes record the CUDA device index and visibility mode, and the widget log identifies the selected model/PCI address before the backend log.

## Verification

Linux Slicer 5.2.2, SlicerDMRI and six RTX 4090 cards:

- Launch with `CUDA_VISIBLE_DEVICES=1,0`; default discovery lists six cards, while inherited visibility lists two.
- Select the physical second card; switching between modes preserves its PCI address while changing its CUDA ordinal between 1 and 0.
- Select CUDA 5 / PCI `0000:39:00.0`; Refresh preserves the selection. Actual Simple2T mixed-precision tracking logs `-> device 5,` and its child environment has no CUDA visibility mask.
- Source and packaged module checks include no-GPU disabling, manual fallback, model controls/presets, three-model tracking and prior GPU-reference geometry/finite recordings, FiberBundle display, Save/reload, cancellation and process-failure cleanup.
- Four platform/runtime unit tests include clearing only the child CUDA mask, preserving parent/container visibility, and forwarding a nonzero device index. Four installer tests passed.

Logs and small artifacts: `/data/yijie/ukf-GPU/validation/gpu-selection-2026-10-14/`.

The numerical backend is unchanged from the previously validated multi-architecture build. These tests do not establish new CPU equivalence. Windows native execution remains unverified; the released Windows archive is source/build material only. AMD/Intel GPUs, macOS and ARM are unsupported by the CUDA backend.
