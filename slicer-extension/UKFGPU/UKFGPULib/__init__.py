"""Command construction shared by the Slicer UI and integration tests."""
import os
import sys
from pathlib import Path, PureWindowsPath

MODELS = {
    "Simple1T": (1, []), "Simple2T": (2, []), "Simple3T": (3, []),
    "Full1T": (1, ["--fullTensorModel"]),
    "Full2T": (2, ["--fullTensorModel"]),
    "Full3T": (3, ["--fullTensorModel"]),
    "Simple1T-FW": (1, ["--freeWater"]),
    "Simple2T-FW": (2, ["--freeWater"]),
    "Full1T-FW": (1, ["--fullTensorModel", "--freeWater"]),
    "Full2T-FW": (2, ["--fullTensorModel", "--freeWater"]),
    "NODDI1F": (1, ["--noddi"]), "NODDI2F": (2, ["--noddi"]),
}
NUMERIC_OPTIONS = ("seedsPerVoxel", "numThreads", "stepLength", "recordLength",
                   "maxHalfFiberLength", "seedingThreshold", "stoppingFA",
                   "stoppingThreshold", "Qm", "Ql", "Qw", "Rs", "Qkappa",
                   "Qvic", "minBranchingAngle", "maxBranchingAngle")
RECORD_OPTIONS = ("recordFA", "recordTrace", "recordTensors", "recordFreeWater",
                  "recordVic", "recordKappa", "recordViso", "recordNMSE",
                  "recordState", "recordCovariance")


def command(executable, dwi, mask, output, model="Simple2T", precision=1,
            device=0, seeds=None, seed_labels="1", options=None):
    """Return argv; all inputs are passed literally, without a shell."""
    tensors, flags = MODELS[model]
    args = [str(executable), "--dwiFile", str(dwi), "--maskFile", str(mask),
            "--tracts", str(output), "--numTensor", str(tensors),
            "--useGPU", "--gpuDevice", str(device), "--gpuPrecision", str(precision)]
    args += flags
    if seeds:
        labels = [int(label.strip()) for label in seed_labels.split(",")]
        args += ["--seedsFile", str(seeds), "--seedLabels", ",".join(map(str, labels))]
    for name, value in (options or {}).items():
        if name in NUMERIC_OPTIONS:
            args += ["--" + name, str(value)]
        elif name in RECORD_OPTIONS:
            if value:
                args += ["--" + name]
        else:
            raise ValueError("Unknown tracking option: " + name)
    return args


def process_environment(startup_environment, executable=None, platform=None):
    """Use Slicer's original environment rather than its loaded library paths."""
    env = dict(startup_environment)
    removed = {"LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH",
               "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"}
    for key in list(env):
        if key.upper() in removed:
            del env[key]
    if (platform or sys.platform) == "win32" and executable:
        windows = next((value for key, value in env.items() if key.upper() == "SYSTEMROOT"), None)
        if not windows:
            raise RuntimeError("Windows SystemRoot is missing from the process environment.")
        for key in list(env):
            if key.upper() == "PATH":
                del env[key]
        # DLLs live beside the executable; exclude Slicer's incompatible DLLs.
        env["PATH"] = ";".join((str(PureWindowsPath(executable).parent), windows + r"\System32", windows))
    return env


def default_executable(module_directory, platform=None):
    name = "UKFTractography_gpu.exe" if (platform or sys.platform) == "win32" else "ukf-gpu"
    bundled = Path(module_directory) / "Runtime" / "bin" / name
    return os.environ.get("UKF_GPU_EXECUTABLE", str(bundled) if bundled.is_file() else "")


def check_result(exit_code, log, output):
    if exit_code != 0:
        raise RuntimeError("UKF GPU exited with code %s. See the process log." % exit_code)
    if "falling back" in log or "tractography done on device" not in log:
        raise RuntimeError("GPU execution was not confirmed. CPU fallback is not accepted.")
    if not Path(output).is_file():
        raise RuntimeError("UKF GPU did not create the fiber output.")
