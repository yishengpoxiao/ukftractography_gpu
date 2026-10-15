"""Command construction shared by the Slicer UI and integration tests."""
import os
import sys
import json
import math
import re
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


def command(executable, dwi, mask, output, model="Simple2T", precision=0,
            device=0, seeds=None, seed_labels="1", options=None):
    """Return argv; all inputs are passed literally, without a shell."""
    validate_options(options or {})
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


def process_environment(startup_environment, executable=None, platform=None, all_gpus=False, gpu_uuids=None, progress=False):
    """Use Slicer's original environment rather than its loaded library paths."""
    env = dict(startup_environment)
    removed = {"LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH",
               "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "UKF_GPU_DEVICES", "UKF_GPU_PROGRESS"}
    if all_gpus or gpu_uuids:
        removed.add("CUDA_VISIBLE_DEVICES")
    for key in list(env):
        if key.upper() in removed:
            del env[key]
    if gpu_uuids:
        env['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_uuids)
        env['UKF_GPU_DEVICES'] = ','.join(str(i) for i in range(len(gpu_uuids)))
    if progress:
        env['UKF_GPU_PROGRESS'] = '1'
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


PRESETS = {
    "Standard": "Standard tracking settings; model-specific filter noise defaults, branching off, and no extra recording.",
    "Quick preview": "Shorter tracks (50 mm per direction) and sparse recording (1.8 mm), for checking inputs and visualization. Not a full tract reconstruction.",
    "Fine sampling": "Half the standard step and recording spacing. Increases computation and memory; does not guarantee better anatomical accuracy.",
}


def preset_values(name, model):
    tensors, _ = MODELS[model]
    values = dict(seedsPerVoxel=1, numThreads=16, stepLength=0.15 if tensors == 3 else 0.3,
                  recordLength=0.45 if tensors == 3 else 0.9, maxHalfFiberLength=250,
                  seedingThreshold=0.18, stoppingFA=0.15, stoppingThreshold=0.1,
                  Qm=0, Ql=0, Qw=0, Rs=0, Qkappa=0, Qvic=0,
                  minBranchingAngle=0, maxBranchingAngle=0)
    if name == "Quick preview":
        values.update(maxHalfFiberLength=50, recordLength=1.8)
    elif name == "Fine sampling":
        values['stepLength'] /= 2
        values['recordLength'] /= 2
    elif name != "Standard":
        raise ValueError('Unknown preset: ' + name)
    return values


def model_options(model, has_seeds):
    tensors, flags = MODELS[model]
    noddi = '--noddi' in flags
    free_water = '--freeWater' in flags
    branching = tensors > 1 and not noddi and has_seeds
    numeric = dict.fromkeys(NUMERIC_OPTIONS, True)
    numeric.update(stoppingFA=not noddi, Ql=not noddi, Qw=free_water,
                   Qkappa=noddi, Qvic=noddi,
                   minBranchingAngle=branching, maxBranchingAngle=branching)
    records = dict.fromkeys(RECORD_OPTIONS, True)
    records.update(recordFA=not noddi, recordTrace=not noddi, recordTensors=not noddi,
                   recordFreeWater=free_water, recordVic=noddi, recordKappa=noddi,
                   recordViso=noddi)
    return numeric, records


def query_cuda_devices(executable):
    """Called only in a child Python process; enumerate runtime-visible ordinals."""
    import ctypes as c
    import ctypes.util
    candidates = list(Path(executable).parent.glob('cudart64_*.dll')) if os.name == 'nt' else list(
        (Path(executable).parent.parent/'lib').glob('libcudart.so*'))
    candidates = sorted({path.resolve() for path in candidates if path.is_file()})
    if len(candidates) > 1:
        raise RuntimeError('Multiple CUDA runtimes beside the backend; use a package containing its matching runtime only.')
    library = next((str(path) for path in candidates if path.is_file()), None)
    library = library or ctypes.util.find_library('cudart')
    if not library:
        raise RuntimeError('CUDA runtime not found near the backend. Choose a packaged backend or enter a CUDA device index manually.')
    runtime = c.CDLL(library)
    def check(status):
        if status:
            raise RuntimeError('CUDA device discovery failed (error %d).' % status)
    count = c.c_int()
    status = runtime.cudaGetDeviceCount(c.byref(count))
    if status == 100:  # cudaErrorNoDevice
        return []
    check(status)
    import uuid
    runtime_version, driver_version = c.c_int(), c.c_int()
    check(runtime.cudaRuntimeGetVersion(c.byref(runtime_version)))
    check(runtime.cudaDriverGetVersion(c.byref(driver_version)))
    devices = []
    if count.value:
        driver = c.WinDLL('nvcuda.dll') if os.name == 'nt' else c.CDLL('libcuda.so.1')
        check(driver.cuInit(0))
        for index in range(count.value):
            bus, name = c.create_string_buffer(64), c.create_string_buffer(256)
            device, memory = c.c_int(), c.c_size_t()
            check(runtime.cudaDeviceGetPCIBusId(bus, len(bus), index))
            check(driver.cuDeviceGetByPCIBusId(c.byref(device), bus))
            check(driver.cuDeviceGetName(name, len(name), device))
            check(driver.cuDeviceTotalMem_v2(c.byref(memory), device))
            identifier = (c.c_ubyte * 16)()
            check(driver.cuDeviceGetUuid(c.byref(identifier), device))
            free, total = c.c_size_t(), c.c_size_t()
            status = runtime.cudaSetDevice(index)
            if not status:
                status = runtime.cudaMemGetInfo(c.byref(free), c.byref(total))
            devices.append(dict(index=index, name=name.value.decode(), memory=memory.value,
                                free_memory=free.value if not status else None, pci_bus=bus.value.decode(),
                                uuid='GPU-' + str(uuid.UUID(bytes=bytes(identifier))),
                                cuda_runtime=runtime_version.value, cuda_driver=driver_version.value))
    return devices


def validate_options(options):
    for name, value in options.items():
        if name not in NUMERIC_OPTIONS:
            continue
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(name + ' must be finite and nonnegative.')
        if name in ('seedsPerVoxel', 'numThreads') and (value < 1 or int(value) != value):
            raise ValueError(name + ' must be a positive integer.')
        if name in ('stepLength', 'recordLength', 'maxHalfFiberLength') and value <= 0:
            raise ValueError(name + ' must be greater than zero.')
        if name in ('stoppingFA', 'stoppingThreshold', 'seedingThreshold') and value > 1:
            raise ValueError(name + ' must be between 0 and 1.')
        if name.endswith('BranchingAngle') and value > 90:
            raise ValueError(name + ' must be between 0 and 90 degrees.')
    if options.get('recordLength', math.inf) < options.get('stepLength', 0):
        raise ValueError('Recording spacing must be at least the tracking step size.')
    if options.get('maxBranchingAngle', 0) and options.get('minBranchingAngle', 0) >= options['maxBranchingAngle']:
        raise ValueError('Branching requires minimum angle < maximum angle.')


class TrackingProgress:
    """Parse complete backend lines, including concurrent per-GPU counters."""
    def __init__(self):
        self.pending = ''
        self.total = 0
        self.completed = {}
        self.stage = 'Preparing inputs'
        self.pass_number = 0

    def feed(self, text):
        self.pending += text
        lines = self.pending.split('\n')
        self.pending = lines.pop()
        for line in lines:
            if line.startswith('[UKF_PROGRESS_PASS] '):
                self.total = int(line.split()[-1])
                self.completed = {}
                self.pass_number += 1
                self.stage = 'GPU tracking'
            match = re.search(r'\[GPU split\].* of (\d+)', line)
            if match:
                total = int(match.group(1))
                if self.stage != 'GPU tracking':
                    self.completed = {}
                self.total = total
                self.stage = 'GPU tracking'
            if line.startswith('[UKF_PROGRESS] '):
                event = json.loads(line[len('[UKF_PROGRESS] '):])
                self.completed[event['device']] = event['completed']
            if '[phase] load files:' in line:
                self.stage = 'Generating seeds'
            if '[phase] tractography:' in line:
                self.stage = 'Preparing fibers'
            if '[phase] vtk write:' in line:
                self.stage = 'Importing result'
        return dict(stage=self.stage, completed=min(sum(self.completed.values()), self.total), total=self.total, pass_number=self.pass_number)


if __name__ == '__main__':
    import json
    print(json.dumps(query_cuda_devices(sys.argv[1])))
