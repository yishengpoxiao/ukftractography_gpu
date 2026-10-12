"""Run inside Slicer to install the offline, version-adaptive UKFGPU bundle."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile


def platform_key(system=None, machine=None):
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system not in ('Linux', 'Windows') or machine not in ('amd64', 'x86_64'):
        raise RuntimeError('UKFGPU requires Linux or Windows x86_64 with an NVIDIA CUDA GPU. '
                           'macOS and ARM backends are not available.')
    return ('linux' if system == 'Linux' else 'windows') + '-x86_64'


def verified_manifest(bundle):
    manifests = list(bundle.glob('manifest-*.json'))
    if len(manifests) != 1:
        raise RuntimeError('Expected one bundle manifest; extract the complete installer ZIP.')
    manifest = json.loads(manifests[0].read_text(encoding='utf-8'))
    if manifest['format'] != 1:
        raise RuntimeError('Unsupported installer manifest format.')
    payload = {p.relative_to(bundle).as_posix() for directory in ('module', 'runtimes', 'docs')
               for p in (bundle/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    payload.add('install.py')
    if payload != set(manifest['sha256']):
        raise RuntimeError('Installer payload does not match its manifest.')
    for name, expected in manifest['sha256'].items():
        path = bundle / name
        if not path.resolve().is_relative_to(bundle.resolve()):
            raise RuntimeError('Invalid bundle path: ' + name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Incomplete or modified installer file: ' + name)
    return manifest


def select_runtime(bundle, manifest, system=None, machine=None):
    key = platform_key(system, machine)
    if key not in manifest['runtimes']:
        raise RuntimeError('This bundle has no native ' + key + ' backend. Available: '
                           + ', '.join(manifest['runtimes'])
                           + '. A Windows source kit is not a Windows runtime; obtain a native build.')
    requirements = manifest['runtimes'][key]
    if key.startswith('linux'):
        name, version = platform.libc_ver()
        required = requirements['glibc_min']
        if name != 'glibc' or tuple(map(int, version.split('.'))) < tuple(map(int, required.split('.'))):
            raise RuntimeError('This Linux backend requires glibc >= ' + required + '; found ' + name + ' ' + version)
    return bundle/'runtimes'/key


def prepare_archive(bundle, destination, version, revision, system=None, machine=None):
    """Create a native Extensions Manager archive without editing Slicer settings."""
    if version[0] != 5 or version[1] < 2:
        raise RuntimeError('This installer targets Slicer 5.2+ within the 5.x series.')
    manifest = verified_manifest(bundle)
    runtime = select_runtime(bundle, manifest, system, machine)
    slicer_version = '%d.%d' % tuple(version)
    with tempfile.TemporaryDirectory(prefix='UKFGPU-package-') as tmp:
        extension = Path(tmp)/'UKFGPU'
        scripts = extension/('lib/Slicer-' + slicer_version)/'qt-scripted-modules'
        shutil.copytree(bundle/'module', scripts, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(runtime, scripts/'Runtime')
        if platform_key(system, machine).startswith('linux'):
            for path in (scripts/'Runtime/bin').iterdir():
                path.chmod(0o755)
        share = extension/('share/Slicer-' + slicer_version)
        share.mkdir(parents=True)
        shutil.copytree(bundle/'docs', share/'UKFGPU')
        metadata = ('scm git\nscmurl https://github.com/yishengpoxiao/ukftractography_gpu\n'
                    'scmrevision {commit}\nslicer_revision {revision}\ndepends NA\n'
                    'homepage https://github.com/yishengpoxiao/ukftractography_gpu\n'
                    'category Diffusion.Tractography\ncontributors UKFTractography authors; UKFGPU maintainers\n'
                    'description Experimental GPU UKF tractography with bundled native dependencies.\n'
                    'status Experimental\nenabled 1\n').format(commit=manifest['source_commit'], revision=revision)
        (share/'UKFGPU.s4ext').write_text(metadata, encoding='utf-8')
        with destination.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as archive:
            archive.add(extension, arcname='UKFGPU')
    return destination


def preflight(bundle, runtime, slicer):
    """Run native checks out of process; never load CUDA into Slicer's GUI."""
    import sys
    sys.path.insert(0, str(bundle/'module'))
    from UKFGPULib import process_environment
    executable = runtime/'bin'/('UKFTractography_gpu.exe' if os.name == 'nt' else 'ukf-gpu')
    env = process_environment(slicer.util.startupEnvironment(), str(executable))
    result = subprocess.run([str(executable), '--help'], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=30)
    if result.returncode:
        raise RuntimeError('Native dependencies could not load:\n' + result.stdout.decode(errors='replace'))
    python = Path(slicer.app.slicerHome)/'bin'/('PythonSlicer.exe' if os.name == 'nt' else 'PythonSlicer')
    result = subprocess.run([str(python), str(bundle/'module/UKFGPULib/__init__.py'), str(executable)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    if result.returncode:
        return 'Backend libraries loaded. CUDA check failed; install/update the NVIDIA driver before tracking.\n' + result.stdout.decode(errors='replace')
    devices = json.loads(result.stdout)
    return 'Backend libraries loaded; %d visible CUDA GPU(s).' % len(devices)


def install_in_slicer(bundle):
    import slicer
    import uuid
    if not hasattr(slicer, 'app'):
        raise RuntimeError('Run install.py inside 3D Slicer, using --python-script or the Python console.')
    version = (slicer.app.majorVersion, slicer.app.minorVersion)
    if version[0] != 5 or version[1] < 2:
        raise RuntimeError('This installer targets Slicer 5.2+ within the 5.x series.')
    manifest = verified_manifest(bundle)
    runtime = select_runtime(bundle, manifest)
    # ZIP extraction tools do not consistently preserve executable permissions.
    if os.name != 'nt':
        for path in (runtime/'bin').iterdir():
            path.chmod(0o755)
    status = preflight(bundle, runtime, slicer)
    packages = bundle/'packages'
    packages.mkdir(exist_ok=True)
    archive = packages/('UKFGPU-Slicer%d.%d-%s-%s.tar.gz' % (*version, platform_key(), uuid.uuid4().hex[:8]))
    manager = slicer.app.extensionsManagerModel()
    prepare_archive(bundle, archive, version, manager.slicerRevision)
    if manager.isExtensionInstalled('UKFGPU'):
        message = ('UKFGPU is already installed. The new matching archive is ready:\n' + str(archive)
                   + '\nRemove the old UKFGPU in Extensions Manager, restart, and install this archive from file.')
    else:
        if not manager.installExtension(str(archive)):
            raise RuntimeError('Extensions Manager rejected the archive: ' + str(archive))
        message = 'UKFGPU installed. Restart Slicer, then open Diffusion > Tractography > UKF GPU Tractography.'
    print(status + '\n' + message, flush=True)
    slicer.util.infoDisplay(status + '\n\n' + message, windowTitle='UKFGPU installer')
    return archive


if __name__ == '__main__':
    try:
        install_in_slicer(Path(__file__).resolve().parent)
    except Exception as exc:
        import slicer
        if hasattr(slicer, 'util'):
            slicer.util.errorDisplay(str(exc), windowTitle='UKFGPU installation failed')
        raise
