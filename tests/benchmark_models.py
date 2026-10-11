#!/usr/bin/env python3
"""Measure CPU/GPU end-to-end and tracking time for a fixed real-DWI seed subset."""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=512)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--models', nargs='*')
    parser.add_argument('--binary', default='../build-cpu/UKFTractography')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = Path('UKFTractography/Data/Input')
    text = (data/'dwi-mask.nhdr').read_text()
    shape = tuple(map(int, re.search(r'^sizes: (.+)$', text, re.M)[1].split()))
    mask = np.frombuffer(gzip.decompress((data/'dwi-mask.raw.gz').read_bytes()),dtype='<i2').reshape(shape[::-1])>0
    interior = mask.copy()
    interior[[0,-1],:,:] = False
    interior[:,[0,-1],:] = False
    interior[:,:,[0,-1]] = False
    for axis in range(3):
        interior &= np.roll(mask,1,axis) & np.roll(mask,-1,axis)
    random = np.random.default_rng(20261010)
    candidates = np.flatnonzero(interior)
    seed = np.zeros(mask.size,dtype='<i2')
    seed[random.choice(candidates,args.seeds,replace=False)] = 1
    (args.output/'seeds.raw.gz').write_bytes(gzip.compress(seed.tobytes()))
    header = re.sub(r'^data file: .+$','data file: seeds.raw.gz',text,flags=re.M)
    (args.output/'seeds.nhdr').write_text(header)
    cases = []
    for n in (1,2,3):
        for full in (False,True):
            for fw in (False,True) if n<3 else (False,):
                name=('full' if full else 'simple')+str(n)+('-fw' if fw else '')
                flags=(['--fullTensorModel'] if full else [])+(['--freeWater','--recordFreeWater'] if fw else [])
                cases.append((name,n,flags+['--recordFA','--recordTrace']))
    for n in (1,2):
        cases.append(('noddi'+str(n),n,['--noddi','--recordVic','--recordKappa','--recordViso']))
    results=[]
    for name,n,flags in cases:
        if args.models and name not in args.models:
            continue
        row={'model':name}
        common=['--dwiFile',str(data/'dwi.nhdr'),'--maskFile',str(data/'dwi-mask.nhdr'),
                '--seedsFile',str(args.output/'seeds.nhdr'),'--seedsPerVoxel','1','--numTensor',str(n),
                '--numThreads',str(args.threads),'--maxHalfFiberLength','80',
                '--minBranchingAngle','0','--maxBranchingAngle','0','--recordNMSE']+flags
        for mode in ('cpu','gpu'):
            extra=[] if mode=='cpu' else ['--useGPU','--gpuDevice','0','--gpuPrecision','0']
            start=time.monotonic()
            run=subprocess.run([args.binary]+common+extra+['--tracts',str(args.output/f'{name}-{mode}.vtk')],
                               stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600)
            wall=time.monotonic()-start
            (args.output/f'{name}-{mode}.log').write_text(run.stdout)
            if run.returncode or (mode=='gpu' and ('falling back' in run.stdout or 'tractography done on device' not in run.stdout)):
                raise RuntimeError(f'{name}/{mode} failed; see log')
            match=re.search(r'\[phase\] tractography: ([\d.]+) s',run.stdout)
            row[mode]={'wall_s':wall,'tracking_s':float(match[1])}
        row['wall_speedup']=row['cpu']['wall_s']/row['gpu']['wall_s']
        row['tracking_speedup']=row['cpu']['tracking_s']/row['gpu']['tracking_s']
        results.append(row)
        (args.output/'timings.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(row),flush=True)


if __name__ == '__main__':
    main()
