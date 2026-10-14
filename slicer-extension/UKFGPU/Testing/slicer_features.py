"""Run in real Slicer with the current backend to validate the interactive features."""
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import qt
import slicer
from vtk.util.numpy_support import vtk_to_numpy
from UKFGPU import UKFGPULogic


def wait_gpu(widget):
    slicer.app.processEvents()
    deadline = time.monotonic()+30
    while widget.gpuProbe is not None:
        slicer.app.processEvents()
        assert time.monotonic()<deadline
        time.sleep(0.01)


def wait_job(widget):
    deadline = time.monotonic()+120
    while widget.logic.process is not None:
        slicer.app.processEvents()
        assert time.monotonic()<deadline
        time.sleep(0.01)
    assert widget.result.currentNode() is not None, widget.status.text
    assert widget.progressBar.value == 100, widget.status.text


try:
    repo=Path(os.environ['UKF_TEST_REPOSITORY'])
    output=Path(os.environ['UKF_TEST_OUTPUT']);output.mkdir(parents=True,exist_ok=True)
    widget=slicer.modules.ukfgpu.widgetRepresentation().self()
    deadline=time.monotonic()+30
    while not widget.gpuDetected:
        slicer.app.processEvents();assert time.monotonic()<deadline,widget.gpuStatus.text;time.sleep(0.01)
    wait_gpu(widget)
    assert len(widget.gpuDevices)>=2
    assert all(d['uuid'].startswith('GPU-') and d['free_memory'] is not None for d in widget.gpuDevices.values())
    widget.recommendGpu()
    assert widget.gpuDevices[widget.gpuDevice()]['free_memory']==max(d['free_memory'] for d in widget.gpuDevices.values())
    print('PASS runtime-independent CUDA discovery, free memory and GPU recommendation',flush=True)
    data=repo/'UKFTractography/Data/Input'
    dwi=slicer.util.loadVolume(str(data/'two_tensor.nhdr'))
    mask=slicer.util.loadLabelVolume(str(data/'mask.nhdr'))
    seeds=slicer.util.loadLabelVolume(str(data/'seed.nhdr'))
    UKFGPULogic.validateInputs(dwi,mask,seeds,'1')
    original=mask.GetOrigin()
    mask.SetOrigin(original[0]+1,original[1],original[2])
    try:
        UKFGPULogic.validateInputs(dwi,mask,seeds,'1')
        raise AssertionError('Mismatched geometry accepted')
    except ValueError:pass
    mask.SetOrigin(original)
    try:
        UKFGPULogic.validateInputs(dwi,mask,seeds,'999')
        raise AssertionError('Empty seed selection accepted')
    except ValueError:pass
    assert widget.parameters['stoppingFA'].maximum==1
    assert widget.parameters['maxBranchingAngle'].maximum==90
    print('PASS aligned inputs, geometry rejection, seed validation and parameter ranges',flush=True)
    widget.dwi.setCurrentNode(dwi);widget.mask.setCurrentNode(mask);widget.seeds.setCurrentNode(seeds)
    widget.model.currentIndex=widget.model.findText('Simple2T')
    widget.precision.currentIndex=1
    widget.records['recordFA'].checked=True
    widget.records['recordNMSE'].checked=True
    widget.records['recordState'].checked=True
    widget.records['recordCovariance'].checked=True
    widget.fiberName.text='Feature single'
    widget.device.currentIndex=widget.device.count-1
    widget.run();wait_job(widget)
    single=widget.result.currentNode()
    assert '[UKF_PROGRESS]' in widget.logic.lastLog
    assert widget.logic.progress.total>0
    points=vtk_to_numpy(single.GetPolyData().GetPoints().GetData()).copy()
    lines=vtk_to_numpy(single.GetPolyData().GetLines().GetData()).copy()
    assert slicer.util.saveNode(single,str(output/'single.vtk'))
    widget.multiGpu.checked=True
    desired=[widget.gpuList.item(1).data(qt.Qt.UserRole),widget.gpuList.item(widget.gpuList.count-1).data(qt.Qt.UserRole)]
    for i in range(widget.gpuList.count):
        item=widget.gpuList.item(i)
        item.setCheckState(qt.Qt.Checked if item.data(qt.Qt.UserRole) in desired else qt.Qt.Unchecked)
    widget.fiberName.text='Feature multi'
    widget.run()
    env=widget.logic.process.processEnvironment()
    assert env.value('UKF_GPU_DEVICES')=='0,1'
    assert len(env.value('CUDA_VISIBLE_DEVICES').split(','))==2
    wait_job(widget)
    multi=widget.result.currentNode()
    assert '-> device 0,' in widget.logic.lastLog and '-> device 1,' in widget.logic.lastLog
    assert np.array_equal(vtk_to_numpy(multi.GetPolyData().GetLines().GetData()),lines)
    assert np.linalg.norm(vtk_to_numpy(multi.GetPolyData().GetPoints().GetData())-points,axis=1).max()<=0.001
    assert len(json.loads(multi.GetAttribute('UKFGPU.GPUUUIDs')))==2
    assert widget.logic.progress.completed and sum(widget.logic.progress.completed.values())==widget.logic.progress.total
    print('PASS actual single/multi-GPU tracking, explicit UUID device set, progress and output ordering',flush=True)
    widget.opacity.value=0.4;widget.lineWidth.value=3
    for mode in ('Direction','FA','Length','Solid'):
        widget.colorMode.currentIndex=widget.colorMode.findText(mode)
        assert multi.GetAttribute('UKFGPU.ColorMode')==mode,widget.status.text
        if mode in ('Direction','Length'):
            assert multi.GetPolyData().GetPointData().GetArray('UKFGPU.'+mode) is not None
    assert abs(multi.GetDisplayNode().GetOpacity()-0.4)<1e-6
    assert multi.GetDisplayNode().GetLineWidth()==3
    widget.showFibers.checked=False
    assert not any(multi.GetNthDisplayNode(i).GetVisibility() for i in range(multi.GetNumberOfDisplayNodes()))
    widget.showFibers.checked=True
    assert any(multi.GetNthDisplayNode(i).GetVisibility() for i in range(multi.GetNumberOfDisplayNodes()))
    print('PASS solid/direction/FA/length coloring, opacity, line width and visibility',flush=True)
    widget.fiberName.text='Saved scene settings 测试'
    widget.parameters['stepLength'].value=0.2
    widget.parameters['recordLength'].value=0.6
    widget.saveParameters()
    settings=json.loads(widget.parameterNode.GetParameter('Settings'))
    assert settings['gpuPci']==desired
    scene=output/'features.mrb';assert slicer.util.saveScene(str(scene))
    slicer.mrmlScene.Clear(0)
    slicer.util.loadScene(str(scene))
    wait_gpu(widget)
    assert widget.fiberName.text=='Saved scene settings 测试'
    assert widget.parameters['stepLength'].value==0.2
    assert widget.parameters['recordLength'].value==0.6
    assert widget.multiGpu.checked and widget.checkedGpuPci()==desired
    assert widget.dwi.currentNode() is not None and widget.mask.currentNode() is not None and widget.result.currentNode() is not None
    assert widget.opacity.value==0.4 and widget.lineWidth.value==3
    print('PASS scene save/clear/reload restores inputs, result, parameters, styles and physical GPUs',flush=True)
    slicer.mrmlScene.Clear(0)
    slicer.app.processEvents()
    assert not widget.sceneRestoring and widget.parameterNode is not None
    widget.fiberName.text='New scene settings'
    widget.saveParameters()
    assert json.loads(widget.parameterNode.GetParameter('Settings'))['fiberName']=='New scene settings'
    print('PASS an empty new scene resumes parameter saving',flush=True)
    slicer.app.exit(0)
except Exception:
    traceback.print_exc();slicer.app.exit(1)
