# Slicer GPU 扩展验证（2026-10-11）

已新增独立的 `UKFGPU` scripted extension，复用现有 GPU 后端，无需把 CUDA、ITK、VTK 链接进 Slicer 进程。原版扩展及算法源码没有因本次接入而改变。

## 交付

- `slicer-extension/UKFGPU/UKFGPU.py`：节点选择、12 模型、精度/设备、跟踪/记录参数、异步 QProcess、取消、日志、输出显示。
- `slicer-extension/UKFGPU/UKFGPULib`：命令构造、隔离运行环境、输出状态检查；所有参数通过 argv 传递。
- `scripts/package_slicer_extension.py`：打包后端及 CUDA runtime、移除构建机 RPATH，生成源码合集及带 `.s4ext` 的 Extensions Manager 安装包。
- `slicer-extension/CMakeLists.txt` 及模块 CMake：标准 Slicer scripted extension 构建入口；未修改原有配置文件。本机无 Slicer SDK，未验证 SDK 构建路线。
- 安装与使用见 [README](README.md)。发布目录为 `/data/yijie/ukf-GPU/bin/slicer-extension-2026-10-11/`。

包内启动器清理 Slicer 注入的动态库、Python、Qt 搜索路径，只为子进程选择私有 GPU/CUDA 库；保留 CUDA_VISIBLE_DEVICES。没有更改 Slicer 主进程的运行库。后端仍使用系统 glibc/libstdc++，当前符号要求分别到 GLIBC_2.29、GLIBCXX_3.4.26。

UKF 输出为 RAS、VTK 文件缺少坐标系标记，因此直接读取 polydata 到 MRML，避免 Slicer 通用模型读取器默认 LPS 造成轴翻转。输入使用 `slicer.util.exportNode`，不会把原节点存储路径改成临时文件。

## 实测

验证目录：`/data/yijie/ukf-GPU/validation/slicer-2026-10-11/`。从官方下载并校验 SHA512 后，使用 Slicer 5.12.4 Linux x86_64（revision 34645）和 RTX 4090 测试。没有操作机器上其他已运行的 Slicer 会话，也没有往其扩展目录安装。安装测试使用独立的日期配置文件和扩展目录。

- `packaged-models.log`：打包后、移除 RPATH 的 GPU 程序，12 模型 × FP64/混合入口，24 组输出几何、连接、记录数组与上一轮 GPU 输出全部相等。
- `integration4.log`：实际注册模块控件；Simple2T mixed、Simple1T-FreeWater FP64、NODDI1F FP64 的 DWI 导出→GPU→结果加载通过；与已有 GPU 参考逐纤维连接一致、位置欧氏误差 ≤0.001 mm；记录数组存在且有限；取消和无效后端启动失败均清理临时目录。
- `install-test2.log`：用真实 `qSlicerExtensionsManagerModel.installExtension` 从安装包安装成功，扩展状态及模块目录正确。
- `installed-integration.log`：从管理器实际安装目录加载，在有主窗口的 Slicer 中测试 Run 按钮、日志、按钮状态恢复；三模型几何/数组、原 DWI 存储路径不变、取消、失败清理全部通过。
- `UKFGPU-Slicer.png`：实际模块及纤维显示截图。
- Python 语法编译及 `git diff --check` 通过。

开发过程中的 `integration.log`、`integration2.log`、`integration3.log`、`install-test.log` 保存了早期失败：PythonQt QByteArray 转换、无主窗口的视图重置、失败进程回调生命周期和 Qt 属性访问。已修复，并由上述最终测试验证；不把早期失败日志作为通过证据。

SlicerDMRI 的专用 FiberBundle 节点显示路径已实现，但本次仅实测 Slicer 自带 Model 节点显示，未宣称验证了 SlicerDMRI 集成。其他 Slicer 版本、Windows/macOS 和其他 GPU 架构未验证。新包没有上传在线扩展服务器。

以上是对界面/打包和既有 GPU 行为的验证，不是 CPU 全模型验收。本次未解除 Full Tensor、双张量 FreeWater、长 NODDI2F、退化 Simple3T、分支、全脑的已知严格精度失败。

## Git

origin 为 `https://github.com/pnlbwh/ukftractography`，当前分支 master，本地无 main；`git show-ref --verify refs/heads/main` 返回 `fatal: 'refs/heads/main' - not a valid ref`。另有此前未提交修改，工作区不满足本次交付之外干净的条件。因此没有提交或推送，全部修改保留。

官方参考：[Extensions](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html)、[Slicer util API](https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/util.py)、[扩展管理器安装实现](https://github.com/Slicer/Slicer/blob/main/Base/QTCore/qSlicerExtensionsManagerModel.cxx)。
