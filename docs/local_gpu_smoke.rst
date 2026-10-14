Windows 本地独立环境与 GPU 冒烟测试
=================================

用途与验证边界
--------------

本节记录本地 Python 实验脚本和预编译包的基础 CUDA 渲染测试。
后面的“Windows 源码编译与上一级工作区”说明源码版的独立加载方式。
``pip`` 安装的 Mitsuba 不会自动包含本仓库中的 C++ 修改，安装包自带的
Python 模块也独立于本仓库。修改渲染器内部代码后，必须重新构建源码，
使用对应的源码构建环境验证，并再次核对 ``mitsuba.__file__``。
不能用这里的 wheel 测试结果替代源码构建或 HPC 正式实验验证。

2026-10-14 查询官方 PyPI（包含预发布版本）后，没有适配 Windows
Python 3.12 的 Mitsuba 3.10 开发版。实际选择 3.9.1 wheel，让它解析
配套 Dr.Jit，再将解析结果锁定到 ``requirements-local-smoke.txt``。
保存 PNG 的 Bitmap 转换需要另加 NumPy。

.. list-table:: 版本差异
   :header-rows: 1

   * - 组件
     - 本地实际安装
     - HPC / 源码目标
   * - Python
     - 3.12.3，64 位
     - 用户报告 HPC 为 3.12.15
   * - Mitsuba
     - 3.9.1，Windows wheel
     - 本地源码标记为 3.10.0.dev1；HPC 配置报告 3.10.0
   * - Dr.Jit
     - 1.5.0，由 Mitsuba 依赖选定
     - 1.6.0.dev1
   * - NumPy
     - 2.5.4
     - 本次没有检查 HPC 环境

没有安装 CUDA Toolkit、LLVM 或大型 C++ 工具链。HPC 首次构建的完成情况
尚未验证，本次没有登录或操作 HPC。

目录与隔离
----------

在仓库根目录执行下文的 PowerShell 命令。假设共享父目录包含：

.. code-block:: text

   reconstruction/
     mitsuba/                 # 当前源码仓库
     envs/mitsuba/            # 本项目 venv
     outputs/mitsuba/local-smoke/

所有安装命令都使用目标环境的 Python。venv 不启用系统 site-packages，
不向系统 Python 或其他项目环境安装项目依赖。安装时关闭 pip 缓存；
测试为 Dr.Jit、CUDA 和 OptiX 设置仓库外的缓存目录。

``.vscode/`` 已被仓库现有 ``.gitignore`` 忽略。机器路径仅保存在本地编辑器
配置、venv 和仓库外报告中。可以通过 Git 同步脚本、此文档和依赖文件；
不要提交环境、缓存、输出或用 Windows wheel 依赖替换 HPC 源码构建依赖。

创建或重建环境
--------------

已有可用环境直接复用。重建时先将旧环境移到备份位置，再执行以下命令；
不要对其他项目环境执行清理。这里使用本机已有的 Python 3.12：

.. code-block:: powershell

   $projectRoot = (Get-Location).Path
   $parentDirectory = Split-Path $projectRoot -Parent
   $environmentDirectory = Join-Path $parentDirectory 'envs\mitsuba'
   $python = Join-Path $environmentDirectory 'Scripts\python.exe'

   if (-not (Test-Path -LiteralPath $environmentDirectory)) {
       py -3.12 -m venv $environmentDirectory
       if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
   }
   & $python -I -c "import sys; print(sys.executable); print(sys.version); assert sys.version_info[:2] == (3, 12); assert sys.prefix != sys.base_prefix"
   if ($LASTEXITCODE -ne 0) { throw 'Check the existing environment first' }
   & $python -m pip --isolated --disable-pip-version-check --no-cache-dir install --only-binary=:all: --index-url https://pypi.org/simple -r requirements-local-smoke.txt
   if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
   & $python -m pip --isolated --disable-pip-version-check check

依赖文件锁定三个项目包的版本。Python 3.12 的补丁版本由解释器决定，
命令会打印它；首次测试使用的是 3.12.3。此锁定不保证不同驱动、GPU 或
软件版本之间的逐位一致性。

激活与运行
----------

激活是可选的，直接调用环境解释器最明确，也不受 PowerShell 激活脚本
执行策略影响。从仓库根目录可直接运行：

.. code-block:: powershell

   & '..\envs\mitsuba\Scripts\python.exe' -I .\resources\local_gpu_smoke.py --gpu 0

``-I`` 让此次 wheel 测试忽略 ``PYTHONPATH`` 和用户 site-packages。
不要将这个解释器和参数组合直接当作源码构建的验证命令。

如果希望激活后使用 ``python``：

.. code-block:: powershell

   & '..\envs\mitsuba\Scripts\Activate.ps1'
   python -I .\resources\local_gpu_smoke.py --gpu 0
   deactivate

若激活被执行策略拦截，使用上面的解释器完整路径命令即可，无需更改
系统执行策略。可用 ``--output-dir <目录>`` 指定其他输出目录。

VS Code
-------

本地 ``.vscode/settings.json`` 配置了此 venv 的
``python.defaultInterpreterPath``、环境搜索路径和终端自动激活。
``.vscode/tasks.json`` 提供 ``Mitsuba: local CUDA smoke test`` 任务，
直接使用该解释器，不依赖终端是否激活或编辑器缓存的解释器选择。
在命令面板运行 ``Tasks: Run Test Task`` 即可启动。

如果当前 VS Code 会话曾选择过另一个解释器，默认路径配置可能不会覆盖
已缓存的选择。此时使用 ``Python: Select Interpreter``，选中此项目
``envs/mitsuba/Scripts/python.exe``。本次没有通过 UI 验证状态栏的选择。

测试内容与实际结果
------------------

脚本只选择 ``cuda_ad_rgb``，失败时返回非零退出码，没有 CPU 回退。
它通过 ``nvidia-smi`` 查询所选 GPU 的 UUID，在导入 Mitsuba/Dr.Jit 前
设置当前进程的 ``CUDA_VISIBLE_DEVICES``，并校验输出张量的 DLPack
设备为 CUDA（类型 2，进程内设备 0）。``--gpu`` 是 nvidia-smi 中的
物理索引；这是单机测试脚本，不应直接覆盖 HPC 调度器提供的设备分配。

场景使用程序生成的 Cornell box，不下载数据集。配置固定为：

* 分辨率 64 × 64，RGB；spp = 4；seed = 42；max_depth = 4。
* 渲染后调用 ``dr.eval(image)`` 和 ``dr.sync_thread()`` 等待计算完成。
* 检查形状为 ``[64, 64, 3]``、全部数值有限且非全黑。
* 同步保存 sRGB PNG，并读回检查尺寸。
* 打印实际 Python 路径、库版本、模块路径和实际渲染模式。

首次本机检查实际发现的是 NVIDIA GeForce RTX 4060 Laptop GPU
（8188 MiB），驱动 610.62；用户确认使用这块 GPU 完成测试。
RTX 4090 未在本机检测到，也没有验证。

2026-10-14 的 CUDA 测试通过。输出目录包含：

* ``cornell-64x64.png``：实际渲染图像。
* ``gpu-smoke.json``：最新运行状态、版本、路径、硬件、数值检查及 PNG 哈希。
* ``run.log``：配置时保存的运行日志；单独重跑脚本主要更新 PNG 和 JSON。
* ``first-attempt.json``、``first-attempt.log``：首次缺少 NumPy 时的错误记录。
* 安装报告与 ``cache/``：仅供本地诊断。

首次运行已完成 CUDA 渲染，在 Bitmap 转换时因为缺少 NumPy 失败；补充
NumPy 后整个测试通过。导入时的 ``LLVM API initialization failed`` 提示
来自未安装的可选 LLVM CPU 后端，本次 CUDA 模式和 CUDA 设备检查均通过。
没有用 CPU 渲染结果代替 GPU 验证。

每次以最新 ``gpu-smoke.json`` 的 ``status`` 和进程退出码判断是否成功；
失败会记录异常，但不会清理旧 PNG。该测试只验证基础前向渲染和图像输出，
这项 wheel 测试不验证梯度、重建算法、C++ 修改、HPC 构建或跨版本数值一致性。

Windows 源码编译与上一级工作区
----------------------------

VS Code 打开共享父目录 ``reconstruction``，因此可以同时看到
``mitsuba``、``envs`` 和 ``outputs``。此节命令均在该父目录中执行，
与前文在仓库根目录执行的 wheel 命令相区别。
父目录的本地 ``.vscode`` 配置不在 Mitsuba Git 仓库中。

本机已有 Visual Studio 2022 Community 17.9、MSVC 19.39 和 CMake 3.29.3。
配置时只有约 3.4 GiB 可用物理内存，因此本地使用 2 路并行，不能照搬
服务器的 8 路并行作为本机默认值。首次配置命令是：

.. code-block:: powershell

   $python = (Resolve-Path '.\envs\mitsuba\Scripts\python.exe').Path
   cmake -S .\mitsuba -B .\mitsuba\build -G 'Visual Studio 17 2022' -A x64 `
     "-DPython_EXECUTABLE=$python" `
     '-DMI_DEFAULT_VARIANTS=scalar_rgb,cuda_ad_rgb' `
     -DMI_SPLIT_MODE=OFF -DMI_ENABLE_EMBREE=OFF
   if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }

``MI_SPLIT_MODE=OFF`` 同时配置 Mitsuba 和 Dr.Jit，避免额外的
``nanobind-backend`` 运行时依赖。关闭 Embree 仅让 CPU 模式使用内置求交
实现，GPU 模式仍使用 OptiX；这个本地构建配置不代表 HPC 配置。
不需要安装 CUDA Toolkit。CMake 提示缺少 NASM 时，JPEG SIMD 优化不可用，
这不影响本次 PNG/CUDA 测试。

``MI_DEFAULT_VARIANTS`` 仅在首次生成 ``build/mitsuba.conf`` 时生效。
以后若要修改变体，需要修改构建目录中的该文件并重新配置。

开始编译或在修改 C++ 后增量编译：

.. code-block:: powershell

   cmake --build .\mitsuba\build --config Release --parallel 2

编译成功后，运行同一个小场景脚本，但明确指定源码产物：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\local_gpu_smoke.py `
     --build-dir .\mitsuba\build\Release `
     --output-dir .\outputs\mitsuba\source-smoke --gpu 0

脚本的 ``--build-dir`` 参数接收包含 ``python/`` 的运行时目录。
它显式设置模块搜索路径及 Windows DLL 搜索目录，并要求 Mitsuba 和
Dr.Jit 都从指定构建目录加载。目录不完整或导入路径不符时直接失败，
不会悄悄使用 venv 中的 3.9.1/1.5.0 wheel。
因此这里可以保留 ``-I``：脚本在进程内部显式指定源码路径，而不是依赖
会被 ``-I`` 忽略的 ``PYTHONPATH``。已有 wheel 环境仍可独立使用。

打开输出图像：

.. code-block:: powershell

   code --reuse-window .\outputs\mitsuba\source-smoke\cornell-64x64.png

父目录工作区任务：

* ``Mitsuba: configure local source``：首次生成构建配置。
* ``Mitsuba: build local source``：默认构建任务，可用 ``Ctrl+Shift+B`` 启动。
* ``Mitsuba: source CUDA smoke test``：默认测试任务，先增量编译，再测试源码版。
* ``Mitsuba: wheel CUDA smoke test``：测试安装在 venv 中的预编译包。

源码测试报告与图片位于 ``outputs/mitsuba/source-smoke``；本地配置与构建
日志位于 ``outputs/mitsuba/local-build``。编译产物在已被 Git 忽略的
``mitsuba/build`` 内。实际使用版本、模块路径与成功状态以最新测试 JSON
为准，不能仅凭构建命令或图片文件存在来判断成功。

2026-10-14 已实际完成上述 Release 构建，源码版 GPU 测试通过：
Mitsuba 3.10.0.dev1、Dr.Jit 1.6.0.dev1 均从 ``build/Release/python``
加载，实际模式为 ``cuda_ad_rgb``，使用 RTX 4060 Laptop GPU。
64 × 64、spp = 4、seed = 42 的图像通过 CUDA 设备、尺寸、有限数值、
非全黑和 PNG 读回检查。此结果验证本次源码基线的基础前向渲染；
梯度、重建算法及 HPC 运行仍未验证。

服务器的增量编译命令保持用户提供的形式；本次不在服务器执行：

.. code-block:: bash

   cmake --build "$HOME/projects/reconstruction/mitsuba/build" --parallel 8

参考
----

* `Mitsuba 安装说明 <https://mitsuba.readthedocs.io/en/stable/>`_
* `Mitsuba 3.9.1 官方 PyPI <https://pypi.org/project/mitsuba/3.9.1/>`_
* `VS Code Python 设置说明 <https://code.visualstudio.com/docs/python/settings-reference>`_
