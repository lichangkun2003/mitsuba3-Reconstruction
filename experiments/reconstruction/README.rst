体素重建实验
============

当前单卡实验从 ``train.py`` 启动，统一参数在该文件顶部的 ``CONFIG`` 赋值区。
切换场景修改 ``CONFIG.data.dataset``，或传入 ``--dataset datasets/lego``；
命令行参数覆盖配置文件与默认配置。RTX 5090 Windows 主机使用这个单卡入口。

目录导航
--------

.. list-table::
   :header-rows: 1

   * - 文件或目录
     - 用途
   * - ``train.py``
     - 单卡训练、保存重建模型，完成后自动出图与评价。
   * - ``evaluate.py``
     - 单独加载已有 ``model.npz``，生成训练集／测试集视角及评价指标。
   * - ``preview.py``
     - 检查点云初始化和一个相机视角的前向结果，不执行训练。
   * - ``train_multigpu.py``
     - Linux Slurm 同节点双 GPU 训练／评价启动器，目前不支持 Windows。
   * - ``reconstruction_*.py``
     - 参考图、优化器、拓扑、checkpoint、性能记录等共用实现。
   * - ``tests/``
     - 数值与功能回归测试，不是日常实验入口。
   * - ``benchmarks/``
     - 保留旧 alpha 监督模式的前向与执行方式基准；当前 RGB-only 速度请用 ``train.py`` 短跑测量。
   * - ``tools/smoke_gpu.py``
     - 程序生成 Cornell box 的基础 CUDA 环境检查。

这些脚本调用本仓库的 C++ ``mi.VoxelReconstruction``。渲染与梯度核心仍在
``src/render/reconstruction.cpp``；目录整理没有将算法改成纯 Python 实现。
``resources/`` 中的原入口已迁移，请更新自己的运行命令。

Windows 新机器准备与源码编译
--------------------------

以下命令均在 **Mitsuba 仓库根目录** 执行，假设它名为 ``mitsuba``。
相邻目录结构保持为：

.. code-block:: text

   reconstruction/
     mitsuba/                 # 本源码仓库
     envs/mitsuba/            # 此机器的独立 Python 环境
     datasets/lego/           # init_points.ply、transforms_train/test.json 及图片
     outputs/                # 各次实验结果

需要 Git、64 位 Python 3.12、CMake，以及包含 C++ 工具和 Windows SDK 的
Visual Studio 2022。使用支持该 GPU 的 NVIDIA 驱动；本说明不代表已经在 RTX 5090
机器上验证。无需为这条构建流程预先安装 CUDA Toolkit。
不要复制其他机器的 ``envs`` 或 ``build``，它们包含本机解释器与构建路径。

已有独立环境先检查后复用；没有时创建。依赖安装明确指向该环境：

.. code-block:: powershell

   if (-not (Test-Path '..\envs\mitsuba')) {
       py -3.12 -m venv '..\envs\mitsuba'
       if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
   }
   $python = (Resolve-Path '..\envs\mitsuba\Scripts\python.exe').Path
   & $python -I -c "import sys; print(sys.executable); print(sys.version); assert sys.version_info[:2] == (3, 12); assert sys.prefix != sys.base_prefix"
   if ($LASTEXITCODE -ne 0) { throw 'Check the project environment first' }
   & $python -m pip install numpy
   if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
   git submodule update --init --recursive
   if ($LASTEXITCODE -ne 0) { throw 'Submodule initialization failed' }

从源码构建 Mitsuba 和配套 Dr.Jit，无需另外安装 Mitsuba wheel。
首次配置采用已有 Windows 构建选项：

.. code-block:: powershell

   cmake -S . -B build -G 'Visual Studio 17 2022' -A x64 `
     "-DPython_EXECUTABLE=$python" `
     '-DMI_DEFAULT_VARIANTS=scalar_rgb,cuda_ad_rgb' `
     -DMI_SPLIT_MODE=OFF -DMI_ENABLE_EMBREE=OFF
   if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
   cmake --build build --config Release --parallel 2
   if ($LASTEXITCODE -ne 0) { throw 'Source build failed' }

``MI_DEFAULT_VARIANTS`` 仅在首次生成 ``build/mitsuba.conf`` 时生效；已有构建应检查
该文件中的启用变体。``--parallel 2`` 是保守的内存用量，可按新主机可用内存增加。
以后 C++ 改动只需再次执行上述 ``cmake --build`` 增量编译命令。

运行脚本会显式加载 ``build/Release/python``，打印并检查 Mitsuba、Dr.Jit 路径与
``cuda_ad_rgb`` 后端。仅用 ``pip install mitsuba`` 不能获得本项目的重建接口。

单卡实验命令
------------

先短跑十轮，保留完整实验的 700 轮调度预算：

.. code-block:: powershell

   & '..\envs\mitsuba\Scripts\python.exe' -I .\experiments\reconstruction\train.py `
     --dataset datasets/lego --gpu 0 --epochs 700 --stop-after-epochs 10 `
     --chunk-size 0 --no-profile-stages

此短跑保存可续训 checkpoint，不发布最终 ``model.npz`` 或自动执行最终评价。
新开完整的 700 轮实验：

.. code-block:: powershell

   & '..\envs\mitsuba\Scripts\python.exe' -I .\experiments\reconstruction\train.py `
     --dataset datasets/lego --gpu 0 --epochs 700 --chunk-size 0 `
     --no-profile-stages --eval-spp 64 --eval-max-views 10

``--no-profile-stages`` 关闭分段诊断，使抽样视角也能使用原生重放，适合速度比较。
若需要定位前向与反向耗时，可重新开启 ``--profile-stages``。
短跑与完整命令分别创建新实验；若要接着短跑训练，使用其实际 checkpoint 路径传给
``--resume``，不要把短跑命令重复执行当作续训。

实验输出仍自动分配为 ``outputs/场景-MM.DD-E编号/``，其中 ``E`` 表示实验编号。
最终 ``model.npz``、训练日志与训练／测试视角评价结果存于同一次实验目录。
所有数据集、JSON 配置和输出相对路径仍以 **仓库的父目录** 为基准，
与终端当前目录、脚本迁移后的目录层级无关。

当前算法范围
------------

当前默认是 RGB-only 第一阶段：800 × 800、每视角 1 spp、opacity 仅优化 DC。
参考 alpha 只在读取时参与白底合成，之后不参与训练监督或筛选。
关闭生长、删除及其证据计算，保留 256 → 512 分裂。
最终出图为 64 spp，训练集与测试集各取前 10 个视角。

旧 alpha 监督 checkpoint 不能直接切换为 RGB-only 续训，应新开实验；旧模型仍可
单独评价。完整说明见 `训练与评价文档 <../../docs/reconstruction_training.rst>`_、
`初始化预览文档 <../../docs/reconstruction_preview.rst>`_ 和
`基础 CUDA 检查文档 <../../docs/local_gpu_smoke.rst>`_。
