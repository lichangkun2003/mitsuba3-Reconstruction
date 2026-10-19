重建初始化的源码 CUDA 预览
=========================

``experiments/reconstruction/preview.py`` 读取已有数据集的 PLY 与 NeRF 相机，
使用源码新增的 ``mi.VoxelReconstruction`` 初始化体素椭球，并渲染一张图。
保留这一现有接口名称与当前算法；这个独立的 Mitsuba C++ 实现对应 Falcor
中 ``VoxelReconstructionNoLightTransport`` pass 的初始化与前向流程；
Falcor 中旧的 ``VoxelReconstruction`` 光传输 pass 不属于这次移植范围。
当前入口用于检查数据坐标、初始化和前向合成，尚未执行损失、反向梯度、
Adam、拓扑更新或 coarse-to-fine 训练。普通 Cornell box 的
``experiments/reconstruction/tools/smoke_gpu.py`` 仍然是独立的基础环境检查。

多视角训练、完整模型保存/恢复及 train/test 多 spp 评价见
:doc:`reconstruction_training`。预览入口继续只执行初始化和前向检查。

编译与运行
----------

在 VS Code 工作区 ``reconstruction`` 的 PowerShell 中运行：

.. code-block:: powershell

   cmake --build .\mitsuba\build --config Release --parallel 2
   if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\preview.py

默认使用 ``datasets/lego/init_points.ply``、
``datasets/lego/transforms_train.json`` 中第 0 个相机，800 × 800、spp=1、
seed=42、grid_resolution=256。使用工作区已有数据，不会下载数据集。

相对路径始终以仓库父目录为基准，与终端当前目录无关：

.. code-block:: text

   reconstruction/
     mitsuba/
     datasets/
       lego/init_points.ply
       lego/transforms_train.json
       lego/train/r_0.png
     envs/mitsuba/
     outputs/mitsuba/reconstruction-preview/

例如显式指定 lego，并在像素中心生成可固定比较的射线：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\preview.py `
     --dataset datasets/lego --camera-index 0 --pixel-sampling center `
     --output-dir outputs/mitsuba/lego-preview

``--build-dir`` 也相对于工作区解释；Windows 默认 ``mitsuba/build/Release``，
其他系统默认 ``mitsuba/build``。它必须指向含 ``python/mitsuba`` 与
``python/drjit`` 的完整源码运行目录，库导入位置会再次检查。
没有源码产物或没有新绑定时直接报错，不使用 venv 中的 wheel 回退。

GPU 选择由当前运行环境决定：

* 未检测到 Slurm 作业时，``--gpu`` 接收本机 ``nvidia-smi`` 的物理编号，
  默认 0。脚本先查询 UUID，再在导入 Mitsuba/Dr.Jit 前设置本进程
  ``CUDA_VISIBLE_DEVICES``。
* 存在 ``SLURM_JOB_ID`` 或 ``SLURM_JOBID`` 时，原样保留调度器提供的
  ``CUDA_VISIBLE_DEVICES`` 与作业环境变量，忽略本地 ``--gpu`` /
  ``runtime.gpu`` 设置并在日志中注明，不调用 ``nvidia-smi`` 按物理编号
  重新选卡。``CUDA_VISIBLE_DEVICES`` 必须是有效的非空数字编号、GPU UUID
  或 MIG 标识列表；缺失、空串、``-1``、``NoDevFiles`` 或非法值均明确
  报错，不回退到其他 GPU 或 CPU。验证时保留原始字符串，不重写其内容。

Slurm 下如果存在 ``SLURM_CPUS_PER_TASK``，它必须是正整数；未提供时
保守使用 1 个 CPU 线程。脚本在导入 Dr.Jit 后立即调用
``dr.set_thread_count()``，再导入 Mitsuba/NumPy，避免使用超出作业分配
的 Dr.Jit CPU 线程。线程限制不改变 GPU 渲染模式。

两种情况都要求实际模式为 ``cuda_ad_rgb``，结果必须位于 CUDA 设备。
Slurm 下根据 DLPack 验证设备属于进程可见设备范围，进程内逻辑编号
不等同于服务器物理编号。多 GPU 分配时也仅在 Dr.Jit 所选的一个可见 GPU
上执行前向计算；本脚本尚无多 GPU 加速。缓存保存在输出目录的 ``cache/`` 中。

在 HPC 上使用同一脚本
--------------------

服务器源码目录为 ``/home/changkun.li/projects/reconstruction/mitsuba``，
环境为 ``/home/changkun.li/envs/mitsuba``。同步源码后，沿用已有增量编译
命令：

.. code-block:: bash

   cmake --build "$HOME/projects/reconstruction/mitsuba/build" --parallel 8

编译成功后，在已经分配的 GPU 计算节点或交互式 ``srun`` shell 中执行：

.. code-block:: bash

   cd "$HOME/projects/reconstruction"
   "$HOME/envs/mitsuba/bin/python" -I mitsuba/experiments/reconstruction/preview.py \
     --dataset datasets/lego --camera-index 0 \
     --output-dir outputs/mitsuba/lego-preview

已有 GPU 分配时无需为这条命令再次申请资源。也可以先执行
``conda activate "$HOME/envs/mitsuba"``，再使用该环境的 ``python -I``
运行同一入口；直接调用上述解释器路径不依赖 shell 激活状态。

Linux 默认源码运行目录为 ``mitsuba/build``，Windows 为
``mitsuba/build/Release``，可用 ``--build-dir`` 显式覆盖。两者都显式
加载并核对源码版 Mitsuba/Dr.Jit，没有预编译 wheel 回退。

``datasets/lego`` 在服务器解析为
``/home/changkun.li/projects/reconstruction/datasets/lego``，输出目录也
相对于同一个 ``reconstruction`` 父目录。这些数据文件需要已经放置在
服务器相应位置；脚本不传输或下载数据。共享配置继续使用相对路径。

这里记录的是待服务器执行的用法；本次没有登录 HPC，也没有确认该节点的
编译、GPU 驱动或实际预览成功。

统一参数入口
------------

脚本开头的 ``CONFIG = PreviewConfig()`` 是统一入口，参数按职责分组：

* ``data``：数据集相对路径、PLY 文件、相机 JSON、第几个相机。
* ``model``：分辨率、包围盒、padding、PLY opacity 覆盖阈值、SH 系数数目。
* ``render``：图像大小、spp、seed、像素采样方式、裁剪面、背景、
  透射率终止阈值及每射线命中上限。
* ``output``：输出目录，是否保存完整初始化参数快照。
* ``runtime``：源码构建目录及本机 GPU 编号；Slurm 下 GPU 编号由调度器决定。

可直接改 dataclass 默认值，也可用 JSON 文件覆盖；CLI 参数最后覆盖。
JSON 只需填写有变化的字段，例如：

.. code-block:: json

   {
     "data": {"dataset": "datasets/lego", "camera_index": 0},
     "model": {"grid_resolution": 256},
     "render": {"width": 800, "height": 800, "spp": 1,
                "seed": 42, "pixel_sampling": "center"},
     "output": {"directory": "outputs/mitsuba/lego-preview"}
   }

保存到工作区的 ``preview-config.json`` 后运行：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\preview.py `
     --config preview-config.json

PLY 与相机文件相对于 ``data.dataset``，相机 JSON 中的图像路径也相对于
该数据集；缺少后缀时补 ``.png``。共享 JSON 配置使用相对路径，无需写入
本机 Windows 路径或服务器部署路径。
输出 ``config.json`` 保留用户请求的相对路径，报告 ``resolved`` 字段
另存实际绝对路径用于诊断。

与 Falcor 的坐标和采样约定
-------------------------

默认包围盒是 ``[-1.6, 1.6]^3``，以中心扩张 1.02 后实际网格为
``[-1.632, 1.632]^3``。这区分手动场景边界和初始化实际网格；原实现
并非直接使用未经扩张的 3.2 宽度。

相机沿用 ``LossPass.cpp``：NeRF/Blender camera-to-world 的第 3 列是
位置，第 1 列是上方，第 2 列取负是 forward，随后将世界坐标
``(x, y, z)`` 转换为 ``(x, z, -y)``。水平 FOV 取 ``camera_angle_x``。
图像原点在左上，射线起点位于 near=0.1 的近裁面。far=1000，射线
``maxt`` 保存近到远的距离；进入网格后保留 Falcor 的
``1e-6 * near_to_far_distance`` 微移。它会影响处在体素边界附近的射线。

``shared_jitter`` 对每个 sample 使用全图同一随机子像素位置，匹配
Falcor 非训练预览的采样布局；这里的 NumPy PCG64 与 Falcor 随机生成器
不同，不能据相同 seed 宣称逐像素一致。``center`` 固定为 (0.5, 0.5)，
供双方固定射线比较；spp 大于 1 时会重复同样射线。本预览没有接入训练时
逐像素 contour importance sampling。

默认原始背景为黑色，与 Falcor 的 ``mClearColor`` 初值一致。另存白底
合成图，使用线性 RGB 中的 ``foreground + (1-alpha)``；参考 RGBA 也先
在线性空间做白底合成。PNG 是显示用的 sRGB，原始浮点数据供数值比较。
参考图保持原始分辨率，不会偷偷缩放；改变渲染分辨率后报告记录两者大小。

输出与验证范围
--------------

默认输出到 ``outputs/mitsuba/reconstruction-preview``：

* ``preview.png``：配置背景下的渲染图。
* ``preview-white.png``：用于查看初始化形状和颜色的白底合成图。
* ``alpha.png``：线性 alpha 可视化，无 sRGB 曲线。
* ``reference-white.png``：原始参考 RGBA 在线性空间白底合成后转 sRGB。
* ``render.npz``：``rgb_linear``、``alpha``、``hit_count_sum``、
  ``rgb_white_linear``、``pixel_offsets``、``reference_rgba_linear``。
  多 spp 时 RGB/alpha 是平均值，hit_count_sum 是样本命中次数之和。
* ``config.json``：展开后的统一参数。
* ``preview-report.json``：解释器与库路径/版本、CUDA 设备、数据集选择、
  实际网格边界、初始化统计、耗时、检查结果和输出 SHA-256。
  ``provenance`` 记录本仓库 Git HEAD、dirty 状态、关键实现文件的相对路径
  与 SHA-256，以及能够确认的实际 variant 扩展文件路径与 SHA-256。
  Git 不可用时记录错误，不影响前向测试；源码哈希代表启动时的工作区，
  不能替代修改 C++ 后重新编译。
  ``gpu.selection`` 区分 ``local`` 与 ``slurm``；
  ``gpu.cuda_visible_devices`` 在 Slurm 下保存原始、未经改写的环境变量，
  本地则记录选中 GPU 的 UUID。Slurm 报告的
  ``gpu.local_gpu_index_ignored`` 记录被忽略的本地编号，
  ``gpu.process_visible_device_index`` 记录通过 DLPack 检查的进程内设备编号。
  ``cpu_threads.requested`` 与 ``cpu_threads.actual`` 分别记录 Slurm
  请求上限和实际 Dr.Jit 线程数；本地保留默认线程配置，仅记录 ``actual``，
  没有 ``requested`` 字段。未完成相应阶段的失败报告可能缺少实际值。
* ``initial-model.npz``：仅 ``--save-model`` 时输出，用于比较初始化参数。
  数组为展平 AoS，rotation 顺序为 wxyz，SH 数目见 config；也包含在报告
  的输出文件 SHA-256 清单中。

渲染后通过 ``dr.eval`` 与 ``dr.sync_thread`` 等待 GPU 完成，验证图像
形状、CUDA 设备、有限数值、alpha 范围及至少一条有效命中。不强制 RGB
非全黑：Gaussian PLY 或没有有效 RGB 的输入在原算法中可能初始化为黑色
辐射，此时 alpha 与命中图仍可验证几何；报告会保留 nonblack 信息。

成功必须同时满足进程退出码为 0 和本次报告 ``status=passed``。失败会
记录异常但不删除旧图片，因此不能仅根据输出文件存在判断成功。当前
预览通过不代表 Falcor 数值等价、梯度正确、训练收敛或 HPC 已验证。

前向与初始化回归检查
--------------------

``experiments/reconstruction/tests/test_reconstruction.py`` 使用生成的小型输入检查原生初始化
和前向行为。这个回归入口仍按本机物理 GPU 编号选卡，仅用于本地检查，
尚未接入上述 Slurm 环境规则；不要直接在 HPC 作业中运行它。
预览、训练和评价入口共用 Slurm 环境规则；这个旧数值回归脚本仍仅用于本地。
先完成源码编译，再在本地工作区父目录运行 CUDA 测试：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\tests\test_reconstruction.py `
     --build-dir .\mitsuba\build\Release --variant cuda_ad_rgb

这个测试脚本的 ``--build-dir`` 与 ``--output-dir`` 相对于终端当前目录，
与预览脚本固定相对于工作区解析的规则不同；上述命令假定当前目录包含
``mitsuba``、``envs`` 和 ``outputs``。

需要额外检查 CPU 标量实现时，独立运行：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\tests\test_reconstruction.py `
     --build-dir .\mitsuba\build\Release --variant scalar_rgb

默认报告位于 ``outputs/reconstruction-tests/native-cuda_ad_rgb.json`` 和
``native-scalar_rgb.json``。CPU 结果仅作为辅助；CUDA 使用独立的设备与
后端验证，CUDA 失败不会回退 CPU，也不能用 CPU 通过代替 GPU 通过。

运行环境规则的单元测试
----------------------

``experiments/reconstruction/tests/test_reconstruction_runtime.py`` 使用 Python 标准库
``unittest`` 模拟环境变量、进程调用、Dr.Jit 线程接口和 DLPack 设备。
它检查本地与 Slurm 的选卡、环境保留、非法输入拒绝、线程上限及设备范围，
不导入 Mitsuba/Dr.Jit/NumPy，不使用真实 GPU，也不渲染。

在本地 ``reconstruction`` 工作区目录运行：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\tests\test_reconstruction_runtime.py

服务器可使用已有环境的解释器执行同一项纯标准库测试，无需为它申请 GPU：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I \
     "$HOME/projects/reconstruction/mitsuba/experiments/reconstruction/tests/test_reconstruction_runtime.py"

该测试验证模拟输入下的运行环境处理逻辑。通过不等同于真实 Slurm GPU
分配、驱动、源码构建或服务器渲染通过；实际验证仍需在获配 GPU 的作业中
执行预览，并检查本次 ``preview-report.json``。
