重建初始化的源码 CUDA 预览
=========================

``resources/reconstruction_preview.py`` 读取已有数据集的 PLY 与 NeRF 相机，
使用源码新增的 ``mi.VoxelReconstruction`` 初始化体素椭球，并渲染一张图。
它用于检查数据坐标、初始化和前向合成，尚未执行损失、反向梯度、Adam、
拓扑更新或 coarse-to-fine 训练。普通 Cornell box 的
``local_gpu_smoke.py`` 仍然是独立的基础环境检查。

编译与运行
----------

在 VS Code 工作区 ``reconstruction`` 的 PowerShell 中运行：

.. code-block:: powershell

   cmake --build .\mitsuba\build --config Release --parallel 2
   if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\reconstruction_preview.py

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

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\reconstruction_preview.py `
     --dataset datasets/lego --camera-index 0 --pixel-sampling center `
     --output-dir outputs/mitsuba/lego-preview

``--build-dir`` 也相对于工作区解释；Windows 默认 ``mitsuba/build/Release``，
其他系统默认 ``mitsuba/build``。它必须指向含 ``python/mitsuba`` 与
``python/drjit`` 的完整源码运行目录，库导入位置会再次检查。
没有源码产物或没有新绑定时直接报错，不使用 venv 中的 wheel 回退。

``--gpu`` 接收本机 ``nvidia-smi`` 的物理编号，默认 0。脚本先查询 UUID，
再在导入 Mitsuba/Dr.Jit 前设置本进程 ``CUDA_VISIBLE_DEVICES``。
实际模式必须为 ``cuda_ad_rgb``，结果必须位于 CUDA 设备。缓存保存在
输出目录的 ``cache/`` 中。这个入口是本地预览工具；检测到 Slurm 作业时
会拒绝覆盖调度器分配，HPC 入口需要另行接入作业环境。

统一参数入口
------------

脚本开头的 ``CONFIG = PreviewConfig()`` 是统一入口，参数按职责分组：

* ``data``：数据集相对路径、PLY 文件、相机 JSON、第几个相机。
* ``model``：分辨率、包围盒、padding、PLY opacity 覆盖阈值、SH 系数数目。
* ``render``：图像大小、spp、seed、像素采样方式、裁剪面、背景、
  透射率终止阈值及每射线命中上限。
* ``output``：输出目录，是否保存完整初始化参数快照。
* ``runtime``：源码构建目录及本机 GPU 编号。

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

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\reconstruction_preview.py `
     --config preview-config.json

PLY 与相机文件相对于 ``data.dataset``，相机 JSON 中的图像路径也相对于
该数据集；缺少后缀时补 ``.png``。配置与示例不包含机器特有绝对路径。
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

``resources/test_reconstruction.py`` 使用生成的小型输入检查原生初始化
和前向行为。先完成源码编译，再在工作区父目录运行 CUDA 测试：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\test_reconstruction.py `
     --build-dir .\mitsuba\build\Release --variant cuda_ad_rgb

这个测试脚本的 ``--build-dir`` 与 ``--output-dir`` 相对于终端当前目录，
与预览脚本固定相对于工作区解析的规则不同；上述命令假定当前目录包含
``mitsuba``、``envs`` 和 ``outputs``。

需要额外检查 CPU 标量实现时，独立运行：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\test_reconstruction.py `
     --build-dir .\mitsuba\build\Release --variant scalar_rgb

默认报告位于 ``outputs/reconstruction-tests/native-cuda_ad_rgb.json`` 和
``native-scalar_rgb.json``。CPU 结果仅作为辅助；CUDA 使用独立的设备与
后端验证，CUDA 失败不会回退 CPU，也不能用 CPU 通过代替 GPU 通过。
