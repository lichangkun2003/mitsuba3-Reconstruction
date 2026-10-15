体素重建训练、保存与评价
======================

训练入口为 ``resources/reconstruction_train.py``，评价入口为
``resources/reconstruction_evaluate.py``。它们共享场景表达、相机、源码运行环境和
前向渲染工具；``reconstruction_preview.py`` 继续用于初始化检查。
原生接口保留名称 ``mi.VoxelReconstruction``，移植对象是 Falcor 的
``VoxelReconstructionNoLightTransport``，不是旧的光传输 pass。

本次算法对照的 Falcor 提交为
``398f634ec788bbb2e86a6f4d0ab78d41158d9905``。随机数生成器、GPU 浮点累加顺序和
运行框架不同，不能把这一移植描述为逐位复现；应结合小场景数值检查和真实训练结果判断。

训练单位与算法范围
------------------

一个 epoch 完整遍历训练相机 JSON 的所有视角。默认每轮打乱视角顺序；每个视角的
每个像素仅发射一条射线，即固定 1 spp。完整视角的梯度累加结束后更新一次参数。
``training.chunk_size=0`` 默认整图提交；800 × 800 时一次提交 640000 条射线。
设为正整数可显式限制每批射线数，不改变每个视角的更新次数。
``data.camera_index`` 是沿用的预览配置字段，训练遍历全部视角，不使用它截取训练集。

训练包括颜色与 alpha 损失、外观与几何梯度、分组 Adam、体素拓扑操作及分辨率调度。
连续外观参数沿用原实现的解析梯度；几何部分沿用原实现针对硬椭球可见性的代理梯度、
支持区间和更新约束，而不是将 ``cuda_ad_rgb`` 当作自动保证所有梯度正确的开关。
几何代理梯度并非硬命中前向在所有位置上的精确导数，边界跳变不能用普通自动微分消除。
评价仍使用原硬命中的前向模型。

训练采样默认逐像素均匀随机；可选的 contour 采样保持提案概率及历史 alpha，
其概率校正用于训练。完整状态会随 checkpoint 保存，升分辨率时按原流程清空历史。
最终评价统一使用独立的均匀
子像素采样，不把训练时的 contour 提案带入测试集。
NumPy PCG64 与 Falcor 的随机数序列不同；相同 seed 不代表与 Falcor 逐射线一致。
沿用的 ``render.pixel_sampling`` 是预览字段；训练和评价实际使用上述逐像素采样，
评价报告的 ``actual_sampling`` 单独记录此约定。

1 spp 随机预测的均方误差包含采样方差项，因此一般不等于先对无限多个样本求均值、
再计算图像均方误差。训练采用用户确定的 1 spp 目标；多 spp 只用于训练后出图与评价，
不声称两种损失完全等价。

训练的参考 RGBA 先解码为线性 RGB，随后严格沿用 Falcor 的固定白底公式
``reference_rgb * alpha + (1 - alpha)``；该训练目标不随 ``render.background`` 改变。
训练默认预测背景也为白色，与这一目标一致。底层接口若传入黑色预测背景，仅改变预测，
不会将训练参考同步改成黑底，因此不应把它当作相同目标下的背景切换。
训练脚本明确要求 ``render.background=[1,1,1]``，非白背景会报错，防止意外优化不一致的目标。
评价是独立流程：评价时的背景设置会同时用于预测与参考合成。此前预览默认黑底，
比较实验必须明确对齐实际背景，不能仅凭 pass 名称认为设置相同。

统一配置
--------

``reconstruction_train.py`` 顶部 ``CONFIG = TrainConfig()`` 及其后面的赋值块为统一入口。
切换训练场景时，直接修改该块中的 ``CONFIG.data.dataset``，无需进入预览脚本查找：

.. code-block:: python

   CONFIG = TrainConfig()
   CONFIG.data.dataset = "datasets/lego"  # 切换场景就在这里改，例如 datasets/ficus
   CONFIG.data.point_cloud = "init_points.ply"
   CONFIG.data.cameras = "transforms_train.json"
   CONFIG.render.width = 800
   CONFIG.render.height = 800
   CONFIG.render.spp = 1
   CONFIG.training.chunk_size = 0  # 整图提交；正整数表示显式分批
   CONFIG.training.profile_views = 0  # 设为 3 可记录前三个视角的编译和设备耗时
   CONFIG.schedule.enabled = True
   CONFIG.schedule.start_resolution = 256
   CONFIG.schedule.target_resolution = 512
   CONFIG.training.num_epochs = 1000
   CONFIG.training.save_count = 3
   CONFIG.training.output = None  # 自动创建 outputs/场景-月.日-E编号/training
   CONFIG.evaluation.enabled = True  # 完成全部训练并保存 model.npz 后自动出图
   CONFIG.evaluation.spp = 64
   CONFIG.evaluation.max_views = 10  # train/test 各取相机 JSON 中前十个视角

``datasets/lego`` 相对于源码仓库的父目录，即 ``reconstruction/datasets/lego``。
点云和相机文件名相对于该场景目录；训练会遍历相机 JSON 中的全部视角。
如果新场景使用相同的文件名，只需改 ``dataset``。还需确认其坐标范围适合模型边界，
参考图尺寸与 ``render.width/height`` 一致。

训练脚本从 ``reconstruction_preview.py`` 导入的是 ``DataConfig`` 等配置类，也就是
字段和默认值的定义；两个脚本分别创建自己的 ``CONFIG`` 对象。因此，修改预览脚本的
``CONFIG.data.dataset`` 只影响预览，不影响训练。训练实验的参数集中改上面这一块。

配置还支持 JSON 分组覆盖及少量命令行覆盖，优先级为：代码中的 ``CONFIG``，然后
``--config`` JSON，最后命令行参数。例如 ``--dataset datasets/ficus`` 会覆盖代码中的 lego。
当前默认是 lego、800 × 800、1 spp、1000 个 epoch、
随机种子 42、整个实验保存三次（含最终模型）。修改 ``--epochs`` 同时设置分辨率调度的
总训练预算，因此不能把不同预算的早期阶段当成相同调度。
训练完成后默认以 64 spp 评价 train/test 各前十个视角；某划分不足十个时使用全部视角。
这里的顺序是对应相机 JSON 的原始顺序，不是训练打乱后的访问顺序。
出图数量不限制训练：每个 epoch 仍遍历全部训练相机，始终为 1 spp。

配置分组：

* ``data``：数据集相对路径、PLY、训练相机 JSON。
* ``model``：初始化边界、padding、SH 数量和固定分辨率。
* ``render``：图像尺寸、裁剪面、背景、命中上限等；训练 ``spp`` 必须为 1。
* ``training``：``num_epochs``、``chunk_size``、``profile_views``、``seed``、``shuffle``、
  ``save_count``（默认 3，包含最后一次）和输出路径 ``output``；``None`` / JSON ``null``
  表示自动分配实验目录，字符串表示使用显式路径。
* ``gradient``：alpha 损失、几何代理宽度/截断、边界和删除证据等参数。
* ``optimizer``：各类参数学习率、Adam、裁剪、步长限制和不透明度调度。
* ``topology``：体素增删及证据条件。
* ``schedule``：coarse-to-fine 开关和分辨率计划；启用时由此组决定起始分辨率。
* ``contour``：训练时可选的 contour 重要性采样。
* ``runtime``：源码构建目录和本地 GPU 编号。
* ``evaluation``：训练结束后出图设置，默认 ``enabled=True``、``spp=64``、
  ``max_views=10``、``chunk_size=65536``、``seed=42``，测试相机文件由
  ``test_cameras="transforms_test.json"`` 指定；``max_views=None`` 表示全部视角。

仅想训练时可传 ``--skip-evaluation``，或设置 ``CONFIG.evaluation.enabled=False``。
``--eval-spp`` 和 ``--eval-max-views`` 可分别覆盖自动评价的采样数与每个划分的视角上限。
训练后评价使用当前实验最终保存的 ``model.npz``，不再初始化体素、不更新参数；
评价设置保存在实验根目录的 ``evaluation-config.json``，训练自己的 ``training/config.json``
及 checkpoint 内的训练配置保持独立。

原 Falcor 宏在本实现中主要转换为运行配置，常用对应关系如下。修改这些配置无需重新
编译 C++，但 Dr.Jit 可能为新配置生成新内核；修改底层 C++ 实现仍需源码编译。

.. list-table:: 宏与配置对应关系
   :header-rows: 1
   :widths: 45 55

   * - Falcor 宏
     - 当前配置
   * - ``RECONSTRUCTION_MODE``
     - ``schedule.enabled``；0 对应 false，1 对应 true
   * - ``GRID_RESOLUTION``
     - 升层模式为 ``schedule.target_resolution``；固定模式为 ``model.grid_resolution``
   * - ``COARSE_TO_FINE_START_RESOLUTION``
     - ``schedule.start_resolution``
   * - ``COARSE_TO_FINE_TOTAL_ITERATIONS``
     - ``training.num_epochs``，入口同步至 ``schedule.total_epochs``
   * - ``COARSE_TO_FINE_LEVEL_WEIGHTS``
     - ``schedule.level_weights``
   * - ``SH_COUNT`` / ``SH_OPACITY_COUNT``
     - ``model.sh_count`` / ``model.opacity_sh_count``；均指系数个数
   * - ``GEOMETRY_TWO_SIDED_BOUNDARY``
     - ``gradient.geometry_two_sided_boundary``
   * - ``GEOMETRY_BOUNDARY_BAND_TAU``
     - ``gradient.geometry_boundary_band_tau``
   * - ``GEOMETRY_TRANSITION_ONLY``
     - ``gradient.geometry_transition_only``
   * - ``GEOMETRY_HIT_EXPANSION_BAND_TAU``
     - ``gradient.geometry_hit_expansion_band_tau``
   * - ``GEOMETRY_ADAM_DIRECTION_GUARD``
     - ``optimizer.geometry_direction_guard``
   * - ``MAX_CONTRIBUTING_VOXELS_PER_RAY`` / ``MAX_CANDIDATES``
     - ``render.max_hits`` / ``gradient.max_candidates``
   * - ``ELLIPSOID_MAX_SCALE_VOXELS``
     - ``optimizer.max_scale_voxels``

例如，保持 800 × 800 并完整遍历全部训练视角，以固定的较小网格进行两轮功能检查：

.. code-block:: json

   {
     "data": {"dataset": "datasets/lego"},
     "model": {"grid_resolution": 32},
     "render": {"width": 800, "height": 800, "spp": 1},
     "training": {
       "num_epochs": 2,
       "save_count": 3,
       "output": null
     },
     "schedule": {"enabled": false}
   }

这仅用于检查程序能否完成训练与恢复，不是原始 256 → 512 正式实验的等价配置。
训练和评价严格要求参考图尺寸与配置一致，不隐式缩放真值。

整图提交、内核复用与回读
----------------------

默认 ``training.chunk_size=0`` 让一个视角的全部射线一起进入前向和梯度计算，随后进行一次 Adam 更新。
整图提交仍会执行多个 GPU 内核，并不等于只启动一个内核，也不保证所有设备上都最快。
它需要更多显存；显存不足时，可以显式设置 ``--chunk-size 65536`` 等正整数，仍保持
800 × 800、1 spp 和每视角一次更新。程序不会自行降低 spp、图像分辨率或回退 CPU。
已有 JSON 配置中的正数 ``chunk_size`` 仍然生效；用 ``--chunk-size 0`` 可以覆盖旧配置，
包括从 checkpoint 恢复时的分批设置。此处的 0 约定仅用于训练入口；评价仍使用其正数分批参数。

相机的位置、方向和视场系数先通过 ``dr.opaque`` 上传为运行时数据，供同一视角的所有批次复用。
这些数值不再作为相机专属的常量嵌入 JIT 内核，因此只换相机数值不需要重新特化内核。
首次运行、批次大小或数据形状变化、网格升层以及算法配置变化仍可能产生新内核，不能把所有
缓存未命中都当成错误。此修改同样用于独立评价的公共射线生成函数。

回读是把 GPU 数据传回 CPU，会产生数据传输和等待。当前回读并非用于可视化：

* 前向、梯度累加和 Adam 更新仍在 GPU；不把梯度数组复制到 CPU 来完成参数更新。
* loss 与 loss/alpha/梯度的有限值检查先在 GPU 归约为两个标量，每个完整视角回读一次，
  用于进度记录和及时发现数值异常。显式分批时，不再逐批回读 loss 或整张 alpha。
* 原始 ``raw_alpha`` 只在 contour 采样器实际产生更新 token 时回读，用于更新采样历史。
* 概率校正后的 alpha 只在本轮需要收集体素生长证据时回读；普通训练轮不搬运这张图。
* 体素删除启用且进入证据收集阶段时，CPU 拓扑管理仍按视角读取证据，保留原来的跨视角
  累积及候选判断顺序。生长和 contour 数据也在需要的视角处理，不能作为纯可视化统一延迟到 epoch 末。

这些调整不在训练的每个视角或每个 epoch 内出图。完成全部训练并保存最终模型后，
训练入口调用评价脚本共用的流程，生成多 spp 图片和 PSNR。

进度现在区分 ``progress 4/100``（本轮已完成四个视角）与 ``camera 19``（相机编号，从 1 开始）。
每视角打印总墙钟时间和参考图读取、梯度阶段、拓扑观察、参数更新的分段时间；每轮 JSONL
记录 ``view_stage_seconds`` 和 ``epoch_topology_seconds``。这些时间包含 Python 调度、JIT、
传输和等待，不是纯 GPU 内核时间；异步执行还会让某阶段发出的工作在后续阶段等待完成。

需要定位编译开销时，额外传入 ``--profile-views 3``，仅记录本次进程最先处理的三个视角。
默认关闭该测量；它会增加测量开销。记录写到训练目录下的 ``profile-training.jsonl``，
恢复运行则使用与本次恢复日志对应的独立文件；``training-report.json`` 的 ``profile_file``
给出实际路径。每条记录包含 JIT 内核数、缓存命中/未命中数、内核哈希、代码生成时间
``codegen_ms``、后端编译时间 ``backend_compile_ms`` 和设备执行时间 ``device_ms``。
设备时间取自 Dr.Jit kernel history，统计的是被记录操作的耗时，不能与各阶段墙钟时间直接等同。

重建结果目录
------------

每次实验使用 ``outputs/场景-月.日-E编号``，例如 ``outputs/lego-10.15-E0``。
``E0`` 表示当天的第一个实验，``E1`` 表示第二个实验；编号与 epoch 无关。
一套配置的一次完整运行属于同一实验，初始化结果和训练后的结果放在一起。
默认 ``training.output=None`` 时，脚本扫描工作区 ``outputs`` 的直接子目录，取当天
所有场景中已有的最大 ``E编号`` 加一；当天没有实验则从 ``E0`` 开始，不填补中间空号。
例如已有 ``lego-10.15-E0`` 和 ``ficus-10.15-E2``，下一次 Lego 实验为
``lego-10.15-E3``。场景名取配置中数据集路径的最后一级，不因数据集目录是链接而改变。
日期统一采用 Asia/Dubai（UTC+4），使 Windows 和 HPC 不受各自系统时区影响。

脚本会在启动时打印实际实验目录，并把分配后的相对输出路径写入保存的配置。
编号分配通过工作区 ``.cache/mitsuba`` 下的短时文件锁保护，同一工作区的并发启动
按顺序分配编号。显式指定 ``--output-dir`` 或 JSON 中的 ``training.output`` 时，
直接使用该路径，不自动生成新编号。
若使用以前保存的配置开启新实验，将其中的 ``training.output`` 改回 ``null``，
即可重新自动编号。启动后即使环境检查或训练失败，也会保留已分配的目录和错误报告，
后续新实验继续递增；已有结果不会因此被删除或覆盖。

建议新实验按以下布局保存：

.. code-block:: text

   outputs/lego-10.15-E1/
     model.npz          # 训练完成后自动导出的完整体素重建模型
     training/          # 训练配置、日志及最多两份中间 checkpoint（默认）
     train/             # 最终模型的训练集视角 PNG 和 metrics.json
     test/              # 最终模型的测试集视角 PNG 和 metrics.json
     evaluation-config.json
     evaluation-report.json
     initial/           # 可选：初始化模型及其 train/、test/ 基线结果

默认自动分配的训练目录是实验根目录下的 ``training/``。全部 epoch 成功完成后，
脚本自动在其父目录（实验根目录）保存 ``model.npz``。如果显式输出目录的最后一级不是 ``training``，
则直接保存到该输出目录下的 ``model.npz``，兼容已有的平铺输出方式。
随后自动评价，在同一实验根目录创建 ``train/``、``test/`` 及评价配置和报告；默认每个划分
分别渲染前十个视角、64 spp。独立评价时指向实验根目录，并用 ``--checkpoint`` 加载这个模型。
训练轮数记录在配置、日志和
checkpoint 中，不通过 ``E编号`` 表示。下文恢复和评价命令中的 ``E1`` 仅为示例，
应替换为启动时打印的实际实验目录。

已经整理的 ``outputs/lego-10.15-E0`` 保存了本地第一次 Lego 实验：
800 × 800、256³ 网格、每视角 1 spp，完整训练 1 个 epoch，即 100 个训练视角、
100 次参数更新。根目录 ``model.npz`` 是训练后模型，``initial/model.npz`` 是
初始化模型；两者的 ``train/``、``test/`` 目前都只有各自第一个视角的 4 spp 图片，
不是完整训练集或测试集的评价。``training-config.json``、``evaluation-config.json``、
``cameras/``、``logs/``、``result.json`` 与 ``README.txt`` 保留配置、子集相机及运行记录。
初始化评价的配置另存为 ``initial/evaluation-config.json``；现在评价脚本统一使用
``evaluation-config.json``，不会覆盖已有训练 ``config.json``。

在 ``reconstruction`` 父目录中，可用保存的模型渲染全部训练和测试视角；
这会更新该实验根目录中的评价图片和报告，不会重新训练：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\reconstruction_evaluate.py `
     --checkpoint outputs/lego-10.15-E0/model.npz --split both --spp 64 `
     --output-dir outputs/lego-10.15-E0

若只需重跑当时保存的两个视角及 4 spp 设置，则使用：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\reconstruction_evaluate.py `
     --config outputs/lego-10.15-E0/evaluation-config.json

``outputs/ficus-10.14-E0`` 保留较早的 Ficus 初始化模型和训练第一个视角的预览，
尚无训练更新或测试集视角结果。冒烟测试、合成场景回归和构建日志不作为重建实验保留。
Dr.Jit、CUDA 和 OptiX 缓存写入源码仓库父目录（``reconstruction`` 工作区）下的
``.cache/mitsuba/``，不再放进 ``outputs``。

本地源码编译与训练
------------------

首次使用原生训练、参数更新和 checkpoint 恢复接口，或修改底层 C++ 后，必须重新编译。
本次自动评价以及此前整图提交、相机运行时数据、按需回读与性能日志的改动仅涉及 Python；已经包含上述接口的
源码构建不需要为这次改动重新编译 C++，同步脚本并重新启动进程即可。
预编译 wheel 不包含这些原生接口，不能用于验证它们。在 ``reconstruction`` 工作区
父目录的 PowerShell 中执行：

.. code-block:: powershell

   cmake --build .\mitsuba\build --config Release --parallel 2
   if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\reconstruction_train.py `
     --dataset datasets/lego --epochs 10

训练 10 轮表示每个视角都会被访问 10 次。省略 ``--output-dir`` 时使用默认的自动实验目录。
训练完成并保存模型后会自动生成 train/test 各前十个视角的 64 spp 图片和 PSNR。
上述命令仍启用配置中的分辨率调度；
要使用前面的固定小网格例子，应将 JSON 保存到工作区，例如 ``train-config.json``，
再以 ``--config train-config.json`` 运行。

HPC Slurm
----------

源码同步由使用者完成。首次使用或 C++ 有修改时先增量编译；在已经获得 GPU 分配的作业步骤中运行：

.. code-block:: bash

   cmake --build "$HOME/projects/reconstruction/mitsuba/build" --parallel 8
   cd "$HOME/projects/reconstruction"
   "$HOME/envs/mitsuba/bin/python" -I mitsuba/resources/reconstruction_train.py \
     --dataset datasets/lego --epochs 1000 --chunk-size 0 --profile-views 3

这里显式选择整图提交，并测量最先处理的三个视角，便于检查服务器上的编译和缓存开销；
不需要性能测量时可省略 ``--profile-views 3``。已经运行的 HPC 进程不会自动加载新脚本，
需同步后重新启动；如恢复已有实验，应使用已保存的完整 epoch checkpoint 和 ``--resume``，
未保存的 epoch 进度仍需重算。实际显存峰值和 A100 提速幅度需要在服务器测量，不能由本地测试推断。

训练、评价和预览共用本地/Slurm 环境规则：Slurm 下保留
``CUDA_VISIBLE_DEVICES`` 原字符串及所有作业变量，不按本地物理编号重新选卡；
缺少有效 GPU 可见列表时明确报错。Dr.Jit CPU 线程数受
``SLURM_CPUS_PER_TASK`` 限制，未提供时使用 1 线程。没有 GPU 或源码绑定时不回退 CPU
或 wheel。多个可见 GPU 不会自动产生多 GPU 训练。

Windows 默认构建目录为 ``mitsuba/build/Release``，Linux 为 ``mitsuba/build``；
``--build-dir`` 可覆盖。所有相对路径固定相对于仓库父目录，包括
``datasets/lego``、checkpoint 与输出。共享配置使用相对路径；无需把本机绝对路径
写入 Git。输出和缓存禁止放进源码仓库。

Checkpoint 与恢复
-----------------

``training.save_count=3`` 表示整个实验保存三次完整状态，包含最后的 ``model.npz``。
保存轮次按总 epoch 数均匀分配并向上取整：1000 轮在第 334、667、1000 轮保存，
30 轮在第 10、20、30 轮保存；不足三轮时每轮保存一次，不重复保存同一轮。
例如前面的本地 10 轮命令会生成 ``training/epoch-00004.npz``、
``training/epoch-00007.npz`` 和实验根目录 ``model.npz``。
不再额外生成 ``latest.npz``、最终轮次的 ``epoch-XXXXX.npz`` 或升分辨率前的分层副本。

训练输出还包括 ``config.json``、``training.jsonl`` 和 ``training-report.json``。
日志仍记录每轮的平均训练损失、分辨率、体素数与拓扑事件，保存次数不改变日志频率、
参数更新或分辨率调度。Checkpoint 在完整 epoch 及该轮拓扑处理结束后保存；它不是任意
视角中途的恢复点。中断后需从最近保存的 checkpoint 重算尚未保存的 epoch；
第一次保存前中断则需要重新开始。中间 checkpoint 保存不触发出图；全部 epoch 完成并保存
``model.npz`` 后，默认自动调用评价流程一次。

最终的 ``model.npz`` 使用同一套原子写入和校验格式，包含完整体素状态、网格元数据、
配置和训练恢复状态。它在全部训练成功完成后、自动评价开始前导出；训练中断时，可从最近保存的
``training/epoch-XXXXX.npz`` 恢复或评价中间结果。恢复一个已经完成的 checkpoint 到新目录时，
也会自动导出 ``model.npz``。训练报告的 ``reconstructed_model`` 记录导出路径、SHA-256、
完成轮数及更新次数。新实验拒绝覆盖已有的最终模型；恢复时也会校验既有模型的配置、
随机种子和训练数据签名，防止把其他实验的结果替换掉。

旧配置中的 ``training.save_every_epochs`` 仍可读取，但该字段不再控制保存间隔；
未显式指定 ``save_count`` 时使用默认三次。``schedule.save_each_level`` 保留旧配置兼容性，
训练入口不再据此额外保存分层副本；分辨率调度与体素更新流程保持不变。
已有的 ``latest.npz`` 和 ``level-R-epoch-XXXXX.npz`` 仍可通过 ``--resume`` 加载，
不会自动删除或改写这些旧结果。旧分层文件中的 ``pending_epoch_transition=true``
表示该轮所有视角已更新完，但尚需执行升层；训练恢复时仍会先完成一次待执行的升层，
再进入下一个 epoch。评价加载旧分层文件时保留保存的本层模型，不自动升层。

Checkpoint 是单个 NPZ，包含 JSON 元数据和数值数组，不使用 pickle。保存过程先在
目标目录写临时文件、刷新，再原子替换目标，避免覆盖过程中留下半个 checkpoint。
加载时检查格式版本、数组 SHA-256、布局、有限值、网格范围及四元数有效性。
拓扑删除可能移除全部体素；零体素是合法状态，checkpoint 保留空数组和网格元数据，
评价时对应仅背景的预测，仍按相同协议计算指标。

保存的内容包括：

* 实际网格分辨率、边界、活跃体素索引和两类 SH 数量。
* 中心、log scale、wxyz 四元数、颜色 SH 和不透明度 SH；数组布局为体素优先的
  平铺 AoS，其中颜色 SH 为系数后 RGB。
* Adam 一、二阶矩及步数、完成的 epoch 数、全局更新次数。
* 随机数状态、contour 历史、拓扑与分辨率调度状态。
* 完整配置、训练相机和参考图签名、源码及已加载原生模块的来源记录。

独立评价不需要再次读取初始化 PLY。新脚本在正确加载源码构建并选择 ``cuda_ad_rgb`` 后，
可以用以下接口恢复体素场景：

.. code-block:: python

   from reconstruction_checkpoint import load_checkpoint
   from reconstruction_common import restore_model

   saved = load_checkpoint(model_path)  # 指向当前实验的 model.npz
   model = restore_model(saved, mi)

这些模块位于 ``resources``，独立脚本需要将该目录加入 Python 导入路径。
模型文件包含渲染所需的全部体素参数；计算 PSNR 仍需数据集的相机 JSON 和真值图片。

例如前面的本地 10 轮实验在第二次保存后中断，可从第 7 轮恢复：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\reconstruction_train.py `
     --resume outputs/lego-10.15-E1/training/epoch-00007.npz

前面的服务器 1000 轮实验对应从第二次保存的第 667 轮恢复：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I mitsuba/resources/reconstruction_train.py \
     --resume outputs/lego-10.15-E1/training/epoch-00667.npz

恢复时默认复用 checkpoint 所在实验，不递增 ``E编号``：中间 checkpoint 使用所在的
训练目录，标准布局根目录的 ``model.npz`` 使用同一实验下的 ``training/``，旧平铺布局
使用原目录。仍可用 ``--output-dir`` 显式指定恢复的目标目录。这里的示例路径必须改为
本次实际保存的 checkpoint 路径。

恢复会校验训练数据签名，并保持原算法设置和总 epoch 预算。运行环境和输出目录可以
换到当前机器；更换算法参数或重新制定分辨率调度应开启新的实验。跨机器恢复保留完整
优化状态，但 GPU 累加次序等差异仍可能改变后续低位数值，不保证逐位确定性。
恢复也固定 ``training.seed``、``training.shuffle`` 和 ``training.save_count``，保持同一
实验的保存计划不变，并检查完整 epoch 数与视角更新
次数是否一致；不会从不完整的视角计数继续。每次恢复写入单独的
``training-resume-TIMESTAMP.jsonl`` 日志，保留此前日志，避免同一文件中重复混入恢复前
已运行但未保存的 epoch。比较曲线时应按本次运行的日志段与恢复点选取数据。
若恢复到已有输出目录，其算法配置必须一致；恢复一个已经完成全部 epoch 的模型到
新目录时，只导出对应实验目录的 ``model.npz``，不再生成额外的 checkpoint 副本。
随后仍按本次评价设置执行自动出图；可传 ``--skip-evaluation`` 仅导出模型。

多 spp 出图与指标
-----------------

训练入口默认在保存最终模型后调用评价流程，使用 64 spp 渲染 train/test 各前十个视角，
并保存每个视角的图片和 PSNR。相机按对应 JSON 顺序选择；不足十个时全取。
这不限制训练视角或改变训练 1 spp。评价只读取 checkpoint，不更新模型或使用测试视角训练。
默认主指标为线性 RGB PSNR，并同时报告 sRGB PSNR。

如果评价失败，已经完成的 ``model.npz`` 保留，进程以非零状态退出；
``training-report.json`` 中的 ``training_status`` 与 ``evaluation`` 分别记录训练和评价结果，
``evaluation-report.json`` 记录具体错误。可以修复数据或运行环境后直接重跑评价，无需重新训练：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\reconstruction_evaluate.py `
     --checkpoint outputs/lego-10.15-E1/model.npz `
     --split both --spp 64 --max-views 10 `
     --output-dir outputs/lego-10.15-E1

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I mitsuba/resources/reconstruction_evaluate.py \
     --checkpoint outputs/lego-10.15-E1/model.npz \
     --split both --spp 64 --max-views 10 \
     --output-dir outputs/lego-10.15-E1

``--split train`` 或 ``--split test`` 只评价一个划分。训练相机默认继承 checkpoint
中的配置，测试相机默认为 ``transforms_test.json``；可用 ``--train-cameras`` /
``--test-cameras`` 覆盖。数据集默认继承 checkpoint 的相对路径，也可用 ``--dataset``
指定当前机器上的同一数据。运行环境按当前平台重新选择，不继承另一台机器的 build
绝对路径。``--background R G B`` 同时修改预测与真值的线性 RGB 背景；默认沿用训练。
独立评价入口省略 ``--max-views`` 时默认评价所选划分的全部视角（除非 ``--config`` 中另有设置）；
训练入口的自动评价默认上限为十个，两者用途不同。报告同时记录划分总视角数、实际选中
的视角及其数量；前十个视角的平均 PSNR 是子集指标，不能当作完整训练集或测试集的结果。

每个视角先对 spp 次预测的线性 RGB 和 alpha 求均值，再计算指标。
每个像素、每个 sample 使用独立的随机子像素位置；seed 由全局 seed、划分和视角索引
确定。评价 ``--chunk-size`` 默认 65536，用于限制射线批次，不改变采样序列。

PSNR 的具体约定：

* MSE 对整张图的所有像素与 RGB 三个通道取平均；不应用前景 mask，峰值取 1。
* 线性指标使用未裁剪的浮点线性 RGB；真值先按相同背景进行 alpha 合成。
* sRGB 指标先将合成后的线性 RGB 裁剪到 [0, 1]，再应用 sRGB 传递函数。
  仍使用浮点数据，不从 8 位 PNG 重新读回计算。
* 报告逐图 PSNR 的算术平均，也单独报告平均 MSE 对应的 PSNR；两者不混用。
* 完全相同图像的 PSNR 为正无穷，JSON 使用 ``null`` 和对应布尔标志表示，避免写入
  非标准的 ``Infinity``。

``train/`` 与 ``test/`` 保存按视角索引命名的 PNG 和各自 ``metrics.json``；
``evaluation-report.json`` 记录完整指标协议、逐视角结果、checkpoint 哈希、源码运行
信息和状态。``--save-linear`` 可额外保存每个视角的线性 RGB/alpha NPZ。
PNG 用于显示；进行论文或其他实现的 PSNR 对比时，应明确选用哪种颜色空间、背景、
分辨率和 spp，并保证双方协议一致。

工程差异与复现范围
------------------

迁移保留 NoLightTransport 的硬椭球前向、外观梯度、几何代理规则、分组更新和拓扑调度，
同时有以下需要在实验中记录的实现差异：

* 前向、梯度和参数更新使用 CUDA/Dr.Jit；拓扑处理在 CPU NumPy 中执行，部分几何运算
  使用 float64，再写回 float32 模型。临界阈值附近可能与 Falcor 的 GPU float32 决策不同，
  拓扑性能也不能假定与原实现相同。
* 多个增长提案竞争同一格时，使用确定性的胜者选择；Falcor 的原子竞争顺序不确定。
  删除后的剩余体素保留顺序，而 Falcor 使用空洞/供体搬移，索引顺序可能不同。
* 对全部活跃体素预分配颜色 Adam 状态，没有复制 D3D 缓冲区分页耗尽时临时退回 SGD
  的资源处理分支；正常 Adam 数学规则保留。
* 相机证据使用动态 bitset，支持不超过 32767 个视角；不采用 Falcor 默认编译容量 128。
* 随机数序列、浮点归约顺序及跨 GPU 数值不同；不能声称逐位相同。
* 按本次约定训练固定每视角 1 spp，未复制原训练多 spp 路径；训练入口强制白底以匹配
  原参考合成公式，多 spp 及可配置背景用于独立评价。

本地验证记录与复现命令
----------------------

2026-10-15，本地 Windows Release 源码构建成功，在 NVIDIA RTX 4060 Laptop GPU
上使用 ``cuda_ad_rgb`` 完成以下检查；CPU 与模拟环境检查单独列出：

下表保留已有验证记录。本次整图提交与按需回读修改后，另行完成了生成小场景的 CUDA 性能回归，
并重跑训练集成检查：两视角三轮、三次模型保存、普通及升层后恢复、独立评价均通过。
未重新运行真实 Lego 实验，也未在 A100 或真实 Slurm 作业中测量吞吐量。

本次自动评价改动另在同一 RTX 4060 的 ``cuda_ad_rgb`` 下通过生成的 16 × 16 场景检查：
训练仍完成全部两视角、三轮共六次更新，随后按默认 64 spp 输出 train/test 各第一个视角；
检查评价子集顺序、训练配置未被覆盖、关闭自动评价、已完成模型恢复后的出图，以及视角
上限大于划分大小时全取。缺失测试相机时按预期返回非零状态，已训练模型完整保留。
独立评价两个完整划分、普通恢复和 coarse-to-fine 恢复也通过。测试使用临时目录并自动清理；
未为此次修改重新运行真实数据集或 HPC 评价。

.. list-table:: 已执行检查
   :header-rows: 1
   :widths: 28 72

   * - 检查
     - 实际结果与范围
   * - 原初始化/前向回归
     - 10 组、236 条射线通过。
   * - 几何代理梯度
     - CUDA 18 项、标量 17 项独立公式检查通过，最大绝对误差约 ``1.455e-6``。
       标量仅为辅助，不替代 CUDA 检查。
   * - 优化器/拓扑
     - 13 项通过，包括独立 NumPy Adam 对照、每组有效步数、方向约束、四元数更新、
       warmup、矩状态迁移、跨视角删除证据、增长证据和 coarse-to-fine。
   * - Checkpoint/指标/contour
     - 18 项 CPU 检查通过，包括原子保存、恢复、无 pickle、PSNR 协议、提案积分和 RNG 恢复。
   * - 本地/Slurm 环境规则
     - 16 项模拟环境检查通过；这不代表真实 Slurm GPU 作业已运行。
   * - 完整小场景训练闭环
     - CUDA 生成场景通过：36 项外观有限差分比较最大绝对误差约 ``9.1158e-6``；
       checkpoint 前向一致、空模型、多 spp 分块不改变射线、两视角三轮训练与恢复一致；
       完成的 checkpoint 恢复到新目录；4 → 8 升层与升层前恢复；开启 contour 后升层
       清空历史；独立 train/test 2 spp 评价及 PSNR 报告。
   * - Lego 本地一轮实验
     - RTX 4060 CUDA 完成 800 × 800、1 spp、256³ 网格的一轮训练：100 个训练视角、
       100 次更新；模型保存、加载和各一个 train/test 视角的 4 spp 出图通过。
       结果位于 ``outputs/lego-10.15-E0``；本次未触发体素增删或升到 512³，
       尚未完成全部视角评价或验证收敛。

在 ``reconstruction`` 工作区父目录 PowerShell 中，完成上述源码编译后可分别复现。
以下前向与几何测试的显式 ``--build-dir`` 相对于当前终端目录；其他新入口的默认
构建路径固定相对于工作区父目录，因此这里统一从父目录运行：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\test_reconstruction.py `
     --build-dir .\mitsuba\build\Release --variant cuda_ad_rgb
   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\test_reconstruction_gradients.py `
     --build-dir .\mitsuba\build\Release --variant cuda_ad_rgb
   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\test_reconstruction_gradients.py `
     --build-dir .\mitsuba\build\Release --variant scalar_rgb
   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\test_reconstruction_optimization.py
   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\test_reconstruction_training.py

新训练闭环测试自行生成小场景，不使用 lego 或下载数据集；重新运行时报告默认写入
``outputs/mitsuba/training-tests/training-tests.json``；优化器/拓扑检查位于
``outputs/mitsuba/optimization-tests``，几何检查位于
``outputs/reconstruction-tests/gradients``。这些可重新生成的测试产物已从当前
``outputs`` 中清理，不属于上面的场景重建结果。

Checkpoint 与 PSNR 的 CPU 单元测试不导入 GPU 后端，也不渲染数据集：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\test_reconstruction_checkpoint.py
   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\test_reconstruction_runtime.py

该测试覆盖模型/优化器/随机数状态往返、空数组与标量布局、原子替换失败保留旧文件、
校验和破坏、pickle 拒绝、元数据与形状错误，以及已知 PSNR、聚合定义和 sRGB 转换。
也检查 contour 提案归一化、PDF 校正后的常量/alpha 积分、仅选中 bin 的原始 alpha
历史更新，以及恢复随机数和历史状态后的采样与射线一致性；这些检查在 CPU 上执行。

整图提交和相机内核复用有独立的 CUDA 小场景回归，可从工作区父目录运行：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\test_reconstruction_performance.py

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I mitsuba/resources/test_reconstruction_performance.py

该检查使用生成的数据和临时目录，比较整图/分批的数值结果，检查按需回读、非有限值拒绝，
以及更换相机后的 JIT 内核缓存命中。它不代表真实 Lego 数据集或 A100 的吞吐量测试。
本次 RTX 4060 CUDA 检查中，16 × 16 整图与每批 37 条射线的梯度最大绝对差为
``1.90735e-6``，参数更新与 contour 历史在测试容差内一致；默认关闭 contour/growth
图像回读时只传回两个统计标量。改变相机位置、方向及视场角后，原生损失/梯度计算的
66 个唯一 JIT 内核哈希保持一致，第二个相机全部命中缓存、后端编译耗时为零。
相同采样下分批方式改变浮点归约顺序，不保证逐位相同。

源码构建、生成的小场景数值测试、checkpoint 往返恢复和完整数据集训练分别验证不同
范围。检查点能够加载、loss 有下降或图像看起来合理，都不能单独证明硬可见性代理梯度
与原 Falcor 所有路径完全一致。真实 lego 收敛、正式参数选择、跨机器数值一致性及
Slurm 上的实际运行，需要在目标数据和硬件上完成验证。
