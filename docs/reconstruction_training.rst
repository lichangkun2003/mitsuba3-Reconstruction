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
   CONFIG.training.profile_interval_epochs = 50  # 新分辨率及每 50 轮抽样一个视角
   CONFIG.training.profile_stages = True  # 抽样视角进一步诊断原生前向/两类反向阶段
   CONFIG.training.reference_cache_mib = 1024  # 参考图 CPU 内存缓存；0 关闭
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
* ``training``：``num_epochs``、``chunk_size``、``profile_views``、``profile_interval_epochs``、
  ``profile_stages``、``reference_cache_mib``、``seed``、``shuffle``、
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

参考图内存缓存
--------------

默认 ``training.reference_cache_mib=1024``，以 MiB 为单位限制训练参考图的 CPU 缓存。
第一次访问一个视角时，仍通过原来的读取流程解码为线性 RGBA float32，并检查尺寸和有限值；
后续访问直接复用只读数组，避免每个 epoch 重复读取和解码 PNG。缓存不占用 CUDA 显存，
也不消除每次训练调用中的参考数据上传。100 张 800 × 800 RGBA float32 约占 976.6 MiB，
默认容量可以全部容纳；这是图像数组的大小，不是整个训练进程的内存占用。

容量不足时按最近使用顺序淘汰旧图，单张图片超过容量则不缓存。可通过
``--reference-cache-mib 2048`` 扩大，或 ``--reference-cache-mib 0`` 关闭。
若数据集超过容量且每轮遍历全部视角，频繁淘汰会降低命中率；应根据日志和可用内存调整。
首次访问仍有解码开销，比较耗时时应观察后续轮次。缓存只对本次训练进程生效，恢复训练后重新填充，
不会写入 checkpoint 或输出目录；恢复时可调整容量，不改变已保存的模型、优化器和随机数状态。

数据集在一次运行期间视为固定输入，不支持训练中修改参考图片。缓存复用相同的 float32 数值，
不改变随机数消耗、训练视角顺序、1 spp、每视角一次更新或拓扑证据规则。
训练后自动评价和独立评价保持原读取流程，不继承这个训练缓存。

每轮 ``training.jsonl`` 和最终 ``training-report.json`` 的 ``reference_cache`` 包含
``capacity_bytes``、``resident_bytes``、``images``、``hits``、``misses``、``evictions``；
命中、未命中与淘汰次数从本次进程开始累计，未命中表示实际调用了图片读取流程。
关闭缓存时所有访问均为未命中，驻留图片数及字节数保持零。

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

原生前向/反向过程中的射线记录按射线和命中槽独占地址写入。对这些已确认无冲突的
写入显式使用 ``ReduceMode::NoConflicts``，使 Dr.Jit 不再为每次记录写入安排不必要的
求值与内核边界。多个射线向同一体素累加梯度仍采用原来的归约规则，不将其声明为无冲突。
本地生成场景的原生内核回归中，唯一 JIT 内核数从此前的 66 降至 18；这表示减少了
内核拆分，不能理解为端到端训练获得相同比例的加速。实际收益仍需结合目标设备上的
``gradients`` 和内核抽样耗时判断。

命中记录同时改为按字段分离（SoA）、字段内按命中序号排列射线的布局，索引为
``field * ray_count * max_hits + hit * ray_count + lane``。同一命中序号下，相邻线程
连续访问同一字段，改善记录读写的访存连续性；记录字段、缓冲元素总数与每条射线的
命中顺序不变，前述 ``NoConflicts`` 写入约定继续保留。这只调整记录的存取方式，
不改变场景参数布局、checkpoint 格式、采样与梯度公式。

默认 256 → 512 调度在 256 阶段不消费删除证据。训练现在将实际证据需求传给原生
``collect_deletion_evidence``，跳过这些未使用的计算；进入需要收集删除证据的目标
分辨率后照常计算，跨视角累计、删除阈值、保护期与执行时机保持不变。256 阶段的生长
仍然启用；跳过删除证据不等于关闭生长或删除功能。

生长证据期间，每批冻结候选的中心和六个轴向采样点在世界空间中只计算一次，后续
视角复用这七点。每个视角仍重新投影并读取当前参考/渲染 alpha，不缓存或跳过证据；
轮末仍依据当时的模型重新生成合法候选。父格与候选格的配对匹配改为无碰撞的 uint64
键和排序查找，保持配对含义及候选原顺序。派生几何缓存不写入 checkpoint，候选批次
变化、升层或恢复后重新构造。

交互终端（Windows PowerShell、Slurm 的 ``srun --pty bash``）在同一行刷新进度，例如
``Epoch 1/1000 | view 4/100 (4%) | cam 19 | loss=0.101282 | 0.09s/view``。
``view`` 表示本轮已完成视角数，``cam`` 是相机编号（从 1 开始）。更新会节流并按终端宽度
截断，避免快速刷新或自动折行；每轮结束只保留一条包含平均 loss、耗时、网格分辨率和
体素数的摘要。拓扑处理期间显示 ``updating topology...``，保存模型时另行提示路径。
输出重定向、管道或非交互式 Slurm 日志中不写回车刷新，只保留轮次摘要和必要提示。
评价阶段同样每个划分使用一行进度，结束后输出一条 PSNR 汇总；异常或中断前结束活动行。

自动耗时记录与性能诊断
----------------------

无需额外命令，训练默认记录耗时到当前实验的 ``training/`` 目录：

* ``training.jsonl``：每个完整 epoch 一条记录，包含参考图读取/缓存、前向与梯度、
  拓扑证据观察、优化器更新的阶段累计时间，以及轮末拓扑处理时间、增删/细分事件、
  活跃体素数和参考图缓存统计。``training_resolution`` 是本轮实际训练的网格分辨率，
  ``resolution`` 仍是执行轮末拓扑操作后的分辨率；升层所在轮不能用后者归类本轮训练。
  ``active_voxels_before`` 记录本轮开始时的体素数。
* ``performance-summary.json``：汇总当前这次进程的耗时，按实际训练分辨率统计
  每轮耗时均值、中位数、p95 和各阶段累计时间，结合冷启动、拓扑事件和缓存命中情况
  分析普通轮次与生长/删除/细分轮次的差别。还记录初始化、checkpoint 保存、自动评价
  的耗时、运行环境和源码版本信息，便于区分训练计算与出图、保存等开销。
  拓扑事件轮包括生长证据采集轮、轮末证据评估轮及实际发生增删/细分的轮次；
  512 阶段每轮常规采集删除证据，本身不将该轮归为拓扑事件轮。
* ``profile-training.jsonl``：稀疏抽样的 Dr.Jit 内核统计，用于判断编译、缓存和设备执行开销；
  不对所有训练视角开启详细内核记录。

逐轮日志沿用 ``view_stage_seconds.gradients`` 字段名，以兼容已有报告。这个计时区间
包含前向遍历与合成、loss、命中反向、未命中候选几何反向，以及归约检查/必要回读，
不是只测梯度公式或原子累加。不能直接把它的全部耗时归因为梯度写冲突。

性能汇总在每个完整 epoch 结束时原子刷新，训练和自动评价结束时再次更新；
正常处理 ``Ctrl+C`` 或运行异常时也尝试保存当前已记录的信息。它不是模型 checkpoint，
不会改变原来的三次模型保存计划，也不能用来恢复未保存的训练进度。
进程被强制结束、Slurm 超时终止或节点故障时，最后一次成功写入的完整记录仍可用于分析，
但不能保证取得正在执行的视角或轮次的完整耗时。

恢复训练为本次进程写独立的 ``performance-summary-resume-TIMESTAMP.json``、
``training-resume-TIMESTAMP.jsonl`` 及对应的 profile 日志，保留恢复前的文件。
各段统计只覆盖各自进程；诊断完整实验时应一起提供，并结合恢复的 checkpoint 区分
有效进度和恢复后重算的轮次，不能直接把各段的 epoch 数当作独立训练进度相加。

默认 ``training.profile_interval_epochs=50``：本次进程首次进入一种训练分辨率时，
以及每第 50 个 epoch，记录其第一个视角的内核信息。``--profile-views 3`` 还可额外记录
本次进程最先处理的三个视角；``profile_views`` 默认仍为 0，两个条件命中同一视角时只记录一次。
``training.profile_stages=True`` 默认只在这些已经选中的抽样视角上增加原生分段计时，
不会额外抽取视角。``--no-profile-stages`` 关闭分段，仍保留抽样视角的总内核统计。
若需要完全关闭内核抽样，可传 ``--profile-interval-epochs 0 --profile-views 0``，
普通逐轮耗时及性能汇总仍保留。恢复时可调整这些性能测量参数，不改变算法设置。
抽样有额外开销，比较优化前后速度时应使用相同的测量设置。

``training-report.json`` 的 ``profile_file`` 给出内核记录的实际路径；终端只提示文件位置。
每条 profile 记录包含各阶段墙钟时间、JIT 内核数、缓存命中/未命中数、内核哈希、
代码生成时间 ``codegen_ms``、后端编译时间 ``backend_compile_ms`` 和设备执行时间
``device_ms``。设备时间取自 Dr.Jit kernel history，只涵盖所记录的操作；这是抽样数据，
不能按它直接外推整个实验的 GPU 时间或多 GPU 加速比。

开启分段后，抽样记录另有 ``native_stages``，包含：

* ``forward``：原生前向路径与候选记录的完整物化。
* ``hit_backward``：loss、命中体素的反向梯度及其累加、后缀记录，以及按需启用的删除证据。
* ``candidate_backward``：未命中候选的几何反向处理。

每段记录 ``wall_seconds``、``device_ms``、``codegen_ms``、``backend_compile_ms``、
``operations``、``jit_kernels``、``cache_hits``、``cache_misses``。
``performance-summary.json`` 的 ``kernel_profiles.overall.native_stages`` 及
``kernel_profiles.by_resolution.<分辨率>.native_stages`` 提供各段的抽样次数、墙钟时间
分布和上述内核指标累计值。没有分段信息的旧日志或关闭分段的视角，不计入各段样本数；
这些细分统计不再次累加到整视角的时间与设备时间中。

分段诊断在边界进行同步，并强制物化完整的前向候选，避免惰性执行把前向工作延迟计入
后续反向阶段。这样的边界会改变正常执行的融合、调度和耗时，因此它只用于定位开销，
不能代表无分段计时的正常训练吞吐。原子累加与公式计算融合在内核中，分段计时不能
单独测出原子操作占比；进一步定位写冲突或寄存器/带宽瓶颈需要更细的硬件画像。

例如在诊断实验中添加 ``--profile-views 3 --profile-interval-epochs 10``，可记录前三个
视角和第 10、20 等轮的首个视角，便于观察初始执行与后续生长轮；进入新分辨率的抽样仍保留。

阶段时间是墙钟时间，包含 Python 调度、JIT、传输和等待，不是纯 GPU 内核时间；
异步执行还会让某阶段发出的工作在后续阶段等待完成。设备时间与主机墙钟时间不能相加，
也不能简单相减后将差值当作纯 CPU 时间。报告中的 GPU 可见设备与 CPU 线程
来源说明实际运行配置，不代表持续采集了 GPU 利用率、显存峰值或多 GPU 通信性能。
目前一个训练进程仍只使用一张 GPU，性能报告本身不会启用分布式训练。

训练完成后，可以直接提供以下小文件，无需发送 ``model.npz`` 或图片：

* ``training/performance-summary.json``：首先查看整体瓶颈和各分辨率阶段。
* ``training/training.jsonl``：核对逐轮变化、体素数量和拓扑事件。
* ``training/profile-training.jsonl``：检查抽样内核的缓存、编译及设备执行时间。
* 实验根目录的 ``evaluation-report.json``：分析 64 spp 出图耗时。

若中途恢复过训练，同时提供相应的 ``*-resume-*`` 文件；仅运行了训练、关闭了内核抽样
或尚未开始评价时，对应文件可能不存在。比较两个实现版本时，应保持数据集、随机种子、
训练预算、采样数、GPU 型号和测量设置一致，并分别查看预热后的普通轮次和拓扑变化轮次。
是否值得增加 CPU、优化单卡或进一步设计多 GPU 梯度汇总，应据这些实测结果判断。

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
       performance-summary.json  # 自动更新的本次进程耗时汇总
       training.jsonl            # 逐轮耗时、损失和拓扑事件
       profile-training.jsonl    # 稀疏抽样的内核性能记录
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
本轮新增原生前向、命中反向和候选反向的诊断计时接口，因此必须增量编译 C++，
再启动使用新 Python 脚本的进程。本轮只增加测量，没有交付梯度算法或执行性能优化；
不能仅同步脚本后沿用旧的原生模块。
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

先在本地提交并推送代码，再在服务器同步。下面命令用于已经获得 GPU 分配的 Slurm
作业步骤，包含 pull、增量编译和生成场景的性能回归。本次包含 C++ 改动，编译不可省略。
若同一源码/构建目录还有训练进程运行，先等该实验结束，再更新和编译；不要在其运行期间
替换正在使用的源码与原生库。若主动中断，应从已有完整 epoch checkpoint 恢复，未保存
的进度需要重算。

.. code-block:: bash

   test -n "${SLURM_JOB_ID:-}" &&
   source /apps/local/conda_init.sh &&
   conda activate "$HOME/envs/mitsuba" &&
   cd "$HOME/projects/reconstruction/mitsuba" &&
   git pull --ff-only &&
   cmake --build build --parallel "${SLURM_CPUS_PER_TASK:-1}" &&
   source build/setpath.sh &&
   "$HOME/envs/mitsuba/bin/python" -I resources/test_reconstruction_performance.py

回归通过后，在同一终端开启新的完整实验：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I resources/reconstruction_train.py \
     --dataset datasets/lego --epochs 1000 --chunk-size 0 --profile-views 3 \
     --profile-interval-epochs 10 \
     --eval-spp 64 --eval-max-views 10

这里显式选择整图提交，并诊断最先处理的三个视角及每第 10 轮的首个视角，
便于区分前向与两类反向开销。省略这两个 profile 参数时，恢复默认的分辨率/每 50 轮抽样。
需要保留内核总统计但关闭同步分段时添加 ``--no-profile-stages``。
已经运行的 HPC 进程不会自动加载新脚本，
需同步后重新启动；如恢复已有实验，应使用已保存的完整 epoch checkpoint 和 ``--resume``，
未保存的 epoch 进度仍需重算。实际显存峰值和 A100 提速幅度需要在服务器测量，不能由本地测试推断。

训练、评价和预览共用本地/Slurm 环境规则：Slurm 下保留
``CUDA_VISIBLE_DEVICES`` 原字符串及所有作业变量，不按本地物理编号重新选卡；
缺少有效 GPU 可见列表时明确报错。Dr.Jit CPU 线程数受
``SLURM_CPUS_PER_TASK`` 限制，未提供时使用 1 线程。没有 GPU 或源码绑定时不回退 CPU
或 wheel。多个可见 GPU 不会自动产生多 GPU 训练。

参考图缓存只需要额外 CPU 内存；当前 100 个 800 × 800 训练视角约需 1 GiB，
无需为了这项优化增加 GPU 数量或 CPU 核数。增加 ``--cpus-per-task`` 会调整 Dr.Jit
线程预算，但不会自动并行 Python 拓扑循环或让同一训练使用多张 GPU。
是否增加资源应根据后续轮次的 ``reference``、``gradients``、``topology_observation``
及 ``epoch_topology_seconds`` 实测判断，不能只用未触发生长的第一轮推断整个训练。
当前仍为单 GPU 训练；这些执行层优化没有引入多 GPU 梯度汇总。A100 与 RTX 5090 的硬件、
运行框架和实际工作负载不同，本地生成场景通过也不保证 A100 上的完整实验追平 5090。

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

训练输出还包括 ``config.json``、``training.jsonl``、``training-report.json`` 和
``performance-summary.json``。
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

多 spp 的四个通道在 GPU 上以 Float64 按原 sample 顺序累积，每个 chunk 求值后释放
其计算图，不保留全部 samples 的射线和计算图。每个 sample 仍先由 NumPy 生成整张图
的相同随机偏移，再分批发射射线；没有改变 RNG、采样位置或逐 sample 加法顺序。
完成整个视角后才显式同步并回读四个完整通道，最终除以 spp 和转换为 Float32 仍使用
原来的 NumPy 运算。这样移除了每个 sample、每个 chunk 的同步与图像回读；它减少的是
评价出图开销，不是训练中每个视角的梯度计算开销。

累积缓冲占 32 bytes/像素，与 spp 无关；800 × 800 时约 19.5 MiB。多 chunk 最后拼接
还需要同等大小的完整通道缓冲，两者同时存在时约 39.1 MiB，另加当前批次射线、命中记录、
模型和运行时分配。这是缓冲大小说明，不是实测进程显存峰值；``chunk_size`` 仍限制射线
工作批次，降低它不会消除整图累积缓冲。RTX 4060 和 A100 的 CUDA 后端支持这里的 Float64
加法，光线追踪本身仍沿用原来的浮点类型。

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
逐视角结果还记录渲染、参考图读取、指标计算和文件写入的墙钟时间，划分汇总记录相应
阶段总时间；这些数据同时写入对应的 ``metrics.json`` 和 ``evaluation-report.json``。
它们用于区分 64 spp 渲染与图片编码/写盘等开销，不改变 spp、射线、指标或输出图像。
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
唯一 JIT 内核哈希保持一致，第二个相机全部命中缓存、后端编译耗时为零。
加入原生记录写入的 ``NoConflicts`` 优化后，相同生成场景的唯一内核数从 66 降至 18。
相同采样下分批方式改变浮点归约顺序，不保证逐位相同。

原生记录布局改动后，另以 24 × 24 多命中场景对照修改前的原生实现，所比较数值的
最大绝对差为 ``2.384185791e-7``。本地 RTX 4060 Laptop 的合成 800 × 800、512 个体素
原生调用基准中，耗时中位数从约 0.2794 秒变为 0.2660 秒，降低约 5%；这是局部调用，
不是完整训练的 epoch 时间。66 → 18 个内核不等于 3.7 倍训练加速。
独立的 128 × 128、64 spp 评价小场景基准中，GPU Float64 累积后的耗时中位数从约
0.0966 秒变为 0.0595 秒。这两项本地局部测量都不能外推 A100 上真实 Lego 的训练或
全部视角出图耗时；正式比较需保持数据、设备和配置一致，并读取实际实验性能报告。

最终布局已通过 CUDA 多命中/分批、18 项几何公式、15 项优化器/拓扑检查，以及包含
保存恢复、升层和自动评价的生成场景训练检查；另通过 17 项 scalar 几何辅助检查。
1/3/64 spp 的六组 GPU 累积结果与原 CPU 累积结果逐位一致。
组合测试曾出现一次 Windows 原生访问异常（0xC0000005）；独立运行和原组合流程重跑
均通过，原因尚未定位，不据此声称该异常已修复。测试产物使用临时目录，未重跑真实 Lego。

源码构建、生成的小场景数值测试、checkpoint 往返恢复和完整数据集训练分别验证不同
范围。检查点能够加载、loss 有下降或图像看起来合理，都不能单独证明硬可见性代理梯度
与原 Falcor 所有路径完全一致。真实 lego 收敛、正式参数选择、跨机器数值一致性及
Slurm 上的实际运行，需要在目标数据和硬件上完成验证。

原生三阶段计时已在本地 RTX 4060 的 ``cuda_ad_rgb`` 源码构建中验证：整图与每批 37 条射线
的分段开关对照中，loss、梯度和有效计数在测试容差内一致，仍只回读两个统计标量；
嵌套内核统计完整保留，空模型的两个反向阶段为零。生成场景的训练、恢复及分段日志汇总检查
通过，另有三项纯 CPU 汇总测试覆盖旧日志兼容和分辨率分组。这些检查不代表 A100 的实际阶段耗时。

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\resources\test_reconstruction_performance_summary.py
