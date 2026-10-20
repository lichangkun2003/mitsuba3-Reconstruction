体素重建实验
============

当前单卡实验从 ``train.py`` 启动，统一参数在该文件顶部的 ``CONFIG`` 赋值区。
切换场景修改 ``CONFIG.data.dataset``，或传入 ``--dataset datasets/lego``；
命令行参数覆盖配置文件与默认配置。RTX 5090 Windows 主机使用这个单卡入口。

两层迭代比例也在 ``train.py`` 顶部配置：

.. code-block:: python

   CONFIG.schedule.start_resolution = 256
   CONFIG.schedule.target_resolution = 512
   CONFIG.schedule.level_weights = (6, 4)  # 分别是粗层、细层的轮数权重
   CONFIG.training.num_epochs = 1000

默认 1000 轮为 256 层级 600 轮、512 层级 400 轮；传 ``--epochs 700`` 时为
420／280 轮。不能整除时，粗层四舍五入，细层取剩余轮数，每层至少一轮。
新实验用两项权重；四项旧配置和 checkpoint 仍按旧分配算法恢复，不自动改成 6:4。
启动日志会打印各分辨率的轮数。512 层级继续允许生长，两层各 warmup 20 轮；
整个训练末尾 100 轮冻结生长和删除，继续优化参数。

参考图协议由 ``CONFIG.data.reference_compositing = "srgb"`` 控制。新实验对 sRGB PNG
先在编码空间用 alpha 合成白背景，再解码为线性 RGB；渲染、训练损失、梯度和拓扑收益
始终在线性空间计算。普通 RGB 图直接解码；线性 EXR 继续在线性空间合成。
预览、训练与评价共用读图实现。参考 alpha 仅用于这一预处理步骤。
``"linear"`` 保留旧的“先解码、再合成”目标，半透明边缘与新协议不同。
旧 checkpoint 缺少该字段时仍按 ``linear`` 续训与评价；比较新协议应新开实验。
默认评价主指标改为 sRGB PSNR，同时保留线性 PSNR；不改变训练的损失空间。

当前为 RGB-only 训练加在线 RGB 生长与删除：保留原来两个合法父体支持、有限边界接触和
外伸的候选规则，利用正常训练前向计算候选的虚拟插入 RGB 收益，不额外渲染确认。
同一空格的所有合法父体提案分别评估，不在投票前只保留第一个父体。
RGB 投票与提交时几何复核后，每格选择跨可靠视角平均收益最高的一份提案，
同分按父体格编号确定顺序，每格最多新增一个体素，再应用总体素容量限制。
父体外伸门槛 0.05、至少两个合法父体支持及形状缩小比例 0.7 保持不变。
粗层、细层分别先冻结生长 20 轮，之后按 10 轮周期处理；各层第一次事件在第 30 轮。
删除包含两条分支：复用已有命中反向的后缀颜色判断 RGB 删除收益，以及持续低 opacity 清理；
两者均不另做前向。
完整训练的最后 100 轮同时停止生长和删除，继续优化已有体素。
分裂机制不变，训练保持 1 spp。配置位置：

.. code-block:: python

   CONFIG.topology.use_rgb_growth = True
   CONFIG.topology.growth_warmup_epochs = 20
   CONFIG.topology.growth_fine_warmup_epochs = 20
   CONFIG.topology.growth_freeze_last_epochs = 100
   CONFIG.topology.growth_rgb_normalization = "observed"
   CONFIG.topology.enable_deletion = True
   CONFIG.topology.use_rgb_deletion = True
   CONFIG.topology.use_low_opacity_deletion = True
   CONFIG.topology.deletion_warmup_epochs = 20
   CONFIG.topology.deletion_interval = 10
   CONFIG.topology.deletion_protection_epochs = 20
   CONFIG.topology.deletion_freeze_last_epochs = 100
   CONFIG.topology.deletion_rgb_confirmation_windows = 2
   CONFIG.topology.deletion_rgb_min_support_views = 3
   CONFIG.topology.deletion_rgb_max_conflict_views = 1
   CONFIG.topology.deletion_rgb_gain_epsilon = 1e-7
   CONFIG.topology.deletion_rgb_min_mean_gain = 1e-7
   CONFIG.topology.deletion_opacity_threshold = 0.005
   CONFIG.topology.deletion_opacity_confirmation_windows = 2

末尾冻结按全局总轮数计算：700 轮实验冻结第 601–700 轮，1000 轮实验冻结第 901–1000 轮，
期间不准备生长候选、不收集生长证据，也不提交新增体素。设为 ``0`` 关闭该窗口；
设为大于或等于总轮数时，全程不生长。层级预算、分裂及层内 warmup 不变，
``--stop-after-epochs`` 不会将暂停位置当作完整训练的结尾。
100 轮是初始对照设置，不保证消除细节模糊；当前默认总轮数为 1000。
续训保留 checkpoint 保存的拓扑配置，缺少此字段的可兼容旧 checkpoint 按 ``0`` 恢复；
启用新规则应新开实验，不在旧实验中途修改。

每视角按候选的有效命中射线数平均收益，跨视角只对可靠观测视角平均，保留负、零收益。
未知射线只放弃自身的证据，同一候选在该视角的其他可靠射线继续参与统计；没有可靠命中时
该视角才因缺少观测而弃权，非有限聚合收益仍按无效数值处理。记录溢出只丢弃未保存的
候选记录，已保存记录仍需通过实际路径及插入后的命中上限、透射率检查。
先合并分批／多 GPU 的逐提案原始收益和计数，再归一化投票。全部合法父体提案参与评估，
通过证据与几何复核后再选同格胜者，只保留原来的总体素容量限制。候选数量与单次新增预算
参数已删除，带旧预算字段的配置及 checkpoint 会报错；比较新规则应使用当前配置新开实验。
``gradient.max_growth_hits=8`` 按每条射线遇到的不同候选格计数，同格替代提案共享记录槽，
但各自保留命中、收益与未知统计。不增加 DDA 或整图渲染；保留更多父体提案仍增加候选
相交及收益归约工作。事件日志同时记录提案数量与不同目标格数量，后者才对应可新增的格子。

可靠射线均值带有条件选择偏差，不是完整视角收益的无偏估计。日志记录已遇到候选命中的
可靠比例，未访问的候选不计入覆盖率；该比例仅用于分析，不设置新门槛。
删除收益为 ``dc=T*a*(B-c)``、``gain=-2*dot(C-reference_rgb,dc)-dot(dc,dc)``：
``B`` 是已有反向中的后方合成颜色，``T`` 为命中前透射率，``a``／``c`` 是该体素的
opacity／颜色。不需要除以 ``1-a``，也不重新追踪，但会增加 GPU 算术与统计归约。
仅正常退出网格或到达 ``ray.maxt`` 的完整路径、单位采样权重和有限数值提供证据；
因透射率或容量提前终止时，删除可能暴露未追踪区域，对应射线弃权。
删除同样先按可靠命中数求视角均值，再按可靠观测视角求均值；不因未知射线取消整票。
每个层级前 20 轮不收集删除证据，第 30、40 等事件轮各遍历全部训练视角。
默认至少 3 个支持视角、没有可靠均值为负的反对视角、跨视角收益超过 ``1e-7``，
且连续两个事件窗口通过才删除；新生体素额外保护 20 个完整轮次。
RGB 分支不因无观测、零收益或低 opacity 删除。
低 opacity 分支在上述事件末检查模型的真实不透明度 ``sigmoid(Y0*b0)``，
连续两次低于 ``0.005`` 才通过，某次不再低于阈值则重置计数。它仅支持 DC opacity，
不使用参考 alpha、不要求 RGB 删除收益为正，也不将未观测视为零贡献。
两条分支共用 warmup、新生保护、冷却和末尾冻结，升层后重新保护与确认，
任一分支确认后即可删除；只有事件末的 opacity 参数读取，不增加前向。
``training.jsonl`` 中 ``topology.deletion_opacity`` 记录新分支统计，
``topology.deleted`` 为两个分支合并去重后的实际删除数。

这些是在线参数快照上的单体素收益，不是固定模型的联合删除证明。多体素分别有益、
一起删除仍可能有害；未知区域也不被可靠射线的均值覆盖。应对照实际图像，不能保证
轮毂改善。``0.005`` 是待验证的初始阈值，很多低 opacity 体素仍可能共同产生可见贡献。
本阶段尚未添加 ``T*a`` 低贡献裁剪。更完整的公式、事件日志和局限见训练文档。

本次多父体生长修正修改了 C++ 候选索引与证据计算，**同步后必须增量编译并重新启动**，
源码构建需满足 ``mi.VoxelReconstruction.rgb_growth_version >= 3``。
使用版本 2 或更早 RGB 生长证据的 checkpoint 不能直接升级续训，应新开实验；
旧模型仍可独立评价。完整投票定义、截断弃权、每射线记录容量与
局限见 ``docs/reconstruction_training.rst`` 开头。
已有原生 RGB 删除要求 ``mi.VoxelReconstruction.rgb_deletion_version >= 1``，
之前的持续低 opacity 分支仅修改 Python，本次生长修正仍需要上述 C++ 增量编译。
旧 checkpoint 缺少 ``use_rgb_deletion`` 时按 ``False`` 恢复，
保留原有已保存配置，不自动给原来关闭删除的实验启用此机制；启用应新开实验。
旧 checkpoint 缺少 ``use_low_opacity_deletion`` 时也按 ``False`` 恢复；
当前新实验默认开启，确认计数保存在 checkpoint 中，续训不悄悄改变删除规则。
仅运行十轮不会覆盖生长／删除；保留 ``--epochs 1000``，另传
``--stop-after-epochs 41 --skip-evaluation`` 可覆盖前两次证据事件，
是否生长或删除仍取决于数据。这是新实验，不会修改已有结果。

Distortion 对照实验
-------------------

保持现有生长、删除和 1 spp 配置，在原训练命令后添加
``--distortion-weight 0.01`` 可尝试沿射线贡献集中正则；``0`` 为默认值，恢复原目标。
代码配置位置为 ``CONFIG.gradient.distortion_loss_weight``。
它使用实际 hard hit 的 ``T*opacity`` 和归一化深度点对距离，反向同时更新 opacity
与深度相关几何，不增加一次前向，也不使用参考 alpha。它不直接给空洞创建新候选。

本功能修改了 C++，同步源码后先执行
``cmake --build build --config Release --parallel 8``，要求
``mi.VoxelReconstruction.distortion_version >= 1``，再新开实验。
权重保存在 checkpoint 中；开启与关闭应分别运行，不在续训时更改。
``training.jsonl`` 分开记录 RGB、未加权 Distortion、加权 Distortion 和总 loss，
便于与原实验比较；``0.01`` 是待验证的初始权重，不保证改善 PSNR 或补洞。
公式、截断路径语义、梯度范围及运行示例见
``docs/reconstruction_training.rst`` 的“可选 Distortion”一节。

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
   * - ``tools/deploy_windows.ps1``
     - Windows 源码部署：工具预检、独立 Python 环境、编译与单卡 CUDA／RGB-only 检查。

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

RTX 5090 新主机快速部署
~~~~~~~~~~~~~~~~~~~~~~

当前定制源码标记为 Mitsuba ``3.10.0.dev1``、Dr.Jit ``1.6.0.dev1``。
RTX 5090 的计算能力为 12.0，配套 Dr.Jit 已包含 Blackwell 支持；
仍应安装 NVIDIA 官网适用于这张卡和 Windows 版本的当前驱动，并在目标机完成检查。
这条源码构建使用 Dr.Jit 运行时生成 GPU 代码，无需设置
``CMAKE_CUDA_ARCHITECTURES=120`` 或单独安装 CUDA Toolkit／OptiX SDK。
参考 `NVIDIA GPU 列表 <https://developer.nvidia.com/cuda/gpus>`_、
`Dr.Jit 更新记录 <https://drjit.readthedocs.io/en/latest/changelog.html>`_ 与
`Mitsuba GPU 构建说明 <https://mitsuba.readthedocs.io/en/latest/src/developer_guide/compiling.html#gpu-variants>`_。

先安装上述工具。Visual Studio Installer 中选择 Visual Studio 2022 或其
Build Tools 的“使用 C++ 的桌面开发”，包含 MSVC v143 x64/x86 工具和 Windows SDK。
安装 Git、CMake 时加入 PATH；安装完成后新开 PowerShell。Python 使用 64 位 3.12。

将 **本项目当前源码** 同步到目标主机。首次获取已推送版本可执行：

.. code-block:: powershell

   # 在目标机器选好的 reconstruction 父目录下执行。
   git clone --recursive https://github.com/lichangkun2003/mitsuba3-Reconstruction.git mitsuba
   if ($LASTEXITCODE -ne 0) { throw 'Source clone failed' }
   Set-Location .\mitsuba

Git clone 只包含已推送的改动；本地新加的部署脚本也需同步。也可以复制完整源码和
子模块（保留 Git 元数据），排除 ``build*``，并在目标机器重新创建 Python 环境。
不要改为克隆上游 Mitsuba，因为本项目的 C++ 重建接口在当前定制仓库中。

从目标机器的 Mitsuba 仓库根目录运行：

.. code-block:: powershell

   # 只检查工具、GPU 和已有环境／构建配置，不安装或编译。
   powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\reconstruction\tools\deploy_windows.ps1 -CheckOnly
   if ($LASTEXITCODE -ne 0) { throw 'Resolve the reported prerequisite first' }

   # 创建／复用独立环境、初始化固定版本的子模块、编译并检查。
   powershell -NoProfile -ExecutionPolicy Bypass -File .\experiments\reconstruction\tools\deploy_windows.ps1 -Gpu 0 -Jobs 2
   if ($LASTEXITCODE -ne 0) { throw 'Deployment failed; inspect the error above' }

``-Gpu`` 对应 ``nvidia-smi`` 的物理编号，多卡机器应先确认 5090 的编号。
``-Jobs 2`` 控制 MSBuild 同时构建的项目数；仓库还启用了 MSVC ``/MP``，
项目内部可以并行编译，因此实际编译进程数可能超过 2。增加它需要足够的系统内存。
这里的执行策略只作用于新开的 PowerShell 进程。
脚本默认使用 ``../envs/mitsuba`` 和 ``build/Release``，不依赖终端当前目录。
若检查发现从其他机器复制的缓存或不匹配的变体，会报告问题，由使用者另行保留、
重命名旧构建目录后再运行。

编译后依次运行 64×64 Cornell box CUDA／OptiX 渲染和六组 RGB-only 重建回归，
验证源码导入位置、定制绑定、颜色／opacity 梯度、训练更新及体素分裂。
检查无需 Lego 数据集；日志和基础渲染报告保存到仓库父目录的
``outputs/mitsuba/`` 下，以脚本打印的路径为准。
只有两项检查都通过才报告部署成功；这不替代真实数据上的短跑或性能测量。

接着同步 ``datasets/lego``，包含 ``init_points.ply``、
``transforms_train.json``、``transforms_test.json`` 及 JSON 引用的训练／测试图片，
再运行下方十轮短跑命令。VS Code 的 Python 解释器选择新主机的
``envs/mitsuba/Scripts/python.exe``，不要沿用本机的绝对路径。

手工环境与编译步骤
~~~~~~~~~~~~~~~~~~

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
``requirements-local-smoke.txt`` 只用于旧版 wheel 环境的基础测试，
不是本定制源码的部署依赖；此处仅需安装 NumPy。
首次配置采用已有 Windows 构建选项：

.. code-block:: powershell

   cmake -S . -B build -G 'Visual Studio 17 2022' -A x64 `
     "-DPython_EXECUTABLE=$python" `
     '-DMI_DEFAULT_VARIANTS=scalar_rgb,cuda_ad_rgb' `
     -DMI_SPLIT_MODE=OFF -DMI_ENABLE_EMBREE=OFF `
     -DCMAKE_POLICY_VERSION_MINIMUM=3.5
   if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
   cmake --build build --config Release --parallel 2
   if ($LASTEXITCODE -ne 0) { throw 'Source build failed' }

``MI_DEFAULT_VARIANTS`` 仅在首次生成 ``build/mitsuba.conf`` 时生效；已有构建应检查
该文件中的启用变体。``--parallel 2`` 限制项目并发数，不限制 ``/MP`` 的内部编译进程数，
可按新主机可用内存调整。
``CMAKE_POLICY_VERSION_MINIMUM=3.5`` 允许 CMake 4.x 配置仓库中较旧的第三方库。
以后 C++ 改动只需再次执行上述 ``cmake --build`` 增量编译命令。

运行脚本会显式加载 ``build/Release/python``，打印并检查 Mitsuba、Dr.Jit 路径与
``cuda_ad_rgb`` 后端。仅用 ``pip install mitsuba`` 不能获得本项目的重建接口。

单卡实验命令
------------

先短跑十轮，保留完整实验的 1000 轮调度预算：

.. code-block:: powershell

   & '..\envs\mitsuba\Scripts\python.exe' -I .\experiments\reconstruction\train.py `
     --dataset datasets/lego --gpu 0 --epochs 1000 --stop-after-epochs 10 `
     --chunk-size 0 --no-profile-stages

此短跑保存可续训 checkpoint，不发布最终 ``model.npz`` 或自动执行最终评价。
新开完整的 1000 轮实验：

.. code-block:: powershell

   & '..\envs\mitsuba\Scripts\python.exe' -I .\experiments\reconstruction\train.py `
     --dataset datasets/lego --gpu 0 --epochs 1000 --chunk-size 0 `
     --no-profile-stages --eval-spp 64 --eval-max-views 10

``--no-profile-stages`` 关闭分段诊断，使抽样视角也能使用原生重放，适合速度比较。
若需要定位前向与反向耗时，可重新开启 ``--profile-stages``。
短跑与完整命令分别创建新实验；若要接着短跑训练，使用其实际 checkpoint 路径传给
``--resume``，不要把短跑命令重复执行当作续训。

已部署主机同步与增量编译
----------------------

先将本地修改提交并 push，再在运行机器更新。若旧训练仍使用同一源码／构建目录，
先等它结束再更新。Windows PowerShell 在仓库根目录执行：

.. code-block:: powershell

   git -c submodule.recurse=false pull --ff-only
   if ($LASTEXITCODE -ne 0) { throw 'Pull failed' }
   git submodule update --init --recursive
   if ($LASTEXITCODE -ne 0) { throw 'Submodule update failed' }
   cmake --build build --config Release --parallel 2
   if ($LASTEXITCODE -ne 0) { throw 'Build failed' }

随后运行上一节的单卡训练命令。HPC 在已经获得 GPU 的 Slurm 作业终端执行：

.. code-block:: bash

   test -n "${SLURM_JOB_ID:-}" &&
   source /apps/local/conda_init.sh &&
   conda activate "$HOME/envs/mitsuba" &&
   cd "$HOME/projects/reconstruction/mitsuba" &&
   git -c submodule.recurse=false pull --ff-only &&
   git submodule update --init --recursive &&
   cmake --build build --parallel "${SLURM_CPUS_PER_TASK:-1}" &&
   source build/setpath.sh

编译完成后新开单卡实验，保持调度器分配的 GPU：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/train.py \
     --dataset datasets/lego --epochs 1000 --chunk-size 0 \
     --no-profile-stages --eval-spp 64 --eval-max-views 10

实验输出仍自动分配为 ``outputs/场景-MM.DD-E编号/``，其中 ``E`` 表示实验编号。
最终 ``model.npz``、训练日志与训练／测试视角评价结果存于同一次实验目录。
所有数据集、JSON 配置和输出相对路径仍以 **仓库的父目录** 为基准，
与终端当前目录、脚本迁移后的目录层级无关。

当前算法范围
------------

当前默认是 RGB-only 加在线生长与删除：800 × 800、每视角 1 spp、opacity 仅优化 DC。
参考 alpha 只在读取时参与白底合成，之后不参与训练监督或筛选；sRGB 图片先合成再解码，
训练输入仍为线性 RGB。
启用同次训练前向中的 RGB 生长证据，以及已有反向中的 RGB 删除证据和持续低 opacity 清理，
每个层级各 warmup 20 轮；保留 256 → 512 分裂。
新实验在全局最后 100 轮冻结生长和删除，继续以 1 spp 优化已有体素。
最终出图为 64 spp，训练集与测试集各取前 10 个视角。

新实验还会在第一次参数更新前，将训练相机 JSON 中视角 0 的初始化体素图保存为
``outputs/场景-日期-E编号/train/0000-initial.png``，可与最终 ``train/0000.png`` 对照。
多层训练还会在粗层全部训练视角优化结束、第一次升层之前，保存同一视角的
``train/0000-coarse.png``；默认对应优化完成、尚未分裂的 256 层模型。
三图使用相同分辨率、背景、评价随机种子约定及 ``--eval-spp``（默认 64），训练保持 1 spp，
额外出图不会改变训练随机序列。
双卡仅由 rank 0 保存初始化图和粗层图，其他工作进程等待；固定单层训练不生成粗层图。
恢复训练不会覆盖已有对照图，也不会把恢复后的模型作为初始化重新出图；
已进入细层的 checkpoint 若缺少粗层图，不能补出过去的粗层状态。
``training-report.json`` 的 ``coarse_render`` 记录粗层出图信息。
``--skip-evaluation`` 同时跳过初始化图、粗层图和最终评价。

旧 alpha 监督 checkpoint 不能直接切换为 RGB-only 续训，应新开实验；旧模型仍可
单独评价。完整说明见 `训练与评价文档 <../../docs/reconstruction_training.rst>`_、
`初始化预览文档 <../../docs/reconstruction_preview.rst>`_ 和
`基础 CUDA 检查文档 <../../docs/local_gpu_smoke.rst>`_。
