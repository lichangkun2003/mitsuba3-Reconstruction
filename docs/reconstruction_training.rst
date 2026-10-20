体素重建训练、保存与评价
======================

RGB-only 训练、在线生长与删除（当前默认）
----------------------------------------

本阶段从新的初始化开始，使用 800 × 800、每像素 1 spp、总计 1000 个 epoch。
每个 epoch 遍历全部训练视角；单卡和双卡使用相同总轮数及 coarse-to-fine 调度。
常用参数集中在 ``experiments/reconstruction/train.py`` 顶部的 ``CONFIG`` 赋值区，
切换场景修改 ``CONFIG.data.dataset``，不要修改预览脚本的 ``CONFIG``。

.. code-block:: python

   CONFIG.training.rgb_only = True
   CONFIG.data.reference_compositing = "srgb"
   CONFIG.model.opacity_sh_count = 1
   CONFIG.render.width = CONFIG.render.height = 800
   CONFIG.render.spp = 1
   CONFIG.training.num_epochs = 1000
   CONFIG.gradient.alpha_loss_weight = 0
   CONFIG.gradient.alpha_geometry_weight = 0
   CONFIG.gradient.background_carve_weight = 0
   CONFIG.gradient.geometry_transition_only = False
   CONFIG.gradient.geometry_two_sided_boundary = False
   CONFIG.topology.enable_growth = True
   CONFIG.topology.enable_deletion = True
   CONFIG.topology.use_rgb_deletion = True
   CONFIG.topology.use_low_opacity_deletion = True
   CONFIG.topology.use_growth_evidence = False
   CONFIG.topology.use_rgb_growth = True
   CONFIG.topology.growth_warmup_epochs = 20
   CONFIG.topology.growth_fine_warmup_epochs = 20
   CONFIG.topology.growth_freeze_last_epochs = 100
   CONFIG.topology.growth_rgb_normalization = "observed"
   CONFIG.schedule.enable_coarse_growth = True
   CONFIG.contour.enabled = False

损失仅为线性 RGB 的平方误差。新实验默认 ``data.reference_compositing="srgb"``：
读取 sRGB RGBA 图片时，先将编码空间的 RGB 与白底按 alpha 合成，再解码为线性 RGB，
随后只保留三个 RGB 通道；普通 RGB 图片直接解码为线性 RGB。
源图片的 alpha 仅参与这一步图像合成，不再作为训练
监督、几何门控、采样依据或删除证据。预测背景固定为白色。底层
``loss_and_grad(..., reference_rgb, reference_alpha, ..., {"rgb_only": 1})``
要求 ``reference_rgb`` 已完成合成，并完全忽略兼容签名中的 ``reference_alpha``。

前向仍为射线与椭球的硬命中及 alpha compositing。每个椭球的 opacity 仅有一个 DC
系数，采用 ``sigmoid(Y0 * b0)``，与观察方向无关。几何反向继续使用 signed-distance
sigmoid surrogate，并保留对称梯度 clamp；RGB-only 取消原来的单向梯度门控与基于参考
alpha 的 carve。这里的“双向”指允许扩张和收缩，不是启用旧的
``geometry_two_sided_boundary`` 窄带分支。该 surrogate 不等于硬可见性目标的精确梯度。
本阶段仍使用 1 spp 的同样本残差，尚未实现多 spp 前缀估计器，也不声明其对无限采样
平均图像 RGB 目标无偏。

训练启用下面的 RGB 生长与删除，保留已有分辨率调度及父体到子体的分裂。
分裂仍使用原有拟合与 opacity 衰减规则，本次不修改。分裂可能改变前向图像，不能视为严格
保持渲染结果。RGB-only 去掉了 alpha 约束，RGB loss 的下降本身不能证明几何或 opacity
唯一可辨识。要复现之前的无生长基线，同时关闭 ``topology.enable_growth``、
``topology.use_rgb_growth`` 和 ``schedule.enable_coarse_growth``，并关闭
``topology.enable_deletion``。``use_rgb_deletion`` 和 ``use_low_opacity_deletion`` 分别控制
RGB 收益与持续低 opacity 两条删除分支，总开关关闭时两者均不执行。
当前 RGB-only 训练入口保留 RGB 删除分支；可单独将 ``use_low_opacity_deletion=False``
以对照新增清理规则的影响。

在线 RGB 生长：复用训练前向，不增加验证渲染
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当前 RGB-only 实验用 RGB 收益投票替换旧的前景／背景 alpha 投票。
几何候选保留原规则：目标空格、26 邻域、有限面／棱／角接触、至少 0.05 个体素宽度的
外伸、同一空格至少两个合法父体支持，以及成熟期与冷却条件。
这些规则仅提出位置，不将“周围有体素”当作缺洞的证明。
候选几何沿用原初始化，颜色继承父体 radiance DC、其余 SH 为零，初始 opacity 为 0.5；
这些候选参数在一次证据事件中冻结，提交时使用同一组参数。

粗层和每个细层开始后的前 20 轮冻结生长。以层内完成轮数计数，冻结后再按 10 轮间隔
安排证据事件，因此两类层级的第一次事件均在层内第 30 轮，之后为第 40、50 等轮。
仅事件轮在正常训练前向中收集证据；其他轮不安装候选。升层时重置层内计数和候选证据，
升层边界及整个训练的最后一轮不再生长。若层级预算不够 30 轮，该层不会生长。
生长 warmup 与优化器的 opacity warmup 是两个独立配置。

新实验另外设置 ``topology.growth_freeze_last_epochs=100``，在完整训练的最后 100 轮
停止准备生长候选、收集生长证据及提交新增体素，继续以 1 spp 优化已有体素。
这是全局末尾窗口，不是每个分辨率层级各冻结 100 轮：总计 700 轮时冻结第 601–700 轮，
总计 1000 轮时冻结第 901–1000 轮。设为 ``0`` 关闭这一额外窗口；大于或等于总轮数时
整个训练都不生长。原来的层内 20 轮 warmup、分裂时机和层级预算均不变。
删除有独立的末尾冻结配置，当前也为 100 轮，见下文。
``--stop-after-epochs`` 只暂停运行，不会重新定义完整训练的末尾。

100 轮是用于检查末尾新增体素是否来不及优化的初始对照设置，不保证消除轮毂模糊；
当前默认总轮数为 1000。该设置保存在配置与 checkpoint 中，续训沿用保存值，
不允许中途悄悄修改拓扑配置。可兼容的旧 checkpoint 若没有此字段，按 ``0`` 恢复，
不会自动套用新实验的 100 轮；比较此规则应新开实验。

在需要证据的轮次，正常 DDA 经过候选空格时额外检查候选椭球相交，并记录该位置前的
合成 RGB ``P`` 和透射率 ``T``。候选并不参与当前渲染、真实命中计数或训练梯度。
利用同次前向的最终颜色 ``C``、候选颜色 ``r``、opacity ``a``，计算单个候选的虚拟增量：

.. code-block:: text

   delta = a * (T * r - (C - P))
   gain  = -2 * dot(C - reference_rgb, delta) - dot(delta, delta)

它等于该固定候选单独插入时的同样本 RGB 平方误差减少量，包含候选对后方颜色的遮挡。
这里不重新发射射线，不调用第二次整图前向，也不在提交前补一次确认渲染。
需要额外的候选相交、临时记录和 GPU 归约计算，不能将其描述为零开销。
训练仍为 1 spp；收益是有噪声的单样本证据，不宣称对无限采样平均颜色的平方误差无偏。

当前新实验使用 ``growth_rgb_normalization="observed"``。对候选 ``t`` 和视角 ``v``，
先把所有可靠命中射线的带符号收益相加，再除以这些射线的数量 ``M[v,t]``：

.. code-block:: text

   G[v,t] = sum_p(gain[v,p,t]) / M[v,t]

分母包含正、负、零收益的全部可靠命中；未命中该候选的像素不计入。扩大图像中与候选
无关的背景区域不会稀释它的分数。不同分批或 GPU 分片先 SUM 合并原始收益和计数，
再归一化并投一次完整视角票，不能先平均分片均值，也不能各分片先投票。
当 ``G > growth_rgb_gain_epsilon`` 时为支持票，``G < -growth_rgb_gain_epsilon``
时为反对票，其余为中性。每个候选在一个视角最多各计一次。

跨视角的最终收益对有可靠观测的视角取平均：``sum_v(G[v,t]) / reliable_views[t]``。
没有可靠命中或聚合收益非有限的视角不进入该分母；可靠但收益为零的视角仍进入分母。
同一候选在一个视角中可以同时有可靠和未知射线，未知射线不取消其他可靠射线的证据。
默认至少需要 3 个支持视角、反对视角少于 2，并要求上述带符号平均收益超过
``growth_rgb_min_mean_gain``，没有可靠观测的候选不会通过。任一视角产生非有限收益时，
该视角不投票，且该候选因无效数值检查被拒绝。
这衡量候选影响区域内的局部平均改善，不再代表整图平均 loss 的减少量；不同视角
等权，也不是把所有视角的射线合在一起求均值。少量 1 spp 射线可能产生较大的均值，
多视角检查不能完全消除这种噪声。整个流程不读取参考 alpha、不按 RGB 是否接近白色分类射线。

射线达到实际命中上限或透射率提前终止时，可能无法获得完整的候选后方信息。
此时仅将对应的候选／射线记录计为 ``growth_unknown``，不额外追踪补齐。
未知射线既不是支持票，也不是反对票，不进入可靠命中均值；同一候选在该视角的
其他可靠射线，无论收益为正、负还是零，仍参与统计。没有任何可靠命中时该视角才因
缺少观测而弃权；聚合收益非有限时仍按前述无效数值规则处理。
候选记录容量溢出只使未能保存的记录未知，已保存记录在其他完整路径检查通过时继续
有效；记录溢出本身不改变真实前向的颜色、透射率或此前保存的 prefix。
即使原射线完整，插入候选后可能触发命中上限或透射率终止时，对应记录也保守弃权。
例如当前候选初始 opacity 为 0.5，会将后方透射率再减半，可能使有用候选也因不满足
完整路径条件而失去部分射线证据；这需要在真实实验中结合未知计数检查。
当前仅支持均匀训练采样；原生证据遇到非单位采样权重同样弃权，不用于 contour 模式。

上述均值是以“证据可靠”为条件的局部均值。路径能否完整、记录是否溢出与遮挡和深度
有关，未知并非随机缺失，因此不能将它称为完整视角收益的无偏估计。只剩少量可靠射线
而大量记录未知时，即使均值很高，证据也可能不足。记录覆盖情况用于分析，不新增覆盖率
硬阈值；多视角支持也不能证明未知部分的收益一定同号。

配置集中在 ``train.py`` 顶部及 ``topology`` 配置组，以下是当前实验参数，不是理论保证：

.. code-block:: python

   CONFIG.topology.growth_warmup_epochs = 20
   CONFIG.topology.growth_fine_warmup_epochs = 20
   CONFIG.topology.growth_freeze_last_epochs = 100
   CONFIG.topology.growth_rgb_normalization = "observed"
   CONFIG.topology.growth_rgb_min_support_views = 3
   CONFIG.topology.growth_rgb_max_conflict_views = 2
   CONFIG.topology.growth_rgb_gain_epsilon = 1e-7
   CONFIG.topology.growth_rgb_min_mean_gain = 1e-7
   CONFIG.gradient.max_growth_hits = 8

生长不再截取前 1024 个候选或限制每次只新增 128 个体素。全部合法目标格参与
评估，全部通过证据与提交时几何复核的目标格均可加入，仍受原来的
``topology.max_active_voxels`` 总体素容量约束；候选总数与单次新增预算参数已删除。
``gradient.max_growth_hits=8`` 是每条射线的临时候选记录容量，本次保持不变；增加
候选数量可能增加相交开销及记录溢出导致的弃权。

两个收益阈值仍为 ``1e-7``，但新模式中的单位改为每条可靠命中射线的 RGB 平方误差
收益及其跨可靠视角均值，不能与旧的全图归一化阈值直接比较。阈值需结合真实实验检查。
兼容模式 ``growth_rgb_normalization="full_image"`` 保留旧的“除以整图像素数，再除以
全部训练视角数”的定义。恢复没有此字段的可兼容 checkpoint 时采用该兼容模式，续训
不允许悄悄更换归一化。带已删除生长预算字段的旧配置及 checkpoint 会报错；比较新
规则应使用当前配置新开实验。

候选在 GPU 上逐视角累积票数与收益，事件结束时才读取候选统计进行决策；安装候选和
拓扑更新仍有 CPU 几何准备、状态快照与传输。没有每视角整图回读来进行生长投票。
通过后正常优化新体素的几何、radiance SH 与 DC opacity，新生体素按原成熟期等待后
才能继续成为父体。RGB-only 删除另外使用下面的收益与保护规则。

``training.jsonl`` 的生长事件在 ``topology.growth_rgb`` 中记录归一化模式、几何／评估
候选数、有效／未知射线与视角数，以及无观测、支持不足、反对、无效数值和均值阈值等
拒绝数量。拒绝检查可能重叠，不能相加当作唯一被拒候选数；几何复核与总体素容量的
拒绝数也分别记录，便于判断补洞仍慢是证据不足还是容量限制。
``views_with_unknown`` 表示存在未知射线的候选／视角记录数，可以与可靠观测同时发生，
不是整票弃权数。``encountered_known_fraction`` 为 ``K / (K + U)``，其中 ``K``、``U``
分别为已遇到候选的可靠、未知命中数；未遇到任何候选时无此比例。射线提前结束后未访问
的候选不在分母内，因此它仅反映已遇到证据的覆盖情况，不代表候选全部受影响射线的覆盖率。

单个候选收益不是多个候选同时加入的联合收益；批量提交仍可能出现候选相互遮挡、
外层增厚或多层低 opacity 雾化。现有 RGB 优化可以降低错误候选的 opacity，但不保证
一定淘汰它们。应通过生长数量、有效／未知证据、损失及图像对照评估此次实验。
分裂保持不变，也不能用本次生长规则声称 RGB 已唯一确定几何。

删除：RGB 收益与持续低 opacity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当前新实验同时启用两条互补的删除规则：持续有正 RGB 删除收益，或者持续低 opacity，
任一分支确认后即可删除。两条规则均不使用参考 alpha、前景／背景射线分类或旧的 alpha 冲突票。
以下首先说明 RGB 收益分支。
在现有命中反向循环中，令 ``T`` 为某体素命中前的透射率，``a`` 为其 opacity，
``c`` 为它的颜色，``B`` 为从该命中之后看向射线末端的合成颜色（包含背景），
``C`` 为当前完整预测，``y`` 为参考 RGB。单独移除这个体素的颜色变化及收益为：

.. code-block:: text

   dc   = T * a * (B - c)
   gain = ||C - y||^2 - ||C + dc - y||^2
        = -2 * dot(C - y, dc) - dot(dc, dc)

这里使用三个 RGB 通道平方误差的和，与训练原生 loss 一致，不额外除以 3。
收益大于零表示在当前射线及参数下，单独删除降低误差；负收益表示保留更好。
``B`` 已由反向合成提供，不通过除以 ``1-a`` 反推，因此不会在 opacity 接近 1 时引入
该除法的不稳定。删除不重新追踪、不延长已结束的 DDA、不额外保存一套后缀；但会增加
GPU 上的算术、按体素归约和事件统计，并非零开销。

确定收益仅来自正常离开网格或到达 ``ray.maxt`` 的完整基线路径，且采样权重为 1、
相关数值有限。因透射率或实际命中容量提前停止的路径，在删除后可能露出未访问区域，
相应命中只计 ``deletion_unknown``；未知按射线弃权，不取消其他可靠射线的证据。
删除减少命中数并增加透射率，所以不套用生长的“新增一个命中的剩余容量”或
“插入后最终透射率仍高于阈值”检查。没有命中、只有未知记录或聚合收益无效的视角
不能提供删除支持；未知不是无害证明。

每视角先 SUM 合并分批／多 GPU 的 ``deletion_gain``、``deletion_observed`` 和
``deletion_unknown``，再以可靠命中数归一化收益，正、负、零收益均参与。视角均值超过
``deletion_rgb_gain_epsilon`` 为支持票，低于其负值为反对票，其余中性；
跨视角再对可靠观测视角均值求平均，可靠零收益视角仍计入分母。
当前要求至少 3 个支持视角、没有反对视角，且跨视角带符号平均收益大于 ``1e-7``。
这里的反对是一个视角的可靠命中平均收益为负，不是任意单条负收益射线直接否决。
任一视角出现非有限聚合收益时，该体素本窗口不通过。

配置集中在 ``train.py`` 的 ``CONFIG.topology``：

.. code-block:: python

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

每个分辨率层级先 warmup 20 轮，然后第 30、40、50 等轮检查删除。RGB 分支的每个窗口只包含
该轮的全部训练视角。体素连续两个完整 RGB 证据窗口通过才由该分支删除，中间不通过或缺少窗口会
重置连续计数。因此初始体素最早可在层内第 40 轮删除；新生体素另外先保护 20 个完整
训练轮次。删除后的格子沿用原冷却期，避免立即被重新生长；升层后重新 warmup，
证据与连续确认状态重置。删除与生长都在 epoch 边界处理，幸存体素保留 Adam 状态。

全局最后 100 轮停止收集及提交删除，配合生长冻结让已有体素继续优化；1000 轮对应
第 901–1000 轮。``deletion_freeze_last_epochs=0`` 关闭该窗口，大于或等于总轮数时全程
不删除。``--stop-after-epochs`` 不改变这个全局窗口。分裂规则、训练的 1 spp 不变。
RGB 分支不因没有观测、零收益或 opacity 较低而删除；持续低 opacity 由独立分支处理。

低 opacity 分支在上述事件的 epoch 结束时，读取当前模型的 DC opacity 参数，
按真实前向公式 ``a = sigmoid(Y0 * b0)`` 计算不透明度，连续两个检查窗口都满足
``a < deletion_opacity_threshold`` 才通过。它比较的是不透明度，不是未经 sigmoid 的 SH 系数，
当前只支持各向同性 DC opacity。某次不再低于阈值会重置连续计数；
非有限参数不能作为低 opacity 证据。
这个分支不要求 RGB 删除收益为正，因此可以清理 RGB 删除收益已接近零的透明体素。
它不把没有命中或没有观测当作零贡献，也不额外运行前向；只有事件末的 opacity 参数读取与检查。

两条分支共用层级 warmup、新生体素 20 轮保护、删除周期、冷却期和末尾冻结。
细分后重新开始保护与确认，避免直接清理经过 opacity 衰减、尚未重新优化的子体素；
幸存体素的连续确认状态随 checkpoint 保存和恢复。
``0.005`` 与两个窗口是初始实验设置，并非已验证的最佳阈值。低 opacity 裁剪是近似清理：
很多低 opacity 体素仍可能共同产生可见贡献，持续低值不构成删除后图像不变的证明。
本阶段尚未实现按 ``T * a`` 的低贡献统计裁剪，保留它作为后续独立对照。

``training.jsonl`` 的 ``topology.deletion_rgb`` 记录有效／未知命中、支持／反对、
保护期、拒绝原因、等待确认与已确认数量；``topology.deletion_opacity`` 另外记录低 opacity
检查与确认统计。两条分支可能命中同一个体素，``topology.deleted`` 是合并后本次实际删除数，
不能将两个分支的确认数直接相加。
这些拒绝原因可能重叠，不能求和当作被拒绝的唯一体素数。可靠比例仍仅覆盖已遇到的
命中，未访问区域不在分母中。条件观测均值不能保证未知区域无损。

训练仍然每视角更新一次参数，所以不同视角的证据对应不同的在线参数快照，不是
同一个固定模型的全数据集联合删除损失。多个体素各自删除有益，也可能一起删除后
损害图像；两窗口确认、反对票及新生保护降低误删风险，但不构成联合下降证明。
当前不额外渲染验证联合删除，也不保证恢复轮毂细节或消除白雾，应通过新实验对照检查。

源码版本、续训与启动
~~~~~~~~~~~~~~~~~~~~

此前按射线处理未知证据时修改了原生候选记录溢出规则，需要相应的 C++ 与绑定。
RGB-only 启动会检查 ``mi.VoxelReconstruction.rgb_only_version``，启用生长还要求
``rgb_growth_version >= 2``；旧二进制不能只更新 Python 后继续使用。
已有 RGB 删除要求 ``mi.VoxelReconstruction.rgb_deletion_version >= 1``；
若尚未编译该原生接口，**同步代码后必须增量编译，再启动新进程**。
本次持续低 opacity 分支只修改 Python 调度，已具备该接口的构建无需重新编译 C++。
checkpoint 保存内部证据版本，使用旧版 RGB 生长证据规则的训练不能直接升级续训，应新开实验；
旧模型仍可独立评价，证据版本不改变已保存的体素渲染参数。
可兼容旧 checkpoint 缺少 ``use_rgb_deletion`` 时恢复为 ``False``，保留原来已保存的
删除开关与调度，原先关闭删除的 RGB 实验不会自动启用新删除。要比较新规则，使用
当前配置新开实验，不在旧 checkpoint 上切换算法。
同样，旧 checkpoint 缺少 ``use_low_opacity_deletion`` 时按 ``False`` 恢复；
只有当前 ``train.py`` 的新实验配置默认启用低 opacity 清理，不会改变既有实验的续训行为。
从仓库目录启动新的单卡实验：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/train.py \
     --rgb-only --dataset datasets/lego --epochs 1000 --chunk-size 0 \
     --eval-spp 64 --eval-max-views 10

在已经分配同节点两张 GPU 的 Linux Slurm 作业内启动新的双卡实验：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/train_multigpu.py \
     --workers 2 --worker-timeout 600 \
     --rgb-only --dataset datasets/lego --epochs 1000 --chunk-size 0 \
     --eval-spp 64 --eval-max-views 10

两条命令都自动创建新的实验目录。需要短跑时添加 ``--stop-after-epochs 10``，保持
``--epochs 1000``，避免改变完整调度。训练后的 64 spp 只用于评价出图，不改变训练 spp。
双卡平台要求和资源分配方式见下文“同节点两张 GPU 训练与评价”。
十轮只检查正常训练路径，不会触发默认 warmup 后的生长或删除。
第一次生长／删除证据在第 30 轮；观察删除的第二次确认机会可短跑 41 轮，
但是否实际删除仍取决于 RGB 收益或持续低 opacity 的确认条件。从仓库根目录执行 Windows 短跑：

.. code-block:: powershell

   & '..\envs\mitsuba\Scripts\python.exe' -I .\experiments\reconstruction\train.py `
     --rgb-only --dataset datasets/lego --epochs 1000 --chunk-size 0 `
     --stop-after-epochs 41 --skip-evaluation --no-profile-stages

已有 alpha 监督 checkpoint 禁止切换为 RGB-only 后续训，包含没有记录
``training.rgb_only`` 的旧 checkpoint；应从初始化开始新的 RGB-only 实验。
RGB-only checkpoint 同样不能切回旧监督模式。旧模型仍可通过
``experiments/reconstruction/evaluate.py --checkpoint <实际模型路径>`` 单独评价。
若要按原算法恢复旧实验，明确使用 ``--no-rgb-only --resume <旧 checkpoint 路径>``。
配置类 ``TrainConfig()`` 及原生接口的默认行为保留 legacy 语义；新实验默认由上面的
``CONFIG`` 显式开启，不能把类默认值当作当前实验配置。

编译后的生成场景 GPU 数值回归命令：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/tests/test_reconstruction_rgb_only.py
   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/tests/test_reconstruction_rgb_growth.py
   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/tests/test_reconstruction_rgb_growth_lifecycle.py
   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/tests/test_reconstruction_rgb_growth_training.py
   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/tests/test_reconstruction_rgb_deletion.py
   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/tests/test_reconstruction_rgb_deletion_driver.py
   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/tests/test_reconstruction_rgb_deletion_lifecycle.py
   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/tests/test_reconstruction_low_opacity_deletion.py
   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/tests/test_reconstruction_rgb_deletion_training.py

该测试不打开真实数据集或既有实验，覆盖参考 alpha 不变性、RGB L2 的颜色与 opacity
有限差分、白色目标及 miss 的几何梯度、DC opacity、三通道缓存、1 → 2 分裂及负 loss
拒绝。负平方误差会直接报错，不会被 clamp 为零；这项检查不能替代对双卡归约异常的调查。
RGB 生长测试另外使用小型合成场景，对照显式插入候选的独立渲染验证虚拟收益，检查
正／负／中性证据、参考 alpha 不变性、原图与梯度不变性、截断弃权、未知与可靠射线共存、
溢出时保留已保存记录、分片先求和后
归一化、局部归一化不随无关像素数变化、可靠视角分母、不限 1024／128 的生长规模、
旧归一化兼容与证据版本不匹配时的续训拒绝，以及粗细层 warmup。
测试中的独立渲染仅是数值验证的参照，
训练投票不执行这些额外渲染。
RGB 删除回归以实际移除体素后重新渲染的结果校验收益，覆盖正／负／零收益、
提前终止与完整边界、未知与可靠射线混合、接近不透明的命中、禁用时的图像与梯度
不变性、分片 SUM、原生重放、空模型和非法设置。生成场景检查不替代真实 Lego 的
删除效果或多卡服务器吞吐测试。
下文保留通用运行细节与旧实现说明，其中 alpha 监督、contour 和旧生长／删除规则
属于 legacy 配置；当前实验以本节及当前 ``CONFIG`` 为准。

单卡训练入口为 ``experiments/reconstruction/train.py``，实验性的同节点双卡启动器为
``experiments/reconstruction/train_multigpu.py``，评价入口为
``experiments/reconstruction/evaluate.py``。它们共享场景表达、相机、源码运行环境和
前向渲染工具；``preview.py`` 继续用于初始化检查。
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
再计算图像均方误差。训练采用用户确定的 1 spp 目标；多 spp 用于初始化对照图及训练后出图与评价，
不声称两种损失完全等价。

新 RGB-only 实验使用 ``data.reference_compositing="srgb"``。对于标记为 sRGB 的输入
（通常为 PNG），设原始编码 RGB 为 ``s``、alpha 为 ``a``，sRGB 解码函数为 ``D``，
白底参考为 ``D(a*s + (1-a))``；合成在输入编码空间进行，随后只把线性 RGB 送给训练。
渲染、损失、梯度，以及生长和删除收益仍全部在线性空间计算，不在 sRGB 上优化。
对于线性输入（如未标记 sRGB 的 EXR），继续在线性空间合成；不会给线性数据套用 sRGB 解码。
背景参数仍表示线性 RGB；评价若指定非白背景，编码空间合成前先把背景编码到 sRGB。

``data.reference_compositing="linear"`` 保留旧协议：先解码，再按
``a*D(s)+(1-a)`` 合成白底。这两个协议在半透明、抗锯齿边缘不等价。
新协议统一的是参考图准备方式，不等价于 NeRF／3DGS 的 sRGB 训练损失。
模式随配置和 checkpoint 保存：旧 checkpoint 缺少该字段时按 ``linear`` 续训和评价，
不会被新的脚本默认值悄悄改变。续训不允许切换参考协议；比较新旧协议应新开实验。
训练的预览、CPU／GPU 参考缓存及自动评价均使用对应配置中的同一协议。

训练脚本明确要求 ``render.background=[1,1,1]``，非白背景会报错，防止意外优化不一致的目标。
评价是独立流程：评价时的背景设置会同时用于预测与参考合成。此前预览默认黑底，
比较实验必须明确对齐实际背景，不能仅凭 pass 名称认为设置相同。

统一配置
--------

``train.py`` 顶部 ``CONFIG = TrainConfig()`` 及其后面的赋值块为统一入口。
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
   CONFIG.training.gpu_reference_cache_mib = 1024  # 参考图显存缓存；0 恢复逐视角上传
   CONFIG.training.freeze_native = True  # 重放原生计算，拓扑变化后重新记录
   CONFIG.schedule.enabled = True
   CONFIG.schedule.start_resolution = 256
   CONFIG.schedule.target_resolution = 512
   CONFIG.training.num_epochs = 1000
   CONFIG.training.save_count = 3
   CONFIG.training.output = None  # 自动创建 outputs/场景-月.日-E编号/training
   CONFIG.evaluation.enabled = True  # 保存初始化对照图，并在训练完成后自动出图
   CONFIG.evaluation.spp = 64
   CONFIG.evaluation.max_views = 10  # train/test 各取相机 JSON 中前十个视角

``datasets/lego`` 相对于源码仓库的父目录，即 ``reconstruction/datasets/lego``。
点云和相机文件名相对于该场景目录；训练会遍历相机 JSON 中的全部视角。
如果新场景使用相同的文件名，只需改 ``dataset``。还需确认其坐标范围适合模型边界，
参考图尺寸与 ``render.width/height`` 一致。

训练脚本从 ``preview.py`` 导入的是 ``DataConfig`` 等配置类，也就是
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
  ``profile_stages``、``reference_cache_mib``、``gpu_reference_cache_mib``、
  ``freeze_native``、``seed``、``shuffle``、
  ``save_count``（默认 3，包含最后一次）和输出路径 ``output``；``None`` / JSON ``null``
  表示自动分配实验目录，字符串表示使用显式路径。
* ``gradient``：alpha 损失、几何代理宽度/截断、边界和删除证据等参数。
* ``optimizer``：各类参数学习率、Adam、裁剪、步长限制和不透明度调度。
* ``topology``：体素增删及证据条件。
* ``schedule``：coarse-to-fine 开关和分辨率计划；启用时由此组决定起始分辨率。
* ``contour``：训练时可选的 contour 重要性采样。
* ``runtime``：源码构建目录和本地 GPU 编号。
* ``evaluation``：初始化对照图及训练结束后出图设置，默认 ``enabled=True``、``spp=64``、
  ``max_views=10``、``chunk_size=65536``、``seed=42``，测试相机文件由
  ``test_cameras="transforms_test.json"`` 指定；``max_views=None`` 表示全部视角。

仅想训练时可传 ``--skip-evaluation``，或设置 ``CONFIG.evaluation.enabled=False``，
同时跳过初始化对照图与最终评价。
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

参考图内存与显存缓存
--------------------

默认 ``training.reference_cache_mib=1024``，以 MiB 为单位限制训练参考图的 CPU 缓存。
第一次访问一个视角时，仍通过原来的读取流程解码为线性 RGBA float32，并检查尺寸和有限值；
后续访问直接复用只读数组，避免每个 epoch 重复读取和解码 PNG。这层缓存使用 CPU 内存；
参考数据的重复上传由独立的显存缓存处理。100 张 800 × 800 RGBA float32 约占 976.6 MiB，
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

默认 ``training.gpu_reference_cache_mib=1024`` 将参考图另外保存在 GPU：当前 RGB-only
使用三个 float32 通道，legacy 模式额外保留参考 alpha，共四通道。
整图直接复用，分批则在 GPU 上读取对应像素。使用
``--gpu-reference-cache-mib 2048`` 调整容量，或 ``--gpu-reference-cache-mib 0``
恢复原来的逐视角上传；CPU 缓存开关不受影响。容量不足时按最近使用顺序淘汰，单图超过
总容量则使用原上传路径。两个缓存都开启时，100 张 800 × 800 图在 RGB-only 下分别占约
732.4 MiB 内存和显存，legacy RGBA 为 976.6 MiB。这里限制的是缓存持有图像的逻辑字节数，
不包括 Dr.Jit 分配器保留的空间、
仍被正在执行的调用引用的淘汰数组，以及模型、梯度和射线记录；不能据此推断显存峰值。

legacy 模式的 CPU 参考图仍用于 contour 和旧生长证据；当前 RGB 生长直接复用原生前向
的参考 RGB 与渲染结果，不另读 CPU 参考图投票。子像素随机采样继续使用原 NumPy 随机数流，
偏移和需要时的 PDF 权重仍需上传；显存缓存只省掉固定参考图的重复上传，不改变采样、
损失、梯度或逐视角更新。缓存不写入 checkpoint，恢复进程重新填充，并在自动评价前释放。
``gpu_reference_cache`` 随逐轮日志、训练报告及性能汇总记录容量、驻留字节数、图片数、
``hits``、``misses``、``uploads``、``uploaded_bytes`` 和 ``evictions``。
上传统计只计算显存缓存负责的上传；关闭缓存或单图过大时原路径的上传不计入该字段。

原生调用重放
------------

默认 ``training.freeze_native=True`` 使用 ``dr.freeze`` 记录原生 ``loss_and_grad``
中的 GPU 工作，兼容的后续调用直接重放，减少重复追踪和组织计算图的主机开销。
它不缓存渲染结果：模型参数、相机射线、参考图和 PDF 每次都是当前调用的输入，Adam
替换后的参数立即参与下一视角。原来的解析外观梯度、几何代理梯度及每视角一次更新保持不变；
这不是将算法改为普通自动微分，也不表示将全部计算融合成一个 GPU 内核。

重放缓存按原生设置、输入宽度、网格和 SH 元数据区分记录，最多保留 8 个配置条目，
每项最多 2 个 Dr.Jit 记录。生长、删除或升层重建模型时先清空旧记录；更换模型对象也会
使记录失效，即使体素数相同。之后首次调用重新记录，checkpoint 恢复亦如此。
缓存只属于本次进程，不保存到模型或优化器状态。``--no-freeze-native`` 可关闭，
``--freeze-native`` 可显式开启；恢复训练时也可调整这两个缓存预算及重放开关。
零活跃体素使用原生 CUDA 空模型路径，不参与重放；当前 Dr.Jit 对零长度模型缓冲的记录存在限制。

``native_replay`` 记录 ``calls``、``recordings``、``replays``、``bypasses``、
``entries``、``cached_recordings``、``clears``、``model_changes``、``evictions``、``empty_model_bypasses`` 等统计。
``recordings`` 来自 Dr.Jit 实际记录计数，不能与内核编译未命中数混为一谈。
预热后观察 ``replays`` 是否增长，并结合普通 epoch 耗时判断收益；首次记录和拓扑变化后的
重新记录会有额外开销。设置 ``profile_stages`` 的抽样视角刻意旁路重放，详见性能诊断。

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
* 原生 ``set_parameters()`` 对本次更新的参数组先合并有限值检查，正常路径只读取一次
  校验标志；失败时再定位具体参数组，且全部检查通过前不改变模型。
* 原始 ``raw_alpha`` 只在 contour 采样器实际产生更新 token 时回读，用于更新采样历史。
* legacy 模式中，概率校正后的 alpha 只在本轮需要收集旧生长证据时分配，训练入口直接传给 GPU 投票，
  不再先下载整张图。参考 alpha 复用 GPU 参考缓存；缓存关闭时按需上传。
* legacy 删除证据在 GPU 的跨视角 bitset 中累积，同一相机重复命中仍只计一次。
  到证据窗口边界才统计唯一视角数并交给原来的候选判定；保存、恢复和体素重排保持原状态格式。
* legacy 生长证据的七个冻结采样点上传一次，各视角在 GPU 上投影并投票，在 epoch 边界取回票数。
  背景否决、前景支持门槛和完整视角集合检查保留。contour 的历史更新仍使用原 CPU 路径。
* 当前 RGB 生长不走上述 alpha／七点投影路径；它在同次训练前向中计算每候选收益，
  GPU 上累计完整视角票数，到生长轮结束才读取候选级统计。
* 当前 RGB 删除复用命中反向，GPU 上按体素累计可靠收益与视角票，到证据轮末读取
  统计、更新连续确认次数；不为投票回读完整图像。

这些调整不在训练的每个视角或每个 epoch 内出图。完成全部训练并保存最终模型后，
训练入口调用评价脚本共用的流程，生成多 spp 图片和 PSNR。

拓扑证据与候选复核的性能对照
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

本小节记录此前 legacy alpha 生长／删除路径的性能优化；当前 RGB 在线生长的定义见文首。
legacy 模式的 GPU 删除证据和生长投票自动用于训练入口，旧的 NumPy 输入路径保留为数值对照。
epoch 结束后，先检查冻结生长提案的票数，只对通过投票的目标格及其所有必要邻居
复核当前几何。支持父体素本身不必通过投票；冷却期、等待期、已占用格、支持数、
背景否决及确定性胜者顺序保持原规则。投票未通过时不再全量重建候选。
首个生长视角仍在 CPU 生成冻结候选，但只回读几何参数，不再为该步骤复制全部 SH 参数。

``training.jsonl`` 的 ``topology_execution``、``training-report.json`` 的同名字段及
``performance-summary.json`` 的 ``details.topology_execution`` 记录本次进程累计的
GPU/CPU 证据处理视角数和实际边界回读字节数。读取这些计数不会同步 GPU。
legacy 训练启用删除/生长时，分别检查 ``deletion_device_views`` / ``growth_device_views`` 增长；
没有启用的步骤保持零。阶段耗时仍是包含异步等待的墙钟时间。
当前 RGB 生长对应 ``rgb_growth_device_views``、``rgb_growth_cpu_views`` 和
``rgb_growth_readback_bytes``，不要用旧 alpha 生长计数判断 RGB 生长是否执行。
RGB 删除对应 ``rgb_deletion_device_views``、``rgb_deletion_cpu_views`` 和
``rgb_deletion_readback_bytes``，与旧 alpha 删除计数分开。

此前这一性能优化没有修改原生接口，但当前新增在线 RGB 生长修改了 C++ 与绑定，必须重新编译。
当前默认 RGB 生长与旧 alpha 生长算法不同，不能直接将它们的耗时差解释为同算法提速。
要比较刚才的 512 阶段，可在同样的双卡分配中从旧实验保存点继续若干轮，保留全部算法设置：

.. code-block:: bash

   # 将下面 checkpoint 和输出路径中的实验名替换为实际名称；输出目录须尚未使用。
   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/train_multigpu.py \
     --workers 2 --resume outputs/原实验/training/epoch-00467.npz --no-rgb-only \
     --output-dir outputs/原实验-topology-check/training \
     --stop-after-epochs 20 --chunk-size 0 --freeze-native \
     --profile-views 3 --profile-interval-epochs 10 --no-profile-stages

不要同时修改总 epoch 或层级比例；恢复过程会校验它们。短测试在完整 epoch 后保存，
不会触发最终评价。分别比较普通轮、证据窗口轮和生长轮的
``view_stage_seconds.topology_observation``、``epoch_topology_seconds`` 以及总 ``seconds``。
本地生成场景回归不能代替双 A100 的实际吞吐和 NCCL 验证。

新增的生成数据回归在仓库根目录运行，测试临时文件写在系统临时目录：

.. code-block:: powershell

   & '..\envs\mitsuba\Scripts\python.exe' -I experiments/reconstruction/tests/test_reconstruction_growth_filter.py
   & '..\envs\mitsuba\Scripts\python.exe' -I experiments/reconstruction/tests/test_reconstruction_growth_device.py
   & '..\envs\mitsuba\Scripts\python.exe' -I experiments/reconstruction/tests/test_reconstruction_topology_device.py

2026-10-19 本地 RTX 4060 Laptop 的独立合成对照：先预热一次，三个重复取中位数，
计时结束等待 GPU 完成。128 万体素、100 个视角的删除证据累积由约 1.976 秒降为
0.018 秒，窗口计数及资格判定由约 0.125 秒降为 0.017 秒；资格及候选结果相同。
10 万个冻结生长提案、20 个视角的七点投票由约 0.486 秒降为 0.012 秒，票数相同。
这些只测证据处理，排除首次候选生成、参考上传、渲染、Adam、NCCL 和完整模型重排，
不是整轮训练或双 A100 的加速比。生成小场景的训练、检查点恢复、GPU 参考缓存关闭、
旧拓扑规则及 RGB 第一阶段回归另行通过；完整 700 轮仍需服务器验证。

原生前向/反向过程中的射线记录按射线和命中槽独占地址写入。对这些已确认无冲突的
写入显式使用 ``ReduceMode::NoConflicts``，使 Dr.Jit 不再为每次记录写入安排不必要的
求值与内核边界。多个射线向同一体素累加梯度仍采用原来的归约规则，不将其声明为无冲突。
早期本地生成场景的原生内核回归中，唯一 JIT 内核数从此前的 66 降至 18；这表示减少了
内核拆分，不能理解为端到端训练获得相同比例的加速。实际收益仍需结合目标设备上的
``gradients`` 和内核抽样耗时判断。

命中记录同时改为按字段分离（SoA）、字段内按命中序号排列射线的布局，索引为
``field * ray_count * max_hits + hit * ray_count + lane``。同一命中序号下，相邻线程
连续访问同一字段，改善记录读写的访存连续性；记录字段、缓冲元素总数与每条射线的
命中顺序不变，前述 ``NoConflicts`` 写入约定继续保留。这只调整记录的存取方式，
不改变场景参数布局、checkpoint 格式、采样与梯度公式。

命中反向现在使用 GPU 内的符号逆序循环，从 ``max_hits - 1`` 遍历到 0，
各射线仍按自己的实际命中数屏蔽无效槽，保留原来的逆序递推顺序。命中记录只读，
候选反向需要的后缀 RGB 和透射率写入独立的四个 float32 字段，替代旧实现中
逐槽覆写命中记录的方式，避免每个命中槽产生一次读写依赖和内核提交。
几何与外观公式、PDF 权重、截断/方向门控、计数及删除证据规则保持不变。
这增加 ``4 * max_hits * ray_count * 4`` 字节的后缀缓冲；800 × 800 整图、
``max_hits=16`` 时为 163840000 字节，即 156.25 MiB，不代表整个训练的显存峰值。

训练原生前向使用 GPU 条件分支区分空格、命中和有效未命中：空格只推进 DDA，
占用体素才计算椭球几何，命中才计算外观并写命中记录，有效未命中才维护候选。
候选容量已满时，先用新分数与当前最差分数比较，仅在严格小于时执行插入；
容量未满时仍正常插入。评分、候选容量、稳定的同分顺序、命中顺序和终止条件保持不变。
这些修改跳过不会影响结果的工作，保留 ``VoxelReconstruction`` 接口、梯度公式及
每视角一次参数更新；没有改变算法或引入空区层次加速结构。实际收益取决于设备、
射线分布和场景稀疏程度，不能保证 A100 的端到端吞吐提升。

legacy 的 256 → 512 调度在 256 阶段不消费删除证据。训练将其实际证据需求传给原生
``collect_deletion_evidence``，跳过这些未使用的计算；进入需要收集删除证据的目标
分辨率后照常计算，跨视角累计、删除阈值、保护期与执行时机保持不变。256 阶段的生长
仍然启用；跳过删除证据不等于关闭生长或删除功能。
当前 RGB 删除使用独立的 ``collect_rgb_deletion_evidence``，粗细层都可在各自 warmup
后的事件轮收集，规则见文首，不能套用上述 legacy 的“仅精细层”限制。

legacy 生长证据期间，每批冻结候选的中心和六个轴向采样点在世界空间中只计算一次，后续
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
  legacy 的 512 阶段每轮常规采集删除证据，本身不将该轮归为拓扑事件轮；
  当前 RGB 删除的周期证据评估轮由独立删除事件记录。
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
不会额外抽取视角。这些视角旁路 ``dr.freeze``，以保留原生阶段边界；普通视角仍使用重放。
``--no-profile-stages`` 关闭分段，仍保留抽样视角的总内核统计，并允许抽样视角使用重放。
若目的是测量重放后的训练吞吐，应使用后者；需要定位前向/反向时再保留分段诊断。
若需要完全关闭内核抽样，可传 ``--profile-interval-epochs 0 --profile-views 0``，
普通逐轮耗时及性能汇总仍保留。恢复时可调整这些性能测量参数，不改变算法设置。
抽样有额外开销，比较优化前后速度时应使用相同的测量设置。

``training-report.json`` 的 ``profile_file`` 给出内核记录的实际路径；终端只提示文件位置。
每条 profile 记录包含各阶段墙钟时间、JIT 内核数、缓存命中/未命中数、内核哈希、
代码生成时间 ``codegen_ms``、后端编译时间 ``backend_compile_ms`` 和设备执行时间
``device_ms``。设备时间取自 Dr.Jit kernel history，只涵盖所记录的操作；这是抽样数据，
不能按它直接外推整个实验的 GPU 时间或多 GPU 加速比。
记录的 ``native_execution`` 区分 ``frozen``、``direct``、``direct_stage_profile`` 和 ``direct_empty_model``，
并附 ``native_replay`` 累计统计。逐轮 ``training.jsonl``、最终 ``training-report.json``
及 ``performance-summary.json`` 也记录重放与显存参考缓存统计，便于确认优化实际启用。
汇总的 ``kernel_profiles.by_native_execution`` 分开统计各执行路径，避免将分段诊断当成重放性能。

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
普通训练入口仍只使用一张 GPU；双 GPU 必须使用下文的专用启动器，
性能报告本身不会启用分布式训练。

训练完成后，可以直接提供以下小文件，无需发送 ``model.npz`` 或图片：

* ``training/performance-summary.json``：首先查看整体瓶颈和各分辨率阶段。
* ``training/training.jsonl``：核对逐轮变化、体素数量和拓扑事件。
* ``training/profile-training.jsonl``：检查抽样内核的缓存、编译及设备执行时间。
* 实验根目录的 ``evaluation-report.json``：分析 64 spp 出图耗时。

若中途恢复过训练，同时提供相应的 ``*-resume-*`` 文件；仅运行了训练、关闭了内核抽样
或尚未开始评价时，对应文件可能不存在。比较两个实现版本时，应保持数据集、随机种子、
训练预算、采样数、GPU 型号和测量设置一致，并分别查看预热后的普通轮次和拓扑变化轮次。
是否值得增加 CPU、优化单卡或进一步设计多 GPU 梯度汇总，应据这些实测结果判断。

固定输入的短基准
----------------

``experiments/reconstruction/benchmarks/benchmark_reconstruction_forward.py`` 用于比较原生实现版本，默认读取
``datasets/lego``，按点云初始化固定的 256³ 网格，测试训练相机索引 0、21、56
（从 0 开始），每个视角为 800 × 800、1 spp。固定随机种子生成射线与参考输入，
在计时前完成输入上传。脚本不更新参数或拓扑，不运行 epoch，也不生成图片或模型结果。

在已完成源码编译的 HPC Slurm 终端执行：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/benchmarks/benchmark_reconstruction_forward.py \
     --dataset datasets/lego

Windows 在 ``reconstruction`` 工作区父目录执行：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\benchmarks\benchmark_reconstruction_forward.py `
     --dataset datasets/lego

默认先预热 3 次，再记录 7 次普通原生调用的墙钟时间；这项时间覆盖完整
``loss_and_grad`` 及其 GPU 计算完成等待，包含反向工作，不是纯前向时间。
输入上传、参考图读取、射线生成、结果回读、优化器和拓扑更新均不在该计时范围。
随后独立预热并记录 3 次分段诊断，提供 ``forward``、``hit_backward`` 和
``candidate_backward`` 的指标；这些分段同步会改变执行方式，应与普通调用分别比较。
报告还提供 ``ordinary_kernel_history_samples`` 与 ``ordinary_kernel_history_medians``：
在关闭 ``profile_stages`` 的完整原生调用上独立预热 1 次、抽样 3 次，记录整体内核统计；
这些抽样与无内核记录的普通墙钟样本分开，不把不同测量方式的时间相加。

终端打印输出位置，默认在系统临时目录创建独立目录并保留 ``benchmark.json``，
不占用重建实验的 E 编号。``--output-dir`` 可指定源码仓库外的报告目录；
``--save-arrays`` 可额外保存每个相机的原生结果和梯度数组，供不同版本数值对照。
``--cameras``、``--warmup`` 和 ``--repeats`` 可调整相机或重复次数。比较版本时保持
这些设置、数据和硬件一致；此短基准不替代完整训练的逐轮耗时、收敛与最终评价验证。

调用重放与显存缓存另有完整视角更新基准：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/benchmarks/benchmark_reconstruction_execution.py \
     --dataset datasets/lego --ablations

该脚本默认使用相同的三个 800 × 800、1 spp Lego 视角，以及固定的初始化 256³ 拓扑。
它计时 ``train_view`` 和一次 Adam 更新，包括 CPU 采样、射线构造、参考上传或缓存访问、
原生前向/反向、标量检查回读以及 GPU 完成等待；不包含 PNG 解码、模型初始化、拓扑操作、
checkpoint 或最终评价。一轮只遍历选定相机，不是完整 epoch。

默认对比全部关闭和全部开启；``--ablations`` 额外测量仅开启重放、仅开启显存缓存。
每种模式从相同模型、空 Adam 状态及相同随机种子开始，视角顺序一致，模式执行顺序轮换；
预热更新在各模式中同样保留。默认预热 1 轮、测量 3 轮，报告同时核对 loss、最终模型和
Adam 状态的数值误差。可调整 ``--cameras``、``--warmup``、``--repeats``、``--chunk-size``；
``--gpu-cache-mib`` 必须足以容纳所有选定参考图。输出默认写入系统临时目录的
``benchmark.json``，不生成重建实验或图片，不占用 E 编号。这个基准用于区分两项优化的
收益，不能直接外推完整 1000 epoch 或跨设备的加速比。

``--repeat-baseline`` 增加一组独立的原执行路径重复实验，可与 ``--ablations`` 合用，
用于区分原路径自身的数值波动与新路径的差异。比较保留严格阈值，并记录每个数组的
超差元素数、RMS、最大误差及对应数值；超差时报告 ``numerical_mismatch`` 并返回非零退出码，
即使计时已经完成也不会把它标成数值验证通过。

2026-10-19 本地 RTX 4060 Laptop、源码 CUDA 构建实测：以上三视角预热 2 轮、计时 5 轮，
每模式共更新 21 次，其中 15 次参与计时。完整步骤墙钟中位数如下；该对照使用同一个已合并
参数校验的 C++ 构建，不单独测量参数校验合并的收益。

.. list-table:: 本地三视角短基准（毫秒/视角）
   :header-rows: 1

   * - 执行方式
     - 中位数
   * - 关闭重放和显存缓存
     - 43.61
   * - 独立重复原路径
     - 44.92
   * - 仅重放
     - 44.58
   * - 仅显存缓存
     - 40.33
   * - 两项开启
     - 40.40

本次合并路径比第一组耗时降低约 7.4%，重放单独没有显示稳定收益。合并路径记录 1 次、
重放 20 次，三张参考图只上传 3 次。各组的 21 个 loss 完全相同，但多步参数/Adam 比较
存在严格阈值超差：原路径重复运行的旋转数组 323036 个元素中 36 个超差，合并路径为 21 个；
对应旋转最大绝对差分别约 ``1.90e-4`` 和 ``1.12e-4``，全部状态的最大差分别约 ``4.83e-4``
和 ``1.12e-4``。因此本次真实场景基准报告保留 ``numerical_mismatch``，不能宣称多步结果逐位一致，
也不能据此确认完整训练收敛等价。A100、512 阶段及完整 1000 epoch 的时间/收敛仍需实测。

生成小场景的重放 45 项、参考缓存、参数事务、训练/跨执行模式断点恢复与升层、
自动评价和现有性能回归均已通过；这些覆盖范围与上述真实场景多步严格对照应分开理解。
另用相同的真实 Lego 初始化模型、相同射线/参考输入对三个视角单独比较，原路径重复与
原路径/重放的原生梯度最大差均为 ``1.9e-6`` 至 ``3.8e-6``，通过 ``atol=1e-5, rtol=3e-5``
检查；一次 Adam 后模型最大差不超过 ``4.77e-7``。提前求值射线也未消除原路径自身的差异。
这些结果支持并行浮点累加及后续更新放大误差的解释，但不替代完整训练的收敛比较。

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
     train/             # 初始化对照图、最终模型训练集视角 PNG 和 metrics.json
       0000-initial.png  # 第一次参数更新前，训练相机索引 0 的初始化体素图
       0000.png         # 同一相机下的最终重建图
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

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\evaluate.py `
     --checkpoint outputs/lego-10.15-E0/model.npz --split both --spp 64 `
     --output-dir outputs/lego-10.15-E0

若只需重跑当时保存的两个视角及 4 spp 设置，则使用：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\evaluate.py `
     --config outputs/lego-10.15-E0/evaluation-config.json

``outputs/ficus-10.14-E0`` 保留较早的 Ficus 初始化模型和训练第一个视角的预览，
尚无训练更新或测试集视角结果。冒烟测试、合成场景回归和构建日志不作为重建实验保留。
Dr.Jit、CUDA 和 OptiX 缓存写入源码仓库父目录（``reconstruction`` 工作区）下的
``.cache/mitsuba/``，不再放进 ``outputs``。

本地源码编译与训练
------------------

首次使用原生训练、参数更新和 checkpoint 恢复接口，或修改底层 C++ 后，必须重新编译。
原生分段计时接口、前向条件分支、候选提前拒绝、命中反向 GPU 逆序循环及参数校验合并均涉及 C++ 修改，必须增量编译，
再启动使用新原生模块的进程；不能仅同步脚本后沿用旧模块。
预编译 wheel 不包含这些原生接口，不能用于验证它们。在 ``reconstruction`` 工作区
父目录的 PowerShell 中执行：

.. code-block:: powershell

   cmake --build .\mitsuba\build --config Release --parallel 2
   if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\train.py `
     --dataset datasets/lego --epochs 1000 --stop-after-epochs 10 --no-profile-stages

上述命令保留 1000 轮分辨率调度，仅执行前 10 轮，即每个视角被访问 10 次，
保存可续训 checkpoint，不生成最终评价。省略 ``--output-dir`` 时自动分配新实验目录。
完整实验去掉 ``--stop-after-epochs 10``，结束并保存模型后会自动生成 train/test
各前十个视角的 64 spp 图片和 PSNR。
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
   git -c submodule.recurse=false pull --ff-only &&
   git submodule sync --recursive &&
   git submodule update --init --recursive &&
   cmake --build build --parallel "${SLURM_CPUS_PER_TASK:-1}" &&
   source build/setpath.sh &&
   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/tests/test_reconstruction_performance.py

回归通过后，在同一终端开启新的完整实验：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/train.py \
     --dataset datasets/lego --epochs 1000 --chunk-size 0 --profile-views 3 \
     --profile-interval-epochs 10 \
     --eval-spp 64 --eval-max-views 10

这里显式选择整图提交，并诊断最先处理的三个视角及每第 10 轮的首个视角，
便于区分前向与两类反向开销。省略这两个 profile 参数时，恢复默认的分辨率/每 50 轮抽样。
测量重放收益时添加 ``--no-profile-stages``，保留内核总统计并让抽样视角也执行重放。
普通训练默认已开启 ``--freeze-native`` 和 1024 MiB 显存参考缓存。
对照旧执行路径可添加 ``--no-freeze-native --gpu-reference-cache-mib 0``，其他配置应相同。
已经运行的 HPC 进程不会自动加载新脚本，
需同步后重新启动；如恢复已有实验，应使用已保存的完整 epoch checkpoint 和 ``--resume``，
未保存的 epoch 进度仍需重算。实际显存峰值和 A100 提速幅度需要在服务器测量，不能由本地测试推断。

训练、评价和预览共用本地/Slurm 环境规则：Slurm 下保留
``CUDA_VISIBLE_DEVICES`` 原字符串及所有作业变量，不按本地物理编号重新选卡；
缺少有效 GPU 可见列表时明确报错。Dr.Jit CPU 线程数受
``SLURM_CPUS_PER_TASK`` 限制，未提供时使用 1 线程。没有 GPU 或源码绑定时不回退 CPU
或 wheel。多个可见 GPU 不会自动产生多 GPU 训练。

CPU 参考缓存和 GPU 参考缓存各自使用额外资源；当前 100 个 800 × 800 训练视角各约需 1 GiB，
显存紧张时可减小或关闭 ``--gpu-reference-cache-mib``。无需为了这项优化增加 GPU 数量或
CPU 核数。增加 ``--cpus-per-task`` 会调整 Dr.Jit
线程预算，但不会自动并行 Python 拓扑循环或让同一训练使用多张 GPU。
是否增加资源应根据后续轮次的 ``reference``、``gradients``、``topology_observation``
及 ``epoch_topology_seconds`` 实测判断，不能只用未触发生长的第一轮推断整个训练。
A100 与 RTX 5090 的硬件、运行框架和实际工作负载不同，本地生成场景通过也不保证
A100 上的完整实验追平 5090。双卡训练是下面单独的实验性入口，不由单卡优化选项开启。

Windows 默认构建目录为 ``mitsuba/build/Release``，Linux 为 ``mitsuba/build``；
``--build-dir`` 可覆盖。所有相对路径固定相对于仓库父目录，包括
``datasets/lego``、checkpoint 与输出。共享配置使用相对路径；无需把本机绝对路径
写入 Git。输出和缓存禁止放进源码仓库。

同节点两张 GPU 训练与评价（实验性）
----------------------------------------

``experiments/reconstruction/train_multigpu.py`` 启动两个独立进程，分别使用调度器可见的逻辑
GPU 0 和 1。当前仅支持 Linux Slurm、同一节点、两个工作进程。普通 Windows 本地训练
继续使用 ``train.py``。这里没有承诺两倍加速：每个视角都需要梯度汇总，
收益取决于射线计算量、模型大小、通信及同步成本。真实双 A100 的 NCCL 通信、数值一致性、
完整收敛和速度尚需在服务器验证；本地的 CPU 模拟通信测试不等于真实双卡验证。

本地 RTX 4060 Laptop 的源码增量构建、CUDA 缓冲外部写入、整图与像素分片数值对照已通过。
另外使用两个进程共享一张 GPU、测试专用的主机归约验证了完整训练、4 → 8 升层、
contour/Adam/RNG 状态和单卡/双进程之间的 checkpoint 恢复；生产路径不使用该主机归约。

双卡拆分的是同一视角内的像素射线。两个进程保持相同模型、相机顺序、CPU 随机采样及
contour 提案，每张卡计算各自不重叠的半幅像素。每个像素仍为 1 spp，每个 epoch
仍遍历全部视角。通过 NCCL 在 GPU 上将梯度、有效计数和损失做 SUM 归约，legacy 布尔删除证据
用 MAX 合并，有限值状态合并检查。梯度不除以卡数，也不把不同视角组成新的训练 batch。
拿到相同的完整视角梯度后，两份模型副本各执行相同的一次 Adam 更新，逻辑全局步数只加 1。
浮点求和的分组和次序会改变，因此不保证与单卡逐位相同。

rank 0 负责实验目录、日志、拓扑生长/删除/升层及 checkpoint；
拓扑改变后将模型与 Adam 状态同步给 rank 1，并清除旧的原生重放记录。需要 contour
历史或 legacy 生长证据时，额外合并预测 alpha；当前 RGB 生长只合并候选级
``growth_gain``、``growth_observed``、``growth_unknown``，然后由 rank 0 投一次完整视角票。
RGB 删除同样对 ``deletion_gain``、``deletion_observed``、``deletion_unknown`` 做 SUM，
在 rank 0 按完整视角的可靠命中数归一化，不能先投分片票再合并。
普通轮次不因此回读完整 RGB 图像。

完整训练结束后，两个进程继续执行最终 train/test 评价，默认仍为每个划分前十个视角、
每视角 64 spp。评价把已选中的 train/test 视角合并为任务列表，再交错分给两个 rank；
每张卡加载完整模型，独立渲染分到的整个视角。这里不拆分同一视角的 samples、不计算梯度，
也不使用 NCCL。每个视角的 seed、64 spp 累积顺序、文件名和 PSNR 协议保持原样。
只有 rank 0 写共享配置、汇总指标和报告，合并后的视角记录恢复为各划分原相机顺序，
汇总指标由逐视角结果重新计算，不直接平均两个 rank 的平均 PSNR。

``test_reconstruction_distributed_evaluation.py`` 已在本地用两个进程共享一张 RTX 4060
验证：奇数视角数、无任务的第二进程、对端参考图失败、主进程独占共享报告，以及训练后
两个进程自动执行 64 spp 评价。生成场景的 PNG、线性数组和指标与单进程结果一致。
测试使用临时目录，不占用实验 outputs；它验证分工及数值，不代表双 A100 的加速幅度。

.. code-block:: powershell

   & '..\envs\mitsuba\Scripts\python.exe' -I experiments/reconstruction/tests/test_reconstruction_distributed_evaluation.py

先申请同一节点的一项任务和两张 GPU，例如：

.. code-block:: bash

   srun --partition=cscc-gpu-p --qos=gpu-debug-qos \
     --nodes=1 --ntasks=1 --gres=gpu:a100-sxm4-40gb:2 --cpus-per-task=16 --mem=80G \
     --time=01:00:00 --pty bash

这是首次短测试的资源示例；正式实验按集群 QoS 允许的时限安排。启动器自身创建两个
进程，不要再用 ``--ntasks=2`` 启动两份启动器。它要求有效的 Slurm 作业、至少两个不同
的 ``CUDA_VISIBLE_DEVICES`` 条目，以及至少 2 个 ``SLURM_CPUS_PER_TASK``。
每个子进程原样继承 ``CUDA_VISIBLE_DEVICES`` 和所有 ``SLURM_*`` 变量，仅通过 Dr.Jit
选择可见逻辑设备，不重写物理卡号。16 个 CPU 配额分为每进程 8 个 Dr.Jit 线程；
奇数配额向下均分，不超过调度器总配额。两份参考图缓存各自占用内存和显存。

本地先提交并 push；服务器在没有旧训练使用同一构建目录时同步和增量编译：

.. code-block:: bash

   test -n "${SLURM_JOB_ID:-}" &&
   source /apps/local/conda_init.sh &&
   conda activate "$HOME/envs/mitsuba" &&
   cd "$HOME/projects/reconstruction/mitsuba" &&
   git -c submodule.recurse=false pull --ff-only &&
   git submodule sync --recursive &&
   git submodule update --init --recursive &&
   cmake --build build --parallel "${SLURM_CPUS_PER_TASK:-1}" &&
   source build/setpath.sh

首次启用双卡功能需要编译原生 CUDA 缓冲和流信息绑定。之后只修改 Python 评价流程时，
无需重新编译已有且匹配当前源码的绑定；上述增量编译命令可以继续使用。双卡训练还需要
与节点驱动兼容的 NCCL 动态库，单独双卡评价不需要 NCCL。优先使用集群现有 NCCL
module 或系统库；若库不在默认搜索路径，
可将 ``MITSUBA_NCCL_LIBRARY`` 设置为实际 ``libnccl.so.2`` 的绝对路径。已安装在项目
Python 环境中的 NVIDIA NCCL 包也会被搜索。启动器不会自动安装依赖，不需要为本功能
预先安装 PyTorch、CUDA Toolkit 或其他大型工具链；找不到库时明确报错。

在这项双 GPU Slurm 分配内先运行实际 NCCL 通信检查：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/tests/test_reconstruction_nccl.py

该检查与 CPU 模拟测试 ``test_reconstruction_collective.py`` 不同；它需要两张实际
CUDA 设备及上述 NCCL 库。通信检查通过后再运行短训练，检查真实场景的行为与吞吐。

先保持 1000 轮的完整分辨率调度，执行前 10 轮并保存恢复点：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/train_multigpu.py \
     --workers 2 --worker-timeout 600 \
     --dataset datasets/lego --epochs 1000 --stop-after-epochs 10 \
     --chunk-size 0 --freeze-native --gpu-reference-cache-mib 1024 \
     --profile-views 3 --profile-interval-epochs 10 --no-profile-stages

``--stop-after-epochs 10`` 表示本次进程最多额外完成 10 轮；恢复时同样从恢复点继续计数。
它仅限制本次执行的轮数边界，不把总训练预算改成 10，也不提前
导出声称完成 1000 轮的 ``model.npz``。它在完整 epoch 和拓扑更新结束后保存中间
checkpoint；新的上述实验保存 ``training/epoch-00010.npz``。到达这个提前停止边界时
自动跳过最终评价，保留配置中的评价开关，恢复到全部轮次完成后仍可自动出图。直接使用
``--epochs 10`` 会重新分配 coarse-to-fine 预算，不能作为上述 1000 轮实验的早期速度对照。

为了比较同一节点和同一版本的单卡基线，可在双卡测试结束后运行：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/train.py \
     --dataset datasets/lego --epochs 1000 --stop-after-epochs 10 \
     --chunk-size 0 --freeze-native --gpu-reference-cache-mib 1024 \
     --profile-views 3 --profile-interval-epochs 10 --no-profile-stages

两个命令分别自动分配实验目录。先比较预热后的普通轮次，再单独比较第 10 轮的拓扑事件；
同时检查 loss、保存参数及有效计数/删除证据，不能只根据每轮时间判断移植正确。
单卡入口即使看见两个 GPU，也只使用一张卡，并保留调度器环境。

短测试确认后，在有足够剩余作业时间的终端开启新的完整双卡实验：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/train_multigpu.py \
     --workers 2 --worker-timeout 600 \
     --dataset datasets/lego --epochs 1000 --chunk-size 0 \
     --freeze-native --gpu-reference-cache-mib 1024 \
     --profile-views 3 --profile-interval-epochs 10 --no-profile-stages \
     --eval-spp 64 --eval-max-views 10

若要继续前面的短实验，可给同一启动器传 ``--resume`` 和实际保存的中间 checkpoint
相对路径，省略 ``--stop-after-epochs``。checkpoint 格式沿用单卡格式；它不是每个 rank
各保存一份模型。恢复后的浮点结果仍需按数值容差和训练曲线检查。

已有最终模型时，可在同一双卡资源分配中单独出图，无需重新训练。从仓库目录运行：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/train_multigpu.py \
     --evaluate --workers 2 --worker-timeout 600 \
     --checkpoint outputs/lego-10.15-E1/model.npz \
     --split both --spp 64 --max-views 10 \
     --output-dir outputs/lego-10.15-E1

把示例实验目录替换为实际目录。``--evaluate`` 将其余评价参数转交给
``evaluate.py``，不进入训练流程；这里使用 ``--spp`` / ``--max-views``，
而训练后自动评价仍使用 ``--eval-spp`` / ``--eval-max-views``。输出保存到同一实验目录的
``train/``、``test/`` 和共享评价报告中，重跑会更新相应图片与指标。两种双卡入口都沿用
同一份 Slurm 分配，16 个 CPU 分为每进程 8 个线程，不另起 Slurm 作业。

双卡 ``training.jsonl`` 增加 ``epoch_communication_seconds``、
``epoch_communication_bytes`` 以及 ``collective`` 的累计统计。通信墙钟包含打包、物化、
调用和同步等待，可能等待此前发出的 GPU 运算；它已经包含在 ``gradients`` 中，不能再次
加到 epoch 总时间，也不能当作纯 NCCL 传输时间。``payload_bytes`` 及每轮的字节差值
表示每个 rank 参与归约的逻辑输入字节数，不是 NVLink/PCIe 实际传输流量。
抽样内核报告仍主要描述 rank 0 的 Dr.Jit 运算，NCCL 内核不会自动被归入 Dr.Jit 内核统计。

双卡评价报告区分从开始到两边完成的总墙钟时间与各工作进程的分阶段耗时之和。
两张卡上的渲染可以重叠，因此累加的渲染、参考图读取、指标和写图耗时不是整个评价的
实际等待时长，也不能把它们再次加到总墙钟时间上。判断出图是否加快，应比较相同模型、
视角和 spp 下的总墙钟；同时查看两个 rank 的任务数与耗时，判断是否存在负载不均。
两个进程共用一张本地 GPU 的数值与输出语义检查，不代表真实双 A100 的速度验证。

任一子进程失败、超时或收到中断，启动器会终止并回收同伴进程，避免一张卡一直等待归约。
默认 ``--worker-timeout 600`` 表示某个进程连续 600 秒没有报告进展时终止；对于很慢的
首次编译、拓扑操作或评价，可显式提高为 ``--worker-timeout 1200``。正常训练每视角报告
进展，父进程心跳消息按秒节流，不增加逐像素通信。

Checkpoint 与恢复
-----------------

``training.save_count=3`` 表示整个实验保存三次完整状态，包含最后的 ``model.npz``。
保存轮次按总 epoch 数均匀分配并向上取整：1000 轮在第 334、667、1000 轮保存，
30 轮在第 10、20、30 轮保存；不足三轮时每轮保存一次，不重复保存同一轮。
例如前面的本地 10 轮命令会生成 ``training/epoch-00004.npz``、
``training/epoch-00007.npz`` 和实验根目录 ``model.npz``。
不再额外生成 ``latest.npz``、最终轮次的 ``epoch-XXXXX.npz`` 或升分辨率前的分层副本。
显式使用 ``--stop-after-epochs`` 做有界短测试时，会在本次结束的完整轮次额外保存
``epoch-XXXXX.npz`` 作为恢复点；这个主动停止点不改变总训练预算和原定三次保存计划。
尚未完成全部预算时，不生成最终的 ``model.npz``，也不运行最终评价。

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

这些模块位于 ``experiments/reconstruction``，独立脚本需要将该目录加入 Python 导入路径。
模型文件包含渲染所需的全部体素参数；计算 PSNR 仍需数据集的相机 JSON 和真值图片。

例如前面的本地 10 轮实验在第二次保存后中断，可从第 7 轮恢复：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\train.py `
     --resume outputs/lego-10.15-E1/training/epoch-00007.npz

前面的服务器 1000 轮实验对应从第二次保存的第 667 轮恢复：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I mitsuba/experiments/reconstruction/train.py \
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

新实验默认在点云初始化完成、第一次参数更新前，额外保存
``train/0000-initial.png``。视角 0 指训练相机 JSON 中的第一个相机，与训练打乱顺序无关。
该图使用 ``evaluation.spp``（默认 64，可由 ``--eval-spp`` 覆盖），
分辨率、背景及随机种子约定与最终 ``train/0000.png`` 一致，便于前后对照；
独立的出图采样不会改变训练随机序列，训练仍为 1 spp。
双卡训练只由 rank 0 保存这一张图。恢复 checkpoint 时保留已有初始化图，
若缺失也不把恢复后的模型当作初始化模型重新出图。``--skip-evaluation`` 同时关闭此功能。

训练入口默认在保存最终模型后调用评价流程，使用 64 spp 渲染 train/test 各前十个视角，
并保存每个视角的图片和 PSNR。相机按对应 JSON 顺序选择；不足十个时全取。
这不限制训练视角或改变训练 1 spp。评价只读取 checkpoint，不更新模型或使用测试视角训练。
默认主指标为 sRGB PSNR，并同时报告线性 RGB PSNR；这不改变线性训练损失。
评价的参考合成默认继承 checkpoint 中的 ``data.reference_compositing``，
缺字段的旧 checkpoint 使用 ``linear``。单卡、双卡及训练后的自动评价共用此规则。

单卡入口使用一张 GPU 出图；经 ``train_multigpu.py`` 启动的完整训练保留两个
工作进程，把上述已选视角分配到两张卡独立出图。选择范围与每视角 spp 不变，不将
``--eval-max-views 10`` 解释为每张卡各十个视角。双卡单独出图的 ``--evaluate`` 命令见
上文“同节点两张 GPU 训练与评价”。

如果评价失败，已经完成的 ``model.npz`` 保留，进程以非零状态退出；
``training-report.json`` 中的 ``training_status`` 与 ``evaluation`` 分别记录训练和评价结果，
``evaluation-report.json`` 记录具体错误。可以修复数据或运行环境后直接重跑评价，无需重新训练：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\evaluate.py `
     --checkpoint outputs/lego-10.15-E1/model.npz `
     --split both --spp 64 --max-views 10 `
     --output-dir outputs/lego-10.15-E1

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I mitsuba/experiments/reconstruction/evaluate.py \
     --checkpoint outputs/lego-10.15-E1/model.npz \
     --split both --spp 64 --max-views 10 \
     --output-dir outputs/lego-10.15-E1

``--split train`` 或 ``--split test`` 只评价一个划分。训练相机默认继承 checkpoint
中的配置，测试相机默认为 ``transforms_test.json``；可用 ``--train-cameras`` /
``--test-cameras`` 覆盖。数据集默认继承 checkpoint 的相对路径，也可用 ``--dataset``
指定当前机器上的同一数据。运行环境按当前平台重新选择，不继承另一台机器的 build
绝对路径。``--background R G B`` 同时修改预测与真值的线性 RGB 背景；默认沿用训练。
``--metric-color-space linear`` 可切换显示的主指标，两个颜色空间的指标始终保留。
独立评价的 ``--reference-compositing srgb``／``linear`` 可显式覆盖参考协议，
用于对照时请写入新的输出目录；它会改变真值和指标，不能与原协议分数直接混用。
``evaluation-report.json`` 的 ``metric_protocol`` 记录实际协议及其来源
（checkpoint、旧 checkpoint 默认或显式覆盖），训练后的自动评价不作此覆盖。
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
* 线性指标使用未裁剪的浮点线性 RGB；真值按保存或显式指定的参考协议准备。
* sRGB 指标先将合成后的线性 RGB 裁剪到 [0, 1]，再应用 sRGB 传递函数。
  新 ``srgb`` 参考协议对 sRGB 图片等价于取编码空间的合成参考（允许浮点转换误差）。
  仍使用浮点数据，不从 8 位 PNG 重新读回计算；预测的多 spp 平均在线性空间完成。
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

* 前向、梯度、参数更新、删除证据累积和生长投票使用 CUDA/Dr.Jit。生长候选生成、
  升层拟合、边界处的生命周期判定和模型重排仍有 CPU NumPy 工作及模型快照，尚未全部迁移到 GPU。
  几何及生长投影保留 float64 精度，再写回 float32 模型；临界阈值附近可能与 Falcor 的
  GPU float32 决策不同，拓扑性能也不能假定与原实现相同。
* 多个增长提案竞争同一格时，使用确定性的胜者选择；Falcor 的原子竞争顺序不确定。
  删除后的剩余体素保留顺序，而 Falcor 使用空洞/供体搬移，索引顺序可能不同。
* 对全部活跃体素预分配颜色 Adam 状态，没有复制 D3D 缓冲区分页耗尽时临时退回 SGD
  的资源处理分支；正常 Adam 数学规则保留。
* 相机证据使用动态 bitset，支持不超过 32767 个视角；不采用 Falcor 默认编译容量 128。
* 随机数序列、浮点归约顺序及跨 GPU 数值不同；不能声称逐位相同。
* 按本次约定训练固定每视角 1 spp，未复制原训练多 spp 路径；训练入口强制白底。
  新实验参考图采用 sRGB 输入先合成、再解码的协议，训练仍优化线性 RGB；
  旧 ``linear`` 协议可显式选择。多 spp 及可配置背景用于独立评价。

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

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\tests\test_reconstruction.py `
     --build-dir .\mitsuba\build\Release --variant cuda_ad_rgb
   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\tests\test_reconstruction_gradients.py `
     --build-dir .\mitsuba\build\Release --variant cuda_ad_rgb
   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\tests\test_reconstruction_gradients.py `
     --build-dir .\mitsuba\build\Release --variant scalar_rgb
   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\tests\test_reconstruction_optimization.py
   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\tests\test_reconstruction_training.py

新训练闭环测试自行生成小场景，不使用 lego 或下载数据集；重新运行时报告默认写入
``outputs/mitsuba/training-tests/training-tests.json``；优化器/拓扑检查位于
``outputs/mitsuba/optimization-tests``，几何检查位于
``outputs/reconstruction-tests/gradients``。这些可重新生成的测试产物已从当前
``outputs`` 中清理，不属于上面的场景重建结果。

Checkpoint 与 PSNR 的 CPU 单元测试不导入 GPU 后端，也不渲染数据集：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\tests\test_reconstruction_checkpoint.py
   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\tests\test_reconstruction_runtime.py

该测试覆盖模型/优化器/随机数状态往返、空数组与标量布局、原子替换失败保留旧文件、
校验和破坏、pickle 拒绝、元数据与形状错误，以及已知 PSNR、聚合定义和 sRGB 转换。
也检查 contour 提案归一化、PDF 校正后的常量/alpha 积分、仅选中 bin 的原始 alpha
历史更新，以及恢复随机数和历史状态后的采样与射线一致性；这些检查在 CPU 上执行。

调用重放、显存参考缓存和参数校验另有生成数据回归。从已编译的仓库目录运行：

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/tests/test_reconstruction_replay.py &&
   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/tests/test_reconstruction_reference.py &&
   "$HOME/envs/mitsuba/bin/python" -I experiments/reconstruction/tests/test_reconstruction_parameters.py

Windows 在工作区父目录将解释器替换为 ``.\envs\mitsuba\Scripts\python.exe``，
脚本路径改为 ``.\mitsuba\experiments\reconstruction\tests\...``。三个测试都只使用生成的数据和自动清理的临时目录，
不写实验 ``outputs``。重放测试对照变化的相机、参考、参数、分批尾部、连续 Adam 更新、
contour 状态、不同拓扑/边界/SH 和分段诊断旁路；显存缓存测试检查全图/分批数值一致、
LRU 淘汰、容量统计和关闭路径；参数测试覆盖标量与 CUDA、非有限值/尺寸/键错误、
失败事务不改变任何参数、部分更新及空模型。真实场景的吞吐仍使用上面的执行基准和训练日志测量。

整图提交和相机内核复用有独立的 CUDA 小场景回归，可从工作区父目录运行：

.. code-block:: powershell

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\tests\test_reconstruction_performance.py

.. code-block:: bash

   "$HOME/envs/mitsuba/bin/python" -I mitsuba/experiments/reconstruction/tests/test_reconstruction_performance.py

该检查使用生成的数据和临时目录，比较整图/分批的数值结果，检查按需回读、非有限值拒绝，
以及更换相机后的 JIT 内核缓存命中。它不代表真实 Lego 数据集或 A100 的吞吐量测试。
早期 RTX 4060 CUDA 检查中，16 × 16 整图与每批 37 条射线的梯度最大绝对差为
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

当时的记录布局已通过 CUDA 多命中/分批、18 项几何公式、15 项优化器/拓扑检查，以及包含
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

   & '.\envs\mitsuba\Scripts\python.exe' -I .\mitsuba\experiments\reconstruction\tests\test_reconstruction_performance_summary.py

前向条件分支与候选提前拒绝的本地对照
----------------------------------

以下保留前向优化阶段的历史测量，尚未包含后续命中反向 GPU 循环。Windows、RTX 4060
Laptop 上，以真实 Lego 的固定初始化模型（80759 个活跃体素、
256³ 网格）和相机 0、21、56 的固定 800 × 800、1 spp 输入比较修改前后。
下表单位均为毫秒：前向 GPU 时间来自独立分段诊断的 3 次中位数，普通原生墙钟
来自关闭分段的完整 ``loss_and_grad`` 调用的 7 次中位数，包含反向计算。

.. list-table:: 固定输入的局部测量
   :header-rows: 1

   * - 相机索引
     - 前向 GPU：修改前
     - 前向 GPU：修改后
     - 普通原生墙钟：修改前
     - 普通原生墙钟：修改后
   * - 0
     - 54.783
     - 29.915
     - 279.200
     - 233.263
   * - 21
     - 59.279
     - 35.697
     - 319.909
     - 331.612
   * - 56
     - 51.500
     - 36.103
     - 325.969
     - 364.428

三个输入的前向诊断时间均降低，但普通完整调用只有相机 0 加快，其余两者变慢，
不能把分段结果描述为整体训练提速。分段边界会改变融合与调度，GPU 数值也不是
epoch 时间；A100 和完整训练的吞吐尚未测量。三个固定输入的 RGB、alpha、loss
及计数逐位一致，全部梯度的最大绝对差为 ``3.8147e-6``；该对照不替代完整训练验证。

该阶段源码构建通过；独立几何检查通过 scalar 20 项、CUDA 24 项（最大误差分别为
``1.18945e-6``、``1.45493e-6``），CUDA 性能回归覆盖整图/分批、分段开关、缓存、
按需回读与非有限值拒绝，训练集成检查通过 36 项外观有限差分（最大误差 ``9.1158e-6``）、
生成场景三轮训练、保存恢复、4 → 8 拓扑调度、自动 64 spp 评价及恢复一致性。
该阶段测试与基准产物使用临时目录，未占用重建实验的 ``outputs`` 目录。

命中反向 GPU 循环的本地对照
--------------------------

本轮在同一台 Windows、RTX 4060 Laptop 上，重新测量已包含前向分支/候选提前拒绝的
版本作为基线，与新增 GPU 逆序循环及独立后缀缓冲的版本对照。仍使用真实 Lego
固定初始化模型、相机 0/21/56、800 × 800、1 spp、256³ 网格和 80759 个活跃体素。
下表为关闭分段计时的完整原生 ``loss_and_grad`` 调用墙钟中位数（7 次，单位毫秒），
包含前向和反向，排除输入上传、读取图片、优化器及拓扑更新。

.. list-table:: 命中反向优化前后的固定输入调用
   :header-rows: 1

   * - 相机索引
     - 修改前
     - 修改后
   * - 0
     - 238.968
     - 33.843
   * - 21
     - 339.621
     - 32.831
   * - 56
     - 367.985
     - 30.612

独立完整调用的内核统计中，JIT 内核数从 18 降到 3；独立分段诊断中，
``hit_backward`` 的内核数从 16 降到 1。这些独立测量不相加，也不能把内核数量比例
直接当作训练加速比。三个输入的 RGB、alpha、loss 及计数逐位一致，全部梯度最大
绝对差为 ``3.8147e-6``。这是单机固定模型的局部调用结果，不是 epoch 耗时或完整
训练吞吐；A100 尚未测量，也不能据此承诺追平 RTX 5090 的 Falcor 训练速度。

本轮源码编译以及 scalar 22 项、CUDA 27 项独立几何检查通过，包含完整 16 次命中、
容量截断、候选后缀、零命中与多射线分歧；删除证据另通过移除单次命中后重新合成
loss 的独立参考计算验证。完整 CUDA 性能回归通过，包括多命中整图/分批、删除证据、
分段计时开关、按需回读和相机内核缓存；``max_hits=1/5/16`` 的命中反向均只执行
1 个 JIT 内核，回归测试限制每种容量至多 2 个，防止退回按槽逐次提交的结构。
生成场景训练集成通过 36 项外观有限差分（最大误差 ``9.115785e-6``）、三轮训练、
checkpoint 保存恢复及优化器/RNG/contour 一致性、4 → 8 升层、自动 64 spp 评价与失败保护。
测试与基准使用临时目录，不占用重建实验的 ``outputs``。
