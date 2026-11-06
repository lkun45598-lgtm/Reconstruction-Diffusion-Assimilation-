"""Main experiment report -> docs/RB实验汇报_稀疏观测同化.tex / .pdf.
Main experiment = EnKF (initial ensemble from ANOTHER case, assimilating from frame 0) vs diffusion DA vs free forecast, K=1,10,30, evaluation frames 400..599 of the 600-frame test trajectories
(outputs/study_rb_othercase_test, outputs/study_rb_dps_test). Divergence: outputs/study_rb_div_test (same runs re-run with the analysis saved, seed 42, 20 cases). All numbers computed from saved outputs. Usage: python analysis/make_report_main.py (needs xelatex)"""
import sys, glob, subprocess, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, 'analysis')
import make_report_tex as M
from make_report_tex import BLUE, GRN, GRY, ORG, PRE
plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'], 'axes.unicode_minus': False, 'font.size': 8, 'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Droid Sans Fallback', 'mathtext.it': 'Droid Sans Fallback:italic', 'mathtext.bf': 'Droid Sans Fallback', 'mathtext.default': 'it'})
DOC = Path('docs'); FIG = DOC / 'report_fig_main'; M.FIG = FIG
EN, DP = Path('outputs/study_rb_othercase_test'), Path('outputs/study_rb_dps_test')
SEEDS, KS, W0, DW0 = (42, 43, 44), (1, 10, 30), 400, 40; f4 = lambda v: f'{v:.4f}'
rd = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))])
DIFF, ENKF, FREE = '扩散同化 (DPS)', 'EnKF', '纯预报'; CL = {DIFF: BLUE, ENKF: GRN, FREE: GRY}
SD = np.load('data/processed/stats.npz')['std']; POOL = float(np.sqrt(np.mean(SD.astype(float) ** 2)))   # pooled channel std, used to express errors relatively


def lag(K, s):
    e = rd(f'{EN}/s{s}_K{K}_c*.csv'); a = rd(f'{DP}/s{s}_K{K}_c*.csv'); fr = rd(f'{DP}/free_s{s}_c*.csv')   # free forecast does not depend on K
    e = e[e.frame >= W0]
    out = pd.concat([a.assign(method=DIFF), e.assign(method=ENKF), fr.assign(method=FREE)])
    return out[['case', 'frame', 'method', 'missing_rmse']], f'{e.infl.iloc[0]:g}'


LAG = {}; INFL = {}
for K in KS:
    for s in SEEDS: LAG[K, s], INFL[K] = lag(K, s)
PCL = {k: d.groupby(['case', 'method']).missing_rmse.mean().unstack() for k, d in LAG.items()}


def summ(pc, K):   # pc: dict (K,s)->per-case table
    g = {m: np.array([pc[K, s][m].mean() for s in SEEDS]) for m in (DIFF, ENKF, FREE)}
    cm = {m: np.mean([pc[K, s][m].values for s in SEEDS], 0) for m in (DIFF, ENKF)}; d = cm[ENKF] - cm[DIFF]; n = len(d)
    return g, d.mean(), stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n), int((d > 0).sum()), n


SL = {K: summ(PCL, K) for K in KS}
T = Path('outputs/study_rb_test'); TS = Path('outputs/study_rb_test_seeds')


DV = rd('outputs/study_rb_div_test/*_s42_K*_c*.csv'); DV['method'] = DV['mode'].map({'dps': DIFF, 'enkf': ENKF}); DV.loc[(DV['mode'] == 'dps') & (DV.K == 0), 'method'] = FREE


def div_row(K):   # mean over the 20 test cases x frames 400..599; free forecast does not depend on K
    d = DV[((DV.K == K) & (DV.method != FREE)) | (DV.method == FREE)]; return d.groupby('method')[['an_div_rel', 'proj_div_rel']].mean()


pct = lambda g: 100 * (1 - g[DIFF].mean() / g[ENKF].mean())
E0 = rd(f'{EN}/s42_K1_c*.csv').query('frame == 0'); ISP, IER = E0.ensemble_spread.mean(), E0.missing_rmse.mean()   # first-frame analysis (after the first assimilation)
TEN = {K: rd(f'{EN}/s*_K{K}_c*.csv').groupby(['seed', 'case']).t_run.first().mean() for K in KS}; TDP = {K: rd(f'{DP}/s*_K{K}_c*.csv').groupby(['seed', 'case']).t_run.first().mean() for K in KS}


def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 1.8)); ax.axis('off'); ax.set_xlim(0, 16); ax.set_ylim(0, 3.5)
    def box(x, w, zh, fc, ec, math=None):
        ax.add_patch(FancyBboxPatch((x, .7), w, 1.4, boxstyle='round,pad=0.04,rounding_size=.15', fc=fc, ec=ec, lw=1)); ax.text(x + w / 2, 1.55 if math else 1.4, zh, ha='center', va='center', fontsize=7)
        if math: ax.text(x + w / 2, 1.05, math, ha='center', va='center', fontsize=8)
    arr = lambda x0, x1, y=1.4: ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=9, lw=1, color='#1F3A5F'))
    box(.1, 2.6, '上一时刻分析场', '#E8F0F8', '#1F3A5F', r'$x^a_{t-1}$'); box(3.5, 3.4, '数值求解器预报\n(Ra = 5e4, 35 步)', '#FFF3E0', ORG)
    box(7.7, 2.4, '背景场', '#E8F0F8', '#1F3A5F', r'$x^b_t$'); box(10.9, 2.9, '无条件扩散先验\n+ DPS 引导采样', '#E3F4EC', GRN); box(14.3, 1.6, '分析场', '#E8F0F8', '#1F3A5F', r'$x^a_t$')
    arr(2.8, 3.4); arr(7.0, 7.6); arr(10.2, 10.8); arr(13.9, 14.2)
    ax.add_patch(FancyBboxPatch((7.7, 2.6), 8.2, .65, boxstyle='round,pad=0.03,rounding_size=.1', fc='#F4F6F8', ec=GRY, lw=.9))
    ax.text(11.8, 2.93, '观测：1% 格点、三通道、高斯噪声 (标准差 0.1)；每 K 帧提供一次', ha='center', va='center', fontsize=6.4)
    ax.add_patch(FancyArrowPatch((12.3, 2.57), (12.3, 2.15), arrowstyle='-|>', mutation_scale=8, lw=1, color='#1F3A5F'))
    ax.text(8, .25, '背景场与观测均作为似然引导项进入 DPS，采样自纯噪声开始；对每个后验样本做无散度、无滑移投影；每个观测时刻循环一次。', ha='center', fontsize=6.6, color='#555'); M.save(fig, 'workflow')


def fig_k():
    fig, a = plt.subplots(figsize=(4.6, 2.6)); x = np.arange(len(KS)); w = .26
    for i, m in enumerate((DIFF, ENKF, FREE)):
        v = np.array([SL[K][0][m] for K in KS]); a.bar(x + i * w, v.mean(1), w, color=CL[m], label=m, yerr=[v.mean(1) - v.min(1), v.max(1) - v.mean(1)], capsize=2, error_kw=dict(lw=.8))
    a.set_xticks(x + w); a.set_xticklabels([f'K={k}' for k in KS]); a.set_xlabel('观测间隔 K（帧）'); a.set_ylabel('未观测点 RMSE'); a.set_ylim(0, 0.062); a.legend(fontsize=6.5, loc='upper left', ncol=3); M.save(fig, 'bar_k')


def fig_curves():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
    for a, K in zip(ax, KS):
        d = pd.concat(LAG[K, s] for s in SEEDS)
        for m in (DIFF, ENKF, FREE): v = d[d.method == m].groupby('frame').missing_rmse.mean(); a.plot(v.index, v, color=CL[m], lw=1.2, label=m)
        if K > 1:
            for f in range(W0, 600, K): a.axvline(f, color='#e3e3e3', lw=.6, zorder=0)
        a.set_title(f'K={K}', fontsize=8); a.set_xlabel('帧序号 t')
    ax[0].set_ylabel('未观测点 RMSE'); ax[0].legend(fontsize=6, loc='upper left'); M.save(fig, 'curves')


def fig_div():
    d = {K: div_row(K) for K in KS}; tv = DV.truth_div_rel.mean()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True); x = np.arange(len(KS)); w = .26
    for a, (col, ttl) in zip(ax, (('an_div_rel', '原始输出'), ('proj_div_rel', '统一投影后'))):
        for i, m in enumerate((DIFF, ENKF, FREE)): a.bar(x + i * w, [d[K].loc[m, col] for K in KS], w, color=CL[m], label=m)
        a.axhline(tv, color='k', ls='--', lw=.8, label='真值'); a.set_yscale('log'); a.set_xticks(x + w); a.set_xticklabels([f'K={k}' for k in KS]); a.set_title(ttl, fontsize=8); a.set_xlabel('观测间隔 K（帧）')
    ax[0].set_ylabel('相对散度'); ax[0].set_ylim(5e-8, 2e-4); ax[1].legend(fontsize=6.5, loc='upper right'); M.save(fig, 'div')


def tab_div():
    d = {K: div_row(K) for K in KS}
    return [f'{m} & ' + ' & '.join(f'{sci(d[K].loc[m, c])}' for K in KS for c in ('an_div_rel', 'proj_div_rel')) for m in (DIFF, ENKF, FREE)]


def sci(v):
    e = int(np.floor(np.log10(v))); return f'${v / 10 ** e:.1f}\\times10^{{{e}}}$'


def tab_main():
    r = []
    for K in KS:
        g, d, h, nb, n = SL[K]; r.append(f'{K} & {INFL[K]} & {g[DIFF].mean():.4f} & {g[ENKF].mean():.4f} & {g[FREE].mean():.4f} & {-pct(g):+.0f}\\% & {nb}/{n}')
    return r


BODY = r'''
\begin{center}{\LARGE\bfseries 稀疏观测下基于无条件扩散先验的循环数据同化\par}\vspace{.4em}{\large Rayleigh--B\'enard 对流实验汇报\par}\end{center}

\begin{center}\textbf{摘\quad 要}\end{center}
\begingroup\small\leftskip=2em\rightskip=2em
数值求解器的预报含有模型误差，观测稀疏且带噪声。在二维 Rayleigh--B\'enard 对流上，以求解器生成的数据作为真值，从中抽取带噪声的稀疏观测。无条件训练的扩散模型作先验，求解器预报（背景场）与稀疏观测都作为引导项，经扩散后验采样（DPS）在每个观测时刻同化，并与 20 成员 EnKF 和纯预报（只用求解器从起始状态向前积分、不再吸收观测）比较。预报求解器的 Rayleigh 数取 $5\times10^4$（真值 $10^5$），观测为 1\% 格点、标准差 0.1 的高斯噪声。EnKF 的初始集合这样得到：随机取另一条轨迹的一帧，用预报求解器向前积分出连续 20 帧，作为 20 个成员。EnKF 从第 0 帧起开始同化，等误差稳定后再评价；扩散同化不需要预热。两者都在第 400--599 帧上评价，测试 20 个轨迹、3 个随机种子。以未观测格点上的均方根误差（RMSE）衡量，观测间隔 $K=1,10,30$ 时扩散同化的误差比 EnKF 分别低 @P1@\%、@P10@\% 与 @P30@\%，20 个轨迹上都更小。
\endgroup

\section{研究目的}
同化把含误差的预报与稀疏、带噪声的观测结合起来。实验采用的同化方法是：先验只在状态上无条件训练，不接触观测与背景；同化时，求解器的预报作为背景场、观测作为另一项似然，二者都以引导项的形式进入扩散后验采样（Diffusion Posterior Sampling，DPS）。在每个观测时刻做一次同化，两次观测之间用求解器预报，如此循环向前推进，记为\textbf{扩散同化 (DPS)}。对照基线为 EnKF，参照为纯预报。其中``预报''指用数值求解器从某一时刻的状态向前积分，得到之后时刻的状态；纯预报是起始帧用该帧观测给出初始状态后，只靠求解器向前积分，不再吸收任何后续观测，反映没有同化时模型误差造成的误差。

研究目的：在求解器有模型误差、观测稀疏且带噪声的条件下，检验这种同化方法得到的分析场是否比 EnKF 更接近真值，并对比两者在不含同化的纯预报之上带来的改进。

\section{数据设置和处理}
\subsection{控制方程与数值求解}
研究对象为二维 Boussinesq Rayleigh--B\'enard 对流，状态为浮力 $b$ 与速度 $\mathbf u=(u_x,u_z)$：
\begin{equation}
\nabla\!\cdot\!\mathbf u=0,\quad \partial_t b-\kappa\nabla^2 b=-\mathbf u\!\cdot\!\nabla b,\quad \partial_t\mathbf u-\nu\nabla^2\mathbf u+\nabla p-b\,\mathbf e_z=-\mathbf u\!\cdot\!\nabla\mathbf u,
\end{equation}
其中 $\kappa=(\mathrm{Ra}\,\mathrm{Pr})^{-1/2}$，$\nu=(\mathrm{Ra}/\mathrm{Pr})^{-1/2}$。Rayleigh 数 Ra 是方程中的一个无量纲系数，表示浮力驱动与黏性、热扩散耗散之比，数值越大对流越剧烈；Prandtl 数 Pr 取 1。$x$ 方向周期（$L_x=3$），$z$ 方向为上下壁面（$L_z=1$），$b(0)=1$、$b(L_z)=0$，壁面无滑移。用 Dedalus v3 求解（$x$ 方向 Fourier、$z$ 方向 Chebyshev，网格 $72\times24$，时间步长 $\mathrm{d}t=0.01$）。保存的相邻两帧相隔 35 个求解器步（0.35 个时间单位）。

\subsection{数据集}
真值数据的 Ra$=10^5$。数据按轨迹划分，训练集每个轨迹 100 帧，验证集和测试集的轨迹延长到 600 帧，状态形状 $72\times24\times3$（$b$、$u_x$、$u_z$），三个通道的标准差分别为 0.256、0.164、0.158。
\begin{table}[H]\centering\small\caption{数据集划分与用途。}
\begin{tabular}{lccp{7.6cm}}\toprule
子集 & 轨迹数 & 帧数/轨迹 & 用途\\\midrule
训练集 & 160 & 100 & 训练扩散先验\\
验证集 & 20 & 600 & 选取扩散同化的引导参数与 EnKF 的膨胀系数\\
测试集 & 20 & 600 & 仅用于最终评价\\\bottomrule
\end{tabular}\end{table}
数据按通道标准化后用于训练扩散先验，评价在物理量纲下进行。每个轨迹由随机初始扰动积分得到，前 120 帧（流动尚未进入统计稳态）丢弃。验证、测试轨迹用相同的初始扰动重新积分到 720 帧后再丢弃前 120 帧，前 100 帧与原数据的最大差异为 $6\times10^{-8}$（浮点舍入）；后 500 帧的通道标准差与前 100 帧相差在 1\% 以内，先验不需重新训练。

\section{实验方法}
\subsection{扩散同化 (DPS)}
\textbf{先验。}U-Net（基础通道数 16），在训练集的帧上按标准扩散目标无条件训练 200 个 epoch。训练中不使用观测、背景场和时间信息，同化时不再训练。

\textbf{后验采样与引导。}设先验给出的得分函数为 $s_\theta(x_t,t)\approx\nabla_{x_t}\log p_t(x_t)$。同化的目标是从后验 $p(x\mid x^b,y)\propto p(x)\,p(x^b\mid x)\,p(y\mid x)$ 中采样，其中 $x^b$ 为背景场（求解器预报），$y$ 为观测。DPS 用去噪网络给出的干净场估计 $\hat x_0(x_t)$ 近似似然，在每个采样步沿着似然的梯度修正轨迹。用 $\hat x_0$ 计算的引导项为
\begin{equation}
\mathcal L(x_t)=\sum_{i\in\mathcal O}\frac{[\hat x_0(x_t)_i-y_i]^2}{\sigma_{\rm obs}^2}+\sum_{i}\frac{[\hat x_0(x_t)_i-x^b_i]^2}{\sigma_{\rm bg}^2}+\lambda\,\bigl\|\nabla\!\cdot\hat{\mathbf u}_0\bigr\|^2,
\end{equation}
第一项只在观测点集合 $\mathcal O$ 上求和（观测项），第二项在全场求和（背景项），第三项为散度惩罚（以速度梯度的均方根作归一化，系数 $\lambda=0.3$）；误差按物理量纲计算，$\sigma_{\rm obs}=0.05$，$\sigma_{\rm bg}=0.3$，观测项的权重约为背景项的 36 倍。采样用 DPM-Solver 一阶的确定性 ODE 形式，从纯噪声 $x_{T}\sim\mathcal N(0,I)$（$T=0.95$）出发，共 100 步，每步为
\begin{equation}
x_{t-\Delta t}=\frac{\alpha_{t-\Delta t}}{\alpha_t}\,x_t+\sigma_{t-\Delta t}\sigma_t\bigl(e^{\Delta\lambda}-1\bigr)\,s_\theta(x_t,t)-\eta\,\nabla_{x_t}\mathcal L(x_t),\qquad \eta=\frac{r\,T}{2N},
\end{equation}
其中 $\alpha_t,\sigma_t$ 为扩散过程的信号与噪声系数，$\Delta\lambda=\tfrac12[\lambda_{t-\Delta t}-\lambda_t]$（$\lambda_t=\log(\alpha_t^2/\sigma_t^2)$），$N=100$ 为步数，$r=2$ 为引导强度系数。梯度 $\nabla_{x_t}\mathcal L$ 通过 $\hat x_0$ 对 $x_t$ 反向传播得到。不强制观测点取观测值。$\sigma_{\rm obs}$ 是引导强度，不等同于观测噪声标准差（后者为 0.1）。每个窗口取 32 个样本，对每个样本做投影（把速度改写成由流函数导出、壁面处为零的形式，使其严格无散度、壁面无滑移），取样本均值作为分析场，即该帧的同化结果。

\textbf{循环。}同化时刻：用求解器把上一次分析场预报到当前帧作为背景，再采样得到分析场；非同化时刻：直接取求解器预报。起始时刻没有背景，只用该帧观测采样。
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{report_fig_main/workflow.pdf}\caption{扩散同化的循环流程。}\end{figure}

\subsection{对照方法}
\textbf{EnKF。}20 个成员的随机 EnKF，状态为 $(b,u_x,u_z)$ 堆叠的整场向量。分析式为
\begin{equation}
\mathbf x^a-\mathbf x^f=\mathbf P^f\mathbf H^{\!\top}(\mathbf H\mathbf P^f\mathbf H^{\!\top}+\mathbf R)^{-1}[\mathbf y-H(\mathbf x^f)],
\end{equation}
$\mathbf P^f$ 为预报集合的样本协方差，$\mathbf H$ 取观测点，$\mathbf R=0.01\,\mathbf I$（观测误差标准差 0.1）。每个成员使用各自叠加 $\mathcal N(0,\mathbf R)$ 噪声的观测，分析场取集合均值。预报协方差做乘性膨胀，不使用局地化。各成员由求解器（与扩散同化相同，Ra$=5\times10^4$）独立预报。

\textbf{EnKF 的初始集合。}集合成员全部相同时协方差为零，观测不起作用，因此成员之间需要有差异。初始集合取自另一条轨迹：从训练集中随机取一个轨迹和其中一帧（第 0--79 帧内随机），用预报求解器（Ra$=5\times10^4$）从这一帧积分出连续 20 帧，作为 20 个成员；此后各成员各自由求解器向前预报，在观测时刻按上式更新。每个测试轨迹各取一次，三个随机种子使用同一个初始集合。起始时集合均值与真值的误差较大，靠之后的同化逐步降低，因此 EnKF 从第 0 帧起同化，评价放在其稳定之后。

\textbf{纯预报。}在起始帧（第 400 帧）用该帧观测得到分析场，此后只由求解器预报，不再同化。

\section{实验设置}
\subsection{误差来源}
真值由求解器生成的数据给出，观测从真值抽取并加噪声，所有误差指标都相对真值计算。各类误差的设置如下。
\begin{table}[H]\centering\small\caption{误差设置。}
\begin{tabular}{>{\raggedright\arraybackslash}p{3cm}>{\raggedright\arraybackslash}p{9cm}}\toprule
设置对象 & 取值\\\midrule
真值 & Ra$=10^5$ 的求解器数据\\
预报求解器（背景场） & Ra$=5\times10^4$，与真值的差别即模型误差\\
观测点 & 1728 个格点中随机取 17 个（1\%），三个通道都观测\\
观测噪声 & 在观测值上加标准差 0.1 的高斯噪声\\
观测间隔 & 每 $K$ 帧提供一次观测，$K=1,10,30$（$K$ 越大观测越稀疏）\\
评价帧 & 第 400--599 帧\\\bottomrule
\end{tabular}\end{table}
观测噪声约为通道标准差（合并值 @POOL@）的 @NOISE@\%；不做同化时预报的 RMSE 约 @FREE@，约为通道标准差的 @FRELP@\%。观测噪声每个同化帧重新抽取，观测点位置由随机种子决定、同一轨迹内不变，同一轨迹、同一种子下三种方法的观测点位置相同；噪声由种子和同化序号决定，扩散同化与纯预报的序号从第 400 帧起算，EnKF 从第 0 帧起算，同一帧上的噪声不是同一组抽样。

\subsection{起止时刻}
EnKF 从第 0 帧起同化，到第 400 帧时已经历 400 帧的同化与预报；评价只取第 400--599 帧（共 200 帧）。扩散同化与纯预报不依赖历史，从第 400 帧开始，起始时只用该帧观测，评价同样取第 400--599 帧。$K>1$ 时 EnKF 的同化时刻为第 $0,K,2K,\dots$ 帧，扩散同化为第 $400,400+K,\dots$ 帧，$K=30$ 时两者的同化相位不同（EnKF 在第 390、420 帧，扩散同化在第 400、430 帧）。

\subsection{评价与调参}
指标为\textbf{未观测点 RMSE}：在没有观测的格点上，三个通道合并，计算分析场（非同化帧为预报场）与真值的均方根误差，物理量纲。评价窗口为第 400--599 帧，观测间隔 $K=1,10,30$。随机种子 42、43、44，每个种子 20 个测试轨迹，表中为三个种子的均值，误差线为三个种子的最小至最大。EnKF 的膨胀系数在验证集前 5 个轨迹上按第 400--599 帧的未观测点 RMSE 选取：$K=1$ 从 1.1、1.3、1.6、2.0 中选，$K=10$ 从 2.0、3.0 中选，$K=30$ 从 5.0、8.0、12.0 中选，依次为 @INFL@；要求 5 个轨迹都已收敛。扩散同化的引导参数在验证集 $K=1$ 下选定，各 $K$ 共用，未按 $K$ 重新调整。

\section{实验结果和分析}
\subsection{观测间隔的影响}
\begin{table}[H]\centering\small\caption{第 400--599 帧未观测点 RMSE（三个种子均值）。``相对 EnKF''为扩散同化的 RMSE 相对 EnKF 的变化，负值表示更低；``更优轨迹数''为 20 个测试轨迹中扩散同化的 RMSE 低于 EnKF 的个数。}
\begin{tabular}{ccccccc}\toprule
$K$ & EnKF 膨胀 & 扩散同化 (DPS) & EnKF & 纯预报 & 相对 EnKF & 更优轨迹数\\\midrule
@TAB_MAIN@
\bottomrule\end{tabular}\end{table}
\begin{figure}[H]\centering\includegraphics[width=.62\linewidth]{report_fig_main/bar_k.pdf}\caption{不同观测间隔下的未观测点 RMSE（误差线为三个种子的最小至最大）。}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{report_fig_main/curves.pdf}\caption{各观测间隔下的逐帧误差（三个种子、20 个轨迹平均）。}\end{figure}
@TXT_K@

\subsection{计算时间}
表~\ref{tab:time} 为每个轨迹的墙钟时间。EnKF 的 20 个成员每帧都要由求解器预报，评价前还要从第 0 帧同化到第 400 帧，共 600 帧，在单个 CPU 线程上运行；扩散同化只算第 400--599 帧共 200 帧，在 GPU 上运行（采样 32 个样本、100 步）。
\begin{table}[H]\centering\small\caption{每个测试轨迹的墙钟时间（秒；3 个种子 20 个轨迹的均值）。括号内为折合到每帧的秒数。}\label{tab:time}
\begin{tabular}{ccc}\toprule
$K$ & 扩散同化 (DPS，200 帧，GPU) & EnKF (600 帧，单 CPU 线程)\\\midrule
@TAB_TIME@
\bottomrule\end{tabular}\end{table}
这些时间是多个任务同时运行时测得的，机器上还有其他用户的任务，只作量级参考，不是独占条件下的计时。

\subsection{不可压缩约束（散度）}
速度场应满足 $\nabla\!\cdot\mathbf u=0$。用相对散度 $\|\nabla\!\cdot\mathbf u\|/\|\nabla\mathbf u\|$ 衡量（导数用 Fourier--Chebyshev 谱方法计算；真值约 $3.9\times10^{-5}$）。扩散模型的采样输出不满足这一条，扩散同化因此在输出前对速度场做投影（把速度改写成由流函数导出、壁面处为零的形式，使其严格无散度）；EnKF 和纯预报的成员由求解器预报，近似无散度。表~\ref{tab:div} 中``原始输出''是各方法实际给出的结果（扩散同化的结果已经过投影，EnKF 与纯预报没有），``统一投影后''是对三种方法的结果都施加同一个投影之后的值。
\begin{table}[H]\centering\small\caption{同化结果的相对散度（第 400--599 帧，20 个测试轨迹的均值，种子 42）。纯预报与 $K$ 无关。}\label{tab:div}
\begin{tabular}{lcccccc}\toprule
 & \multicolumn{2}{c}{$K=1$} & \multicolumn{2}{c}{$K=10$} & \multicolumn{2}{c}{$K=30$}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
方法 & 原始输出 & 统一投影后 & 原始输出 & 统一投影后 & 原始输出 & 统一投影后\\\midrule
@TAB_DIV@
\bottomrule\end{tabular}\end{table}
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{report_fig_main/div.pdf}\caption{同化结果的相对散度（虚线为真值的散度；第 400--599 帧，20 个轨迹平均）。}\end{figure}
@TXT_D@

\subsection{小结}
@TXT_S@
\end{document}
'''


def main():
    FIG.mkdir(parents=True, exist_ok=True); fig_workflow(); fig_k(); fig_curves(); fig_div()
    P = {K: pct(SL[K][0]) for K in KS}; free = np.mean([SL[K][0][FREE].mean() for K in KS])
    txt_k = (f'$K=1,10,30$ 时扩散同化的 RMSE 都低于 EnKF，分别低 {P[1]:.0f}\\%、{P[10]:.0f}\\% 与 {P[30]:.0f}\\%，20 个轨迹中分别有 {SL[1][3]}、{SL[10][3]} 与 {SL[30][3]} 个更优，三个种子的结果一致。'
             f'纯预报为 {free:.4f}；扩散同化在各 $K$ 下都低于纯预报，EnKF 在 $K=30$ 时（{SL[30][0][ENKF].mean():.4f}）略高于纯预报。$K$ 增大时两种同化方法的误差都上升。'
             f'EnKF 在第 0 帧完成第一次同化后，集合离散度约 {ISP:.3f}，集合均值的未观测点误差约 {IER:.3f}（纯预报约 {free:.3f}），初始集合离真值较远，靠之后的同化降低。'
             f'EnKF 的膨胀系数只在上述几个取值中选取，$K=30$ 时 EnKF 是否调到最优没有进一步验证；扩散同化的参数未按 $K$ 重调。')
    txt_s = (f'(1) 在未观测格点上，$K=1,10,30$ 时扩散同化的 RMSE 比 EnKF 分别低 {P[1]:.0f}\\%、{P[10]:.0f}\\% 与 {P[30]:.0f}\\%，20 个轨迹上都更优。'
             '(2) 两种方法都随观测间隔增大而变差，扩散同化在各 $K$ 下都低于纯预报，EnKF 在 $K=30$ 时与纯预报相当。'
             '(3) EnKF 从偏离真值较远的初始集合出发，需要数十至数百帧才稳定，稳定所需帧数随 $K$ 增大而增加；扩散同化不需要预热。'
             '(4) 投影后，扩散同化与 EnKF 的散度相同。')
    D = {K: div_row(K) for K in KS}; tv = DV.truth_div_rel.mean(); fp = D[1].loc[FREE]
    pj = pd.read_csv('outputs/study_rb_proj/all_frames.csv'); pm = pj.groupby('variant').missing_rmse.mean(); pcs = pj.groupby(['case', 'variant']).missing_rmse.mean().unstack(); dd = pcs['A0'] - pcs['A1']
    ab0, ab1, ad0, dmin, dmax = pm['A0'], pm['A1'], pj[pj.variant == 'A0'].an_div_rel.mean(), dd.min(), dd.max()
    txt_d = (f'$K=1$ 时扩散同化原始输出的散度（{sci(D[1].loc[DIFF, "an_div_rel"])}）比 EnKF（{sci(D[1].loc[ENKF, "an_div_rel"])}）低，这是因为 $K=1$ 时每一帧都经过扩散同化的投影，EnKF 没有；对两者施加同一个投影后，散度相同（{sci(D[1].loc[DIFF, "proj_div_rel"])} 对 {sci(D[1].loc[ENKF, "proj_div_rel"])}）。'
             f'$K=10,30$ 时多数帧是求解器的预报输出，三种方法的原始散度都在求解器输出自身的水平（约 {sci(fp["an_div_rel"])}）。'
             f'真值的相对散度约 {sci(tv)}。原始输出的散度在 $10^{{-6}}$ 量级，比真值大约一个量级，主要来自小尺度噪声被求导放大，投影后降到 $10^{{-7}}$ 量级，三种方法相同。'
             f'投影对精度几乎没有影响：在 5 个验证轨迹（$K=1$，100 帧，早先的运行）上，不做投影时未观测点 RMSE 为 {ab0:.4f}、投影后为 {ab1:.4f}，5 个轨迹上投影后都略低（差 {dmin:.5f}--{dmax:.5f}）。')
    ab = {f'P{K}': f'{abs(P[K]):.0f}' for K in KS}
    rep = {'TAB_MAIN': ' \\\\\n'.join(tab_main()) + ' \\\\', 'TXT_K': txt_k, 'TXT_D': txt_d, 'TAB_DIV': ' \\\\\n'.join(tab_div()) + ' \\\\', 'TXT_S': txt_s, 'TAB_TIME': ' \\\\\n'.join(f'{K} & {TDP[K]:.0f} ({TDP[K] / 200:.2f}) & {TEN[K]:.0f} ({TEN[K] / 600:.2f})' for K in KS) + ' \\\\', 'POOL': f'{POOL:.3f}', 'NOISE': f'{100 * .1 / POOL:.0f}',
           'FREE': f'{free:.3f}', 'FRELP': f'{100 * free / POOL:.0f}', 'INFL': '、'.join(INFL[K] for K in KS), **ab}
    body = BODY
    for k in sorted(rep, key=len, reverse=True): body = body.replace(f'@{k}@', rep[k])
    left = re.findall(r'@[A-Z0-9_]+@', body)
    if left: raise SystemExit(f'unfilled placeholders: {left}')
    tex = DOC / 'RB实验汇报_稀疏观测同化.tex'; tex.write_text(PRE + body, encoding='utf-8')
    for _ in range(2):
        r = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', tex.name], cwd=DOC, capture_output=True, text=True)
        if r.returncode: print(r.stdout[-2500:]); raise SystemExit(1)
    for ext in ('aux', 'log', 'out'): tex.with_suffix(f'.{ext}').unlink(missing_ok=True)
    print('saved', tex.with_suffix('.pdf'))
    for K in KS: print('K', K, {m: SL[K][0][m].round(4).tolist() for m in SL[K][0]}, 'paired', round(SL[K][1], 4), round(SL[K][2], 4), SL[K][3], 'pct', round(P[K], 1), 'infl', INFL[K])


if __name__ == '__main__': main()
