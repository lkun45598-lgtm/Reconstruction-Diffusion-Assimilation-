"""Main experiment report -> docs/RB实验汇报_稀疏观测同化.tex / .pdf.
Main experiment = lag-initialised EnKF vs diffusion DA, K=1,5,10,20, frames 40..99 (outputs/study_rb_lag). All numbers computed from saved outputs. Usage: python analysis/make_report_main.py (needs xelatex)"""
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
L = Path('outputs/study_rb_lag')
SEEDS, KS, W0 = (42, 43, 44), (1, 5, 10, 20), 40; f4 = lambda v: f'{v:.4f}'
rd = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))])
DIFF, ENKF, FREE = '扩散同化 (DPS)', 'EnKF', '纯预报'; CL = {DIFF: BLUE, ENKF: GRN, FREE: GRY}
SD = np.load('data/processed/stats.npz')['std']; POOL = float(np.sqrt(np.mean(SD.astype(float) ** 2)))   # pooled channel std, used to express errors relatively


def lag(K, s):
    a = rd(f'{L}/testA_s{s}_K{K}_c*.csv').query("method == 'A1'"); e = rd(f'{L}/testE_s{s}_K{K}_c*.csv'); fr = rd(f'{L}/testA_s{s}_K1_c*.csv').query("method == 'free_run'")   # free_run saved with K=1 only (does not depend on K)
    out = pd.concat([a.assign(method=DIFF), e.assign(method=ENKF), fr.assign(method=FREE)]); out['frame'] += W0
    return out[['case', 'frame', 'method', 'missing_rmse']], e.method.iloc[0].split('_i')[1]


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


def div_frames(K):   # per-frame divergence, runs from frame 0 (only these store divergence), frames >= 40, mean over 3 seeds x 20 cases
    r = []
    for s in SEEDS:
        d = (pd.concat([rd(f'{T}/K{K}/case*/frames.csv'), rd(f'{T}/K{K}_i14/case*/frames.csv')]) if s == 42 else rd(f'{TS}/s{s}_K{K}/case*/frames.csv')).query('frame >= @W0')
        en = 'EnKF' if K == 1 else 'EnKF_i1.4'; r.append(d[d.method.isin(['A1', en, 'free_run'])].replace({'A1': DIFF, en: ENKF, 'free_run': FREE}))
    return pd.concat(r)


def div_row(K): return div_frames(K).groupby('method')[['an_div_rel', 'proj_div_rel']].mean()


pct = lambda g: 100 * (1 - g[DIFF].mean() / g[ENKF].mean())
ISP = np.mean([rd(f'{L}/testE_s42_K{K}_c*.csv').groupby('case').init_spread.first().mean() for K in KS]); IER = np.mean([rd(f'{L}/testE_s42_K{K}_c*.csv').groupby('case').init_err.first().mean() for K in KS])


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
    ax.text(8, .25, '背景场与观测均作为似然引导项进入 DPS，采样自纯噪声开始；对每个后验样本做无散度、无滑移投影；逐帧循环。', ha='center', fontsize=6.6, color='#555'); M.save(fig, 'workflow')


def fig_k():
    fig, a = plt.subplots(figsize=(4.6, 2.6)); x = np.arange(len(KS)); w = .26
    for i, m in enumerate((DIFF, ENKF, FREE)):
        v = np.array([SL[K][0][m] for K in KS]); a.bar(x + i * w, v.mean(1), w, color=CL[m], label=m, yerr=[v.mean(1) - v.min(1), v.max(1) - v.mean(1)], capsize=2, error_kw=dict(lw=.8))
    a.set_xticks(x + w); a.set_xticklabels([f'K={k}' for k in KS]); a.set_xlabel('观测间隔 K（帧）'); a.set_ylabel('未观测点 RMSE'); a.set_ylim(0, 0.062); a.legend(fontsize=6.5, loc='upper left', ncol=3); M.save(fig, 'bar_k')


def fig_curves():
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.2), sharey=True)
    for a, K in zip(ax, KS):
        d = pd.concat(LAG[K, s] for s in SEEDS)
        for m in (DIFF, ENKF, FREE): v = d[d.method == m].groupby('frame').missing_rmse.mean(); a.plot(v.index, v, color=CL[m], lw=1.2, label=m)
        if K > 1:
            for f in range(40, 100, K): a.axvline(f, color='#e3e3e3', lw=.6, zorder=0)
        a.set_title(f'K={K}', fontsize=8); a.set_xlabel('帧序号 t')
    ax[0].set_ylabel('未观测点 RMSE'); ax[0].legend(fontsize=6, loc='upper left'); M.save(fig, 'curves')


def fig_div():
    d = {K: div_row(K) for K in (1, 20)}; tv = pd.read_csv('outputs/study_rb_proj/all_frames.csv').truth_div_rel.mean()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True); x = np.arange(2); w = .26
    for a, (col, ttl) in zip(ax, (('an_div_rel', '原始输出'), ('proj_div_rel', '统一投影后'))):
        for i, m in enumerate((DIFF, ENKF, FREE)): a.bar(x + i * w, [d[K].loc[m, col] for K in (1, 20)], w, color=CL[m], label=m)
        a.axhline(tv, color='k', ls='--', lw=.8, label='真值'); a.set_yscale('log'); a.set_xticks(x + w); a.set_xticklabels(['K=1', 'K=20']); a.set_title(ttl, fontsize=8); a.set_xlabel('观测间隔 K（帧）')
    ax[0].set_ylabel('相对散度'); ax[0].set_ylim(5e-8, 2e-4); ax[1].legend(fontsize=6.5, loc='upper right'); M.save(fig, 'div')


def tab_div():
    d = {K: div_row(K) for K in (1, 20)}
    return [f'{m} & ' + ' & '.join(f'{sci(d[K].loc[m, c])}' for K in (1, 20) for c in ('an_div_rel', 'proj_div_rel')) for m in (DIFF, ENKF, FREE)]


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
数值求解器的预报含有模型误差，观测稀疏且带噪声。在二维 Rayleigh--B\'enard 对流上，以求解器生成的数据作为真值，从中抽取带噪声的稀疏观测。无条件训练的扩散模型作先验，求解器预报（背景场）与稀疏观测都作为引导项，经扩散后验采样（DPS）逐帧循环同化，并与 20 成员 EnKF 和纯预报比较。预报求解器的 Rayleigh 数取 $5\times10^4$（真值 $10^5$），观测为 1\% 格点、标准差 0.1 的高斯噪声。EnKF 的初始集合取同一轨迹真值前 20 帧各自积分到第 40 帧，从第 40 帧起评价第 40--99 帧，测试 20 个轨迹、3 个随机种子。结果：观测间隔 $K=1$ 与 $K=5$ 时扩散同化的未观测点 RMSE 比 EnKF 低 @P1@\% 与 @P5@\%；$K=10$ 与 $K=20$ 时反而高 @P10@\% 与 @P20@\%。EnKF 的初始集合由测试轨迹真值的前 20 帧积分得到，扩散同化不使用这部分信息。
\endgroup

\section{研究目的}
同化把含误差的预报与稀疏、带噪声的观测结合起来。实验采用的同化方法是：先验只在状态上无条件训练，不接触观测与背景；同化时，求解器的预报作为背景场、观测作为另一项似然，二者都以引导项的形式进入扩散后验采样（Diffusion Posterior Sampling，DPS），逐帧循环，记为\textbf{扩散同化 (DPS)}。对照基线为 EnKF，参照为纯预报（不再同化）。

实验考察：观测间隔 $K$（每 $K$ 帧提供一次观测）增大、观测更稀疏时，扩散同化相对 EnKF 的精度如何变化。

\section{数据设置和处理}
\subsection{控制方程与数值求解}
研究对象为二维 Boussinesq Rayleigh--B\'enard 对流，状态为浮力 $b$ 与速度 $\mathbf u=(u_x,u_z)$：
\begin{equation}
\nabla\!\cdot\!\mathbf u=0,\quad \partial_t b-\kappa\nabla^2 b=-\mathbf u\!\cdot\!\nabla b,\quad \partial_t\mathbf u-\nu\nabla^2\mathbf u+\nabla p-b\,\mathbf e_z=-\mathbf u\!\cdot\!\nabla\mathbf u,
\end{equation}
其中 $\kappa=(\mathrm{Ra}\,\mathrm{Pr})^{-1/2}$，$\nu=(\mathrm{Ra}/\mathrm{Pr})^{-1/2}$。Rayleigh 数 Ra 是方程中的一个无量纲系数，表示浮力驱动与黏性、热扩散耗散之比，数值越大对流越剧烈；Prandtl 数 Pr 取 1。$x$ 方向周期（$L_x=3$），$z$ 方向为上下壁面（$L_z=1$），$b(0)=1$、$b(L_z)=0$，壁面无滑移。用 Dedalus v3 求解（$x$ 方向 Fourier、$z$ 方向 Chebyshev，网格 $72\times24$，时间步长 $\mathrm{d}t=0.01$）。保存的相邻两帧相隔 35 个求解器步（0.35 个时间单位）。

\subsection{数据集}
真值数据的 Ra$=10^5$。数据按轨迹划分，每个轨迹 100 帧，状态形状 $72\times24\times3$（$b$、$u_x$、$u_z$），三个通道的标准差分别为 0.256、0.164、0.158。
\begin{table}[H]\centering\small\caption{数据集划分与用途。}
\begin{tabular}{lccp{7.6cm}}\toprule
子集 & 轨迹数 & 帧数/轨迹 & 用途\\\midrule
训练集 & 160 & 100 & 训练扩散先验\\
验证集 & 20 & 100 & 选取扩散同化的引导参数与 EnKF 的膨胀系数\\
测试集 & 20 & 100 & 仅用于最终评价\\\bottomrule
\end{tabular}\end{table}
数据按通道标准化后用于训练扩散先验，评价在物理量纲下进行。

\section{实验方法}
\subsection{扩散同化 (DPS)}
\textbf{先验。}U-Net（基础通道数 16），在训练集的帧上按标准扩散目标无条件训练 200 个 epoch。训练中不使用观测、背景场和时间信息，同化时不再训练。

\textbf{后验采样与引导。}采样自纯噪声开始（100 步，起始时刻 $t=0.4$），每步在先验得分上叠加三类引导：(1) 观测项，只在观测点上计算，权重由 $\sigma_{\rm obs}=0.05$ 决定；(2) 背景项，在全场计算，约束分析场接近求解器预报，权重由 $\sigma_{\rm bg}=0.3$ 决定；(3) 散度惩罚，系数 0.3。引导总强度系数为 2，不强制观测点取观测值。$\sigma_{\rm obs}$ 是引导强度，不等同于观测噪声标准差。每个窗口取 32 个样本，对每个样本做投影（把速度改写成由流函数导出、壁面处为零的形式，使其严格无散度、壁面无滑移），取样本均值作为分析场，即该帧的同化结果。

\textbf{循环。}同化时刻：用求解器把上一次分析场预报到当前帧作为背景，再采样得到分析场；非同化时刻：直接取求解器预报。起始时刻没有背景，只用该帧观测采样。
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{report_fig_main/workflow.pdf}\caption{扩散同化的循环流程。}\end{figure}

\subsection{对照方法}
\textbf{EnKF。}20 个成员的随机 EnKF，状态为 $(b,u_x,u_z)$ 堆叠的整场向量。分析式为
\begin{equation}
\mathbf x^a-\mathbf x^f=\mathbf P^f\mathbf H^{\!\top}(\mathbf H\mathbf P^f\mathbf H^{\!\top}+\mathbf R)^{-1}[\mathbf y-H(\mathbf x^f)],
\end{equation}
$\mathbf P^f$ 为预报集合的样本协方差，$\mathbf H$ 取观测点，$\mathbf R=0.01\,\mathbf I$（观测误差标准差 0.1）。每个成员使用各自叠加 $\mathcal N(0,\mathbf R)$ 噪声的观测，分析场取集合均值。预报协方差做乘性膨胀，不使用局地化。各成员由求解器（与扩散同化相同，Ra$=5\times10^4$）独立预报。

\textbf{EnKF 的初始集合。}集合成员全部相同时协方差为零，观测不起作用，因此成员之间需要有差异。采用滞后积分 (lag)：取同一测试轨迹真值的第 0--19 帧，各自用求解器积分到第 40 帧，得到 20 个成员，各成员的预报时长为 21--40 帧。该集合使用了测试轨迹真值的前 20 帧。

\textbf{纯预报。}只在起始时刻用观测得到分析场，此后只由求解器预报，不再同化。

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
观测间隔 & 每 $K$ 帧一次，$K=1,5,10,20$\\
起始 & 第 40 帧，只用该帧观测，不用真值\\\bottomrule
\end{tabular}\end{table}
观测噪声约为通道标准差（合并值 @POOL@）的 @NOISE@\%；不做同化时预报的 RMSE 约 @FREE@，约为通道标准差的 @FRELP@\%。观测噪声每个同化帧重新抽取，观测点位置由随机种子决定、同一轨迹内不变，同一轨迹、同一种子下三种方法使用相同的观测与噪声。

\subsection{起止时刻}
三种方法在同一轨迹、同一种子下使用相同的观测与噪声，从第 40 帧开始，评价第 40--99 帧（共 60 帧）。EnKF 带着 lag 集合从第 40 帧开始同化；扩散同化与纯预报在第 40 帧起始时只用该帧观测，没有历史信息。

\subsection{评价与调参}
指标为\textbf{未观测点 RMSE}：在没有观测的格点上，三个通道合并，计算分析场（非同化帧为预报场）与真值的均方根误差，物理量纲。评价窗口为第 40--99 帧，观测间隔 $K=1,5,10,20$。随机种子 42、43、44，每个种子 20 个测试轨迹，表中为三个种子的均值，误差线为三个种子的最小至最大。EnKF 的膨胀系数在验证集 5 个轨迹上从 1.0--2.0 中选取，$K=1,5,10,20$ 依次为 @INFL@；扩散同化的引导参数在验证集 $K=1$ 下选定，各 $K$ 共用。

\section{实验结果和分析}
\subsection{观测间隔的影响}
\begin{table}[H]\centering\small\caption{第 40--99 帧未观测点 RMSE（三个种子均值）。``相对 EnKF''为扩散同化的 RMSE 相对 EnKF 的变化，负值表示更低；``更优轨迹数''为 20 个测试轨迹中扩散同化的 RMSE 低于 EnKF 的个数。}
\begin{tabular}{ccccccc}\toprule
$K$ & EnKF 膨胀 & 扩散同化 (DPS) & EnKF & 纯预报 & 相对 EnKF & 更优轨迹数\\\midrule
@TAB_MAIN@
\bottomrule\end{tabular}\end{table}
\begin{figure}[H]\centering\includegraphics[width=.62\linewidth]{report_fig_main/bar_k.pdf}\caption{不同观测间隔下的未观测点 RMSE（误差线为三个种子的最小至最大）。}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{report_fig_main/curves.pdf}\caption{各观测间隔下的逐帧误差（三个种子、20 个轨迹平均）。}\end{figure}
@TXT_K@

\subsection{不可压缩约束（散度）}
速度场应满足 $\nabla\!\cdot\mathbf u=0$。用相对散度 $\|\nabla\!\cdot\mathbf u\|/\|\nabla\mathbf u\|$ 衡量（导数用 Fourier--Chebyshev 谱方法计算；真值约 $3.5\times10^{-5}$）。扩散模型的采样输出不满足这一条，扩散同化因此在输出前对速度场做投影（把速度改写成由流函数导出、壁面处为零的形式，使其严格无散度）；EnKF 和纯预报的成员由求解器预报，近似无散度。表~\ref{tab:div} 中``原始输出''是各方法实际给出的结果（扩散同化的结果已经过投影，EnKF 与纯预报没有），``统一投影后''是对三种方法的结果都施加同一个投影之后的值。
\begin{table}[H]\centering\small\caption{同化结果的相对散度（第 40--99 帧，20 个轨迹、3 个种子的均值）。数据取自从第 0 帧起连续同化的运行，其中 EnKF 以扩散同化第 0 帧的结果为中心加扰动初始化，与 5.1 节的运行不同；$K=20$ 的 EnKF 膨胀系数为 1.4。}\label{tab:div}
\begin{tabular}{lcccc}\toprule
 & \multicolumn{2}{c}{$K=1$} & \multicolumn{2}{c}{$K=20$}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}
方法 & 原始输出 & 统一投影后 & 原始输出 & 统一投影后\\\midrule
@TAB_DIV@
\bottomrule\end{tabular}\end{table}
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{report_fig_main/div.pdf}\caption{同化结果的相对散度（虚线为真值的散度；第 40--99 帧，3 个种子、20 个轨迹平均）。}\end{figure}
@TXT_D@

\subsection{小结}
@TXT_S@
\end{document}
'''


def main():
    FIG.mkdir(parents=True, exist_ok=True); fig_workflow(); fig_k(); fig_curves(); fig_div()
    P = {K: pct(SL[K][0]) for K in KS}; free = np.mean([SL[K][0][FREE].mean() for K in KS])
    txt_k = (f'$K=1$ 和 $5$ 时扩散同化的 RMSE 低于 EnKF，分别低 {P[1]:.0f}\\% 与 {P[5]:.0f}\\%，20 个轨迹中分别有 {SL[1][3]} 个与 {SL[5][3]} 个更优；'
             f'$K=10$ 和 $20$ 时 EnKF 更低，扩散同化的 RMSE 分别高 {-P[10]:.0f}\\% 与 {-P[20]:.0f}\\%，只有 {SL[10][3]} 个与 {SL[20][3]} 个轨迹更优。'
             f'各 $K$ 下两种同化方法都低于纯预报（{free:.4f}）；$K$ 增大时两者都向纯预报靠近。EnKF 的初始集合在第 40 帧的离散度约 {ISP:.3f}，集合均值误差约 {IER:.3f}。')
    txt_s = (f'(1) $K=1,5$ 时扩散同化的未观测点 RMSE 低于 EnKF，分别低 {P[1]:.0f}\\% 与 {P[5]:.0f}\\%。'
             f'(2) $K=10,20$ 时 EnKF 更低，扩散同化的 RMSE 分别高 {-P[10]:.0f}\\% 与 {-P[20]:.0f}\\%；EnKF 的初始集合取自测试轨迹真值的前 20 帧。'
             '(3) 各 $K$ 下两种同化方法都低于纯预报。'
             '(4) 投影后，扩散同化与 EnKF 的散度相同。')
    d1, d20 = div_row(1), div_row(20)
    pj = pd.read_csv('outputs/study_rb_proj/all_frames.csv'); pm = pj.groupby('variant').missing_rmse.mean(); pcs = pj.groupby(['case', 'variant']).missing_rmse.mean().unstack(); dd = pcs['A0'] - pcs['A1']
    ab0, ab1, ad0, dmin, dmax = pm['A0'], pm['A1'], pj[pj.variant == 'A0'].an_div_rel.mean(), dd.min(), dd.max()
    txt_d = (f'$K=1$ 时扩散同化原始输出的散度（{sci(d1.loc[DIFF, "an_div_rel"])}）比 EnKF（{sci(d1.loc[ENKF, "an_div_rel"])}）低，这是因为扩散同化在输出前已做投影，EnKF 没有；对两者施加同一个投影后，散度相同（{sci(d1.loc[DIFF, "proj_div_rel"])} 对 {sci(d1.loc[ENKF, "proj_div_rel"])}）。'
             f'$K=20$ 时多数帧是求解器的预报输出，三种方法的散度都处在求解器输出自身的水平（约 {sci(d20.loc[FREE, "an_div_rel"])}）。'
             f'扩散模型的原始采样输出散度明显偏大，但投影对精度几乎没有影响：在 5 个验证轨迹（$K=1$，100 帧）上，不做投影时未观测点 RMSE 为 {ab0:.4f}、原始散度 {ad0:.3f}，做投影后 RMSE 为 {ab1:.4f}，5 个轨迹上投影后都略低（差 {dmin:.5f}--{dmax:.5f}）。')
    ab = {f'P{K}': f'{abs(P[K]):.0f}' for K in KS}
    rep = {'TAB_MAIN': ' \\\\\n'.join(tab_main()) + ' \\\\', 'TXT_K': txt_k, 'TXT_D': txt_d, 'TAB_DIV': ' \\\\\n'.join(tab_div()) + ' \\\\', 'TXT_S': txt_s, 'POOL': f'{POOL:.3f}', 'NOISE': f'{100 * .1 / POOL:.0f}',
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
    for K in KS: print('lag K', K, {m: SL[K][0][m].round(4).tolist() for m in SL[K][0]}, 'paired', round(SL[K][1], 4), round(SL[K][2], 4), SL[K][3], 'pct', round(P[K], 1), 'infl', INFL[K])


if __name__ == '__main__': main()
