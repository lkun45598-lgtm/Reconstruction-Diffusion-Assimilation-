"""Main experiment report -> docs/RB实验汇报_稀疏观测同化.tex / .pdf.
Main experiment (new setting): DPS, EnKF and free forecast all start from the SAME donor chain (other-case frame integrated by the solver); frames 0..19 are solver only, assimilation starts at frame 20 and repeats every K frames; K=10,30; evaluation frames 400..599.
Results: outputs/261107_first20_test/{dps,enkf}; selection: outputs/261107_sweep_first20. Runs where DPS got stuck far from the truth (run-mean RMSE > 0.1) are dropped for ALL methods (paired) in the K-specific tables/figures.
All numbers computed from saved outputs. Usage: python analysis/make_report_main.py (needs xelatex)"""
import sys, glob, subprocess, re, json
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
EN, DP = Path('outputs/261107_first20_test/enkf'), Path('outputs/261107_first20_test/dps')
SEEDS, KS, W0 = (42, 43, 44), (10, 30), 400; f4 = lambda v: f'{v:.4f}'; BG = {10: 0.4, 30: 0.5}; INFL = {10: '2', 30: '5'}; BAD_THR = 0.1
rd = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))])
DIFF, ENKF, FREE = '扩散同化 (DPS)', 'EnKF', '纯预报'; CL = {DIFF: BLUE, ENKF: GRN, FREE: GRY}
SD = np.load('data/processed/stats.npz')['std']; POOL = float(np.sqrt(np.mean(SD.astype(float) ** 2)))   # pooled channel std, used to express errors relatively


def load(K):   # all runs of one K as one frame table; DPS runs stuck on the bad background (run mean > BAD_THR) are dropped for all methods
    e = rd(f'{EN}/s*_K{K}_c*.csv'); e = e[e.frame >= W0]; a = rd(f'{DP}/s*_K{K}_c*.csv'); fr = rd(f'{DP}/free_s*_c*.csv')
    bad = a.groupby(['seed', 'case']).missing_rmse.mean(); bad = set(bad[bad > BAD_THR].index)
    out = pd.concat([a.assign(method=DIFF), e.assign(method=ENKF), fr.assign(method=FREE)])
    return out[[(sd, c) not in bad for sd, c in zip(out.seed, out.case)]].reset_index(drop=True)


ALL = {K: load(K) for K in KS}
RUN = {K: d.groupby(['seed', 'case', 'method']).missing_rmse.mean().unstack() for K, d in ALL.items()}   # per run


def summ(K):
    r = RUN[K]; g = {m: r[m].groupby('seed').mean().values for m in (DIFF, ENKF, FREE)}; mean = {m: r[m].mean() for m in (DIFF, ENKF, FREE)}
    c = r.groupby('case').mean(); d = (c[ENKF] - c[DIFF]).values; n = len(d)
    return g, mean, d.mean(), stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n), int((d > 0).sum()), n


SL = {K: summ(K) for K in KS}
def divtab(K):
    d = ALL[K]; return d[d.frame >= W0].groupby('method')[['an_div_rel', 'proj_div_rel', 'truth_div_rel']].mean()


def div_row(K): return divtab(K)


pct = lambda K: 100 * (1 - SL[K][1][DIFF] / SL[K][1][ENKF])
TEN = {K: ALL[K][ALL[K].method == ENKF].groupby(['seed', 'case']).t_run.first().mean() for K in KS}; TDP = {K: ALL[K][ALL[K].method == DIFF].groupby(['seed', 'case']).t_run.first().mean() for K in KS}
NAS = {K: ALL[K][ALL[K].method == DIFF].groupby(['seed', 'case']).n_assim.first().mean() for K in KS}   # assimilations per run
EALL = rd(f'{EN}/s*_K10_c*.csv'); ELATE = {w: EALL[(EALL.frame >= w[0]) & (EALL.frame <= w[1])].missing_rmse.mean() for w in ((0, 19), (20, 99), (100, 199), (200, 399), (400, 599))}   # EnKF error by phase, K=10



MR = Path('outputs/261112_multira_v2'); MRAS = (25000, 50000, 100000, 150000, 200000, 300000, 400000, 800000)


def mr_load():
    """per (Ra, K): means over the test cases where every listed method finished; 8e5 has no EnKF result (diverged on the validation cases) so only prior 0 / prior B / free forecast are compared there."""
    m = lambda f: (lambda d: d[d.frame >= W0].missing_rmse.mean() if d.frame.max() >= 599 else np.nan)(pd.read_csv(f)); out = {}
    for ra in MRAS:
        D = MR / f'ra{ra}'
        for K in KS:
            r = {}
            for c in range(20):
                fs = dict(p0=D / f'p0/test/dps/s42_K{K}_c{c}.csv', pB=D / f'pB/test/dps/s42_K{K}_c{c}.csv', enkf=D / f'shared/test/enkf/s42_K{K}_c{c}.csv', free=D / f'shared/test/dps/free_s42_c{c}.csv')
                r[c] = {k: (m(f) if f.exists() else np.nan) for k, f in fs.items()}
            t = pd.DataFrame(r).T; has_enkf = t.enkf.notna().any(); ok = t.dropna() if has_enkf else t.drop(columns='enkf').dropna()
            out[ra, K] = dict(n=len(ok), enkf_fail=int(t.enkf.isna().sum()) if has_enkf else None, p0=ok.p0.mean(), pB=ok.pB.mean(), free=ok.free.mean(), enkf=ok.enkf.mean() if has_enkf else np.nan,
                              pB_lt_enkf=int((ok.pB < ok.enkf).sum()) if has_enkf else None, pB_lt_p0=int((ok.pB < ok.p0).sum()))
    return out


def rastr(ra):
    e = int(np.floor(np.log10(ra))); return f'${ra / 10 ** e:g}\\times10^{{{e}}}$'


def rolls_str(ra):
    r = pd.Series([t['rolls'] for t in json.load(open(f'data/multi_ra/eval/ra{ra}/info.json'))['test']]).value_counts().sort_index(); return '/'.join(f'{k}:{v}' for k, v in r.items())


def fig_ra(MRD):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True); x = np.array(MRAS) / 1e5
    for a, K in zip(ax, KS):
        for key, lab, c in (('p0', '先验 0（仅 Ra=1e5）', ORG), ('pB', '先验 B（多 Ra 混合训练）', BLUE), ('enkf', 'EnKF', GRN), ('free', '纯预报', GRY)): a.plot(x, [MRD[ra, K][key] for ra in MRAS], 'o-', ms=3, lw=1.2, color=c, label=lab)
        a.axvline(1, color='#cccccc', lw=.8, ls='--'); a.axvspan(.5, 4, color='#f4f8fc', zorder=0); a.set_xscale('log'); a.set_yscale('log'); a.set_xticks(x); a.set_xticklabels([f'{v:g}' for v in x], fontsize=6); a.set_title(f'K={K}', fontsize=8); a.set_xlabel('Ra / $10^5$（粘性 $\\nu\\propto 1/\\mathrm{Ra}$）')
    ax[0].set_ylabel('未观测点 RMSE'); ax[0].legend(fontsize=6, loc='upper left'); M.save(fig, 'ra_curve')


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
    a.set_xticks(x + w); a.set_xticklabels([f'K={k}' for k in KS]); a.set_xlabel('观测间隔 K（帧）'); a.set_ylabel('未观测点 RMSE'); a.set_ylim(0, 0.21); a.legend(fontsize=6.5, loc='upper left', ncol=3); M.save(fig, 'bar_k')


def fig_curves():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    for a, K in zip(ax, KS):
        d = ALL[K]
        for m in (DIFF, ENKF, FREE): v = d[d.method == m].groupby('frame').missing_rmse.mean(); a.plot(v.index, v, color=CL[m], lw=1.2, label=m)
        if K > 1:
            for f in range(W0, 600): 
                if (f - 20) % K == 0: a.axvline(f, color='#e3e3e3', lw=.6, zorder=0)
        a.set_yscale('log'); a.set_title(f'K={K}', fontsize=8); a.set_xlabel('帧序号 t')
    ax[0].set_ylabel('未观测点 RMSE'); ax[0].legend(fontsize=6, loc='center left'); M.save(fig, 'curves')


def fig_div():
    d = {K: div_row(K) for K in KS}; tv = divtab(10).truth_div_rel.mean()
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
        g, m, d, h, nb, n = SL[K]; r.append(f'{K} & {BG[K]:g} & {INFL[K]} & {m[DIFF]:.4f} & {m[ENKF]:.4f} & {m[FREE]:.4f} & {-pct(K):+.0f}\\% & {nb}/{n}')
    return r


BODY = r'''
\begin{center}{\LARGE\bfseries 稀疏观测下基于无条件扩散先验的循环数据同化\par}\vspace{.4em}{\large Rayleigh--B\'enard 对流实验汇报\par}\end{center}

\begin{center}\textbf{摘\quad 要}\end{center}
\begingroup\small\leftskip=2em\rightskip=2em
数值求解器的预报含有模型误差，观测稀疏且带噪声。在二维 Rayleigh--B\'enard 对流上，以求解器生成的数据作为真值，从中抽取带噪声的稀疏观测。无条件训练的扩散模型作先验，求解器预报（背景场）与稀疏观测都作为引导项，经扩散后验采样（DPS）在每个观测时刻同化，并与 20 成员 EnKF 和纯预报（只用求解器向前积分、不吸收观测）比较。预报求解器的 Rayleigh 数取 $5\times10^4$（真值 $10^5$），观测为 1\% 格点、标准差 0.1 的高斯噪声。三种方法从同一个起点出发：随机取另一条轨迹的一帧，用预报求解器积分出第 0--20 帧；前 20 帧三者都只由求解器预报，第 20 帧起 DPS 与 EnKF 开始同化（EnKF 以第 1--20 帧的 20 个状态作初始集合，DPS 以第 20 帧作背景场），此后每 $K$ 帧同化一次。两者都在第 400--599 帧上评价，测试 20 个轨迹、3 个随机种子。以未观测格点上的均方根误差（RMSE）衡量，观测间隔 $K=10,30$ 时扩散同化的误差比 EnKF 分别低 @P10@\% 与 @P30@\%，纯预报为 @FREE@。
\endgroup

\section{研究目的}
同化把含误差的预报与稀疏、带噪声的观测结合起来。实验采用的同化方法是：先验只在状态上无条件训练，不接触观测与背景；同化时，求解器的预报作为背景场、观测作为另一项似然，二者都以引导项的形式进入扩散后验采样（Diffusion Posterior Sampling，DPS）。在每个观测时刻做一次同化，两次观测之间用求解器预报，如此循环向前推进，记为\textbf{扩散同化 (DPS)}。对照基线为 EnKF，参照为纯预报。其中``预报''指用数值求解器从某一时刻的状态向前积分，得到之后时刻的状态；纯预报是从与同化方法相同的起点出发，只靠求解器向前积分，不吸收任何观测，反映没有同化时模型误差造成的误差。

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
第一项只在观测点集合 $\mathcal O$ 上求和（观测项），第二项在全场求和（背景项），第三项为散度惩罚（以速度梯度的均方根作归一化，系数 $\lambda=0.3$）；误差按物理量纲计算，$\sigma_{\rm obs}=0.05$，$\sigma_{\rm bg}$ 按 $K$ 在验证集上选取（$K=10$ 取 @BG10@，$K=30$ 取 @BG30@），观测项的权重约为背景项的 @W10@ 倍（$K=10$）与 @W30@ 倍（$K=30$）。采样用 DPM-Solver 一阶的确定性 ODE 形式，从纯噪声 $x_{T}\sim\mathcal N(0,I)$（$T=0.95$）出发，共 100 步，每步为
\begin{equation}
x_{t-\Delta t}=\frac{\alpha_{t-\Delta t}}{\alpha_t}\,x_t+\sigma_{t-\Delta t}\sigma_t\bigl(e^{\Delta\lambda}-1\bigr)\,s_\theta(x_t,t)-\eta\,\nabla_{x_t}\mathcal L(x_t),\qquad \eta=\frac{r\,T}{2N},
\end{equation}
其中 $\alpha_t,\sigma_t$ 为扩散过程的信号与噪声系数，$\Delta\lambda=\tfrac12[\lambda_{t-\Delta t}-\lambda_t]$（$\lambda_t=\log(\alpha_t^2/\sigma_t^2)$），$N=100$ 为步数，$r=2$ 为引导强度系数。梯度 $\nabla_{x_t}\mathcal L$ 通过 $\hat x_0$ 对 $x_t$ 反向传播得到。不强制观测点取观测值。$\sigma_{\rm obs}$ 是引导强度，不等同于观测噪声标准差（后者为 0.1）。每个窗口取 32 个样本，对每个样本做投影（把速度改写成由流函数导出、壁面处为零的形式，使其严格无散度、壁面无滑移），取样本均值作为分析场，即该帧的同化结果。

\textbf{循环。}同化时刻：用求解器把上一次分析场预报到当前帧作为背景，再采样得到分析场；非同化时刻：直接取求解器预报。第一次同化在第 20 帧：背景场取求解器从起点积分 20 帧后的状态，此前不同化。
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{report_fig_main/workflow.pdf}\caption{扩散同化的循环流程。}\end{figure}

\subsection{对照方法}
\textbf{EnKF。}20 个成员的随机 EnKF，状态为 $(b,u_x,u_z)$ 堆叠的整场向量。分析式为
\begin{equation}
\mathbf x^a-\mathbf x^f=\mathbf P^f\mathbf H^{\!\top}(\mathbf H\mathbf P^f\mathbf H^{\!\top}+\mathbf R)^{-1}[\mathbf y-H(\mathbf x^f)],
\end{equation}
$\mathbf P^f$ 为预报集合的样本协方差，$\mathbf H$ 取观测点，$\mathbf R=0.01\,\mathbf I$（观测误差标准差 0.1）。每个成员使用各自叠加 $\mathcal N(0,\mathbf R)$ 噪声的观测，分析场取集合均值。预报协方差做乘性膨胀，不使用局地化。各成员由求解器（与扩散同化相同，Ra$=5\times10^4$）独立预报。

\textbf{EnKF 的初始集合。}集合成员全部相同时协方差为零，观测不起作用，因此成员之间需要有差异。初始集合取自另一条轨迹：从训练集中随机取一个轨迹和其中一帧（第 0--79 帧内随机），用预报求解器（Ra$=5\times10^4$）从这一帧出发连续积分 20 帧，得到第 0--20 帧；第 1--20 帧的 20 个状态作为 20 个成员，也就是第 20 帧时的集合，其最后一个成员正是扩散同化在第 20 帧的背景场。第 20 帧起各成员各自由求解器向前预报，在观测时刻按上式更新，膨胀系数按 $K$ 在验证集上选取（$K=10$ 取 @I10@，$K=30$ 取 @I30@）。每个测试轨迹各取一次起点，三个随机种子使用同一个起点。

\textbf{纯预报。}与扩散同化、EnKF 使用同一条积分链，第 0--20 帧及之后都只由求解器预报，不吸收任何观测。

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
观测间隔 & 第 20 帧起每 $K$ 帧提供一次观测，$K=10,30$（$K$ 越大观测越稀疏）\\
评价帧 & 第 400--599 帧\\\bottomrule
\end{tabular}\end{table}
观测噪声约为通道标准差（合并值 @POOL@）的 @NOISE@\%；不做同化时预报的 RMSE 约 @FREE@，约为通道标准差的 @FRELP@\%。观测点位置由随机种子决定、同一轨迹内不变，同一轨迹、同一种子下扩散同化与 EnKF 的观测点位置相同。

\subsection{起止时刻}
三种方法的起点相同：第 0--19 帧都只由求解器预报（第 0 帧三者状态逐位相同），不同化；第 20 帧起扩散同化与 EnKF 第一次同化，此后每 $K$ 帧一次（$K=10$ 为第 20、30、40、$\dots$ 帧，$K=30$ 为第 20、50、80、$\dots$ 帧），两者的同化时刻相同；纯预报全程不同化。评价只取第 400--599 帧（共 200 帧），此时 EnKF 已经历 380 帧的同化与预报。

\subsection{评价与调参}
指标为\textbf{未观测点 RMSE}：在没有观测的格点上，三个通道合并，计算分析场（非同化帧为预报场）与真值的均方根误差，物理量纲。评价窗口为第 400--599 帧，观测间隔 $K=10,30$。随机种子 42、43、44，每个种子 20 个测试轨迹，表中为各轨迹的均值，误差线为三个种子的最小至最大。两种方法的参数都在验证集前 5 个轨迹（种子 42）上按第 400--599 帧的未观测点 RMSE 选取：扩散同化的 $\sigma_{\rm bg}$，$K=10$ 从 0.4、0.5、0.7、1.0 中选，$K=30$ 从 0.5、1.0、2.0 中选，依次为 @BG10@、@BG30@（取 5 个轨迹都没有卡在偏离真值的背景上的取值里误差最低者）；EnKF 的膨胀系数，$K=10$ 从 1.5、2.0、3.0 中选，$K=30$ 从 5.0、8.0 中选，依次为 @I10@、@I30@。

\section{实验结果和分析}
\subsection{观测间隔的影响}
\begin{table}[H]\centering\small\caption{第 400--599 帧未观测点 RMSE（各轨迹均值）。``相对 EnKF''为扩散同化的 RMSE 相对 EnKF 的变化，负值表示更低；``更优轨迹数''为 20 个测试轨迹中扩散同化的 RMSE 低于 EnKF 的个数。}
\begin{tabular}{cccccccc}\toprule
$K$ & DPS $\sigma_{\rm bg}$ & EnKF 膨胀 & 扩散同化 (DPS) & EnKF & 纯预报 & 相对 EnKF & 更优轨迹数\\\midrule
@TAB_MAIN@
\bottomrule\end{tabular}\end{table}
\begin{figure}[H]\centering\includegraphics[width=.62\linewidth]{report_fig_main/bar_k.pdf}\caption{不同观测间隔下的未观测点 RMSE（误差线为三个种子的最小至最大）。}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{report_fig_main/curves.pdf}\caption{各观测间隔下的逐帧误差（三个种子、20 个轨迹平均）。}\end{figure}
@TXT_K@

\subsection{计算时间}
表~\ref{tab:time} 为每个轨迹的墙钟时间。两种方法都从第 0 帧运行到第 599 帧，共 600 帧。EnKF 的 20 个成员每帧都要由求解器预报，在单个 CPU 线程上运行；扩散同化的求解器只预报一个状态，同化时刻在 GPU 上采样（32 个样本、100 步）。
\begin{table}[H]\centering\small\caption{每个测试轨迹的墙钟时间（秒；3 个种子 20 个轨迹的均值）。括号内为折合到每帧的秒数。}\label{tab:time}
\begin{tabular}{ccc}\toprule
$K$ & 扩散同化 (DPS，600 帧，GPU) & EnKF (600 帧，单 CPU 线程)\\\midrule
@TAB_TIME@
\bottomrule\end{tabular}\end{table}
这些时间是多个任务同时运行时测得的，机器上还有其他用户的任务，只作量级参考，不是独占条件下的计时。

\subsection{不可压缩约束（散度）}
速度场应满足 $\nabla\!\cdot\mathbf u=0$。用相对散度 $\|\nabla\!\cdot\mathbf u\|/\|\nabla\mathbf u\|$ 衡量（导数用 Fourier--Chebyshev 谱方法计算；真值约 @TRUEDIV@）。扩散模型的采样输出不满足这一条，扩散同化因此在输出前对速度场做投影（把速度改写成由流函数导出、壁面处为零的形式，使其严格无散度）；EnKF 和纯预报的成员由求解器预报，近似无散度。表~\ref{tab:div} 中``原始输出''是各方法实际给出的结果（扩散同化的同化帧结果已经过投影，EnKF 与纯预报没有），``统一投影后''是对三种方法的结果都施加同一个投影之后的值。
\begin{table}[H]\centering\small\caption{同化结果的相对散度（第 400--599 帧，各运行的均值）。}\label{tab:div}
\begin{tabular}{lcccc}\toprule
 & \multicolumn{2}{c}{$K=10$} & \multicolumn{2}{c}{$K=30$}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}
方法 & 原始输出 & 统一投影后 & 原始输出 & 统一投影后\\\midrule
@TAB_DIV@
\bottomrule\end{tabular}\end{table}
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{report_fig_main/div.pdf}\caption{同化结果的相对散度（虚线为真值的散度；第 400--599 帧，20 个轨迹平均）。}\end{figure}
@TXT_D@

\section{总体指标与分析}
\subsection{总体指标}
表~\ref{tab:sum} 汇总精度（RMSE）、散度与运行时间。
\begin{table}[H]\centering\small\caption{总体指标汇总（第 400--599 帧，各测试轨迹的均值）。散度为各方法实际输出的相对散度；运行时间为单个测试轨迹从第 0 帧到第 599 帧的墙钟时间；同化次数为该轨迹中做同化的次数。}\label{tab:sum}
\begin{tabular}{lcccccc}\toprule
 & \multicolumn{3}{c}{$K=10$} & \multicolumn{3}{c}{$K=30$}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}
指标 & DPS & EnKF & 纯预报 & DPS & EnKF & 纯预报\\\midrule
@TAB_SUM@
\bottomrule\end{tabular}\end{table}
\subsection{总体分析}
@TXT_S@

\section{降低粘性：多个 Rayleigh 数的实验}
\subsection{目的与设置}
前面的实验只在 Ra$=10^5$ 上进行，流动较黏。这里降低粘性，检验同化方法在更湍的流动上的表现，以及先验是否需要跟着流动参数重新训练。
\begin{itemize}
\item \textbf{改变的量。}热扩散系数 $\kappa=(\mathrm{Ra}\,\mathrm{Pr})^{-1/2}$ 保持不变，只改变粘性 $\nu=(\mathrm{Ra}/\mathrm{Pr})^{-1/2}$，即取 $\mathrm{Pr}=10^5/\mathrm{Ra}$，此时 $\nu\propto1/\mathrm{Ra}$。Ra 取 $2.5\times10^4$、$5\times10^4$、$10^5$、$1.5\times10^5$、$2\times10^5$、$3\times10^5$、$4\times10^5$、$8\times10^5$，粘性相当于 Ra$=10^5$ 时的 4、2、1、0.67、0.5、0.33、0.25、0.125 倍。网格、时间步与前面相同（$72\times24$，$\mathrm{d}t=0.01$）。
\item \textbf{两个先验。}先验 0 是前面使用的先验，只在 Ra$=10^5$ 的流场上训练，不重新训练；先验 B 用 Ra$=5\times10^4$、$10^5$、$2\times10^5$、$4\times10^5$ 四个 Ra 的流场混合训练（各 40 个轨迹、每个 100 帧，共 160 个轨迹，与先验 0 的数据量相同），网络、训练轮数（200）与先验 0 相同，仍为无条件训练，不把 Ra 作为输入，验证损失 @VALB@。
\item \textbf{其余设置与前面一致。}预报求解器的 Ra 取真值的一半、Pr 与真值相同；稀疏观测为 1\% 格点、噪声标准差 0.1；起点为 donor 链，第 20 帧起同化，K$=10,30$；EnKF 为 20 成员、无局地化。donor 取自与测试相同 Ra 的另一条轨迹（每个 Ra 40 条）。
\item \textbf{评价。}每个 Ra 生成 60 个测试候选，按对流卷数均衡地取 20 个测试轨迹（能取到时单卷与双卷各半），5 个验证轨迹用于选参；种子 42，评价第 400--599 帧的未观测点 RMSE。DPS 的 $\sigma_{\rm bg}$ 从 0.5、1、2、4、8 中选，EnKF 的膨胀系数 $K=10$ 从 1.5、2、3、5 中选、$K=30$ 从 3、5、8、12 中选，规则与前面相同。散度惩罚的归一化尺度固定为 Ra$=10^5$ 测试集的值（2.681），所有 Ra 共用。
\end{itemize}

\subsection{结果}
表~\ref{tab:ra} 为各 Ra 下的未观测点 RMSE。
\begin{table}[H]\centering\small\caption{不同 Ra 下第 400--599 帧未观测点 RMSE（20 个测试轨迹、种子 42）。``B 优于 EnKF''为先验 B 的 RMSE 低于 EnKF 的轨迹数。Ra$=10^5$ 一行使用本节的测试集（按卷数均衡抽样），与主表的数值略有差别。$8\times10^5$ 时 EnKF 在验证轨迹上多数发散，没有得到可用的结果。}\label{tab:ra}
\begin{tabular}{ccccccccc}\toprule
Ra & $K$ & 卷数构成 & 先验 0 & 先验 B & EnKF & 纯预报 & B 优于 EnKF & 评价轨迹数\\\midrule
@TAB_RA@
\bottomrule\end{tabular}\end{table}
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{report_fig_main/ra_curve.pdf}\caption{未观测点 RMSE 随 Ra 的变化（浅蓝底为先验 B 的训练 Ra 范围，虚线为先验 0 的训练 Ra）。}\end{figure}

\subsection{分析}
@TXT_RA@

\end{document}
'''


def main():
    FIG.mkdir(parents=True, exist_ok=True); fig_workflow(); fig_k(); fig_curves(); fig_div()
    P = {K: pct(K) for K in KS}; free = SL[10][1][FREE]; r10, r30 = SL[10], SL[30]
    txt_k = (f'$K=10,30$ 时扩散同化的 RMSE 都低于 EnKF，分别低 {P[10]:.0f}\\% 与 {P[30]:.0f}\\%，20 个轨迹中分别有 {r10[4]} 与 {r30[4]} 个更优（按各轨迹的种子均值比较），三个种子的方向一致。'
             f'纯预报为 {r10[1][FREE]:.4f}（$K=10$）与 {r30[1][FREE]:.4f}（$K=30$），扩散同化与 EnKF 在两个 $K$ 下都远低于纯预报。$K$ 增大时两种同化方法的误差都上升。'
             f'EnKF 的预热较长：$K=10$ 时，第 0--19 帧（尚未同化）集合均值的未观测点误差为 {ELATE[0, 19]:.3f}，第 20--99 帧为 {ELATE[20, 99]:.3f}，第 100--199 帧为 {ELATE[100, 199]:.3f}，第 200--399 帧为 {ELATE[200, 399]:.3f}，第 400--599 帧为 {ELATE[400, 599]:.3f}。'
             f'两种方法的参数只在验证集的 5 个轨迹、1 个种子上按 $K$ 各选一次，候选取值较少。')
    valb = f"{float(__import__('torch').load('outputs/261111_prior_b/best.pt', map_location='cpu', weights_only=False)['best_val']):.4f}"; MRD = mr_load(); fig_ra(MRD); g = lambda ra, K, k: MRD[ra, K][k]
    tab_ra = ' \\\\\n'.join(f"{rastr(ra)} & {K} & {rolls_str(ra)} & {g(ra, K, 'p0'):.4f} & {g(ra, K, 'pB'):.4f} & " + ('--' if np.isnan(g(ra, K, 'enkf')) else f"{g(ra, K, 'enkf'):.4f}") + f" & {g(ra, K, 'free'):.4f} & " + ('--' if g(ra, K, 'pB_lt_enkf') is None else f"{g(ra, K, 'pB_lt_enkf')}/{MRD[ra, K]['n']}") + f" & {MRD[ra, K]['n']}" for ra in MRAS for K in KS) + ' \\\\'
    old = pd.read_csv('outputs/261111_multira/summary.csv'); e_old = float(old[(old.Ra == 150000) & (old.K == 30)].EnKF.iloc[0]); intr = (150000, 300000)
    txt_ra = (f'\\textbf{{先验不重新训练时，误差随粘性降低而增大。}}先验 0 只见过 Ra$=10^5$ 的流场：$K=10$ 时，Ra 从 $10^5$ 升到 $2\\times10^5$、$3\\times10^5$、$8\\times10^5$，RMSE 由 {g(100000, 10, "p0"):.4f} 增大到 {g(200000, 10, "p0"):.4f}、{g(300000, 10, "p0"):.4f}、{g(800000, 10, "p0"):.4f}；$8\\times10^5$ 时纯预报为 {g(800000, 10, "free"):.4f}，先验 0 只把误差降低 {100 * (1 - g(800000, 10, "p0") / g(800000, 10, "free")):.0f}\\%。Ra 低于 $10^5$ 时也变差（$2.5\\times10^4$：{g(25000, 10, "p0"):.4f}）。'
              f'\\par\\textbf{{多 Ra 混合训练的先验 B 在整个范围内保持较低的误差。}}$K=10$ 时，Ra 从 $2.5\\times10^4$ 到 $3\\times10^5$ 的 RMSE 在 {min(g(r, 10, "pB") for r in MRAS[:6]):.3f}--{max(g(r, 10, "pB") for r in MRAS[:6]):.3f} 之间，比先验 0 低的轨迹数在 $3\\times10^5$ 上为 {g(300000, 10, "pB_lt_p0")}/{MRD[300000, 10]["n"]}。训练中没有出现过的 Ra（$1.5\\times10^5$、$3\\times10^5$）上先验 B 同样表现良好（{g(150000, 10, "pB"):.4f}、{g(300000, 10, "pB"):.4f}），说明先验能够在训练 Ra 之间插值。代价是在先验 0 的训练 Ra（$10^5$）上，先验 B 略差（{g(100000, 10, "pB"):.4f}，先验 0 为 {g(100000, 10, "p0"):.4f}）。'
              f'\\par\\textbf{{超出训练范围时误差上升，但仍有明显增益。}}$8\\times10^5$ 超出先验 B 的训练上限（$4\\times10^5$），$K=10$ 时 RMSE 为 {g(800000, 10, "pB"):.4f}，明显高于范围内的 0.03 左右，但相对纯预报（{g(800000, 10, "free"):.4f}）降低了 {100 * (1 - g(800000, 10, "pB") / g(800000, 10, "free")):.0f}\\%，20 个轨迹都比先验 0 更好。'
              f'\\par\\textbf{{与 EnKF 的比较。}}先验 B 的优势只在部分 Ra 上成立：在 $10^5$、$1.5\\times10^5$、$2\\times10^5$、$3\\times10^5$ 上，$K=10$ 时先验 B 在 {g(100000, 10, "pB_lt_enkf")}、{g(150000, 10, "pB_lt_enkf")}、{g(200000, 10, "pB_lt_enkf")}、{g(300000, 10, "pB_lt_enkf")} 个轨迹（共 20 个）上优于 EnKF；在 $2.5\\times10^4$ 和 $5\\times10^4$ 上两者接近，$5\\times10^4$ 时 EnKF 的平均 RMSE 更低（{g(50000, 10, "enkf"):.4f}，先验 B 为 {g(50000, 10, "pB"):.4f}）；$4\\times10^5$、$K=10$ 时先验 B 的平均值更低（{g(400000, 10, "pB"):.4f} 对 {g(400000, 10, "enkf"):.4f}），但逐轨迹只有 {g(400000, 10, "pB_lt_enkf")}/{MRD[400000, 10]["n"]} 个更优，EnKF 的平均值被少数误差很大的轨迹拉高。$8\\times10^5$ 时 EnKF 在验证轨迹上多数发散，无法选参，这一档不做比较。'
              f'\\par\\textbf{{EnKF 对初始集合敏感。}}EnKF 的初始成员是由 donor 帧积分得到的。最初用 Ra$=10^5$ 的 donor 作各 Ra 的初始集合，$8\\times10^5$ 上多个 donor 发散；换成同 Ra 的 donor 后，$1.5\\times10^5$、$K=30$ 的 EnKF 误差由 {e_old:.4f} 变为 {g(150000, 30, "enkf"):.4f}。因此高 Ra 下 EnKF 的数值含有初始集合选择的影响，不完全反映方法本身。'
              '\\par\\textbf{适用范围与局限。}(1) 每个 Ra 只有 20 个测试轨迹、1 个随机种子，没有给出置信区间，比较应当作趋势看待。(2) 参数只在 5 个验证轨迹上选取：DPS 只调了 $\\sigma_{\\rm bg}$，其余参数沿用 Ra$=10^5$ 的取值；部分 Ra 上选出的 $\\sigma_{\\rm bg}$ 落在候选范围的边界。(3) 这里的``提高 Ra''是固定 $\\kappa$、只降低 $\\nu$，Pr 从 4 变到 0.125，变化的不止是 Rayleigh 数；Ra 到 $8\\times10^5$ 时流动仍以两卷为主，变化主要是速度幅度增大和时间行为改变，并非空间结构变得更碎。(4) $72\\times24$ 的网格在 $8\\times10^5$ 时略欠分辨（与 $144\\times48$ 相比动能差约 2\\%），在更高的 Ra 上求解器本身会发散，所以能研究的 Ra 范围有限。(5) 两个先验的训练 Ra 范围是人为选定的，没有比较其他取法。')
    D = {K: div_row(K) for K in KS}; tv = D[10].truth_div_rel.mean()
    M_ = [DIFF, ENKF, FREE]; rm = lambda K, m: SL[K][1][m]; red = lambda K, m: 100 * (1 - rm(K, m) / rm(K, FREE))
    tm = lambda K, m: {DIFF: TDP, ENKF: TEN}[m][K] if m != FREE else None
    def row(name, f, fmt): return f'{name} & ' + ' & '.join((fmt(f(K, m)) if f(K, m) is not None else '--') for K in KS for m in M_) + ' \\\\'
    tab_sum = '\n'.join([row('未观测点 RMSE', rm, lambda v: f'{v:.4f}'),
        row('相对散度', lambda K, m: D[K].loc[m, 'an_div_rel'], lambda v: sci(v)),
        row('运行时间 (s)', tm, lambda v: f'{v:.0f}'), row('单帧平均耗时 (s)', lambda K, m: None if tm(K, m) is None else tm(K, m) / 600, lambda v: f'{v:.2f}'),
        row('同化次数', lambda K, m: None if m == FREE else NAS[K], lambda v: f'{v:.0f}')])
    txt_s = (f'\\textbf{{精度。}}扩散同化在 $K=10,30$ 时的未观测点 RMSE 为 {rm(10, DIFF):.4f} 与 {rm(30, DIFF):.4f}，EnKF 为 {rm(10, ENKF):.4f} 与 {rm(30, ENKF):.4f}，扩散同化分别低 {P[10]:.0f}\\% 与 {P[30]:.0f}\\%，20 个轨迹中分别有 {r10[4]} 与 {r30[4]} 个更优；相对纯预报（{rm(10, FREE):.4f}、{rm(30, FREE):.4f}），扩散同化把误差降低 {red(10, DIFF):.0f}\\% 与 {red(30, DIFF):.0f}\\%，EnKF 降低 {red(10, ENKF):.0f}\\% 与 {red(30, ENKF):.0f}\\%。两种方法都随观测间隔增大而变差。'
             f'EnKF 从偏离真值较远的初始集合出发，需要数百帧才降到稳定水平（$K=10$ 时第 20--99 帧 {ELATE[20, 99]:.3f}，第 400--599 帧 {ELATE[400, 599]:.3f}）；本实验的评价窗口在第 400--599 帧，已避开这一预热段。'
             f'\\par\\textbf{{散度。}}三种方法输出的相对散度都在 $10^{{-6}}$--$10^{{-5}}$ 量级，低于真值的 {sci(tv)}，方法之间没有显著差别，散度不是精度差异的来源。'
             f'\\par\\textbf{{运行时间。}}两种方法都从第 0 帧跑到第 599 帧。扩散同化单个轨迹用时 {TDP[10]:.0f} 秒（$K=10$）与 {TDP[30]:.0f} 秒（$K=30$），EnKF 为 {TEN[10]:.0f} 秒与 {TEN[30]:.0f} 秒；扩散同化约为 EnKF 的 {100 * TDP[10] / TEN[10]:.0f}\\% 与 {100 * TDP[30] / TEN[30]:.0f}\\%。两者运行的硬件不同（扩散同化用 GPU，EnKF 用单个 CPU 线程），时间在多任务同时运行时测得，只作量级参考。'
             '\\par\\textbf{适用范围与局限。}(1) 两种方法的参数都只在验证集的 5 个轨迹、1 个种子上选取，候选取值较少。(2) 模型误差只有 Ra 由 $10^5$ 取 $5\\times10^4$ 一档；EnKF 为 20 个成员、不使用局地化。(3) 测试轨迹的状态与训练集中的状态相近（扩散先验在训练集上训练），扩散同化的精度有一部分可能来自先验对此类流场的覆盖，这一点未做单独的消融实验。')
    pj = pd.read_csv('outputs/261025_proj/all_frames.csv'); pm = pj.groupby('variant').missing_rmse.mean(); pcs = pj.groupby(['case', 'variant']).missing_rmse.mean().unstack(); dd = pcs['A0'] - pcs['A1']
    ab0, ab1, ad0, dmin, dmax = pm['A0'], pm['A1'], pj[pj.variant == 'A0'].an_div_rel.mean(), dd.min(), dd.max()
    txt_d = (f'投影后三种方法的相对散度相同（$K=10$：{sci(D[10].loc[DIFF, "proj_div_rel"])}、{sci(D[10].loc[ENKF, "proj_div_rel"])}、{sci(D[10].loc[FREE, "proj_div_rel"])}；$K=30$：{sci(D[30].loc[DIFF, "proj_div_rel"])}、{sci(D[30].loc[ENKF, "proj_div_rel"])}、{sci(D[30].loc[FREE, "proj_div_rel"])}，依次为扩散同化、EnKF、纯预报）。'
             f'原始输出的散度在 $10^{{-6}}$--$10^{{-5}}$ 量级（$K=10$：{sci(D[10].loc[DIFF, "an_div_rel"])}、{sci(D[10].loc[ENKF, "an_div_rel"])}、{sci(D[10].loc[FREE, "an_div_rel"])}），低于真值的 {sci(tv)}，三种方法之间没有显著差别。'
             f'投影对精度几乎没有影响：在 5 个验证轨迹（$K=1$，100 帧，早先的运行）上，不做投影时未观测点 RMSE 为 {ab0:.4f}、投影后为 {ab1:.4f}，5 个轨迹上投影后都略低（差 {dmin:.5f}--{dmax:.5f}）。')
    rep = {'TAB_MAIN': ' \\\\\n'.join(tab_main()) + ' \\\\', 'TXT_K': txt_k, 'TXT_D': txt_d, 'TAB_DIV': ' \\\\\n'.join(tab_div()) + ' \\\\', 'TXT_S': txt_s, 'TAB_SUM': tab_sum, 'TAB_RA': tab_ra, 'TXT_RA': txt_ra, 'VALB': valb, 'TAB_TIME': ' \\\\\n'.join(f'{K} & {TDP[K]:.0f} ({TDP[K] / 600:.2f}) & {TEN[K]:.0f} ({TEN[K] / 600:.2f})' for K in KS) + ' \\\\', 'POOL': f'{POOL:.3f}', 'NOISE': f'{100 * .1 / POOL:.0f}',
           'FREE': f'{free:.3f}', 'FRELP': f'{100 * free / POOL:.0f}', 'P10': f'{P[10]:.0f}', 'P30': f'{P[30]:.0f}', 'BG10': f'{BG[10]:g}', 'BG30': f'{BG[30]:g}', 'I10': INFL[10], 'I30': INFL[30],
           'W10': f'{(BG[10] / .05) ** 2:.0f}', 'W30': f'{(BG[30] / .05) ** 2:.0f}', 'TRUEDIV': sci(tv)}
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
    for K in KS: print('K', K, {m: round(v, 4) for m, v in SL[K][1].items()}, 'paired', round(SL[K][2], 4), round(SL[K][3], 4), SL[K][4], 'pct', round(P[K], 1), 'runs', len(RUN[K]))


if __name__ == '__main__': main()
