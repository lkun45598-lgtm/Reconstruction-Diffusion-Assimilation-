"""Portrait A4 briefing (LaTeX, xelatex) for the RB experiments -> docs/RB实验汇报.tex / .pdf.
Report figures are redrawn here at a fixed 7.2 in width (docs/report_fig/*.pdf) so that fonts stay legible on A4; every
number is read from the result CSVs; projected-divergence numbers from outputs/study_rb_g/unified/summary.txt.
Usage: python analysis/make_report_tex.py      (needs xelatex)
"""
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

R = Path('outputs'); G = R / 'study_rb_g'; DOC = Path('docs'); FIG = DOC / 'report_fig'
DT = 0.35
BLUE, ORG, GRN, GRY = '#0072B2', '#D55E00', '#009E73', '#6B7785'
plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25, 'axes.linewidth': .7, 'lines.linewidth': 1.2})
f4 = lambda x: f'{x:.4f}'
COLS = ['missing_rmse_ch0', 'missing_rmse_ch1', 'missing_rmse_ch2', 'missing_rmse']


def load():
    s = []
    for c in range(3):
        u = pd.read_csv(G / f'unified/c{c}/enkf.csv'); f = pd.read_csv(G / f'fusion_c{c}/fusion.csv')
        s += [u[u.scheme == 'dps'].assign(case=c, arm='A'), f[f.scheme.str.contains('only')].assign(case=c, arm='B'), u[u.scheme == 'enkf_infl_loc'].assign(case=c, arm='C')]
    s = pd.concat(s)
    d = dict(s=s, ch=s.groupby('arm')[COLS].mean(), pc=s.groupby(['arm', 'case']).missing_rmse.mean().unstack(),
             div=s.groupby('arm').an_div_rel.median(), tdiv=s.truth_div_rel.median())
    d['e3'] = pd.read_csv(G / 'enkf3/enkf_all.csv').groupby('scheme').missing_rmse.mean()
    d['noise'] = pd.read_csv(G / 'noise/noise.csv')
    d['long'] = pd.concat([pd.read_csv(f) for f in sorted((R / 'study_rb_long').glob('c*/enkf.csv'))])
    txt = (G / 'unified/summary.txt').read_text()
    d['proj'] = {k: tuple(float(x) for x in (a, b, c, e)) for k, a, b, c, e in re.findall(r'^\s+([ABC]) ([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', txt, re.M)}
    return d


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight'); plt.close(fig)


def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 1.55)); ax.axis('off'); ax.set_xlim(-.1, 15.9); ax.set_ylim(0, 3.2)
    def box(x, w, t, fc, ec, tc='#1F3A5F'):
        ax.add_patch(FancyBboxPatch((x, 0.9), w, 1.3, boxstyle='round,pad=0.04,rounding_size=.15', fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, 1.55, t, ha='center', va='center', fontsize=7, color=tc, linespacing=1.35)
    arrow = lambda x0, x1, y=1.55: ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=9, lw=1.1, color='#1F3A5F'))
    box(.1, 2.5, 'Analysis\n$x^a_{t-1}$', '#E8F0F8', '#1F3A5F')
    box(3.4, 3.6, 'Dedalus forecast\n(Ra = 5e4, 35 steps)', '#FFF3E0', ORG, '#7A3A00')
    box(7.8, 2.4, 'Background\n$x^b_t$', '#E8F0F8', '#1F3A5F')
    box(11.0, 2.6, 'Diffusion prior\n+ DPS guidance', '#E3F4EC', GRN, '#005C42')
    box(14.3, 1.4, 'Analysis\n$x^a_t$', '#E8F0F8', '#1F3A5F')
    arrow(2.7, 3.3); arrow(7.1, 7.7); arrow(10.3, 10.9); arrow(13.7, 14.2)
    ax.add_patch(FancyBboxPatch((7.6, 2.55), 8.1, .6, boxstyle='round,pad=0.03,rounding_size=.1', fc='#F4F6F8', ec=GRY, lw=.9))
    ax.text(11.65, 2.85, 'Observations $y_t$: 1% of grid points, 3 channels, noise-free', ha='center', va='center', fontsize=6.2, color='#333')
    ax.add_patch(FancyArrowPatch((12.3, 2.5), (12.3, 2.25), arrowstyle='-|>', mutation_scale=8, lw=1, color='#1F3A5F'))
    ax.text(7.7, .35, 'Background and observations both enter DPS as likelihood guidance (sampling starts from pure noise); repeated at every frame.', ha='center', fontsize=6.6, color='#555')
    ax.grid(False); save(fig, 'workflow')


def fig_field(frame=59, case=1):
    from rdda.rb_noise import rb_nodes
    srcs = [('A', 'background + observations', BLUE, G / f'unified/c{case}/dps'), ('B', 'observations only', ORG, G / f'fusion_c{case}/obs_only'), ('C', 'EnKF', GRN, G / f'unified/c{case}/enkf_infl_loc')]
    z = rb_nodes(24); xe = np.linspace(0, 3, 73); ze = np.concatenate([[0], (z[1:] + z[:-1]) / 2, [1]]); xc = np.linspace(0, 3, 72, endpoint=False) + 1.5 / 72
    fr = [np.load(s[3] / f'frame_{frame:04d}.npz') for s in srcs]; tr = fr[0]['truth'][..., 0]; mask = fr[0]['mask'][..., 0]; obs = fr[0]['observations'][..., 0]
    errs = [np.abs(f['analysis'][..., 0] - tr) for f in fr]; emax = max(np.percentile(e, 99.5) for e in errs)
    fig = plt.figure(figsize=(7.2, 4.5)); gs = fig.add_gridspec(4, 4, width_ratios=[1, .025, 1, .025], wspace=.22, hspace=.62)
    def panel(r, c, title='', tc='k'):
        ax = fig.add_subplot(gs[r, c]); ax.set_aspect('equal'); ax.set_xlim(0, 3); ax.set_ylim(0, 1); ax.grid(False)
        ax.set_xticks([0, 1, 2, 3] if r == 3 else []); ax.set_yticks([0, 1] if c == 0 else []); ax.set_title(title, fontsize=7, color=tc, loc='left')
        if r == 3: ax.set_xlabel('$x$', fontsize=7)
        if c == 0: ax.set_ylabel('$z$', fontsize=7)
        return ax
    ax = panel(0, 0, 'Truth'); im = ax.pcolormesh(xe, ze, tr.T, cmap='RdBu_r', vmin=0, vmax=1)
    ax = panel(0, 2, 'Observations (17 of 1728 grid points, 1%)'); ax.set_facecolor('#eeeeee'); ii, jj = np.where(mask)
    ax.scatter(xc[ii], z[jj], c=obs[ii, jj], cmap='RdBu_r', vmin=0, vmax=1, s=16, edgecolor='k', linewidth=.4)
    for i, (k, lab, col, _) in enumerate(srcs):
        rm = np.sqrt(np.mean(errs[i][~mask] ** 2))
        ax = panel(1 + i, 0, f'{k}: {lab}  (RMSE {rm:.4f})', col); ax.pcolormesh(xe, ze, fr[i]['analysis'][..., 0].T, cmap='RdBu_r', vmin=0, vmax=1)
        ax = panel(1 + i, 2, f'Absolute error of {k}', col); em = ax.pcolormesh(xe, ze, errs[i].T, cmap='magma', vmin=0, vmax=emax)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0:4, 1])); cb.set_label('Buoyancy $b$', fontsize=7); cb.ax.tick_params(labelsize=6.5)
    cb = fig.colorbar(em, cax=fig.add_subplot(gs[1:4, 3])); cb.set_label('Absolute error of $b$', fontsize=7); cb.ax.tick_params(labelsize=6.5)
    save(fig, 'field')


def fig_unified(d):
    s = d['s']; names = {'A': ('A: background + observations', BLUE), 'B': ('B: observations only', ORG), 'C': ('C: EnKF', GRN)}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw={'width_ratios': [1, 1.3, 1.15]}); w = .27
    for i, (k, (lab, c)) in enumerate(names.items()):
        g = s[s.arm == k]
        ax[0].bar(np.arange(4) + i * w, [g[x].mean() for x in COLS], w, color=c, label=lab)
        m = g.groupby('frame').missing_rmse.mean(); ax[1].plot(m.index * DT, m, color=c)
        e = g.groupby('frame').an_div_rel.mean(); ax[2].plot(e.index * DT, e, color=c)
    ax[2].axhline(d['tdiv'], color='k', ls='--', lw=.9, label='Truth')
    ax[0].set_xticks(np.arange(4) + w, ['$b$', '$u_x$', '$u_z$', 'pooled']); ax[2].set_yscale('log'); ax[2].set_ylim(1e-5, .3)
    ax[0].set_ylabel('RMSE at unobserved points'); ax[1].set_ylabel('RMSE'); ax[2].set_ylabel(r'Relative divergence $\|\nabla\cdot\mathbf{u}\|/\|\nabla\mathbf{u}\|$')
    for a_, t in zip(ax, ('(a) Mean RMSE per channel', '(b) RMSE over time', '(c) Divergence of the analysis')): a_.set_title(t, loc='left', fontweight='bold')
    for a_ in ax[1:]: a_.set_xlabel('Time $t$')
    h, l = ax[0].get_legend_handles_labels(); h2, l2 = ax[2].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.1)); fig.tight_layout(); save(fig, 'unified')


def fig_fusion(d):
    s = d['s']; fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.1), sharey=True)
    for c in range(3):
        for k, col, lab in (('A', BLUE, 'A: background + observations'), ('B', ORG, 'B: observations only')):
            g = s[(s.arm == k) & (s.case == c)].set_index('frame').missing_rmse; ax[c].plot(g.index * DT, g, color=col, lw=1, label=lab)
        a = s[(s.arm == 'A') & (s.case == c)].missing_rmse.mean(); b = s[(s.arm == 'B') & (s.case == c)].missing_rmse.mean()
        ax[c].set_title(f'Test case {c}: {a:.4f} (A) vs {b:.4f} (B)', fontsize=7.5); ax[c].set_xlabel('Time $t$')
    ax[0].set_ylabel('RMSE at unobserved points'); h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.07)); fig.tight_layout(); save(fig, 'fusion')


def fig_noise(d):
    n = d['noise']; S = {'No assimilation (forecast only)': ('Forecast only', ORG), 'DPS, observations every frame': ('DPS', BLUE)}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={'width_ratios': [1.35, 1]})
    for sch, (lab, c) in S.items():
        for lv, ls in ((0.0, '-'), (0.4, '--')):
            g = n[(n.scheme == sch) & (n.level == lv)].groupby('frame').missing_rmse.mean()
            ax[0].plot(g.index * DT, g, color=c, ls=ls, lw=1.1, label=f'{lab} ({lv * 100:.0f}%)')
        m = n[n.scheme == sch].groupby(['level', 'sample']).missing_rmse.mean().groupby('level').agg(['mean', 'min', 'max'])
        ax[1].errorbar(m.index * 100, m['mean'], [(m['mean'] - m['min']).clip(lower=0), (m['max'] - m['mean']).clip(lower=0)], color=c, marker='o', ms=3.5, capsize=2.5, label=lab)
    ax[0].set_xlabel('Time $t$'); ax[0].set_ylabel('RMSE at unobserved points'); ax[1].set_xlabel('Initial noise (% of reference std)'); ax[1].set_ylabel('Mean RMSE over all frames')
    ax[1].set_ylim(0, None)
    for a_, t in zip(ax, ('(a) RMSE over time (solid: 0%, dashed: 40%)', '(b) Mean RMSE vs. noise level')): a_.set_title(t, loc='left', fontweight='bold', fontsize=7.5)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.08)); fig.tight_layout(); save(fig, 'noise')


def fig_long(d):
    lg = d['long']; fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={'width_ratios': [1.6, 1]})
    for sch, lab, c in (('dps', 'Diffusion prior + DPS', BLUE), ('enkf_infl_loc', 'EnKF (inflation + localization)', GRN)):
        g = lg[lg.scheme == sch]; m = g.groupby('frame').missing_rmse.mean()
        ax[0].plot(m.index * DT, m.rolling(3, min_periods=1).mean(), color=c, lw=1.2, label=lab)
        b = g.assign(w=g.frame // 50 * 50).groupby('w').missing_rmse.mean(); ax[1].plot(b.index * DT, b, 'o-', color=c, ms=3.5)
    ax[0].set_xlabel('Time $t$'); ax[0].set_ylabel('RMSE at unobserved points'); ax[1].set_xlabel('Window start $t$ (17.5 time units per window)'); ax[1].set_ylabel('Mean RMSE in the window'); ax[1].set_ylim(0, None)
    for a_, t in zip(ax, ('(a) 400 cycles (140 time units)', '(b) Window means')): a_.set_title(t, loc='left', fontweight='bold')
    ax[0].legend(frameon=False, loc='upper right'); fig.tight_layout(); save(fig, 'long')


PRE = r'''\documentclass[10.5pt,a4paper]{article}
\usepackage[a4paper,left=2.6cm,right=2.6cm,top=2.6cm,bottom=2.5cm]{geometry}
\usepackage{fontspec}
\usepackage{xeCJK}
\setCJKmainfont{AR PL SungtiL GB}[AutoFakeBold=2.5]
\setCJKsansfont{AR PL SungtiL GB}[AutoFakeBold=2.5]
\usepackage{xcolor,graphicx,booktabs,array,amsmath,titlesec,enumitem,caption,float,indentfirst}
\titleformat{\section}{\bfseries\large}{\thesection}{.7em}{}
\titleformat{\subsection}{\bfseries\normalsize}{\thesubsection}{.6em}{}
\titlespacing*{\section}{0pt}{1.4em}{.6em}\titlespacing*{\subsection}{0pt}{1em}{.4em}
\pagestyle{plain}
\renewcommand{\figurename}{图}\renewcommand{\tablename}{表}
\captionsetup{font=small,labelfont=bf,labelsep=quad,skip=4pt}
\setlist[itemize]{leftmargin=1.6em,itemsep=1pt,topsep=2pt,parsep=0pt}
\setlist[enumerate]{leftmargin=1.8em,itemsep=1pt,topsep=2pt,parsep=0pt}
\setlength{\parindent}{2em}\setlength{\parskip}{0pt}\linespread{1.25}
\renewcommand{\arraystretch}{1.15}
\newcommand{\bv}[1]{\textbf{#1}}
\begin{document}
'''

BODY = r'''
\begin{center}
{\LARGE\bfseries 扩散先验稀疏观测同化\par}\vspace{.5em}
{\large Rayleigh--B\'enard 对流实验汇报\par}
\end{center}
\vspace{.3em}

\begin{center}\textbf{摘\quad 要}\end{center}
\begingroup\small\leftskip=2em\rightskip=2em
在仅有 1\% 稀疏观测、数值模式存在模型误差（Rayleigh 数失配）的条件下，比较三种同化方案：A 背景场与稀疏观测联合同化（扩散先验 + DPS），B 稀疏观测直接重建，C 集合卡尔曼滤波（EnKF）。3 个测试 case、每帧进观测的结果表明，A 的未观测点 RMSE 为 @A@，B 为 @B@，C 为 @C@，A 较 B、C 分别降低 @AB@\% 和 @AC@\%；该优势在 3 个 case 中 2 个显著、1 个持平。原始分析场的速度散度上扩散模型大于 EnKF，但经求解器重启所用的流函数投影后三者无明显差异。初值噪声（至 40\% 参考标准差）使纯预报误差增大约 @NUP@\%，同化后几乎不受影响。长时间（400 个循环）下扩散模型方法稳定在约 0.011，EnKF 约 0.019。
\par\endgroup

\section{实验设置}
\subsection{控制方程、数据与时间尺度}
研究对象为二维 Boussinesq Rayleigh--B\'enard 对流，状态为浮力 $b$ 与速度 $\mathbf u=(u_x,u_z)$：
\begin{equation}
\nabla\!\cdot\!\mathbf u=0,\qquad \partial_t b-\kappa\nabla^2 b=-\mathbf u\!\cdot\!\nabla b,\qquad \partial_t\mathbf u-\nu\nabla^2\mathbf u+\nabla p-b\,\mathbf e_z=-\mathbf u\!\cdot\!\nabla\mathbf u,
\end{equation}
其中 $\kappa=(\mathrm{Ra}\,\mathrm{Pr})^{-1/2}$，$\nu=(\mathrm{Ra}/\mathrm{Pr})^{-1/2}$。$x$ 方向周期（$L_x=3$），$z$ 方向为上下壁面（$L_z=1$），边界条件为 $b(0)=1$、$b(L_z)=0$、壁面无滑移。真实流动取 $\mathrm{Ra}=1\times10^{5}$、$\mathrm{Pr}=1$，用 Dedalus v3 求解（$x$ 方向 Fourier、$z$ 方向 Chebyshev，网格 $72\times24$，时间步长 $\mathrm{d}t=0.01$）。

数据集含 200 条轨迹，弃去前 120 帧瞬态后每条保留 100 帧，按 160/20/20 划分训练、验证、测试集，本文使用测试 case 0--2，验证 case 仅用于调参。1 帧对应 35 个求解器步（0.35 个时间单位），100 帧共 3500 步（35 个时间单位）。

\subsection{模型误差、观测与评价指标}
背景模型同为 Dedalus，但 Rayleigh 数取 $5\times10^{4}$（真值 $1\times10^{5}$），这是唯一的模型误差来源，由求解器自身产生，不人为加噪。观测为真值的 1\% 随机格点（1728 个格点中的 17 个，位置固定），三个通道同时观测，无噪声；主对比中每帧进一次观测。

评价指标为未观测格点上的均方根误差（RMSE，三通道合并）。方程平衡用相对散度 $\|\nabla\!\cdot\!\mathbf u\|/\|\nabla\mathbf u\|$ 衡量，导数用 Fourier--Chebyshev 谱方法计算，真值约为 $@TDIV@$。所有方法的参数均在验证 case 上选定，在测试 case 上比较；DPS 与 EnKF 均关闭硬观测覆盖。

\section{同化方法与对照方案}
先验为无条件扩散模型（VP-cosine 噪声调度，U-Net，基础通道数 16，训练 200 个 epoch），只学习流场快照的分布，不接触观测、背景场和时间。同化逐帧循环（图~\ref{fig:wf}）：上一帧分析场经 Dedalus 预报（先做流函数投影后重启，推进 35 步）得到背景场；背景场与当前帧观测一并作为似然引导项进入扩散后验采样（DPS，100 步，8 个样本取均值），采样从纯噪声开始。第 0 帧没有背景场，只用观测。

\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{report_fig/workflow.pdf}\caption{逐帧循环同化流程（方案 A）。}\label{fig:wf}\end{figure}

\begin{table}[htbp]\centering\small\caption{三个对照方案}
\begin{tabular}{@{}c>{\raggedright\arraybackslash}p{3.6cm}>{\raggedright\arraybackslash}p{3.1cm}>{\raggedright\arraybackslash}p{5.2cm}@{}}\toprule
方案 & 方法 & 背景场 & 主要参数（验证 case 上选定）\\ \midrule
A & 背景场 + 稀疏观测联合同化 & Dedalus 预报 & DPS：$b_\mathrm{std}=0.3$，rate $=2$，$o_\mathrm{std}=0.05$ \\
B & 稀疏观测直接重建 & 无 & DPS：$o_\mathrm{std}=0.025$，rate $=2$；每帧独立采样 \\
C & 随机 EnKF & 成员由 Dedalus 预报 & 20 成员；膨胀 2.5；Gaspari--Cohn 局地化 $1.5(x)\times1.2(z)$；$R=0.01$ \\ \bottomrule\end{tabular}\end{table}

EnKF 的更新式为 $\mathbf x^a-\mathbf x^f=\mathbf P^f\mathbf H^{\!\top}(\mathbf H\mathbf P^f\mathbf H^{\!\top}+\mathbf R)^{-1}[\mathbf y-H(\mathbf x^f)]$，采用扰动观测、逐观测点顺序同化，分析场取集合均值。其初始集合以 DPS 第 0 帧的分析场为中心，因此比较的是同一初始分析之后的循环；散度在预报重启之前测量。

\section{结果}
\subsection{重建误差}
表~\ref{tab:res} 给出每帧进观测时三个方案的未观测点 RMSE。A 的合并误差最低，为 @A@，较 B 降低 @AB@\%，较 C 降低 @AC@\%；分通道看，A 在 $b$、$u_x$、$u_z$ 上均最低。分 case 看，A 在 case 1、2 上明显更低，在 case 0 上与 B 基本持平（@A0@ 对 @B0@）。EnKF 的误差集中在 $b$ 通道（@CB@）。每 3 帧进观测时，A 为 @E3A@，EnKF（膨胀加局地化）为 @E3C@，仅膨胀为 @E3I@，无膨胀无局地化为 @E3P@（几乎不吸收观测，误差随时间上涨）。

\begin{table}[htbp]\centering\small\caption{未观测点 RMSE（每帧进观测，3 个测试 case $\times$ 100 帧；粗体为最小值）}\label{tab:res}
\begin{tabular}{@{}lccccccc@{}}\toprule
方案 & $b$ & $u_x$ & $u_z$ & 合并 & case 0 & case 1 & case 2 \\ \midrule
@TAB_RES@ \\ \bottomrule\end{tabular}\end{table}

\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{report_fig/unified.pdf}\caption{三个方案的误差与散度（每帧进观测，3 个测试 case）。}\end{figure}

\subsection{背景场的作用}
图~\ref{fig:fusion} 比较有背景场（A）与无背景场（B）的逐帧误差。背景场平均有益，但并不稳定：case 1、2 显著受益，case 0 持平；循环初期（约前 6 个时间单位），上一步分析尚不准，预报出的背景场反而拖累结果，A 不如 B。此外 B 每帧独立采样、没有时间连续性，A 的优势有一部分来自求解器带来的时间连续性，而非背景场精度本身。

\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{report_fig/fusion.pdf}\caption{A 与 B 的逐帧 RMSE，标题给出各 case 的平均值。}\label{fig:fusion}\end{figure}

\subsection{流场对比}
图~\ref{fig:field} 给出测试 case 1 第 60 帧的浮力场：A 的绝对误差整体最小，B 的残差更大且分布更广，C 在局部出现较大的误差斑块。

\begin{figure}[htbp]\centering\includegraphics[width=.95\linewidth]{report_fig/field.pdf}\caption{测试 case 1、第 60 帧的浮力 $b$：左列为真值与 A/B/C 的分析场，右列为观测位置与三个方案的绝对误差（色标统一）。}\label{fig:field}\end{figure}

\subsection{方程平衡}
表~\ref{tab:div} 给出分析场的相对散度。原始分析场上，A、B 约 @DA@，C 约 @DC@，真值约 $@TDIV@$，扩散模型的分析场不严格满足不可压缩。但每帧预报前求解器重启会做流函数投影，原始分析场的不平衡不会累积；经同样的投影后，三者的散度均约 0.008，贴近对真值做同样投影的下限（0.0079），差异不明显，且投影几乎不改变速度对真值的误差。这说明原始散度偏大主要来自小尺度毛刺被求导放大（给真值速度加 0.6\% 白噪声即可使散度升至 0.06），该解释在 3 个 case、102 帧上验证。

不带局地化的 EnKF 散度约 $10^{-5}$，低于真值，但这并非同化质量所致：成员均由求解器预报、近似无散度，其线性组合亦然（纯预报约 $9\times10^{-6}$）；局地化对不同位置赋予不同权重，破坏了这一性质，散度升至 $10^{-3}$ 量级。

\begin{table}[htbp]\centering\small\caption{分析场的相对散度与速度误差（投影列为 3 个 case、每 3 帧取样的均值）}\label{tab:div}
\begin{tabular}{@{}lcccc@{}}\toprule
方案 & 原始分析场 & 投影后 & 速度 RMSE（投影前） & 速度 RMSE（投影后） \\ \midrule
@TAB_PROJ@ \\ \midrule
真值经同样投影 & $@TDIV@$ & 0.0079 & --- & --- \\ \bottomrule\end{tabular}\end{table}

\subsection{初值噪声}
在 $t=0$ 的真值上叠加平滑噪声：$b$ 的噪声乘壁面包络（壁面处为零）；速度由流函数噪声求导得到，因而无散度且壁面为零，$u_x$、$u_z$ 共用一个缩放系数。噪声幅度为水平乘参考标准差（$b$ 去掉 $x$ 平均后约 0.097，$u_x$、$u_z$ 约 0.166 和 0.169），水平取 0 至 40\%，每档 3 个样本，在测试 case 0 上比较。结果见表~\ref{tab:noise} 与图~\ref{fig:noise}：40\% 噪声使不同化的纯预报全程平均误差增大 @NUP@\%，每帧同化的 DPS 几乎不变。需要说明的是，到末 10 帧纯预报已因模型误差涨至约 0.052，初值噪声的影响被淹没，因此该实验的区分度偏弱。

\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{report_fig/noise.pdf}\caption{$t=0$ 叠加不同水平噪声的影响（误差线为 3 个噪声样本的范围）。}\label{fig:noise}\end{figure}

\begin{table}[htbp]\centering\small\caption{初值噪声对未观测点 RMSE 的影响（测试 case 0）}\label{tab:noise}
\begin{tabular}{@{}lccccc@{}}\toprule
方案 & 无噪声 & 40\% 噪声 & 变化 & 末 10 帧（无噪声） & 末 10 帧（40\%） \\ \midrule
@TAB_NOISE@ \\ \bottomrule\end{tabular}\end{table}

\subsection{长时间表现}
将 3 条测试轨迹重新积分到 520 帧，取其中 400 个循环（约 140 个时间单位），每 3 帧进观测。DPS 约在 100 帧后收敛至约 0.011，EnKF 全程约 0.019（全程平均 @LA@ 对 @LE@），3 个 case 均为 DPS 更低（图~\ref{fig:long}）。真值在该区间已是统计稳态（动能约 0.0276，$b$ 标准差约 0.257，帧间变化约 0.0127），但流场仍在演化。误差走平是否源于模型误差注入与观测修正的平衡，尚未验证（可改变 Rayleigh 数失配量，观察平线高度是否随之变化）。

\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{report_fig/long.pdf}\caption{400 个循环的 RMSE（a）与每 17.5 个时间单位窗口内的平均 RMSE（b）。}\label{fig:long}\end{figure}

\section{结论}
\begin{enumerate}
\item 背景场与稀疏观测联合同化（A）的未观测点 RMSE 最低（@A@），较稀疏观测直接重建降低 @AB@\%，较 EnKF 降低 @AC@\%；对 B 的优势在 3 个 case 中 2 个显著、1 个持平，且循环初期背景场不利。
\item 原始分析场的散度上扩散模型明显大于 EnKF；经求解器重启所用的流函数投影后三者无明显差异。
\item 初值噪声（至 40\%）使纯预报误差增大约 @NUP@\%，同化后几乎不受影响。
\item 长时间下 A 稳定在约 0.011，EnKF 约 0.019。
\end{enumerate}

\section{局限}
结果来自 3 个测试 case、单个随机种子，初值噪声实验仅用测试 case 0。实验为孪生设定：观测无噪声、三通道同时观测、模型误差仅为 Rayleigh 数失配一种，较真实场景容易。EnKF 只用了 20 个成员（未测 40 及以上），更新式按通用形式实现，待与任务给定公式核对。散度偏大的解释仅在 102 帧样本上验证。后续工作包括多种子与更多 case、EnKF 增加成员、改变失配量与观测比例、观测加噪。

\end{document}
'''


def fill(d):
    ch, pc, div, e3, proj = d['ch'], d['pc'], d['div'], d['e3'], d['proj']
    n = d['noise']; nz = n[n.level.isin([0.0, 0.4])]
    mm = nz.groupby(['scheme', 'level']).missing_rmse.mean(); lt = nz[nz.frame >= 90].groupby(['scheme', 'level']).missing_rmse.mean()
    NP, NA, N3 = 'No assimilation (forecast only)', 'DPS, observations every frame', 'DPS, observations every 3 frames'
    A, B, C = (ch.loc[k, 'missing_rmse'] for k in 'ABC')
    lg = d['long']; la = lg[lg.scheme == 'dps'].missing_rmse.mean(); le = lg[lg.scheme == 'enkf_infl_loc'].missing_rmse.mean()
    up = lambda a, b: f'{100 * (a / b - 1):+.0f}\\%'
    rows = [[ch.loc[k, c] for c in COLS] + [pc.loc[k, c] for c in range(3)] for k in 'ABC']
    mn = [min(r[j] for r in rows) for j in range(7)]
    cell = lambda v, j: f'\\bv{{{f4(v)}}}' if v == mn[j] else f4(v)
    tab_res = ' \\\\\n'.join(f'{k}\\ \\ {nm} & ' + ' & '.join(cell(rows[i][j], j) for j in range(7)) for i, (k, nm) in enumerate((('A', '背景场 + 观测'), ('B', '仅观测'), ('C', 'EnKF'))))
    tab_proj = ' \\\\\n'.join(f'{k}\\ \\ {nm} & {div[k]:.4f} & {proj[k][1]:.4f} & {proj[k][2]:.5f} & {proj[k][3]:.5f}' for k, nm in (('A', '背景场 + 观测'), ('B', '仅观测'), ('C', 'EnKF')))
    tab_noise = ' \\\\\n'.join(f'{nm} & {f4(mm[k, 0.0])} & {f4(mm[k, 0.4])} & {up(mm[k, 0.4], mm[k, 0.0])} & {f4(lt[k, 0.0])} & {f4(lt[k, 0.4])}'
                               for k, nm in ((NP, '纯预报'), (NA, 'DPS 每帧观测'), (N3, 'DPS 每 3 帧观测')))
    return {'A': f4(A), 'B': f4(B), 'C': f4(C), 'AB': f'{100 * (1 - A / B):.0f}', 'AC': f'{100 * (1 - A / C):.0f}', 'BUP': up(B, A), 'CUP': up(C, A),
            'DA': f'{div["A"]:.2f}', 'DC': f'{div["C"]:.3f}', 'TDIV': f'{d["tdiv"] / 10 ** np.floor(np.log10(d["tdiv"])):.1f}\\times10^{{{int(np.floor(np.log10(d["tdiv"])))}}}', 'NUP': f'{100 * (mm[NP, 0.4] / mm[NP, 0.0] - 1):.0f}',
            'TAB_RES': tab_res, 'TAB_PROJ': tab_proj, 'TAB_NOISE': tab_noise, 'A0': f4(pc.loc['A', 0]), 'B0': f4(pc.loc['B', 0]), 'CB': f4(ch.loc['C', COLS[0]]),
            'E3A': f4(e3['dps']), 'E3C': f4(e3['enkf_infl_loc']), 'E3I': f4(e3['enkf_inflation']), 'E3P': f4(e3['enkf_plain']), 'LA': f4(la), 'LE': f4(le)}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in FIG.glob('*'): f.unlink()
    d = load()
    fig_workflow(); fig_unified(d); fig_fusion(d); fig_noise(d); fig_long(d); fig_field()
    body = BODY
    for k, v in fill(d).items(): body = body.replace(f'@{k}@', v)
    left = re.findall(r'@[A-Z0-9_]+@', body)
    if left: raise SystemExit(f'unfilled placeholders: {left}')
    tex = DOC / 'RB实验汇报.tex'; tex.write_text(PRE + body, encoding='utf-8')
    for _ in range(2):
        r = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', tex.name], cwd=DOC, capture_output=True, text=True)
        if r.returncode: print(r.stdout[-2500:]); raise SystemExit(1)
    print('saved', tex.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
