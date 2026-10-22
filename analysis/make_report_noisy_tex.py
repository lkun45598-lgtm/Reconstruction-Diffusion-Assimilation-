"""Portrait A4 report (LaTeX, xelatex) for the noisy-observation RB study -> docs/RB实验汇报_带噪声观测.tex / .pdf.
Same layout as make_report_tex.py (which keeps producing the noise-free report). Reads outputs/study_rb_noisy/{main,spinup,noise,
sweep_*}; figures go to docs/report_fig_noisy/. Usage: python analysis/make_report_noisy_tex.py   (needs xelatex)
"""
import re
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import make_report_tex as M
from make_report_tex import BLUE, ORG, GRN, GRY, DT, COLS, f4, PRE

R = Path('outputs/study_rb_noisy'); DOC = Path('docs'); FIG = DOC / 'report_fig_noisy'; M.FIG = FIG
MAP = {'DPS with background (solver forecast + observations)': 'A', 'DPS with observations only (no background)': 'B', 'enkf_infl_loc': 'C'}


def load():
    s = []
    for c in range(3):
        for f in (f'main/c{c}/fusion/fusion.csv', f'main/c{c}/enkf/enkf.csv'):
            x = pd.read_csv(R / f); x['arm'] = x.scheme.map(MAP); s.append(x[x.arm.notna()].assign(case=c))
    s = pd.concat(s)
    d = dict(s=s, ch=s.groupby('arm')[COLS].mean(), pc=s.groupby(['arm', 'case']).missing_rmse.mean().unstack(), div=s.groupby('arm').an_div_rel.median(), tdiv=s.truth_div_rel.median())
    sp0 = pd.concat(pd.read_csv(f) for f in sorted(R.glob('spinup/*/spinup.csv'))).query("init != 'any' or scheme == 'B'").drop_duplicates(['scheme', 'init', 'frame']).assign(case=0)
    sp = pd.concat([sp0] + [pd.read_csv(R / f'spinup_c{c}/spinup.csv').assign(case=c) for c in (1, 2)]); d['sp'] = sp
    d['nz'] = pd.concat(pd.read_csv(f).assign(case=int(f.parent.name[1]), sid=int(f.parent.name[-1])) for f in sorted(R.glob('noise2/*/noise.csv')))
    lg = [pd.read_csv(R / f'long/c{c}/enkf.csv').assign(case=c) for c in range(3)]; d['long'] = pd.concat(lg)
    d['ra'] = {5e4: lg[0]}; d['ra'].update({v: pd.read_csv(R / f'long_ra/ra{k}/enkf.csv') for k, v in (('1e5', 1e5), ('8e4', 8e4), ('2e4', 2e4))})
    txt = (R / 'summary.txt').read_text()
    d['proj'] = {k: tuple(float(x) for x in v) for k, *v in re.findall(r'^\s+([ABC]) ([\d.e-]+)\s+([\d.e-]+)\s+([\d.]+)\s+([\d.]+)', txt, re.M)}
    d['infl'] = pd.read_csv(R / 'sweep_enkf/enkf_sweep.csv').set_index('inflation').mean_rmse
    return d


def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 1.55)); ax.axis('off'); ax.set_xlim(-.1, 15.9); ax.set_ylim(0, 3.2)
    def box(x, w, t, fc, ec, tc='#1F3A5F'):
        ax.add_patch(FancyBboxPatch((x, 0.9), w, 1.3, boxstyle='round,pad=0.04,rounding_size=.15', fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, 1.55, t, ha='center', va='center', fontsize=7, color=tc, linespacing=1.35)
    arrow = lambda x0, x1, y=1.55: ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=9, lw=1.1, color='#1F3A5F'))
    box(.1, 2.5, 'Analysis\n$x^a_{t-1}$', '#E8F0F8', '#1F3A5F'); box(3.4, 3.6, 'Dedalus forecast\n(Ra = 5e4, 35 steps)', '#FFF3E0', ORG, '#7A3A00')
    box(7.8, 2.4, 'Background\n$x^b_t$', '#E8F0F8', '#1F3A5F'); box(11.0, 2.6, 'Diffusion prior\n+ DPS guidance', '#E3F4EC', GRN, '#005C42'); box(14.3, 1.4, 'Analysis\n$x^a_t$', '#E8F0F8', '#1F3A5F')
    arrow(2.7, 3.3); arrow(7.1, 7.7); arrow(10.3, 10.9); arrow(13.7, 14.2)
    ax.add_patch(FancyBboxPatch((7.6, 2.55), 8.1, .6, boxstyle='round,pad=0.03,rounding_size=.1', fc='#F4F6F8', ec=GRY, lw=.9))
    ax.text(11.65, 2.85, r'Observations $y_t$: 1% of grid points, 3 channels, Gaussian error $\sigma_{obs}=0.1$', ha='center', va='center', fontsize=6.2, color='#333')
    ax.add_patch(FancyArrowPatch((12.3, 2.5), (12.3, 2.25), arrowstyle='-|>', mutation_scale=8, lw=1, color='#1F3A5F'))
    ax.text(7.7, .35, 'Background and observations both enter DPS as likelihood guidance (sampling starts from pure noise); repeated at every frame.', ha='center', fontsize=6.6, color='#555')
    ax.grid(False); M.save(fig, 'workflow')


def fig_main(d):
    s = d['s']; names = {'A': ('A: background + observations', BLUE), 'B': ('B: observations only', ORG), 'C': ('C: EnKF', GRN)}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw={'width_ratios': [1, 1.3, 1.15]}); w = .27
    for i, (k, (lab, c)) in enumerate(names.items()):
        g = s[s.arm == k]
        ax[0].bar(np.arange(4) + i * w, [g[x].mean() for x in COLS], w, color=c, label=lab)
        m = g.groupby('frame').missing_rmse.mean(); ax[1].plot(m.index * DT, m, color=c)
        e = g.groupby('frame').an_div_rel.mean(); ax[2].plot(e.index * DT, e, color=c)
    ax[2].axhline(d['tdiv'], color='k', ls='--', lw=.9, label='Truth')
    ax[0].set_xticks(np.arange(4) + w, ['$b$', '$u_x$', '$u_z$', 'pooled']); ax[2].set_yscale('log'); ax[2].set_ylim(1e-5, .6)
    ax[0].set_ylabel('RMSE at unobserved points'); ax[1].set_ylabel('RMSE'); ax[2].set_ylabel(r'Relative divergence $\|\nabla\cdot\mathbf{u}\|/\|\nabla\mathbf{u}\|$')
    for a_, t in zip(ax, ('(a) Mean RMSE per channel', '(b) RMSE over time', '(c) Divergence of the analysis')): a_.set_title(t, loc='left', fontweight='bold')
    for a_ in ax[1:]: a_.set_xlabel('Time $t$')
    h, l = ax[0].get_legend_handles_labels(); h2, l2 = ax[2].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.1)); fig.tight_layout(); M.save(fig, 'main')


def fig_cases(d):
    s = d['s']; fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.1), sharey=True)
    for c in range(3):
        for k, col, lab in (('A', BLUE, 'A: background + observations'), ('B', ORG, 'B: observations only'), ('C', GRN, 'C: EnKF')):
            g = s[(s.arm == k) & (s.case == c)].set_index('frame').missing_rmse; ax[c].plot(g.index * DT, g, color=col, lw=1, label=lab)
        ax[c].set_title(f'Test case {c}', fontsize=7.5); ax[c].set_xlabel('Time $t$')
    ax[0].set_ylabel('RMSE at unobserved points'); h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.07)); fig.tight_layout(); M.save(fig, 'cases')


def fig_spinup(d):
    sp = d['sp']; inits = [('zero', 'Zero field'), ('scale:0.3', 'Truth $\\times$ 0.3'), ('later:60', 'State at frame 60')]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True); b = sp[sp.scheme == 'B'].drop_duplicates('frame')
    cl = {'F': ('F: free forecast', GRY), 'C': ('C: EnKF', GRN), 'A': ('A: background + observations', BLUE)}
    for a_, (i, t) in zip(ax, inits):
        for k, (lab, c) in cl.items():
            g = sp[(sp.scheme == k) & (sp.init == i)]; a_.plot(g.frame * DT, g.missing_rmse, color=c, label=lab)
        a_.plot(b.frame * DT, b.missing_rmse, color=ORG, lw=.9, alpha=.8, label='B: observations only'); a_.axhline(.03, color='k', ls=':', lw=.7)
        a_.set_yscale('log'); a_.set_title(f'First guess: {t}', loc='left', fontweight='bold', fontsize=7.5); a_.set_xlabel('Time $t$')
        a_.set_yticks([.01, .02, .03, .05, .1, .2, .3]); a_.set_yticklabels(['0.01', '0.02', '0.03', '0.05', '0.1', '0.2', '0.3']); a_.minorticks_off()
    ax[0].set_ylabel('RMSE at unobserved points'); h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.1)); fig.tight_layout(); M.save(fig, 'spinup')


def fig_spinup_mean(d):
    sp = d['sp']; inits = [('zero', 'Zero field'), ('scale:0.3', 'Truth $\\times$ 0.3'), ('later:60', 'State at frame 60')]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True); b = sp[sp.scheme == 'B'].groupby('frame').missing_rmse.mean()
    cl = {'F': ('F: free forecast', GRY), 'C': ('C: EnKF', GRN), 'A': ('A: background + observations', BLUE)}
    for a_, (i, t) in zip(ax, inits):
        for k, (lab, c) in cl.items():
            g = sp[(sp.scheme == k) & (sp.init == i)].groupby('frame').missing_rmse.mean(); a_.plot(g.index * DT, g, color=c, label=lab)
        a_.plot(b.index * DT, b, color=ORG, lw=.9, alpha=.8, label='B: observations only'); a_.axhline(.03, color='k', ls=':', lw=.7)
        a_.set_yscale('log'); a_.set_title(f'First guess: {t}', loc='left', fontweight='bold', fontsize=7.5); a_.set_xlabel('Time $t$')
        a_.set_yticks([.01, .02, .03, .05, .1, .2, .3]); a_.set_yticklabels(['0.01', '0.02', '0.03', '0.05', '0.1', '0.2', '0.3']); a_.minorticks_off()
    ax[0].set_ylabel('RMSE at unobserved points (mean of 3 cases)'); h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.1)); fig.tight_layout(); M.save(fig, 'spinup')


def fig_noise2(d):
    n = d['nz']; S = {'No assimilation (forecast only)': ('Forecast only', ORG), 'DPS, observations every frame': ('DPS (observations every frame)', BLUE)}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={'width_ratios': [1.3, 1]}); lv = sorted(n.level.unique())
    for sch, (lab, c) in S.items():
        for l_, ls in ((0.0, '-'), (2.0, '--')):
            g = n[(n.scheme == sch) & (n.level == l_)].groupby('frame').missing_rmse.mean(); ax[0].plot(g.index * DT, g, color=c, ls=ls, label=f'{lab}, {l_ * 100:.0f}%')
        m = n[n.scheme == sch].groupby(['level', 'case', 'sid']).missing_rmse.mean().groupby('level').agg(['mean', 'min', 'max'])
        ax[1].errorbar(range(len(lv)), m['mean'], [m['mean'] - m['min'], m['max'] - m['mean']], color=c, marker='o', ms=3.5, capsize=2.5)
    ax[1].set_xticks(range(len(lv)), [f'{x * 100:.0f}' for x in lv]); ax[1].set_yscale('log'); ax[1].set_yticks([.02, .05, .1, .2]); ax[1].set_yticklabels(['0.02', '0.05', '0.1', '0.2']); ax[1].minorticks_off()
    ax[0].set_xlabel('Time $t$'); ax[0].set_ylabel('RMSE at unobserved points'); ax[1].set_xlabel('Initial noise (% of reference std)'); ax[1].set_ylabel('Mean RMSE over all frames')
    for a_, t in zip(ax, ('(a) RMSE over time (solid: 0%, dashed: 200%)', '(b) Mean RMSE vs. noise level')): a_.set_title(t, loc='left', fontweight='bold', fontsize=7.5)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.09)); fig.tight_layout(); M.save(fig, 'noise')


def fig_ra(d):
    fig, ax = plt.subplots(figsize=(3.5, 2.3)); ras = sorted(d['ra'], reverse=True)
    for sch, lab, c in (('dps', 'A: DPS + background', BLUE), ('enkf_infl_loc', 'C: EnKF', GRN)):
        v = [d['ra'][r][d['ra'][r].scheme == sch].missing_rmse.mean() for r in ras]; ax.plot(range(len(ras)), v, 'o-', color=c, ms=3.5, label=lab)
    ax.set_xticks(range(len(ras)), [f'{r:.0e}'.replace('e+0', 'e') for r in ras]); ax.set_xlabel('Forecast-model Ra (truth 1e5)'); ax.set_ylabel('Mean RMSE (400 cycles)'); ax.set_yscale('log')
    ax.set_yticks([.01, .02, .03, .05, .1]); ax.set_yticklabels(['0.01', '0.02', '0.03', '0.05', '0.1']); ax.minorticks_off(); ax.legend(frameon=False); fig.tight_layout(); M.save(fig, 'ra')


def fig_field(frame=59, case=1):
    from rdda.rb_noise import rb_nodes
    base = R / f'main/c{case}'
    srcs = [('A', 'background + observations', BLUE, base / 'fusion/bg_obs'), ('B', 'observations only', ORG, base / 'fusion/obs_only'), ('C', 'EnKF', GRN, base / 'enkf/enkf_infl_loc')]
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
    ax = panel(0, 2, 'Noisy observations (17 of 1728 grid points)'); ax.set_facecolor('#eeeeee'); ii, jj = np.where(mask)
    ax.scatter(xc[ii], z[jj], c=obs[ii, jj], cmap='RdBu_r', vmin=0, vmax=1, s=16, edgecolor='k', linewidth=.4)
    for i, (k, lab, col, _) in enumerate(srcs):
        rm = np.sqrt(np.mean(errs[i][~mask] ** 2))
        ax = panel(1 + i, 0, f'{k}: {lab}  (RMSE {rm:.4f})', col); ax.pcolormesh(xe, ze, fr[i]['analysis'][..., 0].T, cmap='RdBu_r', vmin=0, vmax=1)
        ax = panel(1 + i, 2, f'Absolute error of {k}', col); em = ax.pcolormesh(xe, ze, errs[i].T, cmap='magma', vmin=0, vmax=emax)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0:4, 1])); cb.set_label('Buoyancy $b$', fontsize=7); cb.ax.tick_params(labelsize=6.5)
    cb = fig.colorbar(em, cax=fig.add_subplot(gs[1:4, 3])); cb.set_label('Absolute error of $b$', fontsize=7); cb.ax.tick_params(labelsize=6.5)
    M.save(fig, 'field')


BODY = r'''
\begin{center}
{\LARGE\bfseries 扩散先验稀疏观测同化（带噪声观测）\par}\vspace{.5em}
{\large Rayleigh--B\'enard 对流实验汇报\par}
\end{center}
\vspace{.3em}

\begin{center}\textbf{摘\quad 要}\end{center}
\begingroup\small\leftskip=2em\rightskip=2em
在仅有 1\% 稀疏观测、观测带高斯误差（标准差 0.1，$R=0.01$）、数值模式存在 Rayleigh 数失配的条件下，比较三种同化方案：A 背景场与稀疏观测联合同化（扩散先验 + DPS），B 稀疏观测直接重建，C 集合卡尔曼滤波（EnKF，20 成员）。3 个测试 case、每帧进观测的结果表明，A 的未观测点 RMSE 为 @A@，B 为 @B@，C 为 @C@，A 较 B、C 分别降低 @AB@\% 和 @AC@\%，且在 3 个 case 中顺序一致。从全零场、减弱场（真值的 0.3 倍）和另一时刻的状态出发，A 在 @FA@ 帧内回到 0.03 以下，EnKF 需要 @FC@ 帧，纯预报在前两种初值下始终不能回到该水平；400 个循环的长时间对比中 A（@LA@）仍优于 EnKF（@LE@）。改变预报模型的 Rayleigh 数后，A 对模型误差不敏感，EnKF 对其敏感：模型误差中等及以上时 A 更优，模型误差很小（Ra 取 $8\times10^4$ 及以上）时 EnKF 更优。原始分析场的散度上 A 为 EnKF 的数十倍，经求解器重启所用的流函数投影后两者接近。
\par\endgroup

\section{实验设置}
\subsection{控制方程、数据与时间尺度}
研究对象为二维 Boussinesq Rayleigh--B\'enard 对流，状态为浮力 $b$ 与速度 $\mathbf u=(u_x,u_z)$：
\begin{equation}
\nabla\!\cdot\!\mathbf u=0,\qquad \partial_t b-\kappa\nabla^2 b=-\mathbf u\!\cdot\!\nabla b,\qquad \partial_t\mathbf u-\nu\nabla^2\mathbf u+\nabla p-b\,\mathbf e_z=-\mathbf u\!\cdot\!\nabla\mathbf u,
\end{equation}
其中 $\kappa=(\mathrm{Ra}\,\mathrm{Pr})^{-1/2}$，$\nu=(\mathrm{Ra}/\mathrm{Pr})^{-1/2}$。$x$ 方向周期（$L_x=3$），$z$ 方向为上下壁面（$L_z=1$），边界条件为 $b(0)=1$、$b(L_z)=0$、壁面无滑移。真实流动取 $\mathrm{Ra}=1\times10^{5}$、$\mathrm{Pr}=1$，用 Dedalus v3 求解（$x$ 方向 Fourier、$z$ 方向 Chebyshev，网格 $72\times24$，时间步长 $\mathrm{d}t=0.01$）。

数据集含 200 条轨迹，弃去前 120 帧瞬态后每条保留 100 帧，按 160/20/20 划分训练、验证、测试集；本文使用测试 case 0--2，验证 case 仅用于调参。1 帧对应 35 个求解器步（0.35 个时间单位），100 帧共 3500 步（35 个时间单位）。

\subsection{模型误差、观测与评价指标}
背景模型同为 Dedalus，但 Rayleigh 数取 $5\times10^{4}$（真值 $1\times10^{5}$），这是唯一的模型误差来源。观测为真值的 1\% 随机格点（1728 个格点中的 17 个，位置固定），三个通道同时观测；\textbf{观测值叠加独立高斯误差 $\varepsilon\sim\mathcal N(0,\sigma_o^2)$，$\sigma_o=0.1$（即 $R=\mathrm{diag}(0.01)$）}，每帧重新抽取，三种方案使用同一组噪声。主对比中每帧进一次观测。$\sigma_o$ 相对三个通道的标准差（$b$ 约 0.257，$u_x$、$u_z$ 约 0.17）约为 40\%--60\%，噪声不小。

评价指标为未观测格点上的均方根误差（RMSE，三通道合并）。方程平衡用相对散度 $\|\nabla\!\cdot\!\mathbf u\|/\|\nabla\mathbf u\|$ 衡量，导数用 Fourier--Chebyshev 谱方法计算，真值约为 $@TDIV@$。所有方法的参数均在验证 case 上选定，在测试 case 上比较；DPS 与 EnKF 均关闭硬观测覆盖。

\section{同化方法与对照方案}
先验为无条件扩散模型（VP-cosine 噪声调度，U-Net，基础通道数 16，训练 200 个 epoch），只学习流场快照的分布，不接触观测、背景场和时间。同化逐帧循环（图~\ref{fig:wf}）：上一帧分析场经 Dedalus 预报（先做流函数投影后重启，推进 35 步）得到背景场；背景场与当前帧观测一并作为似然引导项进入扩散后验采样（DPS，100 步，8 个样本取均值），采样从纯噪声开始。第 0 帧没有背景场，只用观测。

\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{report_fig_noisy/workflow.pdf}\caption{逐帧循环同化流程（方案 A）。}\label{fig:wf}\end{figure}

\begin{table}[htbp]\centering\small\caption{三个对照方案}
\begin{tabular}{@{}c>{\raggedright\arraybackslash}p{3.6cm}>{\raggedright\arraybackslash}p{3.1cm}>{\raggedright\arraybackslash}p{5.2cm}@{}}\toprule
方案 & 方法 & 背景场 & 主要参数（验证 case 上选定）\\ \midrule
A & 背景场 + 稀疏观测联合同化 & Dedalus 预报 & DPS：$b_\mathrm{std}=0.3$，rate $=2$，$o_\mathrm{std}=0.05$ \\
B & 稀疏观测直接重建 & 无 & DPS：$o_\mathrm{std}=0.025$，rate $=1$；每帧独立采样 \\
C & 随机 EnKF & 成员由 Dedalus 预报 & 20 成员；膨胀 1.1；Gaspari--Cohn 局地化 $1.5(x)\times1.2(z)$；$R=0.01$ \\ \bottomrule\end{tabular}\end{table}

EnKF 的更新式为 $\mathbf x^a-\mathbf x^f=\mathbf P^f\mathbf H^{\!\top}(\mathbf H\mathbf P^f\mathbf H^{\!\top}+\mathbf R)^{-1}[\mathbf y-H(\mathbf x^f)]$，\textbf{每个成员使用各自加了 $\mathcal N(0,R)$ 扰动的观测}（随机 EnKF），逐观测点顺序同化，分析场取集合均值。主对比中其初始集合以 DPS 第 0 帧的分析场为中心；坏初值实验中集合以坏初值为中心、初始离散度取 1 倍参考标准差，并在 $t=0$ 同化。散度在预报重启之前测量。

\paragraph{参数选择。}带噪声观测后在验证 case 上重新选参。EnKF 膨胀系数取 1.0、1.1、1.3、1.5、2.0、2.5 时的 RMSE 依次为 @I10@、@I11@、@I13@、@I15@、@I20@、@I25@，最优为 1.1（原先的 1.3 为 @I13@，较差）。DPS 的 $o_\mathrm{std}$ 取 0.05 最优，比真实噪声标准差 0.1 小；原因未查明，推测与 DPS 的似然权重和采样步数有关，而非 $\sigma_o$ 本身。该选择仅基于一个验证 case。

\section{结果}
\subsection{重建误差}
表~\ref{tab:res} 给出每帧进观测时三个方案的未观测点 RMSE。A 的合并误差最低，为 @A@，较 B 降低 @AB@\%，较 C 降低 @AC@\%；分通道看，A 在 $b$、$u_x$、$u_z$ 上均最低，C 的差距集中在 $b$ 通道（@CB@，A 为 @AB0@）。三个测试 case 上顺序一致（A $<$ C $<$ B，图~\ref{fig:cases}）。加噪声后 B 的误差大幅上升且逐帧波动大，因为它每帧只有 17 个带噪观测，没有时间上的约束。

\begin{table}[htbp]\centering\small\caption{未观测点 RMSE（每帧进观测，$\sigma_o=0.1$，3 个测试 case $\times$ 100 帧；粗体为最小值）}\label{tab:res}
\begin{tabular}{@{}lccccccc@{}}\toprule
方案 & $b$ & $u_x$ & $u_z$ & 合并 & case 0 & case 1 & case 2 \\ \midrule
@TAB_RES@ \\ \bottomrule\end{tabular}\end{table}

\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{report_fig_noisy/main.pdf}\caption{三个方案的误差与散度（每帧进观测，3 个测试 case，观测噪声 $\sigma_o=0.1$）。}\end{figure}

\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{report_fig_noisy/cases.pdf}\caption{三个测试 case 各自的逐帧 RMSE。}\label{fig:cases}\end{figure}

\subsection{流场对比}
图~\ref{fig:field} 给出测试 case 1 第 60 帧的浮力场：A 的绝对误差整体最小，B 的残差更大且分布更广，C 的误差更集中在局部斑块。

\begin{figure}[htbp]\centering\includegraphics[width=.95\linewidth]{report_fig_noisy/field.pdf}\caption{测试 case 1、第 60 帧的浮力 $b$：左列为真值与 A/B/C 的分析场，右列为带噪声的观测位置与三个方案的绝对误差（色标统一）。}\label{fig:field}\end{figure}

\subsection{从较差的初值恢复}
为检验观测能否把偏离很大的初值拉回来，在 3 个测试 case 上令三个方案从三种较差的初值出发：全零场、真值的 0.3 倍（减弱）、同一轨迹第 60 帧的状态（相位不同）；另设纯预报（F，从该初值自由积分，不同化）作参照。观测同前（$\sigma_o=0.1$，每帧）。结果取 3 个 case 的平均（表~\ref{tab:spin}、图~\ref{fig:spin}），三个 case 的趋势一致。

A 在 @FA@ 帧内回到 0.03 以下，之后的稳态误差（后 50 帧均值 @SA@）与初值无关；EnKF 需要 @FC@ 帧才降到 0.03 以下，稳态后为 @SC@，优于 B 但不及 A；纯预报在全零场和减弱场下始终无法回到该水平（全零场下后 50 帧仍约 @SF@），说明恢复来自观测而非求解器本身。全零场不满足无散度约束，其相对散度恒为 1，A 与 EnKF 在同化一步后即消除。EnKF 收敛慢的一部分原因可能是初始离散度和成员数未调，本实验未区分。

\begin{table}[htbp]\centering\small\caption{坏初值恢复（3 个测试 case 的平均；“降至 0.03”为 RMSE 连续 5 帧低于 0.03 的起始帧，给出 3 个 case 的范围，--- 表示未达到）}\label{tab:spin}
\begin{tabular}{@{}llccccc@{}}\toprule
初值 & 方案 & 第 0 帧 & 第 5 帧 & 第 20 帧 & 后 50 帧均值 & 降至 0.03 \\ \midrule
@TAB_SPIN@ \\ \bottomrule\end{tabular}\end{table}

\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{report_fig_noisy/spinup.pdf}\caption{从三种较差初值出发的未观测点 RMSE（3 个 case 平均；虚线：0.03）。B 不依赖初值。}\label{fig:spin}\end{figure}

\subsection{方程平衡}
表~\ref{tab:div} 给出分析场的相对散度。原始分析场上，A 约 @DA@，B 约 @DB@，C 约 @DC@，真值约 $@TDIV@$：扩散模型的分析场不严格满足不可压缩，B 最差。但每帧预报前求解器重启会做流函数投影，原始分析场的不平衡不会累积；经同样的投影后，A 降至 @PA@，C 为 @PC@，差异不大，B 为 @PB@；投影几乎不改变速度对真值的误差。这与此前无噪声观测下的结论一致：原始散度偏大主要来自小尺度毛刺被求导放大。需要说明，投影后 A 与 C 仍有小差距，不能说完全无差异。

EnKF 的原始散度低，是因为成员由求解器预报、近似无散度，其线性组合亦然（纯预报约 $9\times10^{-6}$）；局地化使不同位置的权重不同，破坏了这一性质，散度升至 $10^{-3}$ 量级。因此 EnKF 的散度低不代表它同化得更好。

\begin{table}[htbp]\centering\small\caption{分析场的相对散度与速度误差（投影列为 3 个 case、每 3 帧取样的均值；原始列为中位数）}\label{tab:div}
\begin{tabular}{@{}lcccc@{}}\toprule
方案 & 原始分析场 & 投影后 & 速度 RMSE（投影前） & 速度 RMSE（投影后） \\ \midrule
@TAB_PROJ@ \\ \midrule
真值经同样投影 & $@TDIV@$ & 0.0079 & --- & --- \\ \bottomrule\end{tabular}\end{table}

\subsection{初值噪声}
在 $t=0$ 的真值上叠加平滑噪声：$b$ 的噪声乘壁面包络；速度由流函数噪声求导得到，因而无散度，$u_x$、$u_z$ 共用一个缩放系数。噪声幅度为水平乘参考标准差（$b$ 约 0.097，$u_x$、$u_z$ 约 0.166 和 0.169），水平取 0、5\%、10\%、20\%、40\%、100\%、200\%，3 个测试 case $\times$ 3 个噪声样本，观测带 $\sigma_o=0.1$ 噪声。结果见表~\ref{tab:noise} 与图~\ref{fig:noise}：噪声增大时不同化的纯预报误差明显上升（200\% 时为无噪声的 @NF@ 倍，各次运行范围 @NFR@），每帧同化的 DPS 变化很小（200\% 时 @ND@），第 0 帧之后迅速被观测消掉。40\% 以内的档位对预报本身影响也不大，加到 100\%、200\% 才有区分度。

\begin{table}[htbp]\centering\small\caption{初值噪声对未观测点 RMSE 的影响（全程平均，9 次运行的均值）}\label{tab:noise}
\begin{tabular}{@{}lccccccc@{}}\toprule
方案 & 0\% & 5\% & 10\% & 20\% & 40\% & 100\% & 200\% \\ \midrule
@TAB_NOISE@ \\ \bottomrule\end{tabular}\end{table}

\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{report_fig_noisy/noise.pdf}\caption{$t=0$ 叠加不同水平噪声的影响（误差线为 9 次运行的范围）。}\label{fig:noise}\end{figure}

\subsection{长时间表现}
将 3 条测试轨迹重新积分到 520 帧，取其中 400 个循环（约 140 个时间单位），每 3 帧进观测，观测带 $\sigma_o=0.1$ 噪声，预报模型 $\mathrm{Ra}=5\times10^{4}$。结果见表~\ref{tab:long} 与图~\ref{fig:long}：A 全程平均 @LA@，EnKF @LE@，3 个 case 均为 A 更低。A 约在 100 帧后走平在约 0.016，EnKF 走平在约 0.026。

\begin{table}[htbp]\centering\small\caption{400 个循环的未观测点 RMSE}\label{tab:long}
\begin{tabular}{@{}lcccccc@{}}\toprule
方案 & case 0 & case 1 & case 2 & 前 100 帧 & 后 150 帧 \\ \midrule
@TAB_LONG@ \\ \bottomrule\end{tabular}\end{table}

\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{report_fig_noisy/long.pdf}\caption{400 个循环的 RMSE（a，3 个 case 平均）与每 17.5 个时间单位窗口内的平均 RMSE（b）。}\label{fig:long}\end{figure}

\subsection{对模型误差的敏感性}
为检验结论是否依赖模型误差的大小，在测试 case 0 上把预报模型的 Rayleigh 数取为 $10^{5}$（无失配）、$8\times10^{4}$、$5\times10^{4}$（主设置）、$2\times10^{4}$，其余设置不变，各跑 400 个循环（表~\ref{tab:ra}、图~\ref{fig:ra}）。A 对失配不敏感：$\mathrm{Ra}$ 从 $10^{5}$ 到 $5\times10^{4}$，A 的误差几乎不变，到 $2\times10^{4}$ 才明显升高；EnKF 随失配增大迅速变差。因此两者的优劣随模型误差而变：失配中等及以上（$5\times10^{4}$、$2\times10^{4}$）时 A 更优；失配很小（$8\times10^{4}$）或没有时 EnKF 更优。这也说明 A 的误差平台在小失配范围内不由模型误差决定，其成因（例如观测噪声）本文未进一步验证。EnKF 的膨胀与局地化在 $5\times10^{4}$ 下选定，在小失配下的最优值可能不同，故 EnKF 在 $10^{5}$、$8\times10^{4}$ 的数字不一定是其上限。

\begin{table}[htbp]\centering\small\caption{预报模型 Rayleigh 数的影响（测试 case 0，400 个循环）}\label{tab:ra}
\begin{tabular}{@{}lcccc@{}}\toprule
预报 Ra & A 全程 & C 全程 & A 后 150 帧 & C 后 150 帧 \\ \midrule
@TAB_RA@ \\ \bottomrule\end{tabular}\end{table}

\begin{figure}[htbp]\centering\includegraphics[width=.5\linewidth]{report_fig_noisy/ra.pdf}\caption{未观测点 RMSE 随预报模型 Rayleigh 数的变化。}\label{fig:ra}\end{figure}

\section{结论}
\begin{enumerate}
\item 观测带 $\sigma_o=0.1$ 误差时，背景场与稀疏观测联合同化（A）的未观测点 RMSE 最低（@A@），较稀疏观测直接重建降低 @AB@\%，较 EnKF 降低 @AC@\%，3 个测试 case 顺序一致。
\item 从全零场、减弱场或另一时刻的状态出发，A 在 @FA@ 帧内恢复，EnKF 需要 @FC@ 帧；纯预报在全零场和减弱场下不能恢复（第 60 帧状态下约 40 帧后进入 0.03 以下）；A 的稳态误差与初值无关。
\item 原始分析场的散度上 A 明显大于 EnKF（@DA@ 对 @DC@）；经求解器重启所用的流函数投影后差距缩小（@PA@ 对 @PC@），速度误差几乎不变。
\item $t=0$ 噪声（至 200\% 参考标准差）使纯预报误差增大至 @NF@ 倍，对同化后的 DPS 影响很小（@ND@）。
\item 400 个循环的长时间对比中 A（@LA@）仍优于 EnKF（@LE@）。A 对预报模型的 Rayleigh 数失配不敏感，EnKF 敏感：失配中等及以上时 A 更优，失配很小时 EnKF 更优，A 的优势因此是对模型误差的稳健性，而不是在任何条件下都更准。
\end{enumerate}

\section{局限}
结果来自 3 个测试 case、单个观测噪声实现；Rayleigh 数敏感性仅用测试 case 0。观测比例（1\%）、观测噪声水平（0.1）和 EnKF 成员数（20）均未改变，观测噪声水平的扫描未做。EnKF 只用 20 个成员，膨胀与局地化仅在 $\mathrm{Ra}=5\times10^{4}$ 的一个验证 case 上选定，其收敛慢与 $b$ 通道误差偏大可能部分源于参数而非方法本身；DPS 的 $o_\mathrm{std}$ 小于真实噪声的原因未查明。A 的误差平台成因未查明。EnKF 更新式按通用形式实现，待与任务给定公式核对。

\end{document}
'''


def fill(d):
    ch, pc, div, proj, infl, sp = d['ch'], d['pc'], d['div'], d['proj'], d['infl'], d['sp']
    A, B, C = (ch.loc[k, 'missing_rmse'] for k in 'ABC')
    SPN = {'zero': '全零场', 'scale:0.3': '真值 $\\times$ 0.3', 'later:60': '第 60 帧状态'}; rows = []; first = {}; last = {}
    def ser(k, i, c): return sp[(sp.scheme == k) & (sp.init == i) & (sp.case == c)].sort_values('frame').missing_rmse.values
    for i, nm in SPN.items():
        for k in 'AFC':
            g = np.mean([ser(k, i, c) for c in range(3)], 0)
            ths = [next((j for j in range(100) if (ser(k, i, c)[j:j + 5] < .03).all()), None) for c in range(3)]; first[i, k] = ths; last[i, k] = g[50:].mean()
            rng_ = '---' if all(t is None for t in ths) else ('/'.join('---' if t is None else str(t) for t in ths) if None in ths else f'{min(ths)}--{max(ths)}')
            rows.append(f'{nm if k == "A" else ""} & {k} & {f4(g[0])} & {f4(g[5])} & {f4(g[20])} & {f4(g[50:].mean())} & {rng_} \\\\' + (' \\addlinespace' if k == 'C' and i != 'later:60' else ''))
    b = np.mean([sp[(sp.scheme == 'B') & (sp.case == c)].sort_values('frame').missing_rmse.values for c in range(3)], 0)
    rows.append(f'\\midrule 任意 & B & {f4(b[0])} & {f4(b[5])} & {f4(b[20])} & {f4(b[50:].mean())} & --- \\\\')
    cells = [[ch.loc[k, c] for c in COLS] + [pc.loc[k, c] for c in range(3)] for k in 'ABC']; mn = [min(r[j] for r in cells) for j in range(7)]
    cell = lambda v, j: f'\\bv{{{f4(v)}}}' if v == mn[j] else f4(v)
    nm3 = (('A', '背景场 + 观测'), ('B', '仅观测'), ('C', 'EnKF'))
    tab_res = ' \\\\\n'.join(f'{k}\\ \\ {nm} & ' + ' & '.join(cell(cells[i][j], j) for j in range(7)) for i, (k, nm) in enumerate(nm3))
    tab_proj = ' \\\\\n'.join(f'{k}\\ \\ {nm} & {div[k]:.4f} & {proj[k][1]:.4f} & {proj[k][2]:.5f} & {proj[k][3]:.5f}' for k, nm in nm3)
    n = d['nz']; NP, ND = 'No assimilation (forecast only)', 'DPS, observations every frame'; lv = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0]
    mm = n.groupby(['scheme', 'level', 'case', 'sid']).missing_rmse.mean()
    mean = lambda k, l: mm[k, l].mean()
    tab_noise = ' \\\\\n'.join(f'{nm} & ' + ' & '.join(f4(mean(k, l)) for l in lv) + '' for k, nm in ((NP, '纯预报'), (ND, 'DPS 每帧观测')))
    ratio = mm[NP, 2.0] / mm[NP, 0.0].values.mean()
    lg = d['long']
    def lrm(sch, c=None, lo=0, hi=400):
        g = lg[(lg.scheme == sch)] if c is None else lg[(lg.scheme == sch) & (lg.case == c)]
        return g[(g.frame >= lo) & (g.frame < hi)].missing_rmse.mean()
    LA, LE = lrm('dps'), lrm('enkf_infl_loc')
    tab_long = ' \\\\\n'.join(f'{nm} & ' + ' & '.join(f4(lrm(k, c)) for c in range(3)) + f' & {f4(lrm(k, None, 0, 100))} & {f4(lrm(k, None, 250, 400))}' for k, nm in (('dps', 'A\\ \\ 背景场 + 观测'), ('enkf_infl_loc', 'C\\ \\ EnKF')))
    def rar(r, sch, lo=0):
        g = d['ra'][r]; g = g[g.scheme == sch]; return g[g.frame >= lo].missing_rmse.mean()
    tab_ra = ' \\\\\n'.join(f'${r / 10 ** int(np.floor(np.log10(r))):.0f}\\times10^{{{int(np.floor(np.log10(r)))}}}$ & {f4(rar(r, "dps"))} & {f4(rar(r, "enkf_infl_loc"))} & {f4(rar(r, "dps", 250))} & {f4(rar(r, "enkf_infl_loc", 250))}' for r in (1e5, 8e4, 5e4, 2e4))
    tdiv = d['tdiv']; e = int(np.floor(np.log10(tdiv)))
    allA = [t for i in SPN for t in first[i, 'A'] if t is not None]; allC = [t for i in SPN for t in first[i, 'C'] if t is not None]
    return {'A': f4(A), 'B': f4(B), 'C': f4(C), 'AB': f'{100 * (1 - A / B):.0f}', 'AC': f'{100 * (1 - A / C):.0f}', 'CB': f4(ch.loc['C', COLS[0]]), 'AB0': f4(ch.loc['A', COLS[0]]),
            'DA': f'{div["A"]:.2f}', 'DB': f'{div["B"]:.2f}', 'DC': f'{div["C"]:.4f}', 'PA': f'{proj["A"][1]:.4f}', 'PB': f'{proj["B"][1]:.4f}', 'PC': f'{proj["C"][1]:.4f}',
            'TDIV': f'{tdiv / 10 ** e:.1f}\\times10^{{{e}}}', 'NF': f'{ratio.mean():.1f}', 'NFR': f'{ratio.min():.1f}--{ratio.max():.1f}',
            'ND': f'{100 * (mean(ND, 2.0) / mean(ND, 0.0) - 1):+.0f}\\%',
            'FA': f'{min(allA)}--{max(allA)}', 'FC': f'{min(allC)}--{max(allC)}', 'SA': f4(np.mean([last[i, 'A'] for i in SPN])), 'SC': f4(np.mean([last[i, 'C'] for i in SPN])),
            'SF': f4(last['zero', 'F']), 'LA': f4(LA), 'LE': f4(LE),
            'TAB_RES': tab_res, 'TAB_PROJ': tab_proj, 'TAB_NOISE': tab_noise, 'TAB_SPIN': '\n'.join(rows), 'TAB_LONG': tab_long, 'TAB_RA': tab_ra,
            **{f'I{int(round(k * 10))}': f4(infl[k]) for k in (1.0, 1.1, 1.3, 1.5, 2.0, 2.5)}}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in FIG.glob('*'): f.unlink()
    d = load(); fig_workflow(); fig_main(d); fig_cases(d); fig_spinup_mean(d); fig_field(); fig_noise2(d); M.fig_long(d); fig_ra(d)
    body = BODY
    for k, v in fill(d).items(): body = body.replace(f'@{k}@', v)
    left = re.findall(r'@[A-Z0-9_]+@', body)
    if left: raise SystemExit(f'unfilled placeholders: {left}')
    tex = DOC / 'RB实验汇报_带噪声观测.tex'; tex.write_text(PRE + body, encoding='utf-8')
    for _ in range(2):
        r = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', tex.name], cwd=DOC, capture_output=True, text=True)
        if r.returncode: print(r.stdout[-2500:]); raise SystemExit(1)
    for ext in ('aux', 'log', 'out'): tex.with_suffix(f'.{ext}').unlink(missing_ok=True)
    print('saved', tex.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
