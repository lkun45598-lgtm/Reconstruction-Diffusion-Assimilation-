"""Report for the lag-initialised EnKF comparison (analysis/lag_eval.py) -> docs/RB实验汇报_lag初始化.tex / .pdf. All numbers come from
outputs/study_rb_lag (test 20 cases x seeds 42/43/44 x K=1/5/10/20; val for inflation choice) and outputs/study_rb_test* (center+perturbation EnKF, same-window comparison).
Usage: python analysis/make_report_lag.py   (needs xelatex)"""
import sys, glob, subprocess, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'analysis')
import make_report_tex as M
from make_report_tex import BLUE, GRN, GRY, PRE
plt.rcParams.update({'font.family': ['DejaVu Sans', 'Droid Sans Fallback'], 'axes.unicode_minus': False})
DOC = Path('docs'); FIG = DOC / 'report_fig_lag'; M.FIG = FIG
L = Path('outputs/study_rb_lag'); T = Path('outputs/study_rb_test'); TS = Path('outputs/study_rb_test_seeds')
KS, SEEDS = (1, 5, 10, 20), (42, 43, 44); f4 = lambda v: f'{v:.4f}'
rd = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))])
A = {(K, s): rd(f'{L}/testA_s{s}_K{K}_c*.csv') for K in KS for s in SEEDS}
E = {(K, s): rd(f'{L}/testE_s{s}_K{K}_c*.csv') for K in KS for s in SEEDS}
pc = {}
for (K, s), a in A.items():
    g = a.groupby(['case', 'method']).missing_rmse.mean().unstack(); e = E[K, s].groupby('case').missing_rmse.mean()
    g['EnKF'] = e; pc[K, s] = g
for (K, s), g in pc.items():  # free_run is only saved in the K=1 files; it assimilates frame 40 once, so it does not depend on K
    if 'free_run' not in g: g['free_run'] = pc[1, s]['free_run']
INFL = {K: E[K, 42].method.iloc[0].split('_i')[1] for K in KS}


def pct(K, s): g = pc[K, s]; return 100 * (1 - g['A1'].mean() / g['EnKF'].mean())


def fig_main():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={'width_ratios': [1.3, 1, 1]}); w = .26
    for i, (nm, c, key) in enumerate((('A1', BLUE, 'A1'), ('EnKF (lag)', GRN, 'EnKF'), ('纯预报', GRY, 'free_run'))):
        v = np.array([[pc[K, s][key].mean() for s in SEEDS] for K in KS])
        ax[0].bar(np.arange(4) + i * w, v.mean(1), w, color=c, label=nm, yerr=[v.mean(1) - v.min(1), v.max(1) - v.mean(1)], capsize=2, error_kw=dict(lw=.8))
    ax[0].set_xticks(np.arange(4) + w); ax[0].set_xticklabels([f'K={k}' for k in KS]); ax[0].set_ylabel('未观测点 RMSE'); ax[0].legend(fontsize=6.5); ax[0].set_title('3 个种子均值（须为范围）', fontsize=8)
    for a, K in zip(ax[1:], (1, 20)):
        for key, d, c, lb in (('A1', A[K, 42], BLUE, 'A1'), ('EnKF', E[K, 42], GRN, 'EnKF (lag)'), ('free_run', A[1, 42], GRY, '纯预报')):
            m = d[d.method == (key if key != 'EnKF' else d.method.iloc[0])].groupby('frame').missing_rmse.mean(); a.plot(m.index + 40, m, color=c, label=lb)
        a.set_title(f'K={K}：逐帧误差（种子 42）', fontsize=8); a.set_xlabel('帧')
    M.save(fig, 'main')


def fig_cases():
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.0))
    for a, K in zip(ax, KS):
        g = pc[K, 42]; lim = [0.01, g[['EnKF', 'A1']].max().max() * 1.1]
        a.scatter(g['EnKF'], g['A1'], s=12, color=BLUE); a.plot(lim, lim, color=GRY, lw=.8); a.set_xlim(lim); a.set_ylim(lim); a.set_title(f'K={K}', fontsize=8); a.set_xlabel('EnKF RMSE')
    ax[0].set_ylabel('A1 RMSE（对角线下方 A1 更优）'); M.save(fig, 'cases')


def main_rows():
    r = []
    for K in KS:
        j = lambda k: ' / '.join(f4(pc[K, s][k].mean()) for s in SEEDS)
        im = ' / '.join(f'{pct(K, s):+.0f}\\%' for s in SEEDS); nb = ' / '.join(f'{int((pc[K, s]["EnKF"] > pc[K, s]["A1"]).sum())}' for s in SEEDS)
        r.append(f'{K} & {INFL[K]} & {j("A1")} & {j("EnKF")} & {j("free_run")} & {im} & {nb}')
    return r


def val_rows():
    r = []
    for K in KS:
        fs = glob.glob(f'{L}/val_s42_K{K}_c*.csv') + glob.glob(f'{L}/val2/val_s42_K{K}_c*.csv')
        if not fs: continue
        d = pd.concat([pd.read_csv(f) for f in fs]); d = d[d.method.str.startswith('EnKF')]
        m = d.groupby(['method', 'case']).missing_rmse.mean().groupby('method').mean().sort_index()
        r.append(f'{K} & ' + '，'.join(f'{k.split("_i")[1]}: {v:.4f}' for k, v in m.items()))
    return r


def same_window():
    """center+perturbation EnKF (main experiment) restricted to frames 40..99, 3 seeds."""
    r = []
    for K, b in ((1, 'EnKF'), (20, 'EnKF_i1.4')):
        v = {'A1': [], 'EnKF': []}
        for s in SEEDS:
            d = (pd.concat([rd(f'{T}/K{K}/case*/frames.csv'), rd(f'{T}/K{K}_i14/case*/frames.csv')]) if s == 42 else rd(f'{TS}/s{s}_K{K}/case*/frames.csv'))
            d = d[d.frame >= 40]
            v['A1'].append(d[d.method == 'A1'].missing_rmse.mean()); v['EnKF'].append(d[d.method == b].missing_rmse.mean())
        a, e = np.mean(v['A1']), np.mean(v['EnKF']); r.append(f'{K} & {f4(a)} & {f4(e)} & {100 * (1 - a / e):+.0f}\\%')
    return r


BODY = r'''
\begin{center}{\LARGE\bfseries Rayleigh--B\'enard 稀疏观测同化：\\滞后（lag）初始化 EnKF 对比\par}\vspace{.4em}{\small 基于已保存的实验结果；初始化对 EnKF 含有真值信息，见第 5 节}\end{center}

\section{目的与假设}
前期主实验中 EnKF 的初始集合以 A1 第 0 帧分析为中心加扰动。该集合起点离散度小、且借用了 A1 的结果，与 EnKF 的常规用法不同。本文用另一种构造：\textbf{取同一条轨迹的第 0--19 帧，各自用求解器积分到第 40 帧，作为 20 个成员；从第 40 帧开始同化，评价第 40--99 帧（共 60 帧）。}假设是：在这样的集合下，观测间隔 $K$ 增大时 A1 相对 EnKF 的优势会发生变化。若 A1 在所有 $K$ 下仍占优，则此前的结论稳固；若在大 $K$ 下 EnKF 更优，则此前``K 大时 A1 略优''的结论依赖于初始化。

\section{设置}
\begin{itemize}
\item 状态、先验、观测噪声（0.1）、观测比例（1\%）、预报求解器（Ra$=5\times10^4$，真值 $10^5$）、A1 的全部参数均与主实验相同，未为本实验重调。
\item \textbf{EnKF 初始集合：}真值第 $i$ 帧（$i=0,\dots,19$）经求解器积分 $40-i$ 帧至第 40 帧，得到 20 个成员；不额外加噪声；EnKF 在第 40 帧也做同化；20 成员、乘性膨胀、不用局地化。
\item \textbf{A1 与纯预报：}从第 40 帧冷启动。第 40 帧无背景、只用该帧观测；之后每 $K$ 帧同化一次。纯预报为第 40 帧同化后不再同化。
\item 测试 20 个轨迹，种子 42/43/44；观测位置、噪声、采样噪声由种子决定，三种方法在同一轨迹、同一种子下看到相同的观测。
\item EnKF 膨胀在 5 个验证轨迹（种子 42）上选，见下表；测试轨迹只用于最终评估。
\end{itemize}
\begin{table}[H]\centering\small\caption{验证轨迹上各膨胀系数的 EnKF 平均 RMSE（第 40--99 帧）。}
\begin{tabular}{cp{10.5cm}}\toprule
$K$ & 膨胀系数: RMSE\\\midrule
@TAB_VAL@
\bottomrule\end{tabular}\end{table}
$K=10,20$ 的最优膨胀落在扫描区间的上边界（2.0），未继续扩大，因此这两个 $K$ 下的 EnKF 可能仍有改善空间。

\section{结果}
\begin{table}[H]\centering\small\caption{测试集 20 个轨迹的未观测点 RMSE（第 40--99 帧，种子 42 / 43 / 44）。``降低''为 A1 相对 EnKF 的降低，负值表示 A1 更差；``A1 更优''为 20 个轨迹中 A1 误差更小的个数。}
\resizebox{\linewidth}{!}{\begin{tabular}{cccccccc}\toprule
$K$ & 膨胀 & A1 & EnKF (lag) & 纯预报 & 降低 & A1 更优\\\midrule
@TAB_MAIN@
\bottomrule\end{tabular}}\end{table}
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{report_fig_lag/main.pdf}\caption{左：各观测间隔下的误差；中、右：$K=1$、$K=20$ 的逐帧误差。}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\linewidth]{report_fig_lag/cases.pdf}\caption{逐轨迹配对比较（种子 42）。}\end{figure}
@TEXT_MAIN@

\subsection{与主实验初始化的对照}
同样取第 40--99 帧，主实验（EnKF 以 A1 第 0 帧分析为中心加扰动、第 0 帧起连续同化；$K=20$ 用膨胀 1.4）的结果如下，3 个种子均值：
\begin{table}[H]\centering\small\caption{主实验初始化下第 40--99 帧的 RMSE。}
\begin{tabular}{cccc}\toprule
$K$ & A1 & EnKF（中心加扰动） & A1 降低\\\midrule
@TAB_SW@
\bottomrule\end{tabular}\end{table}
两种初始化下 $K=1$ 时 A1 都占优；$K=20$ 时结论相反。这说明大 $K$ 下的 A1--EnKF 比较\textbf{取决于 EnKF 的初始化}，不能单独引用任一组数字作为总体结论。

\section{EnKF 初始集合的性质}
lag 集合在第 40 帧的离散度（spread）约 @SP0@，集合均值对真值的误差约 @ER0@，即误差约为 spread 的 @RATIO@ 倍，集合明显偏自信（离散度小于实际误差）；观测更新量因此偏小，实验中靠较大的膨胀系数补偿。主实验中的中心加扰动集合：spread 约 0.014、误差约 0.038（比值约 2.7）。

\section{需要如实说明的限制}
\begin{enumerate}
\item \textbf{信息泄漏：}lag 集合使用了测试轨迹真值的前 20 帧，A1 没有用到。EnKF 在 $K\ge10$ 时更优，有多少来自这部分真值信息，本文\textbf{没有做拿掉真值的对照}，不能断言是 EnKF 方法本身更好。
\item \textbf{起点不对等：}A1 在第 40 帧冷启动，没有历史。对 $K=20$ 的单种子检查显示：A1 冷启动与从第 0 帧连续同化相比，起点后前 10 帧误差高约 13\%（0.0375 对 0.0332），之后 50 帧并不更差（0.0342 对 0.0369）。这是单种子、两次运行随机数流不同的比较，差异在噪声范围内，不是严格结论。
\item \textbf{A1 的参数在 $K=1$ 下选定，}$K=5,10,20$ 及 lag 设置下没有重调；$K=10,20$ 的 EnKF 膨胀在扫描边界。两边调参预算不对等。
\item 评价窗口只有 60 帧，$K=20$ 时仅 3 次同化（第 40、60、80 帧），统计量有限，不能与 100 帧的主表直接比较。
\item 成员的积分时长不同（21--40 帧），不是同一预报时长的标准集合。
\item 3 个种子、20 个轨迹；Ra 失配只有 Ra$=5\times10^4$ 一档；模型误差较弱（纯预报约 0.041）也限制了不同方法之间的区分度。
\end{enumerate}

\section{结论}
\begin{itemize}
\item 观测密（$K=1,5$）时，A1 在两种 EnKF 初始化下都优于调优的 EnKF。
\item 观测稀疏（$K\ge10$）时，在 lag 初始化下 EnKF 更优；在中心加扰动初始化下 A1 略优。两种设置的差别包含真值信息泄漏，目前无法分离原因。
\item 下一步应补做不含真值的 EnKF 初始化（例如用独立预热的集合），并在相同起点下重调 A1 的 $K=5,10,20$ 参数，再下结论。
\end{itemize}
\end{document}
'''


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_main(); fig_cases()
    sp0 = np.nanmean([E[K, 42].groupby('case').init_spread.first().mean() for K in KS]); er0 = np.nanmean([E[K, 42].groupby('case').init_err.first().mean() for K in KS])
    ks = {K: [100 * (1 - pc[K, s]['A1'].mean() / pc[K, s]['EnKF'].mean()) for s in SEEDS] for K in KS}
    txt = ('$K=1,5$ 时 A1 更优（误差分别低 ' + f'{min(ks[1]):.0f}--{max(ks[1]):.0f}\\% 与 {min(ks[5]):.0f}--{max(ks[5]):.0f}\\%' + '）；$K=10,20$ 时 EnKF 更优（A1 误差分别高 '
           + f'{-max(ks[10]):.0f}--{-min(ks[10]):.0f}\\% 与 {-max(ks[20]):.0f}--{-min(ks[20]):.0f}\\%' + '，三个种子的范围）。逐轨迹看，大 $K$ 下 A1 只在少数轨迹上更优。')
    rep = {'TAB_VAL': ' \\\\\n'.join(val_rows()) + ' \\\\', 'TAB_MAIN': ' \\\\\n'.join(main_rows()) + ' \\\\', 'TAB_SW': ' \\\\\n'.join(same_window()) + ' \\\\',
           'TEXT_MAIN': txt, 'SP0': f'{sp0:.4f}', 'ER0': f'{er0:.4f}', 'RATIO': f'{er0 / sp0:.0f}'}
    body = BODY
    for k in sorted(rep, key=len, reverse=True): body = body.replace(f'@{k}@', rep[k])
    left = re.findall(r'@[A-Z0-9_]+@', body)
    if left: raise SystemExit(f'unfilled placeholders: {left}')
    tex = DOC / 'RB实验汇报_lag初始化.tex'; tex.write_text(PRE + body, encoding='utf-8')
    for _ in range(2):
        r = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', tex.name], cwd=DOC, capture_output=True, text=True)
        if r.returncode: print(r.stdout[-2500:]); raise SystemExit(1)
    for ext in ('aux', 'log', 'out'): tex.with_suffix(f'.{ext}').unlink(missing_ok=True)
    print('saved', tex.with_suffix('.pdf'))


if __name__ == '__main__': main()
