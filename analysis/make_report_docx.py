"""Write the RB experiment report (Word). Tables are computed from the result CSVs in outputs/; prose numbers that come
from analysis/unified_summary.py (stream-function projection) are taken from outputs/study_rb_g/unified/summary.txt.
Usage: python analysis/make_report_docx.py --out docs/RB实验报告.docx
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

R = Path('outputs')
G = R / 'study_rb_g'
CH = ('b', 'u_x', 'u_z')


def font(run, size=None, bold=None):
    run.font.name = 'Times New Roman'; run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    if size: run.font.size = Pt(size)
    if bold is not None: run.bold = bold


class Doc:
    def __init__(self):
        d = self.d = Document()
        s = d.sections[0]
        s.orientation = WD_ORIENT.LANDSCAPE; s.page_width, s.page_height = Cm(29.7), Cm(21.0)
        for m in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'): setattr(s, m, Cm(2.0))
        st = d.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(11)
        st.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        for n in ('Heading 1', 'Heading 2', 'Heading 3'):
            h = d.styles[n]; h.font.name = 'Times New Roman'; h.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        self.nfig = 0; self.ntab = 0

    def h(self, text, lv=1): self.d.add_heading(text, lv)

    def p(self, text, bold=False, size=None, indent=True):
        para = self.d.add_paragraph(); r = para.add_run(text); font(r, size, bold)
        para.paragraph_format.space_after = Pt(4)
        if indent: para.paragraph_format.first_line_indent = Pt(22)
        return para

    def bullets(self, items):
        for t in items:
            para = self.d.add_paragraph(style='List Bullet'); font(para.add_run(t))

    def eq(self, text):
        para = self.d.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = para.add_run(text); r.font.name = 'Cambria Math'; r.font.size = Pt(11)

    def table(self, header, rows, caption, widths=None):
        self.ntab += 1
        c = self.d.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        font(c.add_run(f'表 {self.ntab}  {caption}'), 10, True)
        t = self.d.add_table(rows=1, cols=len(header)); t.style = 'Table Grid'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, x in enumerate(header):
            cell = t.rows[0].cells[i]; cell.text = ''; font(cell.paragraphs[0].add_run(str(x)), 10, True)
        for row in rows:
            cells = t.add_row().cells
            for i, x in enumerate(row):
                cells[i].text = ''; font(cells[i].paragraphs[0].add_run(str(x)), 10)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths): row.cells[i].width = Cm(w)
        self.d.add_paragraph()

    def fig(self, path, caption, width=24.5):
        self.nfig += 1
        para = self.d.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.add_run().add_picture(str(path), width=Cm(width))
        c = self.d.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        font(c.add_run(f'图 {self.nfig}  {caption}'), 10, True)


def f4(x): return f'{x:.4f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--out', default='docs/RB实验报告.docx'); a = ap.parse_args()
    D = Doc()

    # ---------------- data -----------------
    arms = {'A': 'A  背景场 + 稀疏观测联合同化（扩散先验 + DPS）', 'B': 'B  稀疏观测直接重建（扩散先验 + DPS，无背景场）', 'C': 'C  EnKF 同化（膨胀 + 局地化）'}
    s = []
    for c in range(3):
        u = pd.read_csv(G / f'unified/c{c}/enkf.csv'); f = pd.read_csv(G / f'fusion_c{c}/fusion.csv')
        s += [u[u.scheme == 'dps'].assign(case=c, arm='A'), f[f.scheme.str.contains('only')].assign(case=c, arm='B'), u[u.scheme == 'enkf_infl_loc'].assign(case=c, arm='C')]
    s = pd.concat(s)
    cols = ['missing_rmse_ch0', 'missing_rmse_ch1', 'missing_rmse_ch2', 'missing_rmse']
    ch = s.groupby('arm')[cols].mean(); pc = s.groupby(['arm', 'case']).missing_rmse.mean().unstack()
    div = s.groupby('arm').an_div_rel.median(); truth_div = s.truth_div_rel.median()
    A, B, C = (ch.loc[k, 'missing_rmse'] for k in 'ABC')
    e3 = pd.read_csv(G / 'enkf3/enkf_all.csv'); e3m = e3.groupby('scheme').missing_rmse.mean()
    noise = pd.read_csv(G / 'noise/noise.csv')
    long = pd.concat([pd.read_csv(f) for f in sorted((R / 'study_rb_long').glob('c*/enkf.csv'))])
    lw = long.assign(w=long.frame // 50 * 50).groupby(['scheme', 'w']).missing_rmse.mean().unstack(0)

    # ---------------- title -----------------
    t = D.d.add_heading('Rayleigh–Bénard 对流的扩散先验稀疏观测同化实验报告', 0); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    D.p('项目：Reconstruction Diffusion Assimilation（rdda）。本文档描述实验设置、同化方法、对照方案与全部实验结果。'
        '所有结果均来自孪生实验（观测为真值的随机抽样），所有数字均取自 outputs/ 下的结果文件，下文逐项给出来源。', indent=False)

    # ---------------- 1 -----------------
    D.h('1  研究目的')
    D.p('在只有约 1% 的稀疏观测、且数值模式存在模型误差的条件下，得到既贴合观测、又符合物理的完整流场分析场。'
        '本文方法的思路是：用一个只学习"流场长什么样"的无条件扩散模型作为先验，用数值求解器把上一帧的分析场推进到当前帧作为背景场，'
        '再把背景场与当前帧稀疏观测一起，作为似然引导项放进扩散后验采样（DPS）中，逐帧循环。要回答的问题有四个：')
    D.bullets(['背景场 + 稀疏观测联合同化，相比只用稀疏观测直接重建，误差降低多少？',
               '与经典的集合卡尔曼滤波（EnKF）相比，误差如何？',
               '同化之后是否破坏了方程的平衡？用速度场的散度来衡量。',
               't=0 的初值含有不同程度的噪声时，同化结果是否受影响？'])

    # ---------------- 2 -----------------
    D.h('2  物理问题与数据')
    D.h('2.1  控制方程', 2)
    D.p('二维 Rayleigh–Bénard 对流，Boussinesq 近似、无量纲形式，状态为浮力 b 与速度 u = (u_x, u_z)（b 是浮力，不是原始温度）：')
    D.eq('∇·u = 0')
    D.eq('∂b/∂t − κ ∇²b = − u·∇b')
    D.eq('∂u/∂t − ν ∇²u + ∇p − b e_z = − u·∇u')
    D.eq('κ = (Ra·Pr)^(−1/2),   ν = (Ra/Pr)^(−1/2)')
    D.p('边界条件：x 方向周期（长度 L_x = 3）；z 方向为上下壁面（L_z = 1），b(z=0) = 1、b(z=L_z) = 0，壁面上速度为 0（无滑移）。'
        '真实流动取 Ra = 1×10⁵、Pr = 1。方程由 Dedalus v3 求解：x 方向 Fourier、z 方向 Chebyshev，网格 72 × 24（z 方向为 Chebyshev–Gauss 非均匀节点），'
        '去混叠因子 1.5，时间步长 dt = 0.01，RK222 时间推进。源码：src/rdda/solver/rb_equations.py。')
    D.h('2.2  数据集与时间尺度', 2)
    D.bullets([
        '数据生成：200 个随机种子，每个种子从"线性静力 b(z) + 壁面处为零的随机扰动（幅度 0.1）、速度为零"的初值出发，每 35 个求解器步保存一帧，共 220 帧；'
        '丢弃前 120 帧（t < 42，瞬态），每个 case 保留 100 帧。',
        '一帧 = 35 个求解器步 = 0.35 个无量纲时间单位；一个 case 的 100 帧 = 3500 个求解器步 = 35 个无量纲时间单位。',
        '数据划分：训练 / 验证 / 测试 = 160 / 20 / 20 个 case。验证 case 只用于调参，测试 case 只用于出结果；本文用测试 case 0、1、2。',
        '通道：(b, u_x, u_z)，每个 case 形状为 (100, 72, 24, 3)。',
        '统计性质（长时间真值，测试 case 对应的轨迹再往后积分到 520 帧，取瞬态之后 400 帧）：动能约 0.0276、b 的标准差约 0.257，在 400 帧内基本不变，'
        '帧间变化约 0.0127 也不变。即流动处于统计意义上的稳态，但流场每帧都在演化，不是固定点。'])
    D.h('2.3  数值求解器与模型误差', 2)
    D.p('背景场由 Dedalus 求解器产生，但模型参数故意与真值失配：背景模型取 Ra = 5×10⁴，真值为 1×10⁵。这是本实验唯一的模型误差来源，'
        '误差由求解器自身产生，不是人为往数据里加噪声。从真值出发只预报一帧，误差约 0.01；在循环中误差逐帧累积，纯预报（不同化）的误差最终涨到约 0.055。'
        '每帧预报的做法：取上一帧分析场，先对速度做流函数投影（在壁面处为零的 Chebyshev 基下最小二乘拟合，使速度近似无散度并满足壁面条件），'
        '压力与 tau 项置零，重启求解器后推进 35 步。这不是严格的 restart。')

    # ---------------- 3 -----------------
    D.h('3  同化方法')
    D.h('3.1  无条件扩散先验', 2)
    D.p('先验 p_θ(X) 是一个无条件扩散模型（VP-cosine 噪声调度，U-Net 去噪网络，基础通道数 16），训练时只输入完整流场快照，不带观测、背景场和时间。'
        '训练设置：200 个 epoch，每卡 batch 32，学习率 1×10⁻⁴，4 卡训练，输入按训练集逐通道均值和标准差归一化。权重：outputs/rb_prior_4gpu/best.pt。')
    D.h('3.2  逐帧循环同化（方案 A）', 2)
    D.p('第 t 帧的流程如下：')
    D.bullets(['背景场：背景[t] = Dedalus(分析场[t−1])，Ra = 5×10⁴，推进 35 步（第 0 帧没有背景，只用观测）。',
               '后验采样：以背景[t] 与观测[t] 为似然项做 DPS 引导采样（采样从纯噪声起，100 个去噪步，8 个样本）。每个去噪步用当前对干净场的估计与观测、背景的偏差的梯度引导采样。',
               '分析场：取 8 个样本的均值作为分析场[t]。',
               '不做硬观测覆盖：观测点上的分析值也由采样给出，不被观测值直接覆盖（DPS 与 EnKF 一致）。'])
    D.p('DPS 参数（在验证 case 上选定）：背景误差尺度 bg_std = 0.3、引导强度 dps_rate = 2、观测误差尺度 obs_std = 0.05。'
        '观测是无噪声真值，所以 obs_std 只是似然的权重，不是噪声水平。')
    D.h('3.3  观测', 2)
    D.bullets(['观测比例 1%：72 × 24 = 1728 个格点中随机取 17 个位置，位置在整个实验中固定（同一掩码）。',
               '三个通道（b, u_x, u_z）在这 17 个位置同时被观测，观测值为无噪声的真值。',
               '观测频率：主对比中每一帧（35 个求解器步，0.35 时间单位）进一次观测；另有每 3 帧（1.05 时间单位）一次的补充实验。'])

    # ---------------- 4 -----------------
    D.h('4  对照方案与评价指标')
    D.table(['方案', '方法', '背景场', '观测', '关键参数（均在验证 case 上选定）'], [
        ['A', '扩散先验 + DPS', '有（Dedalus 预报，Ra=5e4）', '每帧 1%', 'bg_std 0.3、dps_rate 2、obs_std 0.05；8 样本、100 步'],
        ['B', '扩散先验 + DPS', '无', '每帧 1%', 'obs_std 0.025、dps_rate 2；8 样本、100 步（每帧独立采样，不用求解器）'],
        ['C', '随机 EnKF', '有（成员由 Dedalus 预报）', '每帧 1%', '20 成员；膨胀 2.5；局地化 1.5(x)×1.2(z)；观测误差 R=0.01'],
    ], '三个对照方案', widths=[1.5, 4, 5.5, 3, 11])
    D.p('EnKF（方案 C）的实现：随机 EnKF，更新公式 xᵃ − xᶠ = Pᶠ Hᵀ (H Pᶠ Hᵀ + R)⁻¹ [y − H(xᶠ)]，每个成员使用扰动观测 y + ε，逐观测点顺序同化；'
        '膨胀指把成员相对均值的偏差放大 2.5 倍；局地化用 Gaspari–Cohn 权重，距离按物理坐标计算（x 周期，z 取真实 Chebyshev 坐标），半径 1.5（x）× 1.2（z）。'
        '成员各自用 Dedalus 推进。初始集合以 DPS 第 0 帧的分析场为中心并加 10% 参考标准差的扰动，且跳过 t=0 的 EnKF 更新，因此比较的是同一个初始分析之后的循环，而不是两种方法各自独立的初始化。'
        '分析场取集合均值，其散度在预报重启之前测量。（说明：EnKF 公式按通用形式实现，使用时请与任务给定的公式核对。）')
    D.h('4.1  评价指标', 2)
    D.bullets(['RMSE：只在未被观测的格点上计算，(b, u_x, u_z) 三个通道合并；也给出分通道结果。观测点上的误差不计入，避免被观测值抬高或压低。',
               '散度一致性：相对散度 ‖∇·u‖ / ‖∇u‖，导数用谱方法（x 方向 Fourier、z 方向 Chebyshev）。0 表示严格不可压缩；真值约 3.4×10⁻⁵。',
               '公平性：所有方法的参数都在验证 case 上调好，在 3 个测试 case 上比较；DPS 与 EnKF 的硬观测策略一致（都关）；EnKF 的散度在重启前测量。'])

    # ---------------- 5 -----------------
    D.h('5  实验结果')
    D.h('5.1  三个方案的误差对比（每帧观测，100 帧，3 个测试 case）', 2)
    D.p(f'对比结果发现，方案 A 的误差最小：三个通道合并的未观测点 RMSE 为 {f4(A)}，方案 B 和 C 分别为 {f4(B)}、{f4(C)}'
        f'（A 比 B 低 {100 * (1 - A / B):.0f}%，比 C 低 {100 * (1 - A / C):.0f}%）。整个演化过程共 3500 个求解器步，对应 35 个无量纲时间单位，每 35 个求解器步（每一帧）进一次观测。')
    D.table(['方案', 'b', 'u_x', 'u_z', '三通道合并'], [[arms[k]] + [f4(ch.loc[k, c]) for c in cols] for k in 'ABC'],
            '未观测点 RMSE（3 个测试 case、100 帧的平均）；数据：outputs/study_rb_g/unified、fusion_c*', widths=[11, 3, 3, 3, 3.5])
    D.table(['方案', '测试 case 0', '测试 case 1', '测试 case 2'], [[arms[k]] + [f4(pc.loc[k, c]) for c in range(3)] for k in 'ABC'],
            '分 case 的合并 RMSE', widths=[11, 3.5, 3.5, 3.5])
    D.bullets([
        f'A 在 3 个 case 中的 2 个（case 1、2）明显优于 B 和 C；在 case 0 上 A（{f4(pc.loc["A", 0])}）与 B（{f4(pc.loc["B", 0])}）基本持平，A 并不是处处最优。',
        f'C 的分通道结果不均衡：b 的误差 {f4(ch.loc["C", cols[0]])} 明显高于 A、B，u_x、u_z 与 B 接近。',
        '早期不稳定：循环开始后约前 5 到 7 个时间单位，A 的误差反而不如 B（上一步分析还不准，预报出的背景拖累结果）；之后 A 更好。',
        'B 每帧独立采样、没有时间连续性，A 相对 B 的优势有一部分来自求解器带来的时间连续性，不全是背景场本身的质量。'])
    D.fig(G / 'unified/unified.png', '三个方案的误差与散度（每帧观测）。(a) 分通道平均 RMSE；(b) RMSE 随时间；(c) 分析场的相对散度（虚线为真值）。')
    D.fig(G / 'unified/field_b_c1_preview.png', '测试 case 1、第 60 帧的 b 场：真值、观测位置、A/B/C 的分析场与绝对误差，底部为 RMSE 随时间曲线。完整动画见 outputs/study_rb_g/gif/。', width=17)
    D.fig(G / 'fusion/fusion.png', '加背景（A）与不加背景（B）的 RMSE 随时间，3 个测试 case，每个标题给出该 case 的平均值。', width=24)
    D.p(f'补充（每 3 帧观测一次，即 1.05 个时间单位）：A 的平均 RMSE 为 {f4(e3m["dps"])}，EnKF（膨胀 + 局地化）为 {f4(e3m["enkf_infl_loc"])}，'
        f'只加膨胀的 EnKF 为 {f4(e3m["enkf_inflation"])}，无膨胀无局地化的 EnKF 为 {f4(e3m["enkf_plain"])}。无膨胀的 EnKF 几乎不吸收观测，误差一路上涨；膨胀与局地化都是必要的。'
        '每 3 帧的对比图见 outputs/study_rb_g/enkf3/enkf3.png。')

    D.h('5.2  散度一致性（方程平衡）', 2)
    D.p('同一份分析场的相对散度分两种口径给出：（1）模型输出的原始分析场；（2）经过与求解器重启相同的流函数投影之后（投影是每一帧预报前都会做的操作，因此原始分析场的不平衡不会累积）。'
        f'参照：真值的散度约 {truth_div:.1e}；对真值做同样的投影后为 0.0079，这是投影算子本身的下限。')
    D.table(['方案', '原始分析场散度（中位数）', '投影后散度', '投影前速度 RMSE', '投影后速度 RMSE'], [
        [arms['A'], f'{div["A"]:.3f}', '0.0084', '0.01348', '0.01335'],
        [arms['B'], f'{div["B"]:.3f}', '0.0084', '0.01744', '0.01690'],
        [arms['C'], f'{div["C"]:.4f}', '0.0075', '0.01562', '0.01564']],
        '相对散度（每帧观测；投影一列为 3 个 case、每 3 帧取样的均值，来源 outputs/study_rb_g/unified/summary.txt）', widths=[11, 4.5, 3.5, 3.8, 3.8])
    D.bullets([
        '经过求解器同款投影之后，三个方案的散度约 0.0075 到 0.0084，与对真值做同样投影的下限（0.0079）无明显差异，且投影几乎不改变速度对真值的误差。在这个意义上，扩散模型的重建结果与 EnKF 在散度一致性上无明显差异。',
        '但原始分析场上差别明显：A、B 约 0.07，局地化 EnKF 约 0.006，真值约 3.4×10⁻⁵。扩散模型的分析场不严格满足不可压缩。已知这个指标对小尺度噪声极其敏感（给真值的速度加 0.6% 的白噪声，散度就从 3.6×10⁻⁵ 升到 0.06），原始散度偏大与约 1% 量级的小尺度毛刺被求导放大有关，上表中投影不改变速度误差也支持这一点；但该解释只在 3 个 case、每 3 帧取样共 102 帧上验证过。',
        '不带局地化的 EnKF 散度约 1×10⁻⁵（每 3 帧观测实验），甚至低于真值，这不是它同化得好：成员都是求解器刚预报出来的、本身近似无散度，分析场是成员的线性组合，仍近似无散度；纯 Dedalus 预报的散度本来就是约 9×10⁻⁶。局地化对不同位置乘不同权重，破坏了这一性质，散度升到 10⁻³ 量级。'])
    D.h('5.3  t = 0 初值噪声', 2)
    D.p('在 t=0 的真值上叠加不同程度的噪声，比较不同化的纯预报与每帧同化的扩散模型方法（方案 A 的设置）。噪声的构造：')
    D.bullets(['b 的噪声：平滑随机场（高斯滤波，相关长度 3 个格点）乘壁面包络 4s(1−s)（s 为归一化的 z 坐标），壁面处为 0。',
               '速度的噪声：不直接加在 u_x、u_z 上，而是先造流函数的平滑噪声（壁面包络 (s(1−s))²），再按 u_x = ∂ψ/∂z、u_z = −∂ψ/∂x 求导，得到的速度噪声无散度且壁面为 0；u_x 与 u_z 共用一个缩放系数。',
               '幅度：噪声标准差 = 噪声水平 × 参考标准差。参考标准差为 b 去掉 x 方向平均后的标准差约 0.097，u_x、u_z 分别约 0.166 与 0.169。噪声水平取 0、5%、10%、20%、40%，每档 3 个随机噪声样本，测试 case 0。'])
    nz = noise[noise.level.isin([0.0, 0.4])]
    mm = nz.groupby(['scheme', 'level']).missing_rmse.mean()
    lt = nz[nz.frame >= 90].groupby(['scheme', 'level']).missing_rmse.mean()
    rows = []
    for sch, nm in (('No assimilation (forecast only)', '不同化（纯预报）'), ('DPS, observations every frame', 'DPS，每帧观测'), ('DPS, observations every 3 frames', 'DPS，每 3 帧观测')):
        rows.append([nm, f4(mm[sch, 0.0]), f4(mm[sch, 0.4]), f'{100 * (mm[sch, 0.4] / mm[sch, 0.0] - 1):+.0f}%', f4(lt[sch, 0.0]), f4(lt[sch, 0.4])])
    D.table(['方案', '无噪声（全程平均）', '40% 噪声（全程平均）', '变化', '无噪声（最后 10 帧）', '40% 噪声（最后 10 帧）'], rows,
            '初值噪声对未观测点 RMSE 的影响（测试 case 0，3 个噪声样本的平均；数据 outputs/study_rb_g/noise/noise.csv）', widths=[7, 3.8, 3.8, 2.5, 3.8, 3.8])
    D.p(f'在噪声方面，初始误差为 40% 参考标准差时，不同化的纯预报全程平均 RMSE 从 {f4(mm["No assimilation (forecast only)", 0.0])} 升到 {f4(mm["No assimilation (forecast only)", 0.4])}'
        f'（+{100 * (mm["No assimilation (forecast only)", 0.4] / mm["No assimilation (forecast only)", 0.0] - 1):.0f}%）；每帧同化的 DPS 从 {f4(mm["DPS, observations every frame", 0.0])} 到 {f4(mm["DPS, observations every frame", 0.4])}，几乎不变，说明同化能够快速消除初值误差的影响。')
    D.bullets(['限制：到最后 10 帧时，纯预报在无噪声时也已经靠模型误差涨到约 0.052，初值噪声的影响被淹没（0.0522 对 0.0553），所以这个实验对"初值噪声"的区分度偏弱；这与 RB 流动变化慢有关。',
               '初值噪声只对扩散模型方法做了测试，没有对 EnKF 做。',
               '同化后速度场的散度不随初始噪声变化（见 outputs/study_rb_g/noise/noise.png 下排）。'])
    D.fig(G / 'noise/noise.png', '初值噪声实验：上排为未观测点 RMSE（橙：不同化，蓝：DPS，线为平均、带为 3 个噪声样本的范围），下排为同化后的相对散度。', width=25)

    D.h('5.4  长时间表现（400 个循环）', 2)
    D.p('为回答"误差随时间还能不能继续下降"，把 3 个测试 case 对应的原始轨迹重新积分到 520 帧（前 100 帧与原测试数据数值一致，最大差 4×10⁻⁹），从第 120 帧起共 400 个循环（约 140 个时间单位），只比较 DPS 与局地化 EnKF（每 3 帧观测一次，参数同前）。')
    D.table(['帧区间', 'A（DPS）', 'EnKF（膨胀 + 局地化）'], [[f'{w}–{w + 49}', f4(lw.loc[w, 'dps']), f4(lw.loc[w, 'enkf_infl_loc'])] for w in lw.index],
            '长时间每 50 帧的平均 RMSE（3 个 case 平均；数据 outputs/study_rb_long）', widths=[5, 5, 7])
    la = long[long.scheme == 'dps'].missing_rmse.mean(); le = long[long.scheme == 'enkf_infl_loc'].missing_rmse.mean()
    D.bullets([f'DPS 前 100 帧下降最快，之后缓慢下降，300 帧以后稳定在约 0.011；EnKF 全程约 0.019，不再下降。全程平均 DPS {f4(la)}、EnKF {f4(le)}，DPS 低约 {100 * (1 - la / le):.0f}%，3 个 case 都是 DPS 更低。',
               '真值本身在这段时间内已是统计稳态（动能约 0.0276、b 标准差约 0.257 基本不变）。误差走平是否由"误差注入（Ra 失配）与观测修正达到平衡"造成，只是一个合理的解释，尚未验证（验证办法：改变 Ra 失配的大小，看平线高度是否跟着变）。'])
    D.fig(R / 'study_rb_long/long.png', '长时间表现：(a) 400 个循环的 RMSE；(b) 每 17.5 个时间单位窗口内的平均 RMSE。', width=22)

    # ---------------- 6 -----------------
    D.h('6  结论')
    D.bullets([
        f'误差：背景场 + 稀疏观测联合同化（A）的未观测点 RMSE 最小（{f4(A)}），稀疏观测直接重建（B）为 {f4(B)}，EnKF（C）为 {f4(C)}。A 平均比 B 低约 {100 * (1 - A / B):.0f}%、比 C 低约 {100 * (1 - A / C):.0f}%；长时间（400 帧）下 A 比 EnKF 低约 {100 * (1 - la / le):.0f}%。',
        'A 相对 B 的优势在 3 个 case 中有 2 个明显，1 个持平，且循环早期背景场反而有害，所以优势不稳定。',
        '散度一致性：经与求解器重启相同的投影后，A、B、C 的散度无明显差异（约 0.0075–0.0084，与真值投影下限 0.0079 相当）；原始分析场上扩散模型约 0.07、局地化 EnKF 约 0.006，扩散模型明显更大。',
        '噪声：初值噪声（至 40% 参考标准差）使纯预报误差增大约 24%，而每帧同化后几乎不受影响。',
        '以上结论的适用范围见第 7 节。'])

    # ---------------- 7 -----------------
    D.h('7  局限与待办')
    D.bullets(['样本量小：只有 3 个测试 case、单个随机种子；初值噪声实验只用测试 case 0。',
               '孪生实验：观测是无噪声真值的 1% 随机点，三个通道同时被观测，模型误差只有 Ra 失配一种且大小固定，比真实场景容易。',
               'EnKF 只用了 20 个成员，没有测 40 个或更多；增加成员后差距可能缩小。EnKF 的更新公式按通用形式实现，需要与任务给定的公式核对。',
               '用于对比的 EnKF 是从 DPS 的第 0 帧分析场起步的，比较的是同一个初始分析之后的循环。',
               '散度的解释（原始散度偏大来自小尺度噪声、投影不改变速度误差）只在 3 个 case、每 3 帧取样的 102 帧上验证过，没有对每一帧系统验证。',
               '本文的实验时间跨度为 3500 个求解器步（35 个时间单位）和补充的 14000 个求解器步（140 个时间单位），不是 1000 步。',
               '另有一套 NS2D 问题的代码、数据和先验保留在项目中，但它的 EnKF 对比的硬观测策略未与 DPS 对齐，结论待核实，不在本文范围。'])

    # ---------------- 8 -----------------
    D.h('8  文件位置与复现')
    D.table(['内容', '位置'], [
        ['方程与求解器', 'src/rdda/solver/rb_equations.py、forecast_solver.py'],
        ['扩散先验权重', 'outputs/rb_prior_4gpu/best.pt'],
        ['数据', 'data/processed（训练/验证/测试）；data/rb200_raw（原始 200 个种子）；data/long_raw（长时间 3 条轨迹）'],
        ['三方案对比（A/B/C）结果与图', 'outputs/study_rb_g/unified/、outputs/study_rb_g/fusion_c0..2/'],
        ['初值噪声', 'outputs/study_rb_g/noise/'],
        ['长时间', 'outputs/study_rb_long/；脚本 scripts_long_run.sh'],
        ['动画（3 个 case × 3 个变量）', 'outputs/study_rb_g/gif/case{0,1,2}/rb_{b,ux,uz}.gif'],
        ['实验与绘图脚本', 'analysis/study_rb.py、plot_study_rb.py、make_rb_gif.py、unified_summary.py'],
        ['详细说明', 'README.md 第 6 节']], '主要文件位置', widths=[8, 17])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); D.d.save(out); print('saved', out)


if __name__ == '__main__':
    main()
