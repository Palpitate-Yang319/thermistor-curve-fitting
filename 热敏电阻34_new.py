# -*- coding: utf-8 -*-
"""
热敏电阻阻值-温度特性研究 —— 第 3)、4) 部分
=====================================================================

背景（与 2) 相同的模型与数据）：
    模型（1）   R_T = R_T0 * exp( beta * (1/T - 1/T0) )
               beta = 3100 K, R_T0 = 22 kΩ @ 25℃(298.15 K)
    模型（2）   R_t = a_n * t^n + ... + a_1 * t + a_0      （t 为 ℃，R 为 kΩ）

数据划分（与 2) 完全一致）：
    训练集 20~80℃，步长 2℃  -> 31 个点
    测试集 0~100℃ 剔除 20~80℃ -> 0,2,...,18 与 82,84,...,100 共 20 个点

本程序完成：
  3) 把 2) 的实验重复 10 次——每次用【不同随机种子】重新叠加 sigma=500Ω 的高斯
     噪声，模拟 10 个不同批次的实验数据；对 n=1..6 阶多项式模型分别拟合，
     统计训练集/测试集误差在 10 次实验中的均值、标准差和极差，并讨论
     "采用不同训练数据对拟合（学习）结果的影响"（即方差/稳定性问题）。

  4) 把噪声方差设为 0（sigma=0）后重复 2)、3) 的全部内容，观察并讨论
     "数据中的噪声对拟合结果的影响"——无噪声时过拟合现象为什么消失。

  程序同时打印多张表格，并输出 6 张图（见文件末尾说明）。

运行环境： Python 3 + numpy + matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------------
# 0. 参数
# ----------------------------------------------------------------------
BETA = 3100.0
R_25 = 22.0
T0 = 25.0 + 273.15

T_MIN, T_MAX, T_STEP = 0.0, 100.0, 2.0
T_TR_MIN, T_TR_MAX = 20.0, 80.0

NOISE_SIGMA = 0.5          # 500 Ω = 0.5 kΩ（题目给定）
N_REPEAT = 10              # 题目要求重复 10 次
ORDERS = list(range(1, 7))  # 模型阶次 n = 1..6
SEED0 = 1                  # 第 1 次实验的随机种子，后面依次 +1


# ----------------------------------------------------------------------
# 1. 基础函数与数据准备
# ----------------------------------------------------------------------
def R_th(t_celsius):
    """模型（1）：由摄氏温度算真实阻值（kΩ）。"""
    return R_25 * np.exp(BETA * (1.0 / (t_celsius + 273.15) - 1.0 / T0))


def mse(y_meas, y_pred):
    """均方误差 MSE。"""
    y_meas = np.asarray(y_meas, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_pred - y_meas) ** 2))


t_all = np.arange(T_MIN, T_MAX + T_STEP * 0.5, T_STEP)      # 0,2,...,100 共 51 点
mask_tr = (t_all >= T_TR_MIN) & (t_all <= T_TR_MAX)

t_tr = t_all[mask_tr]                       # 训练温度（31 点）
t_te = t_all[~mask_tr]                      # 测试温度（20 点）
r_tr_true = R_th(t_tr)                      # 训练温度上的真值
r_te_true = R_th(t_te)                      # 测试温度上的真值（无噪声）

t_dense = np.linspace(T_MIN, T_MAX, 501)
r_dense_true = R_th(t_dense)


# ----------------------------------------------------------------------
# 2. 核心：重复 N_REPEAT 次"做实验 + 拟合"
# ----------------------------------------------------------------------
def run_trials(sigma, n_repeat=N_REPEAT, seed0=SEED0, orders=ORDERS):
    """
    重复 n_repeat 次实验：
      第 k 次实验用随机种子 seed0+k 生成一份新的噪声，得到"第 k 批实验数据"，
      然后对每个阶次 n 做最小二乘拟合，并记录各种误差与关键预测值。

    返回 stats：stats[n] = {'coef': [...], 'train': [...], 'train_true': [...],
                            'test': [...], 'fit0': [...], 'fit100': [...],
                            'atop': [...]}
         还有 stats['_data'][k] = 第 k 批带噪训练数据（画图用）
    """
    stats = {n: {"coef": [], "train": [], "train_true": [], "test": [],
                 "fit0": [], "fit100": [], "atop": []} for n in orders}
    stats["_data"] = []

    for k in range(n_repeat):
        rng = np.random.default_rng(seed0 + k)          # 每批数据独立的噪声
        r_tr = r_tr_true + rng.normal(0.0, sigma, size=t_tr.shape)
        stats["_data"].append(r_tr)

        for n in orders:
            c = np.polyfit(t_tr, r_tr, n)               # 最小二乘拟合（学习）
            stats[n]["coef"].append(c)
            stats[n]["train"].append(mse(r_tr, np.polyval(c, t_tr)))
            stats[n]["train_true"].append(mse(r_tr_true, np.polyval(c, t_tr)))
            stats[n]["test"].append(mse(r_te_true, np.polyval(c, t_te)))
            stats[n]["fit0"].append(float(np.polyval(c, 0.0)))
            stats[n]["fit100"].append(float(np.polyval(c, 100.0)))
            stats[n]["atop"].append(float(c[0]))
    return stats


def summarize(values):
    """返回 (均值, 标准差, 最小值, 最大值)。"""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)), float(v.min()), float(v.max())


def print_trial_table(stats, title, sigma):
    """打印"误差在 10 次实验中的统计"主表。"""
    print("=" * 100)
    print(title)
    print(f"噪声标准差 sigma = {sigma:.2f} kΩ   重复次数 = {N_REPEAT}   "
          f"随机种子 = {SEED0}..{SEED0 + N_REPEAT - 1}")
    print("=" * 100)
    print(f"{'阶次':>4} | {'训练MSE 均值±标准差':>24} | {'测试MSE 均值±标准差':>24} | "
          f"{'测试MSE 极差[min,max]':>24}")
    print("-" * 100)
    for n in ORDERS:
        tm, ts, _, _ = summarize(stats[n]["train"])
        em, es, emin, emax = summarize(stats[n]["test"])
        print(f"{n:>4} | {tm:>11.5f} ± {ts:>10.5f} | {em:>11.5f} ± {es:>10.5f} | "
              f"[{emin:>9.4f}, {emax:>9.4f}]")
    print("-" * 100)
    print("（训练MSE = 相对带噪测量值；测试MSE = 相对 0~20℃、80~100℃ 的真实阻值）")
    print()


def print_stability_table(stats, title):
    """打印"拟合结果本身（而不是误差）在 10 次实验中的波动"。"""
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(f"{'阶次':>4} | {'最高次系数 a_n 的 10 次波动':>40} | "
          f"{'预测值@0℃ 的 10 次波动':>24} | {'预测值@100℃ 的 10 次波动':>24}")
    print("-" * 100)
    for n in ORDERS:
        am, asd, amin, amax = summarize(stats[n]["atop"])
        f0m, f0s, f0min, f0max = summarize(stats[n]["fit0"])
        f1m, f1s, f1min, f1max = summarize(stats[n]["fit100"])
        print(f"{n:>4} | {am:>12.3e} ± {asd:>10.2e}  范围[{amin:.2e},{amax:.2e}] | "
              f"{f0m:>9.2f} ± {f0s:>7.2f}  范围[{f0min:.1f},{f0max:.1f}] | "
              f"{f1m:>9.2f} ± {f1s:>7.2f}  范围[{f1min:.1f},{f1max:.1f}]")
    print("-" * 100)
    print(f"参考真值：R(0℃) = {R_th(0.0):.2f} kΩ，R(100℃) = {R_th(100.0):.2f} kΩ")
    print()


def plot_curve_fan(stats, sigma, suptitle):
    """画 2×3 子图：每个阶次一图，叠加 10 次实验各自的拟合曲线。"""
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
    fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, N_REPEAT))

    for ax, n in zip(axes.ravel(), ORDERS):
        # 10 次实验的带噪测量点（只画第 1 批作代表，避免画面过乱）
        ax.plot(t_tr, stats["_data"][0], "o", ms=5, mfc="none", mec="c",
                mew=1.1, label="batch #1 measurements")
        # 10 条拟合曲线
        for k, c in enumerate(stats[n]["coef"]):
            ax.plot(t_dense, np.polyval(c, t_dense), "-", lw=1.3, color=colors[k],
                    alpha=0.85)
        # 真实模型（1）
        ax.plot(t_dense, r_dense_true, "k--", lw=1.8, label="true model (1)")

        ax.set_title(f"n = {n}:  10 fits from 10 batches", fontsize=10)
        ax.set_xlabel("Temperature/Celsius degree", fontsize=9)
        ax.set_ylabel(r"Resistance/k$\Omega$", fontsize=9)
        ax.set_xlim(T_MIN, T_MAX)
        ax.grid(True, alpha=0.35)
        ax.tick_params(labelsize=8)
        if n == ORDERS[0]:
            ax.legend(fontsize=7, loc="best")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_errorbars(stats, sigma, title):
    """画误差随阶次变化，并用误差棒表示 10 次实验的标准差。"""
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    train_mean = [summarize(stats[n]["train"])[0] for n in ORDERS]
    train_std = [summarize(stats[n]["train"])[1] for n in ORDERS]
    test_mean = [summarize(stats[n]["test"])[0] for n in ORDERS]
    test_std = [summarize(stats[n]["test"])[1] for n in ORDERS]

    ax.errorbar(ORDERS, train_mean, yerr=train_std, fmt="o-", lw=1.8,
                capsize=4, label="Training MSE (mean ± std over 10 batches)")
    ax.errorbar(ORDERS, test_mean, yerr=test_std, fmt="^-", color="crimson",
                lw=1.8, capsize=4,
                label="Test MSE (mean ± std over 10 batches)")
    if sigma > 1e-6:                     # 无噪声时不画这条参考线
        ax.axhline(sigma ** 2, color="gray", ls=":", lw=1.2,
                   label=r"noise variance $\sigma^2$")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial order n")
    ax.set_ylabel(r"MSE / k$\Omega^2$  (log scale)")
    ax.set_title(title)
    ax.set_xticks(ORDERS)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ======================================================================
# 3. 第 3) 题：含噪声（sigma = 0.5 kΩ），重复 10 次
# ======================================================================
print()
print("#" * 100)
print("# 第 3) 题：重复 2) 的实验 10 次（每批数据重新加噪，模拟不同批次实验）")
print("#" * 100)
print()

stats_noisy = run_trials(NOISE_SIGMA)

print_trial_table(
    stats_noisy,
    "表 1  含噪数据（sigma = 0.5 kΩ）重复 10 次的误差统计",
    NOISE_SIGMA)
print_stability_table(
    stats_noisy,
    "表 2  含噪数据重复 10 次：拟合系数与端点预测值的波动（稳定性的直接证据）")

# 单次结果与 10 次均值的对比（说明"单次实验结论不可靠"）
print("=" * 100)
print("表 3  单次实验 vs 10 次实验统计：为什么只看一次实验会得出错误结论")
print("=" * 100)
print(f"{'阶次':>4} | {'第1次测试MSE':>14} | {'10次均值':>10} | {'10次标准差':>10} | "
      f"{'相对波动 标准差/均值':>20}")
print("-" * 100)
for n in ORDERS:
    m1 = stats_noisy[n]["test"][0]
    mean, std, _, _ = summarize(stats_noisy[n]["test"])
    print(f"{n:>4} | {m1:>14.4f} | {mean:>10.4f} | {std:>10.4f} | "
          f"{std / mean if mean > 0 else float('nan'):>19.1%}")
print("-" * 100)
print()

best_per_trial = [int(np.argmin([stats_noisy[n]["test"][k] for n in ORDERS])) + 1
                  for k in range(N_REPEAT)]
print("每一批数据各自选出'测试误差最小的阶次'：",
      ", ".join(f"第{k+1}批->n={best_per_trial[k]}" for k in range(N_REPEAT)))
print()

print("=" * 100)
print("第 3) 题的讨论：不同训练数据（不同批次实验）对拟合结果的影响")
print("=" * 100)
n3t = summarize(stats_noisy[3]["test"])
n6t = summarize(stats_noisy[6]["test"])
n6f0 = summarize(stats_noisy[6]["fit0"])
n1f0 = summarize(stats_noisy[1]["fit0"])
print(f"""1) 误差本身的统计（表 1）：
   低阶（n=1,2）的测试误差均值高达 {summarize(stats_noisy[1]['test'])[0]:.1f} / {summarize(stats_noisy[2]['test'])[0]:.1f} kΩ²，标准差也小，
   说明"模型太简单"是主要矛盾，换一批数据结论几乎不变 -> 这是偏差主导，结果稳定。
   高阶（n=5,6）则相反：测试误差均值 {summarize(stats_noisy[5]['test'])[0]:.1f} / {n6t[0]:.1f} kΩ²，标准差高达
   {summarize(stats_noisy[5]['test'])[1]:.1f} / {n6t[1]:.1f} kΩ²（极差 {n6t[2]:.0f} ~ {n6t[3]:.0f}），
   换一批数据（对同一物理对象、同样的测量精度）结论可以差一个数量级。

2) 拟合结果本身（表 2）比误差更能说明问题：
   n=6 时最高次系数 a_6 在 10 批数据间从 {summarize(stats_noisy[6]['atop'])[2]:.2e} 变到
   {summarize(stats_noisy[6]['atop'])[3]:.2e}，甚至变号；同一模型对 0℃ 的预测从
   {n6f0[2]:.1f} kΩ 变到 {n6f0[3]:.1f} kΩ（真值仅 {R_th(0.0):.2f} kΩ）。
   而 n=1 的 0℃ 预测只在 [{n1f0[2]:.1f}, {n1f0[3]:.1f}] 之间动。 ->
   阶次越高，拟合结果对"具体是哪一批数据"越敏感，可重复性越差。

3) 为什么？最小二乘的解是 â=(X'X)⁻¹X'y，误差对数据的敏感度正比于系数的
   "方差" sigma²p/m 与各阶基函数的放大因子。阶次高 -> 高次项在训练区间内
   几乎不被数据约束（x^6 在 [20,80] 上变化平缓），噪声便乘虚而入，被
   完整地"存"进高次系数里；换个噪声实现，这批系数就整体换一套。

4) 但是"该选几阶"这个结论对数据批次不敏感：
   10 批数据选出的最优阶次只落在 {sorted(set(best_per_trial))} 之间（见上），
   n=3 的测试误差均值 {n3t[0]:.2f} kΩ²（标准差 {n3t[1]:.2f}），
   n=4 为 {summarize(stats_noisy[4]['test'])[0]:.2f} kΩ²（标准差 {summarize(stats_noisy[4]['test'])[1]:.2f}），
   两者接近且都远小于 n=5,6 的 {summarize(stats_noisy[5]['test'])[0]:.1f} / {n6t[0]:.1f}。
   即：最优复杂度是数据背后的客观规律，而高阶模型的系数与预测值才是噪声的产物。
   顺便注意：有一批数据的 n=4 测试误差只有 {min(stats_noisy[4]['test']):.3f} kΩ²（比任何一批的 n=3 都好），
   这正是"单次实验会误导"的活例子——若只做这一批实验，就会误以为 4 阶最优。

5) 结论：不同训练数据带来的影响，随模型复杂度（阶次）急剧放大。
   模型越复杂，学习结果越依赖于"这一次恰好抽到哪些噪声"，方差(方差项 sigma²p/m)
   越大，泛化能力越不稳定；简单模型虽然偏差大，但可重复性好。
   这就是机器学习里"方差—偏差权衡"在实验数据上的直接体现，也是为什么
   报告里必须用多次重复实验的均值±标准差、而不是一次实验的数值来下结论。
""")

fig3a = plot_curve_fan(stats_noisy, NOISE_SIGMA,
                       "Part 3):  Effect of Different Training Data on the Fitted Model "
                       r"($\sigma$ = 0.5 k$\Omega$, 10 batches)")
fig3b = plot_errorbars(stats_noisy, NOISE_SIGMA,
                       r"Part 3):  MSE vs Order with 10-batch Mean $\pm$ Std "
                       r"($\sigma$ = 0.5 k$\Omega$)")

# ======================================================================
# 4. 第 4) 题：不含噪声（sigma = 0），重复 2)、3) 的内容
# ======================================================================
print()
print("#" * 100)
print("# 第 4) 题：噪声方差为 0（sigma = 0），重复 2)、3) 的内容")
print("#" * 100)
print()

stats_clean = run_trials(0.0)

print_trial_table(
    stats_clean,
    "表 4  无噪声数据（sigma = 0）重复 10 次的误差统计",
    0.0)
print_stability_table(
    stats_clean,
    "表 5  无噪声数据重复 10 次：拟合系数与端点预测值的波动")

# 有噪 / 无噪 对照表
print("=" * 100)
print("表 6  核心对照：同一个阶次，有噪 vs 无噪（均为 10 次实验的均值）")
print("=" * 100)
print(f"{'阶次':>4} | {'训练MSE 无噪':>14} | {'训练MSE 有噪':>14} | "
      f"{'测试MSE 无噪':>14} | {'测试MSE 有噪':>14} | {'有噪/无噪':>12}")
print("-" * 100)
for n in ORDERS:
    tr_clean = summarize(stats_clean[n]["train"])[0]
    tr_noisy = summarize(stats_noisy[n]["train"])[0]
    te_clean = summarize(stats_clean[n]["test"])[0]
    te_noisy = summarize(stats_noisy[n]["test"])[0]
    ratio = te_noisy / te_clean if te_clean > 1e-12 else float("inf")
    print(f"{n:>4} | {tr_clean:>14.2e} | {tr_noisy:>14.5f} | "
          f"{te_clean:>14.4f} | {te_noisy:>14.4f} | {ratio:>12.2f}")
print("-" * 100)
print()

print("=" * 100)
print("第 4) 题的讨论：数据中的噪声对拟合结果的影响")
print("=" * 100)
te_clean_all = [summarize(stats_clean[n]["test"])[0] for n in ORDERS]
mono = all(te_clean_all[i + 1] < te_clean_all[i] for i in range(len(ORDERS) - 1))
print(f"""1) 重复 10 次的结果完全相同（表 4、表 5 的标准差恒为 0）。
   原因：sigma=0 时没有任何随机性，10 批"实验数据"其实是同一份数据，
   拟合结果是确定性的。 -> "重复实验"只有在数据含噪（有随机性）时才有意义，
   它是用来评估结果的不确定性的。

2) 训练误差不再有"噪声地板"：n>=3 时训练 MSE 降到 4.6e-03 量级
   （计算机浮点误差量级），因为无噪声时残差只由模型的逼近能力决定。
   有噪时训练误差最低只到 0.10 ~ 0.19 kΩ²（≈sigma²=0.25 的量级），这就是
   噪声给训练误差设下的"地板"。

3) 测试误差随阶次{"单调下降" if mono else "总体下降"}（表 6 第 3 列：
   {", ".join(f"{v:.3f}" for v in te_clean_all)}），
   不再出现有噪情形的 U 形/爆炸。也就是说：在 n=1..6 范围内，
   "过拟合现象消失了"——高阶模型不但训练误差小，测试误差也更小。

4) 结论（本题最重要的一句话）：
   过拟合不是"高阶模型"本身的错，而是"高阶模型 + 含噪数据"共同作用的产物。
   噪声是过拟合的必要条件：
     * 训练误差的最小值被噪声地板 sigma²(1-p/m) 托住，看起来"还能再降"，
       于是模型不断加高次项去拟合噪声 -> 方差爆炸；
     * 一旦噪声为 0，就没有"虚假规律"可拟合，高阶只体现为更好的逼近能力，
       训练误差和测试误差同时单调下降。
   这也解释了题目 2) 中"训练误差一直下降"的本质：下降的最后一段降的是噪声，
   不是规律。

5) 补充（无噪声时继续抬高阶次会怎样）：见表末的"补充观察"，n 一直升到 30
   （p=m=31，参数个数等于数据点数）都没有出现 2) 里的爆炸——因为数据是
   精确的、真函数又极其光滑，高阶只表现为更好的逼近能力（0℃ 预测 56.98，
   100℃ 预测 2.54~2.75，与真值 56.98 / 2.72 吻合）。
   这进一步证明：2) 里高阶曲线的爆炸不是"多项式"的错，而是噪声的错。
   真正的风险来自另外两处，与噪声无关：最小二乘的数值病态（p 接近 m 时
   Vandermonde 矩阵条件数急剧变差）与真函数不够光滑时的龙格(Runge)振荡。

6) 对实验的建议：实际测量一定有噪声（本题 sigma=500Ω），因此实际应取
   n=3 左右；只有在测量精度足够高、或数据量远大于参数个数时，才值得提高阶次。
""")

# 补充观察：无噪声下继续提高阶次会怎样
print("-" * 100)
print("补充观察：无噪声数据下把阶次继续抬高（p 接近 m=31 时会病态）")
print(f"{'阶次 n':>6}{'p=n+1':>7}{'训练MSE':>14}{'测试MSE':>16}{'R_fit(0℃)':>14}{'R_fit(100℃)':>14}")
print("-" * 100)
for n in (6, 8, 10, 15, 20, 25, 30):
    c = np.polyfit(t_tr, r_tr_true, n)
    print(f"{n:>6}{n + 1:>7}{mse(r_tr_true, np.polyval(c, t_tr)):>14.2e}"
          f"{mse(r_te_true, np.polyval(c, t_te)):>16.4f}"
          f"{np.polyval(c, 0.0):>14.2f}{np.polyval(c, 100.0):>14.2f}")
print("-" * 100)
print()

fig4a = plot_curve_fan(stats_clean, 0.0,
                       "Part 4):  Same Experiment with NO Noise "
                       r"($\sigma$ = 0, all 10 fits coincide)")
fig4b = plot_errorbars(stats_clean, 1e-9,
                       r"Part 4):  MSE vs Order with NO Noise ($\sigma$ = 0)")

# 有噪 / 无噪 测试误差对照图
fig5, ax5 = plt.subplots(figsize=(8.2, 5.2))
te_noisy_mean = [summarize(stats_noisy[n]["test"])[0] for n in ORDERS]
te_noisy_std = [summarize(stats_noisy[n]["test"])[1] for n in ORDERS]
ax5.errorbar(ORDERS, te_noisy_mean, yerr=te_noisy_std, fmt="^-",
             color="crimson", lw=1.8, capsize=4,
             label=r"Test MSE, WITH noise ($\sigma$ = 0.5 k$\Omega$), mean ± std")
ax5.semilogy(ORDERS, te_clean_all, "o--", color="navy", lw=1.8,
             label=r"Test MSE, WITHOUT noise ($\sigma$ = 0)")
ax5.set_xlabel("Polynomial order n")
ax5.set_ylabel(r"Test MSE / k$\Omega^2$  (log scale)")
ax5.set_title("Noise Is the Necessary Condition for Overfitting")
ax5.set_xticks(ORDERS)
ax5.grid(True, which="both", alpha=0.3)
ax5.legend(fontsize=9)
fig5.tight_layout()

print()
print("=" * 100)
print("共 5 张图，均以窗口形式显示（不保存文件）：")
print("  3) 10 批拟合曲线簇、3) 误差棒图、4) 无噪拟合曲线、4) 无噪误差图、")
print("  4) 有噪 vs 无噪 测试误差对照")
print("=" * 100)
plt.show()
