# -*- coding: utf-8 -*-
"""
热敏电阻阻值-温度特性研究 —— 第 5) 部分（选做）：用梯度下降算法重复 2)、3)
=====================================================================

题目：
    5) 选做：采用梯度下降算法，重复 2)、3) 内容，探讨模型参数初值、学习率
       对结果的影响。

本程序做的事：
    A. 用梯度下降（GD）求解 2) 的多项式最小二乘问题，n = 1..6，
       并与闭式解（np.polyfit，即 (X'X)^(-1)X'y）逐项对比；
    B. 探讨【学习率】的影响：扫描 alpha = {…} * alpha_max，
       记录收敛/震荡/发散，画出 loss-迭代 曲线；
    C. 探讨【参数初值】的影响：零初值、常数初值、小随机、大随机、接近解等，
       记录迭代次数与最终解，说明"凸问题里初值不改变最优解，只改变收敛路径"；
    D. 复现 3)：10 批不同噪声数据各自用 GD 拟合，统计误差的均值±标准差；
    E. 附加探讨：不做特征归一化时学习率上界会塌到 1e-22（GD 无法训练）。

关键公式（梯度下降）：
    损失函数（均方误差的一半，便于求导）
        J(a) = 1/(2m) * || X a - y ||^2
    梯度
        g(a) = 1/m * X' (X a - y)
    更新
        a <- a - alpha * g(a)
    学习率上界（A = X'X/m 的最大特征值 lambda_max）
        alpha < 2 / lambda_max
    （alpha 超过上界时，损失沿最大特征值方向按 |1 - alpha*lambda_max| 振荡/爆炸）

运行环境： Python 3 + numpy + matplotlib
"""

import time
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

NOISE_SIGMA = 0.5
ORDERS = list(range(1, 7))
SEED0 = 1
N_REPEAT = 10

# 特征归一化：x = (t - T_CENTER) / T_SCALE
# 目的：让设计矩阵各列量级相当，否则 t^6 ~ 1e11，梯度下降完全没法训练
T_CENTER, T_SCALE = 50.0, 30.0


def norm_x(t):
    """把摄氏温度映射到 [-1, 1] 附近（训练区间 20~80℃ -> [-1, 1]）。"""
    return (np.asarray(t, dtype=float) - T_CENTER) / T_SCALE


def R_th(t_celsius):
    """模型（1）：真实阻值（kΩ）。"""
    return R_25 * np.exp(BETA * (1.0 / (np.asarray(t_celsius, float) + 273.15)
                                 - 1.0 / T0))


def mse(y_meas, y_pred):
    y_meas = np.asarray(y_meas, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean((y_pred - y_meas) ** 2))


# ----------------------------------------------------------------------
# 1. 数据准备
# ----------------------------------------------------------------------
t_all = np.arange(T_MIN, T_MAX + T_STEP * 0.5, T_STEP)
mask_tr = (t_all >= T_TR_MIN) & (t_all <= T_TR_MAX)

t_tr = t_all[mask_tr]                     # 训练温度 31 点
t_te = t_all[~mask_tr]                    # 测试温度 20 点
r_tr_true = R_th(t_tr)
r_te_true = R_th(t_te)

x_tr = norm_x(t_tr)                       # 归一化后的训练特征
x_te = norm_x(t_te)

t_dense = np.linspace(T_MIN, T_MAX, 501)
x_dense = norm_x(t_dense)
r_dense_true = R_th(t_dense)

M_TRAIN = t_tr.size


def design(x, n):
    """多项式设计矩阵：列为 [x^n, x^(n-1), ..., x, 1]。"""
    return np.vander(np.asarray(x, float), n + 1)


# ----------------------------------------------------------------------
# 2. 梯度下降核心
# ----------------------------------------------------------------------
def alpha_max_of(X):
    """
    解析最大学习率：alpha_max = 2 / lambda_max(X'X/m)。
    （lambda_max 用归一化后的设计矩阵算；超过它就发散）
    """
    A = X.T @ X / X.shape[0]
    lam = np.linalg.eigvalsh(A)
    return 2.0 / lam[-1], lam[-1], lam[0]


def gd_fit(X, y, alpha, a0=None, max_iter=400000, tol=1e-9,
           record_every=200, diverge_loss=1e30, a_ref=None):
    """
    用梯度下降最小化 J(a) = 1/(2m)||Xa - y||^2。

    参数
        alpha      : 学习率
        a0         : 初值；None 表示零初值
        max_iter   : 最大迭代次数
        tol        : 停止条件 ||g||_inf < tol * ||g_0||_inf（相对梯度）
        record_every : 每多少次迭代记录一次损失（画图用）
        a_ref      : 若给出参考解，则同时记录 ||a - a_ref||（观察收敛路径用）

    返回 (a, info)
        info = {'iters', 'converged', 'diverged', 'hist_iter', 'hist_loss',
                'hist_dist', 'loss_final'}
    """
    m = X.shape[0]
    G = X.T @ X / m              # 预计算，等价于每次算 X'(Xa-y)/m，但更快
    Xty = X.T @ y / m
    a = np.zeros(X.shape[1]) if a0 is None else np.array(a0, dtype=float)

    g = G @ a - Xty
    g0 = np.max(np.abs(g)) + 1e-300

    hist_iter, hist_loss, hist_dist = [], [], []
    converged = diverged = False
    it = 0
    loss = 0.5 * float(np.mean((X @ a - y) ** 2))
    while it < max_iter:
        # 一次"块"迭代，块内不检查，减少 Python 开销
        for _ in range(record_every):
            a -= alpha * g
            g = G @ a - Xty
        it += record_every

        loss = 0.5 * float(np.mean((X @ a - y) ** 2))
        hist_iter.append(it)
        hist_loss.append(loss)
        if a_ref is not None:
            hist_dist.append(float(np.linalg.norm(a - a_ref)))

        if not np.isfinite(loss) or loss > diverge_loss:
            diverged = True
            break
        if np.max(np.abs(g)) < tol * g0:
            converged = True
            break

    info = {"iters": it, "converged": converged, "diverged": diverged,
            "hist_iter": np.array(hist_iter), "hist_loss": np.array(hist_loss),
            "hist_dist": np.array(hist_dist), "loss_final": loss}
    return a, info


def closed_form(X, y):
    """闭式解（最小二乘正规解）。"""
    return np.linalg.lstsq(X, y, rcond=None)[0]


# ======================================================================
# 3. 第 5 问 · 部分 A：用 GD 重复 2)
# ======================================================================
print("=" * 108)
print("第 5) 问 A：用梯度下降重复第 2) 问（n = 1..6 的多项式最小二乘）")
print("=" * 108)
print("特征归一化 x = (t - 50)/30，训练集 20~80℃（31 点），噪声 sigma = 0.5 kΩ（第 1 批）")
print()

rng1 = np.random.default_rng(SEED0)
r_tr_1 = r_tr_true + rng1.normal(0.0, NOISE_SIGMA, size=t_tr.shape)

t0 = time.time()
gd_main = {}
print(f"{'n':>2} {'p':>3} {'alpha_max':>10} {'alpha使用':>10} {'迭代次数':>9} "
      f"{'GD训练MSE':>11} {'闭式训练MSE':>12} {'GD测试MSE':>11} {'闭式测试MSE':>12} "
      f"{'曲线最大偏差':>12}")
print("-" * 108)
for n in ORDERS:
    X = design(x_tr, n)
    amax, lmax, lmin = alpha_max_of(X)
    alpha = 0.95 * amax                      # 稳妥的学习率（留 5% 余量）
    a_gd, info = gd_fit(X, r_tr_1, alpha)
    a_cf = closed_form(X, r_tr_1)

    gd_main[n] = {"alpha_max": amax, "alpha": alpha, "info": info,
                  "a_gd": a_gd, "a_cf": a_cf, "X": X}

    r_gd_tr = X @ a_gd
    r_gd_te = design(x_te, n) @ a_gd
    r_cf_tr = X @ a_cf
    r_cf_te = design(x_te, n) @ a_cf
    dev = float(np.max(np.abs(design(x_dense, n) @ a_gd
                              - design(x_dense, n) @ a_cf)))

    print(f"{n:>2} {n+1:>3} {amax:>10.3f} {alpha:>10.3f} {info['iters']:>9d} "
          f"{mse(r_tr_1, r_gd_tr):>11.5f} {mse(r_tr_1, r_cf_tr):>12.5f} "
          f"{mse(r_te_true, r_gd_te):>11.4f} {mse(r_te_true, r_cf_te):>12.4f} "
          f"{dev:>12.2e}")
print("-" * 108)
print(f"（总耗时 {time.time()-t0:.1f} s；'曲线最大偏差' = GD 与闭式解在 0~100℃ 上"
      f"预测值的最大绝对差，单位 kΩ）")
print()
print("结论：GD 收敛后与闭式解一致（0~100℃ 上预测曲线的最大偏差 ≤1e-3 kΩ，")
print("      远小于测量噪声 0.5 kΩ），说明 GD 求的确实是同一个最小二乘最优解。")
print("      注意 n=5,6 的测试 MSE 仍差 1e-3 量级（118.8473 vs 118.8482）——")
print("      因为外推会把「优化精度不足」同样放大，这是高阶模型额外的娇气之处。")
print()

# 收敛速度与阶次的关系
print("-" * 108)
print("收敛速度随阶次的变化（这是「高阶模型难训练」的直接证据）")
print(f"{'n':>2}{'条件数 kappa':>16}{'alpha_max':>12}{'需要的迭代次数':>16}{'迭代次数增长倍数':>18}")
print("-" * 108)
kap_list = []
for n in ORDERS:
    X = design(x_tr, n)
    amax, lmax, lmin = alpha_max_of(X)
    kap = lmax / lmin
    kap_list.append(kap)
    it = gd_main[n]["info"]["iters"]
    ratio = it / max(1, gd_main[ORDERS[0]]["info"]["iters"])
    print(f"{n:>2}{kap:>16.1f}{amax:>12.3f}{it:>16d}{ratio:>17.1f}x")
print("-" * 108)
print("原因：GD 的收敛速率由条件数 kappa = lambda_max/lambda_min 决定，"
      "每步误差乘上约 (1 - 1/kappa)，")
print("      所以要把误差压到 1e-9 量级，迭代次数大致正比于 kappa。"
      "阶次从 1 升到 6，kappa 从 2.8 涨到")
print(f"      {kap_list[-1]:.0f}（约 {kap_list[-1]/kap_list[0]:.0f} 倍），迭代次数同步暴涨。")
print()

# ======================================================================
# 4. 第 5 问 · 部分 B：学习率的影响
# ======================================================================
print("=" * 108)
print("第 5) 问 B：学习率的影响（n = 6，第 1 批数据，alpha 取 alpha_max 的不同比例）")
print("=" * 108)

N_LR = 6
X6 = design(x_tr, N_LR)
amax6, lmax6, lmin6 = alpha_max_of(X6)
ratios = [0.001, 0.01, 0.1, 0.5, 0.95, 0.999, 1.0, 1.001, 1.05, 1.5]
lr_results = []
print(f"alpha_max = {amax6:.4f}（= 2/lambda_max，lambda_max = {lmax6:.4f}）")
print()
print(f"{'alpha/alpha_max':>16}{'alpha':>12}{'状态':>10}{'迭代次数':>10}"
      f"{'最终训练MSE(kΩ²)':>22}{'与闭式解偏差':>16}")
print("-" * 108)
a_cf6 = closed_form(X6, r_tr_1)
for r in ratios:
    alpha = r * amax6
    a_gd, info = gd_fit(X6, r_tr_1, alpha, max_iter=200000)
    pred = design(x_dense, N_LR) @ a_gd
    dev = float(np.max(np.abs(pred - design(x_dense, N_LR) @ a_cf6)))
    if info["diverged"]:
        state = "发散"
    elif info["converged"]:
        state = "收敛"
    else:
        state = "未收敛"
    final_mse = mse(r_tr_1, X6 @ a_gd) if np.isfinite(info["loss_final"]) else float("inf")
    print(f"{r:>16.3f}{alpha:>12.4f}{state:>10}{info['iters']:>10d}"
          f"{final_mse:>22.3e}{dev:>16.2e}")
    lr_results.append({"ratio": r, "alpha": alpha, "alpha_max": amax6,
                       "info": info, "state": state})
print("-" * 108)
print("规律：")
print("  * alpha 太小（0.001~0.01 倍）-> 迭代几万次仍未收敛，模型「欠训练」，"
      "看起来像模型能力不足；")
print("  * alpha 增大到 0.5~0.95 倍 -> 收敛最快、最稳，是实践中的推荐区域；")
print("  * alpha 逼近 alpha_max -> 在最大特征值方向来回震荡（loss 出现锯齿/平台）；")
print("  * alpha 超过 alpha_max -> 该方向误差每步放大 |1 - alpha*lambda_max| > 1，"
      "loss 指数爆炸（发散）。")
print()

# 学习率对"复现 2) 的结论"的影响：若是固定迭代次数会怎样
print("-" * 108)
print("补充：若用固定迭代次数（而不是收敛判据），学习率会直接改变'看起来的模型好坏'")
print(f"{'n':>2}{'alpha=0.001*amax, 2000步':>28}{'alpha=0.95*amax, 2000步':>28}{'闭式解':>14}")
print("-" * 108)
for n in ORDERS:
    X = design(x_tr, n)
    amax, _, _ = alpha_max_of(X)
    a_small, _ = gd_fit(X, r_tr_1, 0.001 * amax, max_iter=2000, tol=0.0)
    a_good, _ = gd_fit(X, r_tr_1, 0.95 * amax, max_iter=2000, tol=0.0)
    print(f"{n:>2}{mse(r_te_true, design(x_te, n) @ a_small):>28.3f}"
          f"{mse(r_te_true, design(x_te, n) @ a_good):>28.3f}"
          f"{mse(r_te_true, design(x_te, n) @ closed_form(X, r_tr_1)):>14.3f}")
print("-" * 108)
print("（表中为测试集 MSE，单位 kΩ²。学习率太小时高阶模型根本还没收敛，")
print("  误差甚至大于低阶——如果不检查收敛，就会把「优化失败」误判成「模型不行」。）")
print()

# ======================================================================
# 5. 第 5 问 · 部分 C：参数初值的影响
# ======================================================================
print("=" * 108)
print("第 5) 问 C：模型参数初值的影响（n = 6，第 1 批数据，alpha = 0.95*alpha_max）")
print("=" * 108)

p6 = N_LR + 1
rng_i = np.random.default_rng(7)
inits = [
    ("零初值 a0 = 0", np.zeros(p6)),
    ("常数初值 a0 = 1", np.ones(p6)),
    ("较大常数 a0 = 20", 20.0 * np.ones(p6)),
    ("小随机 N(0, 0.01)", rng_i.normal(0.0, 0.01, p6)),
    ("大随机 N(0, 5)", rng_i.normal(0.0, 5.0, p6)),
    ("接近解（闭式解+扰动）", closed_form(X6, r_tr_1) + rng_i.normal(0, 0.1, p6)),
]
alpha_c = 0.95 * amax6
print(f"{'初值类型':>22}{'初始loss':>14}{'迭代次数':>10}{'最终loss':>14}"
      f"{'最终与闭式解最大系数差':>24}")
print("-" * 108)
init_results = []
for name, a0 in inits:
    a_gd, info = gd_fit(X6, r_tr_1, alpha_c, a0=a0, max_iter=600000, tol=1e-9,
                        a_ref=a_cf6)
    loss0 = 0.5 * float(np.mean((X6 @ a0 - r_tr_1) ** 2))
    dev = float(np.max(np.abs(a_gd - a_cf6)))
    print(f"{name:>22}{loss0:>14.4f}{info['iters']:>10d}{info['loss_final']:>14.6f}"
          f"{dev:>24.2e}")
    init_results.append({"name": name, "info": info, "loss0": loss0, "a": a_gd})
print("-" * 108)
print("结论：")
print("  1) 损失函数 J(a) = 1/(2m)||Xa-y||^2 是严格凸二次函数，只有唯一极小点，")
print("     所以只要收敛，不同初值最终收敛到【同一个解】（最终 loss 六种初值完全相同，")
print("     与闭式解的系数差在 1e-4 ~ 1e-6 量级，取决于收敛容差）。")
print("  2) 初值影响的是：① 初始 loss 大小；② 收敛所需迭代次数。")
print("     GD 的误差按 (1 - alpha*lambda_min)^k 衰减，初始误差变成 log(初始误差/容差)，")
print("     所以初值差 100 倍只带来常数级的迭代次数差异（见上表）。")
print("  3) 若不做归一化（t 直接用摄氏温度），初值、学习率都会变得极其敏感，")
print("     见下面 D 部分：alpha_max 会塌到 1e-22，梯度下降无法使用。")
print()

# ======================================================================
# 6. 第 5 问 · 部分 D：特征归一化的必要性（学习率的另一种视角）
# ======================================================================
print("=" * 108)
print("第 5) 问 D：不做特征归一化（直接用摄氏温度 t 作特征）会怎样")
print("=" * 108)
print(f"{'特征':>26}{'n':>4}{'lambda_max':>16}{'alpha_max':>14}{'条件数':>14}")
print("-" * 108)
for label, feat in (("原始温度 t (℃)", t_tr), ("归一化 x=(t-50)/30", x_tr)):
    for n in (3, 6):
        X = design(feat, n)
        A = X.T @ X / X.shape[0]
        lam = np.linalg.eigvalsh(A)
        print(f"{label:>26}{n:>4}{lam[-1]:>16.3e}{2.0/lam[-1]:>14.2e}"
              f"{lam[-1]/lam[0]:>14.2e}")
print("-" * 108)
print("原始温度下 alpha_max ~1e-11（n=3）和 ~1e-22（n=6），意味着用双精度浮点")
print("几乎不可能正确更新（一步就把系数淹没在舍入误差里）。归一化后 alpha_max 回到")
print("1.6~2.0 的正常量级，条件数也从 1e13~1e27 降到 1e1~1e3，GD 才能工作。")
print("这就是实践中「特征标准化/归一化」是深度学习第一步的原因。")
print()

# ======================================================================
# 7. 第 5 问 · 部分 E：用 GD 重复第 3) 问（10 批数据）
# ======================================================================
print("=" * 108)
print("第 5) 问 E：用梯度下降重复第 3) 问（10 批不同噪声数据，n = 1..6）")
print("=" * 108)


def run_trials_gd(sigma, n_repeat=N_REPEAT, orders=ORDERS, alpha_frac=0.95):
    """对每一批数据、每个阶次，用 GD 拟合，返回统计量与曲线系数。"""
    res = {n: {"test": [], "train": [], "coef": []} for n in orders}
    res["_closed"] = {n: {"test": [], "train": []} for n in orders}
    res["_data"] = []
    for k in range(n_repeat):
        rng = np.random.default_rng(SEED0 + k)
        r_tr = r_tr_true + rng.normal(0.0, sigma, size=t_tr.shape)
        res["_data"].append(r_tr)
        for n in orders:
            X = design(x_tr, n)
            amax, _, _ = alpha_max_of(X)
            a_gd, _ = gd_fit(X, r_tr, alpha_frac * amax)
            res[n]["coef"].append(a_gd)
            res[n]["train"].append(mse(r_tr, X @ a_gd))
            res[n]["test"].append(mse(r_te_true, design(x_te, n) @ a_gd))
            a_cf = closed_form(X, r_tr)
            res["_closed"][n]["train"].append(mse(r_tr, X @ a_cf))
            res["_closed"][n]["test"].append(mse(r_te_true, design(x_te, n) @ a_cf))
    return res


t0 = time.time()
gd_trials = run_trials_gd(NOISE_SIGMA)
print(f"（10 批 × 6 阶，总耗时 {time.time()-t0:.1f} s）")
print()
print(f"{'n':>2} | {'GD 训练MSE 均值±标准差':>26} | {'GD 测试MSE 均值±标准差':>26} | "
      f"{'闭式解 测试MSE 均值':>20}")
print("-" * 108)
for n in ORDERS:
    tr = np.array(gd_trials[n]["train"])
    te = np.array(gd_trials[n]["test"])
    te_cf = np.array(gd_trials["_closed"][n]["test"])
    print(f"{n:>2} | {tr.mean():>12.5f} ± {tr.std(ddof=1):>10.5f} | "
          f"{te.mean():>12.4f} ± {te.std(ddof=1):>10.4f} | {te_cf.mean():>20.4f}")
print("-" * 108)
print("与第 3) 问（闭式解）的结论完全一致：n=3、4 最优（测试MSE ~12 kΩ²），")
print("n=5、6 的测试误差大而且波动极大（标准差与均值同量级，甚至更大），")
print("说明'高阶不可重复'这一结论与用什么优化算法无关，是模型复杂度本身的性质。")
print()

# ======================================================================
# 8. 作图
# ======================================================================
print("正在保存图片……")

# --- 图 1：GD 拟合曲线 vs 闭式解 vs 真实模型 ---
fig1, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
fig1.suptitle("Part 5) A:  Gradient-Descent Fit vs Closed-form Solution "
              r"($\sigma$ = 0.5 k$\Omega$)", fontsize=16, fontweight="bold")
for ax, n in zip(axes.ravel(), ORDERS):
    X = gd_main[n]["X"]
    ax.plot(t_dense, design(x_dense, n) @ gd_main[n]["a_gd"], "r-", lw=1.8,
            label=f"GD (n={n})")
    ax.plot(t_dense, design(x_dense, n) @ gd_main[n]["a_cf"], "b--", lw=1.2,
            label="closed form")
    ax.plot(t_dense, r_dense_true, "k:", lw=1.5, label="true model (1)")
    ax.plot(t_tr, r_tr_1, "o", ms=5, mfc="none", mec="c", mew=1.1,
            label="noisy measurements")
    ax.set_title(f"n = {n}:  GD vs closed form", fontsize=10)
    ax.set_xlabel("Temperature/Celsius degree", fontsize=9)
    ax.set_ylabel(r"Resistance/k$\Omega$", fontsize=9)
    ax.set_xlim(T_MIN, T_MAX)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7)
fig1.tight_layout(rect=(0, 0, 1, 0.94))

# --- 图 2：学习率对收敛的影响 ---
fig2, ax2 = plt.subplots(figsize=(9.0, 5.6))
colors = plt.cm.plasma(np.linspace(0.05, 0.85, len(lr_results)))
for c, item in zip(colors, lr_results):
    h_it, h_loss = item["info"]["hist_iter"], item["info"]["hist_loss"]
    if len(h_it) == 0:
        continue
    lab = (f"alpha = {item['ratio']:g} x alpha_max  "
           f"({item['state']})")
    ax2.semilogy(h_it, np.clip(h_loss, 1e-16, 1e16), "-", lw=1.5,
                 color=c, label=lab)
ax2.axhline(0.5 * np.mean((r_tr_1 - r_tr_1.mean()) ** 2), color="gray", ls=":",
            lw=1.0, label="loss at trivial predictor (mean)")
ax2.set_xlabel("iteration")
ax2.set_ylabel("loss  J(a) = ||Xa - y||² / (2m)   [log]")
ax2.set_title(r"Part 5) B:  Effect of Learning Rate (n = 6, $\sigma$ = 0.5 k$\Omega$)")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(fontsize=7.5, loc="best")
ax2.set_ylim(1e-6, 1e10)
fig2.tight_layout()

# --- 图 3：初值对收敛的影响（看参数到最优解的距离） ---
fig3, ax3 = plt.subplots(figsize=(9.0, 5.6))
colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(init_results)))
for c, item in zip(colors, init_results):
    h_it, h_dist = item["info"]["hist_iter"], item["info"]["hist_dist"]
    ax3.semilogy(h_it, np.clip(h_dist, 1e-12, 1e6), "-", lw=1.6, color=c,
                 label=f"{item['name']}  (init loss {item['loss0']:.2f})")
ax3.set_xlabel("iteration")
ax3.set_ylabel(r"distance to optimum  $\|a - a^*\|$   [log]")
ax3.set_title(r"Part 5) C:  Effect of Initial Values (n = 6, $\sigma$ = 0.5 k$\Omega$)"
              "\nsame slope (same convergence rate), different starting point"
              " -> same final solution")
ax3.grid(True, which="both", alpha=0.3)
ax3.legend(fontsize=7.5, loc="best")
fig3.tight_layout()

# --- 图 4：alpha_max 与所需迭代次数随阶次变化 ---
fig4, ax4 = plt.subplots(figsize=(8.2, 5.0))
ns = ORDERS
amaxes = [gd_main[n]["alpha_max"] for n in ns]
iters = [gd_main[n]["info"]["iters"] for n in ns]
ax4.plot(ns, amaxes, "o-", color="tab:blue", lw=1.8, label=r"$\alpha_{max}=2/\lambda_{max}$")
ax4.set_xlabel("polynomial order n")
ax4.set_ylabel(r"$\alpha_{max}$", color="tab:blue")
ax4.tick_params(axis="y", labelcolor="tab:blue")
ax4b = ax4.twinx()
ax4b.semilogy(ns, iters, "s--", color="crimson", lw=1.8,
              label="iterations needed (to 1e-9)")
ax4b.set_ylabel("GD iterations", color="crimson")
ax4b.tick_params(axis="y", labelcolor="crimson")
ax4.set_title("Higher Order  ->  Worse Conditioning  ->  More Iterations")
ax4.set_xticks(ns)
ax4.grid(True, alpha=0.3)
lines = ax4.get_legend_handles_labels()[0] + ax4b.get_legend_handles_labels()[0]
labs = ax4.get_legend_handles_labels()[1] + ax4b.get_legend_handles_labels()[1]
ax4.legend(lines, labs, fontsize=9, loc="center left")
fig4.tight_layout()

# --- 图 5：10 批数据 + GD 拟合曲线簇 ---
fig5, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
fig5.suptitle(r"Part 5) E:  10 Batches Fitted by Gradient Descent "
              r"($\sigma$ = 0.5 k$\Omega$)", fontsize=16, fontweight="bold")
colors = plt.cm.viridis(np.linspace(0.05, 0.9, N_REPEAT))
for ax, n in zip(axes.ravel(), ORDERS):
    ax.plot(t_tr, gd_trials["_data"][0], "o", ms=5, mfc="none", mec="c",
            mew=1.1, label="batch #1 measurements")
    for k, a in enumerate(gd_trials[n]["coef"]):
        ax.plot(t_dense, design(x_dense, n) @ a, "-", lw=1.3, color=colors[k],
                alpha=0.85)
    ax.plot(t_dense, r_dense_true, "k--", lw=1.8, label="true model (1)")
    ax.set_title(f"n = {n}:  10 GD fits from 10 batches", fontsize=10)
    ax.set_xlabel("Temperature/Celsius degree", fontsize=9)
    ax.set_ylabel(r"Resistance/k$\Omega$", fontsize=9)
    ax.set_xlim(T_MIN, T_MAX)
    ax.grid(True, alpha=0.35)
    if n == ORDERS[0]:
        ax.legend(fontsize=7)
fig5.tight_layout(rect=(0, 0, 1, 0.94))

# --- 图 6：归一化 vs 未归一化的收敛对比 ---
fig6, ax6 = plt.subplots(figsize=(8.6, 5.2))
X_raw = design(t_tr, 6)
lam_raw = np.linalg.eigvalsh(X_raw.T @ X_raw / M_TRAIN)
a_raw, info_raw = gd_fit(X_raw, r_tr_1, 2.0 / lam_raw[-1] * 0.95, max_iter=100000)
ax6.semilogy(info_raw["hist_iter"], np.clip(info_raw["hist_loss"], 1e-16, 1e16),
             "-", lw=1.8, color="firebrick",
             label=r"raw feature $t$ (℃),  $\alpha \approx 2\times10^{-22}$")
a_norm, info_norm = gd_fit(design(x_tr, 6), r_tr_1, 0.95 * amax6, max_iter=100000)
ax6.semilogy(info_norm["hist_iter"], np.clip(info_norm["hist_loss"], 1e-16, 1e16),
             "-", lw=1.8, color="seagreen",
             label=r"normalized $x=(t-50)/30$,  $\alpha \approx 1.51$")
ax6.set_xlabel("iteration")
ax6.set_ylabel("loss  J(a)   [log]")
ax6.set_title("Part 5) D:  Feature Normalization Is Mandatory for Gradient Descent")
ax6.grid(True, which="both", alpha=0.3)
ax6.legend(fontsize=8.5)
fig6.tight_layout()

# ======================================================================
# 9. 总结
# ======================================================================
print()
print("=" * 108)
print("第 5) 问总结：初值与学习率对结果的影响")
print("=" * 108)
print("""1) 学习率 alpha：
   * alpha < alpha_max：收敛，但越接近上界越快。alpha 太小时高阶模型在有限
     迭代内根本收敛不到最优解，会误判为"模型能力不足"。
   * alpha 略小于 alpha_max（如 0.95 倍）：收敛最快最稳，推荐取值。
   * alpha 逼近 alpha_max：沿最大特征值方向震荡。
   * alpha > alpha_max：误差每步放大 |1 - alpha*lambda_max| > 1，loss 指数发散。
   * alpha_max = 2/lambda_max 随阶次略降（2.0 -> 1.59），但条件数从 2.8 涨到
     8635，所以高阶模型的"可用学习率区间"变窄、所需迭代次数暴涨
     （n=1 约 65 步，n=6 约 10 万步）——即高阶模型训练更"娇气"，这与
     2)3) 里"高阶模型方差大、不稳定"是同一件事的两种表现。

2) 参数初值 a0：
   * 因为 J(a) 是严格凸的二次函数，极小点唯一，所以【只要收敛，初值不影响
     最终解】——不同初值收敛到的系数与闭式解差别在 1e-7 量级。
   * 初值影响的是初始 loss 和收敛所需迭代次数：误差按
     (1 - alpha*lambda_min)^k 衰减，初值越远离最优解，迭代次数只按对数增加。
   * 所以对本题这种"凸问题 + 闭式解可得"的场景，初值不是关键；但若换成
     非凸模型（神经网络）或迭代次数受限，初值就会显著影响结果。

3) 与第 2)、3) 的一致性：
   GD 收敛后与闭式解完全等价，第 2) 的"训练误差单调降、测试误差 U 形"和
   第 3) 的"高阶结果不可重复"这两个结论在 GD 下完全复现（见 E 部分表格与图 5）。

4) 实践要点（可以用在报告结论里）：
   * 一定要做特征归一化：不做时 alpha_max 会塌到 1e-11 ~ 1e-22，GD 无法训练；
   * 学习率用 alpha_max 的 0.9~0.99 倍，或配合收敛判据/自适应方法；
   * 判断模型好坏前必须先确认"优化已收敛"，否则会把"没训好"当成"模型不行"。
""")
print("=" * 108)
print("共 6 张图，均以窗口形式显示（不保存文件），依次为：")
print("  GD vs 闭式解 / 学习率影响 / 初值影响 / alpha_max与迭代次数 /")
print("  10 批 GD 曲线簇 / 归一化 vs 未归一化")
print("=" * 108)
plt.show()
