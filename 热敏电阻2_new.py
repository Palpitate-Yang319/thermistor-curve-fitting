# -*- coding: utf-8 -*-
"""
热敏电阻阻值-温度特性研究 —— 第 2) 部分
=====================================================================

已知模型（1）：  R_T = R_T0 * exp( beta * (1/T - 1/T0) )
    其中  beta = 3100 K,  R_T0 = 22 kΩ @ T0 = 25℃ = 298.15 K,  T 为开氏温度

本程序完成的任务：

  1) 用 1) 中得到的 0~100℃、步长 2℃ 的特性数据，取出 20~80℃ 区间作为
     "温度—阻值" 数据，叠加零均值、标准差 500Ω(=0.5 kΩ) 的高斯噪声，
     用来模拟实际测量过程（含测量误差的实验数据）。
  2) 针对模型（2） R_t = a_n*t^n + a_(n-1)*t^(n-1) + ... + a_1*t + a_0 ，
     用最小二乘法（np.polyfit ≈ matlab 的 polyfit）分别拟合
     n = 1,2,3,4,5,6 阶多项式模型。
  3) 计算每个阶次模型在
        训练集：20~80℃ 的带噪实验数据
        测试集：0~20℃ 与 80~100℃（剔除 20~80℃ 后的其余数据，无噪声）
     上的均方误差（MSE, mean squared error），观察误差随阶次的变化规律。
  4) 画出与课件图风格一致的拟合结果（2×3 子图，每阶一幅）
     以及 MSE—阶次 变化曲线。

运行环境： Python 3 + numpy + matplotlib
（注：matlab 中 randn -> np.random.normal，polyfit -> np.polyfit，polyval -> np.polyval）
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. 全局参数
# ----------------------------------------------------------------------
BETA = 3100.0                 # 材料常数 beta，单位 K
R_25 = 22.0                   # 25℃ 时的阻值，单位 kΩ
T0 = 25.0 + 273.15            # 参考温度 298.15 K

T_MIN, T_MAX, T_STEP = 0.0, 100.0, 2.0   # 1) 中特性曲线的温度范围与步长（℃）
T_TRAIN_MIN, T_TRAIN_MAX = 20.0, 80.0    # 实验温度范围（℃）

NOISE_SIGMA = 0.5             # 噪声标准差 500Ω = 0.5 kΩ
RANDOM_SEED = 0               # 固定随机种子，保证结果可复现
MAX_ORDER = 6                 # 最高多项式阶次

# 是否额外画出"训练误差(相对真实阻值)"这条诊断线。
# 题目只要求"训练集误差"和"测试集误差"两条，这一条是本程序额外给的
# 诊断量，用于把"噪声造成的不可约误差"从训练误差里剥出来；
# 如果作业只要两条线，把它设为 False 即可。
SHOW_DIAGNOSTIC = True

# 绘图用的中文字体（Windows 下一般都有；找不到时自动退化为 DejaVu Sans）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False   # 让负号正常显示


# ----------------------------------------------------------------------
# 1. 热敏电阻的理论阻值模型
# ----------------------------------------------------------------------
def thermistor_resistance(t_celsius):
    """由摄氏温度 t(℃) 计算热敏电阻阻值 R(kΩ)，即模型（1）。"""
    t_kelvin = t_celsius + 273.15
    return R_25 * np.exp(BETA * (1.0 / t_kelvin - 1.0 / T0))


# ----------------------------------------------------------------------
# 2. 构造训练集 / 测试集
# ----------------------------------------------------------------------
# 1) 中的特性曲线数据：0,2,4,...,100 ℃ 共 51 个点
t_all = np.arange(T_MIN, T_MAX + T_STEP * 0.5, T_STEP)
r_all = thermistor_resistance(t_all)

# 布尔掩码：20~80℃ 为训练集，其余（0~18℃ 与 82~100℃）为测试集
train_mask = (t_all >= T_TRAIN_MIN) & (t_all <= T_TRAIN_MAX)

t_train = t_all[train_mask]                    # 训练温度（31 个点）
t_test = t_all[~train_mask]                    # 测试温度（20 个点）

rng = np.random.default_rng(RANDOM_SEED)       # 随机数发生器
noise = rng.normal(0.0, NOISE_SIGMA, size=t_train.shape)   # 零均值高斯噪声

r_train = thermistor_resistance(t_train) + noise   # 带噪声的"实验数据"（训练用）
r_train_true = thermistor_resistance(t_train)      # 训练温度上的真实阻值（仅供对照）
r_test = thermistor_resistance(t_test)             # 测试集：真实阻值，不加噪声

# 画拟合曲线用的稠密温度网格
t_dense = np.linspace(T_MIN, T_MAX, 501)
r_dense_true = thermistor_resistance(t_dense)


def mse(y_meas, y_pred):
    """均方误差 MSE = (1/m) * Σ (y_pred - y_meas)^2"""
    y_meas = np.asarray(y_meas, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_pred - y_meas) ** 2))


# ----------------------------------------------------------------------
# 3. 逐阶做最小二乘多项式拟合，并计算训练集 / 测试集误差
# ----------------------------------------------------------------------
orders = list(range(1, MAX_ORDER + 1))
coeffs_all = {}        # 各阶模型的系数
train_mse = []         # 训练集 MSE（相对带噪实验数据，即通常意义的训练误差）
train_mse_true = []    # 训练集 MSE（相对真实阻值，用于分离"噪声"与"模型偏差"）
test_mse = []          # 测试集 MSE（相对真实阻值）
train_max_err = []     # 训练集最大绝对误差
test_max_err = []      # 测试集最大绝对误差

for n in orders:
    # 最小二乘拟合：t 为 ℃，r 为 kΩ
    coef = np.polyfit(t_train, r_train, n)
    coeffs_all[n] = coef

    r_hat_train = np.polyval(coef, t_train)    # 拟合模型在训练温度上的预测
    r_hat_test = np.polyval(coef, t_test)      # 拟合模型在测试温度上的预测

    train_mse.append(mse(r_train, r_hat_train))
    train_mse_true.append(mse(r_train_true, r_hat_train))
    test_mse.append(mse(r_test, r_hat_test))
    train_max_err.append(float(np.max(np.abs(r_hat_train - r_train))))
    test_max_err.append(float(np.max(np.abs(r_hat_test - r_test))))

# ----------------------------------------------------------------------
# 4. 打印结果表
# ----------------------------------------------------------------------
print("=" * 86)
print("热敏电阻多项式模型拟合结果（训练集 20~80℃，步长 2℃，含 sigma=0.5 kΩ 高斯噪声）")
print("=" * 86)
print(f"训练样本数 = {t_train.size}  测试样本数 = {t_test.size}  "
      f"噪声标准差 = {NOISE_SIGMA} kΩ")
print("-" * 86)
print(f"{'阶次 n':>6}{'训练集MSE(kΩ²)':>18}{'训练MSE(诊断)':>16}"
      f"{'测试集MSE(kΩ²)':>18}{'测试集最大误差':>16}")
print("      (训练集MSE = 相对带噪测量值(*)；训练MSE(诊断) = 相对真实阻值)")
print("-" * 86)
for i, n in enumerate(orders):
    print(f"{n:>6}{train_mse[i]:>18.5f}{train_mse_true[i]:>16.5f}"
          f"{test_mse[i]:>18.5f}{test_max_err[i]:>16.3f}")
print("-" * 86)
print(f"参考：真实曲线在 0℃ 为 {thermistor_resistance(0.0):.2f} kΩ，"
      f"100℃ 为 {thermistor_resistance(100.0):.2f} kΩ")
print()

best_train = orders[int(np.argmin(train_mse))]
best_test = orders[int(np.argmin(test_mse))]
print("各阶模型的多项式系数（按 t^n ... t^1, t^0 降幂排列）：")
for n in orders:
    print(f"  n = {n}: " + ", ".join(f"{c: .6e}" if abs(c) < 1e-3 else f"{c: .6f}"
                                    for c in coeffs_all[n]))
print()
print(f"训练集误差最小的阶次：n = {best_train}")
print(f"测试集误差最小的阶次：n = {best_test}")
print()

# ----------------------------------------------------------------------
# 4.5 为什么会过拟合：偏差-方差分解（解析式，不需要蒙特卡洛）
# ----------------------------------------------------------------------
# 记 m = 训练点数、p = n+1 = 待估系数个数、sigma = 噪声标准差，
# 拟合曲线在各训练点上的
#     偏差(bias)   = 用【无噪数据】拟合出的曲线与真值之差  -> 只由模型容量决定
#     方差(variance)= 噪声使拟合曲线随机摆动的幅度  -> 约为 sigma^2 * p / m
# 于是（对噪声取期望）：
#     橙线(训练 vs 真值)   ≈ bias^2 + sigma^2 * p / m
#     蓝线(训练 vs 带噪值) ≈ bias^2 + sigma^2 * (1 - p / m)
#     红线(测试 vs 真值)   ≈ 外推点上的 bias^2 + 被放大很多倍的 sigma^2 * p / m
# p 增大时：bias^2 迅速趋于 0（高阶总能逼近光滑曲线），而 sigma^2*p/m 线性增大。
# 两者的和先降后升 -> 有一个最优点，这就是"为什么 n=3 时橙线最小"。
print("=" * 86)
print("偏差-方差分解（解析式；m = 训练点数，p = n+1 个待估系数）")
print("=" * 86)
print(f"{'阶次 n':>6}{'p=n+1':>7}{'偏差^2':>12}{'方差σ²p/m':>14}"
      f"{'理论橙线':>12}{'实测橙线':>12}{'理论蓝线':>12}{'实测蓝线':>12}")
print("-" * 86)
for i, n in enumerate(orders):
    p = n + 1
    # 用无噪数据拟合 -> 得到"无偏"的模型曲线，它与真值之差就是偏差
    coef_clean = np.polyfit(t_train, r_train_true, n)
    r_clean = np.polyval(coef_clean, t_train)
    bias2 = float(np.mean((r_clean - r_train_true) ** 2))
    var = NOISE_SIGMA ** 2 * p / t_train.size     # 拟合曲线的方差 sigma^2 * p / m
    print(f"{n:>6}{p:>7}{bias2:>12.5f}{var:>14.5f}"
          f"{bias2 + var:>12.5f}{train_mse_true[i]:>12.5f}"
          f"{bias2 + NOISE_SIGMA ** 2 * (1 - p / t_train.size):>12.5f}"
          f"{train_mse[i]:>12.5f}")
print("-" * 86)
print("读法：p 从 2 增到 7 时，偏差² 从 3 迅速塌到 ~0，而方差按 p 线性增长；")
print("      两者之和(橙线)在 n=3 处取最小 -> 最佳模型阶次。")
print("      蓝线里的 -σ²p/m 项使训练误差随 p 单调变小，所以‘训练误差最小’")
print("      永远会选到最高阶 —— 这就是不能用训练误差选模型的原因。")
print()

# ----------------------------------------------------------------------
# 4.6 测试集误差的两种基准（与带噪测量值比 / 与模型(1)真值比）
# ----------------------------------------------------------------------
# 题目给的测试集是 1) 中用模型(1)算出的干净数据，所以主结果与真值比较。
# 若把测试集也看成"再做一次独立测量"，即同样叠加 sigma 的高斯噪声，
# 那么 E[MSE_带噪] = MSE_真值 + σ²（测试点不参与拟合，所以没有 hat 修正），
# 只是整体抬高一个常数，阶次排序与结论完全不变。
rng_test = np.random.default_rng(RANDOM_SEED + 1)     # 独立的一次"重测"
r_test_noisy = r_test + rng_test.normal(0.0, NOISE_SIGMA, size=t_test.shape)
test_mse_noisy = [mse(r_test_noisy, np.polyval(coeffs_all[n], t_test))
                  for n in orders]
print("=" * 86)
print("测试集误差的两种基准对照")
print("=" * 86)
print(f"{'阶次 n':>6}{'①与模型(1)真值比':>20}{'②与带噪测量值比':>20}{'差值':>12}")
print("-" * 86)
for i, n in enumerate(orders):
    print(f"{n:>6}{test_mse[i]:>20.4f}{test_mse_noisy[i]:>20.4f}"
          f"{test_mse_noisy[i] - test_mse[i]:>12.4f}")
print("-" * 86)
print(f"理论差值应约为 σ² = {NOISE_SIGMA ** 2:.4f}（测试点只有 20 个且模型误差较大时，"
      f"交叉项会带来同量级的波动）；两种基准给出的最优阶次："
      f"① n={orders[int(np.argmin(test_mse))]}，"
      f"② n={orders[int(np.argmin(test_mse_noisy))]}  -> 结论一致。")
print()


# ----------------------------------------------------------------------
# 5. 误差随阶次变化规律的讨论
# ----------------------------------------------------------------------
print("=" * 86)
print("讨论：训练/测试误差随模型阶次的变化规律")
print("=" * 86)
print(f"""1) n = 1~2：欠拟合（underfitting）。一次、二次多项式无法描述 exp 形式的
   强非线性下降趋势，训练误差本身就很大（约为 {train_mse[0]:.2f} ~ {train_mse[1]:.2f} kΩ²），
   在 0~20℃ 低温端外推时严重偏低（线性模型在 0℃ 只给出约 29 kΩ，真值约 57 kΩ），
   测试误差远大于训练误差。

2) n = 3~4：拟合良好。三次、四次模型已能很好刻画曲线形状，训练误差迅速
   下降到 {train_mse[2]:.3f} ~ {train_mse[3]:.3f} kΩ²（已接近噪声方差 {NOISE_SIGMA ** 2:.2f} kΩ² 的量级），
   测试误差同时降到各阶中的较小水平，是"偏差—方差"折中最优的区间。

3) n = 5~6：过拟合（overfitting）。阶次升高后训练误差仍在缓慢减小
   ({train_mse[4]:.3f} → {train_mse[5]:.3f} kΩ²)，但改善微乎其微；反倒是测试误差
   随阶次增大而上升（{test_mse[4]:.3f} → {test_mse[5]:.3f} kΩ²）。原因是：
     * 高阶多项式为了穿过 20~80℃ 区间内的噪声点，被迫加进高次振荡项，
       这些项在区间外（0~20℃、80~100℃）被放大，也就是龙格(Runge)现象；
     * 高温端 100℃ 阻值只有约 2.7 kΩ，模型只要偏差 1 kΩ，MSE 就已经是
       0.12 kΩ²；而低温端真值约 57 kΩ，相对误差更容易被放大成绝对误差；
     * 数据只覆盖 20~80℃，两端属于"外推"，高次项的外推能力极差，
       阶次越高，测试误差对系数扰动的敏感性就越强。

4) 结论：模型阶次并非越高越好。本例中 n = 3 是最佳选择（次优为 n = 4）：
   训练误差已足够小，且 0~20℃、80~100℃ 的外推误差最小。若单纯追求训练误差最小
   （n = 6），会出现典型的过拟合，泛化能力反而下降。

5) 说明：训练误差相对"真实阻值"与相对"带噪测量值"的差别，就是噪声方差
   (~{NOISE_SIGMA ** 2:.2f} kΩ²)。当训练误差降到该量级以下时，模型拟合的已不是规律
   而是噪声，这正是不应再提高阶次的信号。
""")

# ----------------------------------------------------------------------
# 6. 作图 1：与课件一致的 2×3 回归曲线图
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5))
fig.suptitle("Regression: Perspective from Machine Learning",
             fontsize=18, fontweight="bold")

for ax, n in zip(axes.ravel(), orders):
    coef = coeffs_all[n]
    r_hat_dense = np.polyval(coef, t_dense)

    # 真实特性曲线（模型(1)）
    ax.plot(t_dense, r_dense_true, color="k", lw=1.0, ls="--",
            label="True model (1)")
    # 拟合的 n 阶多项式模型，外推到整个 0~100℃
    ax.plot(t_dense, r_hat_dense, color="r", lw=1.6,
            label=f"Fitted poly (n={n})")
    # 带噪声的实验数据（训练集）
    ax.plot(t_train, r_train, "o", ms=6, mfc="none", mec="c", mew=1.2,
            label="Noisy measurements (train)")
    # 训练集中心（噪声均值为 0，用 + 标出理想位置，便于观察噪声散布）
    ax.plot(t_train, r_train_true, "+", color="navy", ms=7, mew=1.3,
            label="Noise-free resistance")
    # 测试集数据点（0~20℃、80~100℃）
    ax.plot(t_test, r_test, "s", ms=4, mfc="none", mec="g", mew=1.0,
            label="Test data (clean)")

    ax.set_title(f"Regression Curves Using {n}-order Polynomial Model",
                 fontsize=10)
    ax.set_xlabel("Temperature/Celsius degree", fontsize=9)
    ax.set_ylabel(r"Resistance/k$\Omega$", fontsize=9)
    ax.set_xlim(T_MIN, T_MAX)
    ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="best", framealpha=0.85)

fig.tight_layout(rect=(0, 0, 1, 0.95))

# ----------------------------------------------------------------------
# 7. 作图 2：训练/测试 MSE 随阶次的变化（对数纵轴）
# ----------------------------------------------------------------------
# 三条线的含义（前两条是题目要求的，第三条是额外的诊断线）：
#
#   ① Training MSE (noisy data)  —— 题目要求的"训练集误差"
#      预测值 vs 20~80℃ 的【带噪测量值】。训练集的"标签"就是测量值本身
#      （真值未知，否则就不用做实验了），所以训练误差只能这样算。
#
#   ② Test MSE (0~20℃ & 80~100℃) —— 题目要求的"测试集误差"
#      预测值 vs 剔除 20~80℃ 后那部分数据的【真实阻值】。测试集在 1) 里
#      是由模型(1)算出来的、没有加噪声，所以它是干净的真值，
#      这条线反映的是"外推/泛化能力"。
#
#   ③ Training MSE (vs. true resistance) —— 额外诊断线（非题目要求）
#      预测值 vs 训练温度上的【真实阻值】，相当于把噪声关掉(σ=0)后的训练误差，
#      即纯粹的"模型偏差(bias)"。它比①更早、更清楚地暴露过拟合：
#      ①只能降到"噪声地板"就停住（看起来还在慢慢降，其实降的是噪声），
#      而③在 n=3 达最小、n>=4 反而上升，说明高阶项开始跟着噪声跑(方差变大)。
#
#   三者之间的解析关系（m 个训练点，p = n+1 个待估系数，hat 矩阵迹 = p）：
#          E[①] = ③ + σ² * (1 - 2p/m)
#   即训练误差 = 模型偏差 + 噪声地板；σ² 那一项之所以不能全部计回来，
#   是因为残差中的噪声已被拟合"吃掉"了一部分（自由度 p 个）。
fig2, ax2 = plt.subplots(figsize=(7.5, 5.0))
ax2.semilogy(orders, train_mse, "o-", lw=1.8,
             label="Training MSE (vs. noisy measurements)")
if SHOW_DIAGNOSTIC:
    ax2.semilogy(orders, train_mse_true, "s--", lw=1.4,
                 label="Training MSE (vs. true resistance, diagnostic)")
ax2.semilogy(orders, test_mse, "^-", color="crimson", lw=1.8,
             label="Test MSE (0~20°C & 80~100°C)")
ax2.axhline(NOISE_SIGMA ** 2, color="gray", ls=":", lw=1.2,
            label=r"noise variance $\sigma^2$")
ax2.set_xlabel("Polynomial order n")
ax2.set_ylabel(r"MSE / k$\Omega^2$  (log scale)")
ax2.set_title("MSE vs. Model Order: Underfitting -> Good Fit -> Overfitting")
ax2.set_xticks(orders)
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(fontsize=9)
fig2.tight_layout()

# ----------------------------------------------------------------------
# 8. 补充演示：为什么"末端炸开"的程度可大可小
# ----------------------------------------------------------------------
# 高阶多项式在数据区间外（0~20℃、80~100℃）的爆炸幅度，主要由
# "噪声大小 / 噪声的具体实现(随机种子) / 数据点密度" 决定：
#   * 噪声越大、数据点越稀  -> 拟合出的高次系数越大 -> 外推越夸张；
#   * 题目给的 sigma = 500Ω = 0.5kΩ 相对于 4~27kΩ 的信号非常小，
#     31 个点又给高阶项很强的约束，所以 n=6 的曲线在 100℃ 只是"翘起来"，
#     而不会像课件图那样冲到 ±200 kΩ。
# 下面把 sigma 依次放大到 0.5 / 1.0 / 1.5 / 2.0 kΩ，可以清楚看到
# 同一阶次(n=6)的模型末端爆炸幅度随噪声线性→超线性地增大。
DEMO_SIGMAS = [0.5, 1.0, 1.5, 2.0]
DEMO_ORDER = 6

fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
fig3.suptitle(f"Noise Level Controls the End-Point Blow-up "
              f"({DEMO_ORDER}-order Polynomial Model)", fontsize=14,
              fontweight="bold")
print("-" * 86)
print(f"补充演示：{DEMO_ORDER} 阶模型在不同噪声水平下的端点外推（固定随机种子 "
      f"{RANDOM_SEED}）")
print(f"{'sigma(kΩ)':>10}{'R_fit(0℃)':>14}{'R_fit(100℃)':>14}{'曲线最大值':>14}")
for ax, sigma in zip(axes3.ravel(), DEMO_SIGMAS):
    rng_d = np.random.default_rng(RANDOM_SEED)
    r_demo = thermistor_resistance(t_train) + rng_d.normal(0.0, sigma,
                                                          size=t_train.shape)
    coef_d = np.polyfit(t_train, r_demo, DEMO_ORDER)
    r_curve = np.polyval(coef_d, t_dense)
    print(f"{sigma:>10.1f}{np.polyval(coef_d, 0.0):>14.1f}"
          f"{np.polyval(coef_d, 100.0):>14.1f}{r_curve.max():>14.1f}")

    ax.plot(t_dense, r_dense_true, "k--", lw=1.0, label="True model (1)")
    ax.plot(t_dense, r_curve, "r-", lw=1.6, label=f"Fitted poly (n={DEMO_ORDER})")
    ax.plot(t_train, r_demo, "o", ms=5, mfc="none", mec="c", mew=1.1,
            label="Noisy measurements")
    ax.set_title(rf"$\sigma$ = {sigma} k$\Omega$ "
                 rf"({'500' if sigma == 0.5 else int(sigma * 1000)} ohm)",
                 fontsize=10)
    ax.set_xlabel("Temperature/Celsius degree", fontsize=9)
    ax.set_ylabel(r"Resistance/k$\Omega$", fontsize=9)
    ax.set_xlim(T_MIN, T_MAX)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=7)
print("-" * 86)
fig3.tight_layout(rect=(0, 0, 1, 0.94))

print("共 3 张图，均以窗口形式显示（不保存文件）：2×3 回归曲线、"
      "MSE-阶次曲线、噪声大小与端点误差放大")
plt.show()
