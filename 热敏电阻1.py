# 引入头文件
import numpy as np
import matplotlib.pyplot as plt

# 建立摄氏度数组 t_c
t_c = np.arange(0, 102, 2)

# 将摄氏度转化为开尔文温度
T = 273.15 + t_c
# 确立参数
T0 = 273.15 + 25
beta = 3100
R0 = 22

# 计算热敏电阻的公式
R = R0 * np.exp(beta * (1/T - 1/ T0))

# 画图
plt.plot(t_c, R)
plt.xlabel('Temperature (°C)')
plt.ylabel('Resistance (kΩ)')
plt.title('Thermistor Characteristics')
plt.grid(True)
plt.show()