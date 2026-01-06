import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# 文件路径
file_path = 'MATD3_reward_6.csv'

# 从文件路径中提取文件名（不带扩展名）
file_name = os.path.splitext(os.path.basename(file_path))[0]

# # 定义函数从字符串中提取数字
# def extract_number(cell):
#     # 使用正则表达式匹配浮点数
#     match = re.search(r'tensor\(([\d.]+)', str(cell))
#     if match:
#         return float(match.group(1)) / 200
#     return None

# 读取 CSV 文件，并手动指定列名为 'value'
data = pd.read_csv(file_path, header=None, names=['value'])

# 提取所有单元格中的数字
data['value'] = data['value']

# 添加时间列
data['time'] = range(len(data))

# 设置窗口大小
window_size = 30

# 计算滑动均值和标准差，设置 min_periods=1 以从第一个点开始计算
data['mean'] = data['value'].rolling(window=window_size, center=True, min_periods=1).mean()
data['std'] = data['value'].rolling(window=window_size, center=True, min_periods=1).std()

# 计算波动范围
data['upper'] = data['mean'] + data['std']
data['lower'] = data['mean'] - data['std']

# 绘制图形
plt.figure(figsize=(10, 6))

# 绘制均值曲线（颜色为蓝色，线条稍粗）
plt.plot(data['time'], data['mean'], color='blue', linewidth=2, label='Mean Curve')

# 绘制波动范围（颜色为蓝色，透明度很低）
plt.fill_between(data['time'], data['lower'], data['upper'], color='blue', alpha=0.1, label='Fluctuation Range')

# 添加图例和标签
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')

# 设置图片标题为文件名
plt.title(file_name)
plt.xlim(-5, 2005)

# 显示图形
plt.show()