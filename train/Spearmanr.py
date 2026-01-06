import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载数据并进行数据处理
def load_data(file1, file2):
    """加载两个CSV文件，确保数据长度一致，并处理kl_1数据"""
    df1 = pd.read_csv(file1, header=None)  # vary_L数据
    df2 = pd.read_csv(file2, header=None)  # kl_1数据

    # 检查数据长度是否相同
    assert len(df1) == len(df2), "两个文件的数据长度不一致！"

    # 对kl_1数据从第2302行开始乘以200（注意Python是0-based索引）
    if len(df2) >= 2302:
        df2.iloc[2301:, 0] = df2.iloc[2301:, 0] * 160

    return df1[0].values, df2[0].values  # 返回处理后的数据

# 2. 计算Spearman相关系数
def compute_spearman(x, y):
    """计算Spearman相关系数及p值"""
    corr, p_value = spearmanr(x, y)
    return corr, p_value

# 3. 绘制散点图与趋势线
def plot_scatter_with_correlation(x, y, corr):
    """绘制带相关系数和拟合公式的散点图"""
    plt.figure(figsize=(10, 7))

    # 散点图
    plt.scatter(x, y, alpha=0.6, edgecolors='w', s=80)

    # 添加趋势线（使用numpy的线性拟合）
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), 'r--', linewidth=2)

    # 生成拟合公式文本
    fit_expression = f'y = {z[0]:.4f}x + {z[1]:.4f}'

    # 标注相关系数和拟合公式
    plt.text(0.05, 0.95,
             f'Spearman ρ = {corr:.3f}\n{fit_expression}',
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.xlabel('Q', fontsize=12)
    plt.ylabel('kl', fontsize=12)  # 注明数据经过缩放
    plt.title('Spearman Correlation between beta and eta', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 文件路径（替换为实际路径）
    file_vary_L = 'L_0.csv'
    file_kl_1 = 'lr_L_0.csv'

    # 加载数据（会自动处理kl_1数据）
    B, Lr = load_data(file_vary_L, file_kl_1)

    # 计算Spearman相关系数（基于处理后的数据）
    corr, p_value = compute_spearman(B, Lr)
    print(f"Spearman相关系数: {corr:.4f}")
    print(f"P值: {p_value:.4e}")

    # 可视化
    plot_scatter_with_correlation(B, Lr, corr)