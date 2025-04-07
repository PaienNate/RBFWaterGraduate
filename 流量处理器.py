# inflow_processor.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional

def 处理流量数据(
        窗口大小: int,
        入库流量: np.ndarray
) -> Dict[str, np.ndarray]:
    """主处理流程

    参数:
        处理器: 流量处理器字典
        原始流量: 原始入库流量序列（形状 [n_steps]）
        是否绘图: 是否生成可视化图表
        绘图保存路径: 图表保存路径（如None则不保存）

    返回:
        处理后的分量字典：
        {
            "平滑后": 平滑后序列,
            "趋势": 趋势分量,
            "季节性": 季节性分量,
            "残差": 残差分量
        }
    """
    # 1. 平滑处理
    平滑后 = 移动平均平滑(入库流量, 窗口大小)

    # 2. 趋势分解
    趋势 = 提取趋势分量(平滑后)
    季节性, 残差 = 分解季节性(平滑后, 趋势)

    return {
        "平滑后": 平滑后,
        "趋势": 趋势,
        "季节性": 季节性,
        "残差": 残差
    }


def 移动平均平滑(数据: np.ndarray, 窗口大小: int) -> np.ndarray:
    """移动平均平滑"""
    return pd.Series(数据).rolling(
        window=窗口大小,
        center=True,
        min_periods=1
    ).mean().values


def 提取趋势分量(数据: np.ndarray) -> np.ndarray:
    """线性趋势提取"""
    x轴 = np.arange(len(数据))
    系数 = np.polyfit(x轴, 数据, deg=1)
    return np.polyval(系数, x轴)


def 分解季节性(数据: np.ndarray, 趋势: np.ndarray) -> (np.ndarray, np.ndarray):
    """经典加法模型分解季节性"""
    去趋势 = 数据 - 趋势
    # 假设季节性周期为12个月
    季节性 = 计算季节性分量(去趋势, 周期=12)
    残差 = 去趋势 - 季节性
    return 季节性, 残差


def 计算季节性分量(数据: np.ndarray, 周期: int) -> np.ndarray:
    """计算季节性分量（按周期平均值）"""
    if len(数据) % 周期 != 0:
        补齐长度 = 周期 - (len(数据) % 周期)
        补齐后 = np.pad(数据, (0, 补齐长度), mode='edge')
    else:
        补齐后 = 数据

    重塑后 = 补齐后.reshape(-1, 周期)
    季节性 = np.mean(重塑后, axis=0)
    return np.tile(季节性, len(补齐后) // 周期)[:len(数据)]


def 绘制分解图(
        原始入库流量: np.ndarray,
        平滑后: np.ndarray,
        趋势: np.ndarray,
        季节性: np.ndarray,
        残差: np.ndarray,
        窗口大小: int,
        保存路径: Optional[str] = None
):
    """绘制流量分解图"""
    plt.figure(figsize=(12, 8))

    # 子图1：原始 vs 平滑
    plt.subplot(2, 2, 1)
    plt.plot(原始入库流量, label="原始流量", alpha=0.6)
    plt.plot(平滑后, label=f"平滑 (窗口={窗口大小})")
    plt.title("原始与平滑对比")
    plt.legend()

    # 子图2：趋势分解
    plt.subplot(2, 2, 2)
    plt.plot(平滑后, label="平滑后", alpha=0.6)
    plt.plot(趋势, label="趋势")
    plt.title("趋势分量")
    plt.legend()

    # 子图3：季节性
    plt.subplot(2, 2, 3)
    plt.plot(季节性, label="季节性")
    plt.title("季节性分量")

    # 子图4：残差
    plt.subplot(2, 2, 4)
    plt.plot(残差, label="残差")
    plt.title("残差分量")

    plt.tight_layout()

    if 保存路径:
        plt.savefig(保存路径)
    plt.show()