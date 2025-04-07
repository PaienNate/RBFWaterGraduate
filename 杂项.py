import matplotlib
import matplotlib.pyplot as plt
def 初始化可视化相关():
    # 解决pycharm不显示问题
    matplotlib.use('TkAgg')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
