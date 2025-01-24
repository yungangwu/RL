import logging
import matplotlib.pyplot as plt
from enum import Enum

class Player(Enum):
    WHITE = 1
    BLACK = -1

def get_logger(module_name, log_level=logging.INFO):
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="./data/log/error_log.txt",  # 日志文件路径
        filemode="w"  # 追加模式
    )
    logger = logging.getLogger(module_name)
    # 创建logger
    logger.setLevel(log_level)

    # 检查是否已经有处理器，避免重复添加
    if not logger.handlers:
        # 创建控制台处理器，并设置级别为INFO
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 创建formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 将formatter添加到处理器
        ch.setFormatter(formatter)

        # 将处理器添加到logger
        logger.addHandler(ch)

    return logger

def plot_loss(losses_dict, save_path=None, title="Training Loss", xlabel="Epochs", ylabel="Loss"):
    """
    绘制模型训练过程中的损失曲线

    参数:
        losses: 记录损失值的列表
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
    """
    plt.figure(figsize=(10, 5))  # 设置图表的大小

    # 找到最大长度，进行填充
    # max_length = max(len(losses) for losses in losses_dict.values())

    # for label, losses in losses_dict.items():
    #     # 填充较短的序列
    #     padded_losses = losses + [np.nan] * (max_length - len(losses))
    #     plt.plot(padded_losses, label=label)  # 绘制每条损失曲线

    # 按最短的长度，进行截断
    min_length = min(len(losses) for losses in losses_dict.values())

    for label, losses in losses_dict.items():
        # 填充较短的序列
        padded_losses = losses[:min_length]
        plt.plot(padded_losses, label=label)  # 绘制每条损失曲线

    plt.title(title)  # 设置图表标题
    plt.xlabel(xlabel)  # 设置x轴标签
    plt.ylabel(ylabel)  # 设置y轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    if save_path:
        plt.savefig(save_path)
    plt.show()  # 显示图表