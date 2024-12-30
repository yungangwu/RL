import logging
from enum import Enum

class Player(Enum):
    WHITE = 1
    BLACK = -1


def get_logger(module_name):
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="./data/log/error_log.txt",  # 日志文件路径
        filemode="w"  # 追加模式
    )
    logger = logging.getLogger(module_name)
    # 创建logger
    logger.setLevel(logging.INFO)

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