# 初始化日志
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("水电优化.log"), logging.StreamHandler()]
)
日志记录器 = logging.getLogger(__name__)