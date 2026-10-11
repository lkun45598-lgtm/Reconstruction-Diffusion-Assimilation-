import logging

def get_logger():
    """
    配置并返回一个 logger 对象
    """
    # 1. 配置基础设置
    logging.basicConfig(
        level=logging.INFO,  
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 2. 获取 logger 对象
    logger = logging.getLogger()
    
    return logger

# 如果你希望导入这个文件时自动配置，也可以直接在这里调用（但通常推荐用函数调用）
logger = get_logger() 

'''
logger.info 
logger.warning
logger.error
logger.exception
'''