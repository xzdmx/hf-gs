import numpy as np
def get_dynamic_weight(iteration, total_iterations=1500, initial_weight=10.0, final_weight=0.1):
    """
    计算动态权重的指数衰减函数

    Args:
        iteration: 当前迭代次数
        total_iterations: 总迭代次数，默认1500
        initial_weight: 初始权重，默认10.0
        final_weight: 最终权重，默认0.1 (小于1，使修复图片在后期影响较小)

    Returns:
        当前迭代应该使用的权重值
    """
    # 使用指数衰减公式
    decay_rate = -np.log(final_weight / initial_weight) / (0.7 * total_iterations)
    current_weight = initial_weight * np.exp(-decay_rate * iteration)

    # 确保权重不会小于最终权重
    return max(current_weight, final_weight)