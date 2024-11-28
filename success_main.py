# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:07:11 2024

@author: lei
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import math
from success_main import ff  # 假设ff是你定义的一些功能的模块

def success_main():
    # 设置随机种子以保证结果可重复
    random_seed = random.randint(1, 100)
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(f'the random seed is: {random_seed}')
    
    # 配置参数
    num_case = 6
    grid_sizes = [100, 200, 300, 400, 500, 600, 800, 1000]
    num_robots = 5
    step_size = 2
    target_step_size_iterations = 0.8
    target_step_size_success_rate = 1
    max_iterations = 1500
    stop_step = step_size
    scope = 10
    
    # 初始化存储结构
    avg_iterations = []
    success_rates_by_grid_size = {}
    
    # 辅助函数：执行单次实验并返回迭代次数和是否成功
    def run_single_experiment(grid_size, num_robots, robot_rectangle, num_case, max_iterations, step_size, target_step_size, scope, stop_step):
        num_obstacles = int(grid_size / 20)
        obstacle_size = int(grid_size / 20)
        map_grid = ff.generate_map(grid_size, num_obstacles, obstacle_size)
        robots_positions, target_position = ff.place_robots_and_target(map_grid.copy(), num_robots, robot_rectangle, random.randint(1, num_case))
        _, _, _, _, iteration = ff.gwo_algorithm(
            map_grid.copy(), robots_positions, target_position, max_iterations, step_size, target_step_size, scope, stop_step
        )
        return iteration, iteration <= max_iterations
    
    # 主循环
    for grid_size in grid_sizes:
        robot_rectangle = (1, 1, grid_size-2, grid_size-2)
        iterations = []
        successes = []
        
        for _ in range(5):
            iteration, success = run_single_experiment(
                grid_size, num_robots, robot_rectangle, num_case, max_iterations, step_size, target_step_size_success_rate, scope, stop_step
            )
            iterations.append(iteration)
            successes.append(success)
        
        avg_iterations.append(sum(iterations) / len(iterations))
        success_rate = sum(successes) / len(successes) * 100
        success_rates_by_grid_size[grid_size] = success_rate
        print(f"The average success rate = {success_rate:.2f}% when grid_size = {grid_size}")
    
    # 绘制图表
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # 折线图
    axs[0].plot(grid_sizes, avg_iterations, marker='x', linestyle='-', color='red')
    axs[0].set_title('Average Iterations vs Grid Size')
    axs[0].set_xlabel('Grid Size')
    axs[0].set_ylabel('Average Iterations')
    axs[0].grid(True)
    
    # 柱状图
    grid_sizes = list(success_rates_by_grid_size.keys())
    success_rates = list(success_rates_by_grid_size.values())
    axs[1].bar(grid_sizes, success_rates, color='blue', width=10)
    axs[1].set_title('Success Rate vs Grid Size')
    axs[1].set_xlabel('Grid Size')
    axs[1].set_ylabel('Success Rate (%)')
    axs[1].set_xticks(grid_sizes)
    axs[1].set_yticks(np.arange(0, 101, 10))
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# 假设这是调用函数的地方
if __name__ == "__main__":
    success_main()