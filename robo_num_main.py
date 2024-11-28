# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:25:10 2024

@author: lei
"""

'''
不同机器人数量对迭代次数与成功率的影响
'''
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import forehand_function as ff

def robo_num_main():
    # 设置随机种子以保证结果可重复
    random_seed = random.randint(1, 100)
    random.seed(random_seed)
    np.random.seed(random_seed)
    num_case =6
    grid_sizes = [300]
    robot_counts = [3, 6, 9 ,12]
    robot_test = [3,6]
    step_size = 2
    target_step_size = 1
    scope = 10
    
    for grid_size in grid_sizes:
        robot_rectangle = (1, 1, grid_size-2, grid_size-2)
        num_obstacles = int(grid_size / 20)
        obstacle_size = int(grid_size / 20)
        max_iterations = 300
        
        # 存储不同机器人数量的成功率和迭代次数
        success_rates_dict = {num_robots: [] for num_robots in robot_counts}
        iteration_counts_dict = {num_robots: [] for num_robots in robot_counts}
        
        for num_robots in robot_counts:
            for j in range(1):  # 每次搜索三轮
                iteration_to_plot = []
                iteration_counts = []
                
                for i in range(5):  # 每轮5次
                    # 生成地图
                    map_grid = ff.generate_map(grid_size, num_obstacles, obstacle_size)
                    
                    # 放置机器人和目标
                    robots_positions, target_position = ff.place_robots_and_target(map_grid.copy(), num_robots, robot_rectangle, random.randint(1, num_case))
                    
                    # 记录初始机器人位置
                    initial_robots_positions = robots_positions[:]
                    
                    # 运行GWO算法
                    final_positions, paths, final_target_position, target_path, iteration = ff.gwo_algorithm(
                        map_grid.copy(), robots_positions, target_position, max_iterations, step_size, target_step_size, scope
                    )
                    
                    # 更新地图上的机器人位置
                    for pos in initial_robots_positions:
                        map_grid[pos] = 0
                    for pos in final_positions:
                        map_grid[pos] = 2
                    
                    # 绘制最终地图及路径
                    # ff.plot_map(map_grid, initial_robots_positions, final_positions, target_path, paths)
                    
                    iteration_to_plot.append(iteration)
                    iteration_counts.append(iteration)
                
                # 输出成功率，并存储
                iteration_success_rate, success_rates = ff.success_rate_output(iteration_to_plot, [])
                success_rates_dict[num_robots].append(iteration_success_rate)
                iteration_counts_dict[num_robots].extend(iteration_counts)
                print(f"Time of {j + 1} success rate with {num_robots} robots is {iteration_success_rate * 100}%")
            
            # 计算平均成功率
            avg_success_rate = sum(success_rates_dict[num_robots]) / len(success_rates_dict[num_robots])
            print(f"The average success rate with {num_robots} robots is {avg_success_rate * 100}% when grid_size = {grid_size}")
        
        # 绘制不同机器人数量的成功率图和迭代次数图
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # 成功率图
        for num_robots in robot_counts:
            x = range(len(success_rates_dict[num_robots]))
            axs[0].scatter(x, success_rates_dict[num_robots], label=f'{num_robots} robots')
        
        axs[0].set_xlabel('Experiment Number')
        axs[0].set_ylabel('Success Rate')
        axs[0].set_title(f'Success Rates for Different Robot Counts (Grid Size = {grid_size})')
        axs[0].legend()
        
        # 迭代次数图
        for num_robots in robot_counts:
            x = range(len(iteration_counts_dict[num_robots]))
            axs[1].scatter(x, iteration_counts_dict[num_robots], label=f'{num_robots} robots')
        
        axs[1].set_xlabel('Experiment Number')
        axs[1].set_ylabel('Iteration Count')
        axs[1].set_title(f'Iteration Counts for Different Robot Counts (Grid Size = {grid_size})')
        axs[1].legend()
        
        plt.tight_layout()
        plt.show()

# if __name__ == "__main__":
#     main_succ_rate_plus_ite_times()



