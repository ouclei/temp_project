# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:12:15 2024

@author: lei
"""

'''
地图大小对成功率的影响
'''
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import forehand_function as ff

def success_main():
    # 设置随机种子以保证结果可重复
    random_seed = random.randint(1, 100)
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(f'the random seed is :{random_seed}')
    num_case = 6
    grid_sizes = [100, 200, 300, 400, 500, 600, 800, 1000]
    grid_test_iterations = [100, 200, 300]
    grid_size_success_rate = [400]
    num_robots = 5
    step_size = 2
    target_step_size_iterations = 0.8
    target_step_size_success_rate = 1
    
    max_iterations = 1500 # 设定最大迭代次数
    stop_step = step_size
    scope = 10
    
    # 存储每个网格大小的平均迭代次数和成功率
    avg_iterations = []
    success_rates_by_grid_size = {}
    
    print("now it's for iteration")
    for grid_size in grid_sizes:
        iteration_results = []
        robot_rectangle = (1, 1, grid_size-2, grid_size-2)
        for _ in range(1):
            num_obstacles = int(grid_size / 20)
            obstacle_size = int(grid_size / 20)
            
            iteration_to_plot = []
            
            for _ in range(5):
                map_grid = ff.generate_map(grid_size, num_obstacles, obstacle_size)
                robots_positions, target_position = ff.place_robots_and_target(map_grid.copy(), num_robots, robot_rectangle, random.randint(1,num_case))
                initial_robots_positions = robots_positions[:]
                
                final_positions, paths, final_target_position, target_path, iteration = ff.gwo_algorithm(
                    map_grid.copy(), robots_positions, target_position, max_iterations, step_size, target_step_size_iterations, scope, stop_step
                )
                
                for pos in initial_robots_positions:
                    map_grid[pos] = 0
                for pos in final_positions:
                    map_grid[pos] = 2
                
                iteration_to_plot.append(iteration)
            
            avg_iteration = sum(iteration_to_plot) / len(iteration_to_plot)
            iteration_results.append(avg_iteration)
        
        avg_iterations.append(sum(iteration_results) / len(iteration_results))
    
    print("now it's for success")
    for grid_size in grid_size_success_rate:
        robot_rectangle = (1, 1, grid_size - 2, grid_size - 2)
        success_rates = []
        num_obstacles = int(grid_size / 20)
        obstacle_size = int(grid_size / 20)
        #max_iterations = int(grid_size * math.sqrt(2) * 1)
        scope = 10
        
        for j in range(1):  
            iteration_to_plot = []
            # print("\niterations")
            for i in range(5):  
                map_grid = ff.generate_map(grid_size, num_obstacles, obstacle_size)
                robots_positions, target_position = ff.place_robots_and_target(map_grid.copy(), num_robots, robot_rectangle,random.randint(1,8))
                initial_robots_positions = robots_positions[:]
                
                final_positions, paths, final_target_position, target_path, iteration = ff.gwo_algorithm(
                    map_grid.copy(), robots_positions, target_position, max_iterations, step_size, target_step_size_success_rate, scope, stop_step
                )
                
                for pos in initial_robots_positions:
                    map_grid[pos] = 0
                for pos in final_positions:
                    map_grid[pos] = 2
                
                # ff.plot_map(map_grid, initial_robots_positions, final_positions, target_path, paths)
                
                iteration_to_plot.append(iteration)
            
            iteration_success_rate, success_rates = ff.success_rate_output(iteration_to_plot, success_rates)
            print(f"Time of {j + 1} success rate is {iteration_success_rate * 100:.2f}%")
        
        avg_success_rate = sum(success_rates) / len(success_rates)
        success_rates_by_grid_size[grid_size] = avg_success_rate
        print(f"The average success rate = {avg_success_rate * 100:.2f}% when grid_size = {grid_size}")
    
    # 创建子图
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    print("now it's painting")
    # 绘制折线图
    axs[0].plot(grid_test_iterations, avg_iterations, marker='x', linestyle='-', color='red')
    axs[0].set_title('Average Iterations vs Grid Size')
    axs[0].set_xlabel('Grid Size')
    axs[0].set_ylabel('Average Iterations')
    axs[0].grid(True)
    
    # 绘制柱状图
    grid_sizes = list(success_rates_by_grid_size.keys())
    success_rates = list(success_rates_by_grid_size.values())
    
    axs[1].bar(grid_sizes, success_rates, color='blue', width=10)
    axs[1].set_title('Success Rate vs Grid Size')
    axs[1].set_xlabel('Grid Size')
    axs[1].set_ylabel('Success Rate (%)')
    axs[1].set_xticks(grid_sizes)
    axs[1].set_yticks(np.arange(0, 1.1, 0.1))
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     main()


