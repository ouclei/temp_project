import random
import numpy as np
import matplotlib.pyplot as plt
import forehand_function as  ff  # 假设ff是你定义的一些功能的模块

def robo_num_main():
    # 设置随机种子以保证结果可重复
    random_seed = random.randint(1, 100)
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(f'the random seed is: {random_seed}')
    
    # 配置参数
    num_case = 6
    grid_sizes = [300]
    robot_counts = [3, 6, 9, 12]
    step_size = 2
    target_step_size = 1
    scope = 10
    max_iterations = 1500
    
    # 初始化存储结构
    success_rates_dict = {num_robots: [] for num_robots in robot_counts}
    iteration_counts_dict = {num_robots: [] for num_robots in robot_counts}
    
    # 辅助函数：执行单次实验并返回迭代次数和是否成功
    def run_single_experiment(grid_size, num_robots, robot_rectangle, num_case, max_iterations, step_size, target_step_size, scope):
        num_obstacles = int(grid_size / 20)
        obstacle_size = int(grid_size / 20)
        map_grid = ff.generate_map(grid_size, num_obstacles, obstacle_size)
        robots_positions, target_position = ff.place_robots_and_target(map_grid.copy(), num_robots, robot_rectangle, random.randint(1, num_case))
        initial_robots_positions = robots_positions[:]
        final_positions, paths, final_target_position, target_path, iteration = ff.gwo_algorithm(
            map_grid.copy(), robots_positions, target_position, max_iterations, step_size, target_step_size, scope
        )
        for pos in initial_robots_positions:
            map_grid[pos] = 0
        for pos in final_positions:
            map_grid[pos] = 2
        return iteration, iteration <= max_iterations
    
    # 主循环
    for grid_size in grid_sizes:
        robot_rectangle = (1, 1, grid_size-2, grid_size-2)
        
        for num_robots in robot_counts:
            for j in range(1):  # 每次搜索三轮
                iteration_to_plot = []
                iteration_counts = []
                
                for i in range(5):  # 每轮5次
                    iteration, success = run_single_experiment(
                        grid_size, num_robots, robot_rectangle, num_case, max_iterations, step_size, target_step_size, scope
                    )
                    iteration_to_plot.append(iteration)
                    iteration_counts.append(iteration)
                
                # 输出成功率，并存储
                iteration_success_rate = sum(iteration_to_plot) / len(iteration_to_plot) <= max_iterations
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
        axs[0].scatter(x, [rate * 100 for rate in success_rates_dict[num_robots]], label=f'{num_robots} robots')
    
    axs[0].set_xlabel('Experiment Number')
    axs[0].set_ylabel('Success Rate (%)')
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

# # 假设这是调用函数的地方
# if __name__ == "__main__":
#     robo_num_main()