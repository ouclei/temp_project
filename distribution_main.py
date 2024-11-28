import random
import numpy as np
import matplotlib.pyplot as plt
import math
import forehand_function as ff

# 设置随机种子以保证结果可重复
def set_random_seed():
    random_seed = random.randint(1, 100)
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(f"Random seed used: {random_seed}")

def distribution_main1():
    set_random_seed()
    
    # 参数配置
    num_case = 6
    grid_size = 300
    num_robots = 5
    step_size = 2
    target_step_size = 0.8
    max_iterations = 1500  # 最大迭代次数
    stop_step = step_size
    scope = 10

    # 定义机器人和障碍物相关参数
    robot_rectangle = (1, 1, grid_size - 2, grid_size - 2)
    num_obstacles = int(grid_size / 20)
    obstacle_size = int(grid_size / 20)

    # 存储每个CASE的平均迭代次数（仅支持1到8）
    avg_iterations_by_case = {case: [] for case in range(1, num_case )}

    # 辅助函数：执行单次实验并返回迭代次数
    def run_single_experiment(case):
        try:
            # 生成地图
            map_grid = ff.generate_map(grid_size, num_obstacles, obstacle_size)

            # 放置机器人和目标
            robots_positions, target_position = ff.place_robots_and_target(
                map_grid.copy(), num_robots, robot_rectangle, case
            )

            # 记录初始机器人位置
            initial_robots_positions = robots_positions[:]

            # 运行灰狼优化算法 (GWO)
            final_positions, paths, final_target_position, target_path, iteration = ff.gwo_algorithm(
                map_grid.copy(), robots_positions, target_position,
                max_iterations, step_size, target_step_size, scope, stop_step
            )

            # 检查结果是否合理
            if not final_positions or iteration >= max_iterations:
                print(f"Warning: Algorithm did not converge in case {case}")
                return None

            # 更新地图上的机器人位置
            for pos in initial_robots_positions:
                if 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size:
                    map_grid[pos] = 0  # 清除旧位置
            for pos in final_positions:
                if 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size:
                    map_grid[pos] = 2  # 标记新位置

            return iteration

        except Exception as e:
            print(f"Error occurred in case {case}: {e}")
            return None

    # 遍历每种CASE
    for case in range(1, num_case ):
        print(f'\nCase {case}')
        iteration_to_plot = []

        # 重复实验10次
        for _ in range(10):
            iteration = run_single_experiment(case)
            if iteration is not None:
                iteration_to_plot.append(iteration)

        # 计算当前CASE的平均迭代次数
        if iteration_to_plot:
            avg_iterations_by_case[case] = sum(iteration_to_plot) / len(iteration_to_plot)
        else:
            avg_iterations_by_case[case] = None

    # 输出每个CASE的平均迭代次数
    for case, avg_iter in avg_iterations_by_case.items():
        if avg_iter is not None:
            print(f"Average iterations for Case {case}: {avg_iter:.2f}")
        else:
            print(f"Average iterations for Case {case}: No valid data")

    # 准备绘制折线图的数据
    cases = [case for case in avg_iterations_by_case.keys() if avg_iterations_by_case[case] is not None]
    avg_iterations = [avg_iterations_by_case[case] for case in cases]

    # 绘制折线图
    plt.figure(figsize=(12, 8))
    plt.plot(cases, avg_iterations, marker='o', linestyle='-', color='red')
    plt.title('Average Iterations vs Distribution Case on a Fixed 300x300 Grid')
    plt.xlabel('Distribution Case')  # x轴表示案例编号
    plt.ylabel('Average Iterations')  # y轴表示平均迭代次数
    plt.xticks(cases)  # 设置x轴刻度为案例编号
    plt.grid(True)
    plt.show()



def distribution_main():
    set_random_seed()
    
    # 参数配置
    num_case = 6
    grid_size = 300
    num_robots = 5
    step_size = 2
    target_step_size = 0.8
    max_iterations = 1500  # 最大迭代次数
    stop_step = step_size
    scope = 10

    # 定义机器人和障碍物相关参数
    robot_rectangle = (1, 1, grid_size - 2, grid_size - 2)
    num_obstacles = int(grid_size / 20)
    obstacle_size = int(grid_size / 20)

    # 存储每个CASE的平均迭代次数（仅支持1到8）
    avg_iterations_by_case = {case: [] for case in range(1, num_case + 1)}

    # 辅助函数：执行单次实验并返回迭代次数
    def run_single_experiment(case):
        try:
            # 生成地图
            map_grid = ff.generate_map(grid_size, num_obstacles, obstacle_size)

            # 放置机器人和目标
            robots_positions, target_position = ff.place_robots_and_target(
                map_grid.copy(), num_robots, robot_rectangle, case
            )

            # 记录初始机器人位置
            initial_robots_positions = robots_positions[:]

            # 运行灰狼优化算法 (GWO)
            final_positions, paths, final_target_position, target_path, iteration = ff.gwo_algorithm(
                map_grid.copy(), robots_positions, target_position,
                max_iterations, step_size, target_step_size, scope, stop_step
            )

            # 检查结果是否合理
            if not final_positions or iteration >= max_iterations:
                print(f"Warning: Algorithm did not converge in case {case}")
                return None

            # 更新地图上的机器人位置
            for pos in initial_robots_positions:
                if 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size:
                    map_grid[pos] = 0  # 清除旧位置
            for pos in final_positions:
                if 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size:
                    map_grid[pos] = 2  # 标记新位置

            # 绘制地图
            ff.plot_map(map_grid, initial_robots_positions, final_positions, target_path, paths)  # 新增绘图功能

            return iteration

        except Exception as e:
            print(f"Error occurred in case {case}: {e}")
            return None

    # 遍历每种CASE
    for case in range(1, num_case + 1):
        print(f'\nCase {case}')
        iteration_to_plot = []

        # 重复实验10次
        for _ in range(10):
            iteration = run_single_experiment(case)
            if iteration is not None:
                iteration_to_plot.append(iteration)

        # 计算当前CASE的平均迭代次数
        if iteration_to_plot:
            avg_iterations_by_case[case] = sum(iteration_to_plot) / len(iteration_to_plot)
        else:
            avg_iterations_by_case[case] = None

    # 输出每个CASE的平均迭代次数
    for case, avg_iter in avg_iterations_by_case.items():
        if avg_iter is not None:
            print(f"Average iterations for Case {case}: {avg_iter:.2f}")
        else:
            print(f"Average iterations for Case {case}: No valid data")

    # 准备绘制折线图的数据
    cases = [case for case in avg_iterations_by_case.keys() if avg_iterations_by_case[case] is not None]
    avg_iterations = [avg_iterations_by_case[case] for case in cases]

    # 绘制折线图
    plt.figure(figsize=(12, 8))
    plt.plot(cases, avg_iterations, marker='o', linestyle='-', color='red')
    plt.title('Average Iterations vs Distribution Case on a Fixed 300x300 Grid')
    plt.xlabel('Distribution Case')  # x轴表示案例编号
    plt.ylabel('Average Iterations')  # y轴表示平均迭代次数
    plt.xticks(cases)  # 设置x轴刻度为案例编号
    plt.grid(True)
    plt.show()
# if __name__ == "__main__":
#     distribution_main()