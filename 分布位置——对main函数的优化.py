# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 01:14:37 2024

@author: lei
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 05:15:56 2024

@author: lei
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import time
import math




def generate_map(grid_size, num_obstacles, obstacle_size):
    map_grid = np.zeros((grid_size, grid_size))
    
    # 生成障碍物
    for _ in range(num_obstacles):
        x, y = random.randint(0, grid_size - obstacle_size), random.randint(0, grid_size - obstacle_size)
        if all(map_grid[x+dx, y+dy] == 0 for dx in range(obstacle_size) for dy in range(obstacle_size)):
            for dx in range(obstacle_size):
                for dy in range(obstacle_size):
                    map_grid[x+dx, y+dy] = 1
    
    return map_grid

# 随机生成矩形内的坐标
# 随机生成矩形内的坐标
def get_random_points_inside_rectangle(rectangle, num_points, case):
    """
    rectangle: (x_min, y_min, x_max, y_max) 矩形边界
    num_points: 生成的随机点数量
    case: 分布类型对应的数字 (1-8)
    """
    x_min, y_min, x_max, y_max = rectangle
    points = []

    if case == 1:
        # 1. 底部集中分布
        points = [
            (random.randint(x_min, x_max), random.randint(y_max - int(0.2 * (y_max - y_min)), y_max))
            for _ in range(num_points)
        ]
    
    elif case == 2:
        # 2. 底部分散分布
        points = [
            (random.randint(x_min, x_max), random.randint(y_max - int(0.3 * (y_max - y_min)), y_max))
            for _ in range(num_points)
        ]
    
    elif case == 3:
        # 3. 区域内随机分散分布
        points = [
            (random.randint(x_min, x_max), random.randint(y_min, y_max))
            for _ in range(num_points)
        ]
    
    elif case == 4:
        # 4. 区域内随机集中分布
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        half_width = (x_max - x_min) // 4
        half_height = (y_max - y_min) // 4
        
        points = [
            (random.randint(center_x - half_width, center_x + half_width),
             random.randint(center_y - half_height, center_y + half_height))
            for _ in range(num_points)
        ]
    
    elif case == 5:
        # 5. 正对角线分布
        diagonal_length = max(abs(x_max - x_min), abs(y_max - y_min))
        points = [
            (int(x_min + i * (x_max - x_min) / (num_points - 1)),
             int(y_min + i * (y_max - y_min) / (num_points - 1)))
            for i in range(num_points)
        ]
    
    elif case == 6:
        # 6. 反对角线分布
        diagonal_length = max(abs(x_max - x_min), abs(y_max - y_min))
        points = [
            (int(x_min + i * (x_max - x_min) / (num_points - 1)),
             int(y_max - i * (y_max - y_min) / (num_points - 1)))
            for i in range(num_points)
        ]
    
    elif case == 7:
        # 7. 过区域中心点平行X轴分布
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        points = [
            (random.randint(x_min, x_max), center_y)
            for _ in range(num_points)
        ]
    
    elif case == 8:
        # 8. 过中心点平行Y轴分布
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        points = [
            (center_x, random.randint(y_min, y_max))
            for _ in range(num_points)
        ]
    
    else:
        raise ValueError("Invalid case number. Choose a number between 1 and 8.")

    return points

# 检查坐标是否有效
def check_positions(map_grid, positions):
    """
    检查给定坐标是否在障碍物区域内
    map_grid: 地图数组
    positions: 待检测的坐标点列表
    返回有效的点
    """
    valid_positions = []
    for pos in positions:
        x, y = pos
        if map_grid[x, y] == 0:  # 确保不在障碍物区域
            valid_positions.append(pos)
    return valid_positions

# 主检测逻辑
# 主检测逻辑
def generate_valid_positions(map_grid, rectangle, num_robots, case):
    """
    生成有效的机器人初始坐标点
    map_grid: 地图数组
    rectangle: 矩形边界 (x_min, y_min, x_max, y_max)
    num_robots: 需要的机器人数量
    case: 分布类型对应的数字 (1-8)
    返回有效坐标点列表
    """
    robots_positions = set()
    
    while len(robots_positions) < num_robots:
        # 生成随机点
        candidate_positions = get_random_points_inside_rectangle(rectangle, num_robots, case)
        # 检测随机点是否有效
        valid_positions = check_positions(map_grid, candidate_positions)
        # 添加有效点到最终集合
        robots_positions.update(valid_positions)
    
    return list(robots_positions)

# 生成随机点


def place_robots_and_target(map_grid, num_robots, robot_rectangle, case):
    """
    在地图上放置机器人和目标。
    
    :param map_grid: 地图网格
    :param num_robots: 机器人数量
    :param robot_rectangle: 机器人活动范围矩形 (x_min, y_min, x_max, y_max)
    :param case: 分布类型对应的数字 (1-8)
    :return: 机器人位置列表和目标位置
    """
    robots_positions = generate_valid_positions(map_grid, robot_rectangle, num_robots, case)
    
    # 确保目标位置不在障碍物区域且不在机器人位置
    target_position = None
    while target_position is None or map_grid[target_position] != 0 or target_position in robots_positions:
        x = random.randint(robot_rectangle[0], robot_rectangle[2] - 1)
        y = random.randint(robot_rectangle[1], robot_rectangle[3] - 1)
        target_position = (x, y)
    
    return robots_positions, target_position


def plot_map(map_grid, initial_robots_positions, final_robots_positions, target_path, paths):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制网格
    cmap = plt.colormaps['Greys'].copy()
    cmap.set_over(color='black')  # 设置障碍物颜色为黑色
    cmap.set_under(color='white')  # 设置背景色为白色
    cmap.set_bad(color='white')   # 设置默认背景色为白色
    
    ax.imshow(map_grid, cmap=cmap, interpolation='nearest', vmin=-0.5, vmax=1.5)
    
    # 绘制机器人路径
    for i, path in enumerate(paths):
        path_x = [pos[1] for pos in path]
        path_y = [pos[0] for pos in path]
        ax.plot(path_x, path_y, label=f'Robot {i+1} Path')
    
    # 绘制初始位置的机器人
    initial_robot_x = [pos[1] for pos in initial_robots_positions]
    initial_robot_y = [pos[0] for pos in initial_robots_positions]
    ax.scatter(initial_robot_x, initial_robot_y, c='blue', s=100, label='Initial Robot Positions', zorder=5)
    
    # 绘制最终位置的机器人
    final_robot_x = [pos[1] for pos in final_robots_positions]
    final_robot_y = [pos[0] for pos in final_robots_positions]
    # ax.scatter(final_robot_x, final_robot_y, c='green', s=100, label='Final Robot Positions', zorder=5)
    
    # 绘制目标路径
    target_path_x = [pos[1] for pos in target_path]
    target_path_y = [pos[0] for pos in target_path]
    ax.plot(target_path_x, target_path_y, color='red', linestyle='--', label='Target Escape Path', linewidth=2)
    
    # 绘制目标起点和终点
    start_target = target_path[0]
    end_target = target_path[-1]
    ax.scatter(start_target[1], start_target[0], c='red', s=100, label='Start Target Position', zorder=5)
    # ax.scatter(end_target[1], end_target[0], c='purple', s=100, label='End Target Position', zorder=5)
    
    # # 添加图例
    # ax.legend(loc='upper right')
    
    # 设置轴标签和标题
    ax.set_xticks(np.arange(0, map_grid.shape[1], 50))
    ax.set_yticks(np.arange(0, map_grid.shape[0], 50))
    ax.set_xticklabels(np.arange(0, map_grid.shape[1], 50))
    ax.set_yticklabels(np.arange(0, map_grid.shape[0], 50))
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    plt.show()

def calculate_fitness(positions, target_position):
    distances = [np.sqrt((pos[0] - target_position[0])**2 + (pos[1] - target_position[1])**2) for pos in positions]
    fitness_values = [1 / d if d != 0 else float('inf') for d in distances]
    return fitness_values

# 检查区域是否有效（不在障碍物内且在地图范围内）
def is_valid_region(new_position, map_grid, obstacle_size):
    x, y = new_position
    grid_size = map_grid.shape[0]

    # 判断整个正方形区域是否在地图范围内
    if x < 0 or y < 0 or x + obstacle_size > grid_size or y + obstacle_size > grid_size:
        return False

    # 检查正方形区域内是否存在障碍物
    for dx in range(obstacle_size):
        for dy in range(obstacle_size):
            if map_grid[x + dx, y + dy] == 1:  # 存在障碍物
                return False

    return True

from heapq import heappop, heappush

# 启发式函数：曼哈顿距离
def heuristic(a, b):
    """
    启发式函数，计算点a到点b的曼哈顿距离。
    a, b: 坐标点 (x, y)
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A*路径规划算法
def a_star_search(map_grid, start, goal):
    """
    使用A*算法在地图上找到从start到goal的最短路径。
    
    map_grid: 2D地图数组，0表示可通行，1表示障碍物
    start: 起点坐标 (x, y)
    goal: 目标点坐标 (x, y)
    
    返回:
    - 路径: 包含从start到goal的所有坐标点列表。如果无法到达目标，返回空列表。
    """
    grid_size = map_grid.shape[0]
    open_set = []
    heappush(open_set, (0, start))  # 优先级队列，存储 (优先级, 当前点)
    came_from = {}  # 跟踪路径
    cost_so_far = {start: 0}  # 起点到各点的当前累计代价

    while open_set:
        # 取出优先级最低（估算总代价最小）的节点
        _, current = heappop(open_set)

        if current == goal:  # 找到目标
            break

        # 遍历当前节点的邻居（上下左右四个方向）
        neighbors = [
            (current[0] + dx, current[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]
        # 筛选有效邻居节点（在地图范围内且非障碍物）
        neighbors = [
            n for n in neighbors
            if 0 <= n[0] < grid_size and 0 <= n[1] < grid_size and map_grid[n[0], n[1]] == 0
        ]

        for neighbor in neighbors:
            # 当前路径到邻居的代价为1（可以根据需求调整代价）
            new_cost = cost_so_far[current] + 1
            # 如果邻居未被访问过，或找到更优的路径，则更新
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current

    # 如果目标点未被访问，返回空路径
    if goal not in came_from:
        return []

    # 通过回溯找到完整路径
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.reverse()  # 起点到目标的顺序
    return path


def gwo_algorithm(map_grid, robots_positions, target_position, max_iterations=1000, step_size=2, target_step_size=1, scope=100,stop_step=2):
    num_robots = len(robots_positions)
    alpha_pos = [(float('inf'), float('inf'))] * num_robots
    beta_pos = [(float('inf'), float('inf'))] * num_robots
    delta_pos = [(float('inf'), float('inf'))] * num_robots
    
    alpha_score = [float('-inf')] * num_robots
    beta_score = [float('-inf')] * num_robots
    delta_score = [float('-inf')] * num_robots
    
    grid_size = map_grid.shape[0]
    
    # 初始化路径记录
    paths = [[] for _ in range(num_robots)]
    for i in range(num_robots):
        paths[i].append(robots_positions[i])
    
    # 记录目标路径
    target_path = [target_position]
    
    # 自适应惯性权重参数
    w_max = 0.9
    w_min = 0.4
    
    for iteration in range(max_iterations):
        fitness_values = calculate_fitness(robots_positions, target_position)
        
        # Update Alpha, Beta, Delta
        for i in range(num_robots):
            if fitness_values[i] > alpha_score[i]:
                alpha_score[i] = fitness_values[i]
                alpha_pos[i] = robots_positions[i]
            
            if fitness_values[i] > beta_score[i] and fitness_values[i] <= alpha_score[i]:
                beta_score[i] = fitness_values[i]
                beta_pos[i] = robots_positions[i]
            
            if fitness_values[i] > delta_score[i] and fitness_values[i] <= beta_score[i]:
                delta_score[i] = fitness_values[i]
                delta_pos[i] = robots_positions[i]
        
        a = 2 - iteration * (2 / max_iterations)  # linearly decreased from 2 to 0
        
        # 自适应惯性权重
        w = w_max - ((w_max - w_min) * (iteration / max_iterations))
        
        for i in range(num_robots):
            A1 = 2 * a * random.random() - a
            C1 = 2 * random.random()
            D_alpha = [abs(C1 * alpha_pos[i][d] - robots_positions[i][d]) for d in range(2)]
            X1 = [alpha_pos[i][d] - A1 * D_alpha[d] for d in range(2)]
            
            A2 = 2 * a * random.random() - a
            C2 = 2 * random.random()
            D_beta = [abs(C2 * beta_pos[i][d] - robots_positions[i][d]) for d in range(2)]
            X2 = [beta_pos[i][d] - A2 * D_beta[d] for d in range(2)]
            
            A3 = 2 * a * random.random() - a
            C3 = 2 * random.random()
            D_delta = [abs(C3 * delta_pos[i][d] - robots_positions[i][d]) for d in range(2)]
            X3 = [delta_pos[i][d] - A3 * D_delta[d] for d in range(2)]
            
            new_position = [
                int(round((X1[d] + X2[d] + X3[d]) / 3)) for d in range(2)
            ]
            
            # Optimal Learning Search Strategy
            best_position = alpha_pos[i]
            learning_factor = 0.8
            optimal_position = (
                int(best_position[0] + learning_factor * (target_position[0] - best_position[0])),
                int(best_position[1] + learning_factor * (target_position[1] - best_position[1]))
            )
            
            # Ensure the optimal position is within bounds and not on an obstacle
            if 0 <= optimal_position[0] < grid_size and 0 <= optimal_position[1] < grid_size and map_grid[optimal_position[0], optimal_position[1]] == 0:
                new_position = optimal_position
            
            # Calculate the direction vector
            direction_vector = [new_position[d] - robots_positions[i][d] for d in range(2)]
            distance = np.linalg.norm(direction_vector)
            
            if distance > 0:
                unit_direction = [direction_vector[d] / distance for d in range(2)]
                
                # Adaptive Speed Adjustment Strategy
                adaptive_step_size = step_size * (1 + random.uniform(-0.1, 0.1))
                step = [int(unit_direction[d] * adaptive_step_size) for d in range(2)]
                new_position = [robots_positions[i][d] + step[d] for d in range(2)]
            
            # Ensure the new position is within bounds and not on an obstacle
            if 0 <= new_position[0] < grid_size and 0 <= new_position[1] < grid_size and map_grid[new_position[0], new_position[1]] == 0 :
                robots_positions[i] = tuple(new_position)
                paths[i].append(robots_positions[i])
            else:
                # Escape Mechanism
                # escape_range = 20
                # escape_x = random.randint(-escape_range, escape_range)
                # escape_y = random.randint(-escape_range, escape_range)
                # new_escape_position = (
                #     min(max(robots_positions[i][0] + escape_x, 0), grid_size - 1),
                #     min(max(robots_positions[i][1] + escape_y, 0), grid_size - 1)
                # )
                # if map_grid[new_escape_position[0], new_escape_position[1]] == 0:
                #     robots_positions[i] = new_escape_position
                #     paths[i].append(robots_positions[i])
                
                
                #a* Mechanism
                path = a_star_search(map_grid, robots_positions[i], target_position)
                if path:
                    next_position = path[0]  # 获取路径中的下一步
                    robots_positions[i] = next_position
                    paths[i].append(next_position)
                else:
                    print(f"Robot {i} failed to find a valid path.")
                    
        
                
        # Check termination condition
        min_distance = min([np.sqrt((pos[0] - target_position[0])**2 + (pos[1] - target_position[1])**2) for pos in robots_positions])
        if min_distance < stop_step:
            print(f"Termination Condition Met: Distance to Target < 5 after {iteration} iterations.")
            break
        elif iteration >= max_iterations - 1:
            print("Termination Condition Met: Maximum Iterations Reached.")
            iteration = -1
        
        # Check if any robot is within the scope of the target
        target_within_scope = False
        closest_robot_index = None
        closest_distance = float('inf')
        for i, pos in enumerate(robots_positions):
            distance_to_target = np.sqrt((pos[0] - target_position[0])**2 + (pos[1] - target_position[1])**2)
            if distance_to_target < scope:
                target_within_scope = True
                if distance_to_target < closest_distance:
                    closest_distance = distance_to_target
                    closest_robot_index = i
        
        if target_within_scope:
            # Calculate escape direction using AFP artificial potential field method
            escape_direction = [-(robots_positions[closest_robot_index][d] - target_position[d]) for d in range(2)]
            escape_magnitude = np.linalg.norm(escape_direction)
            if escape_magnitude > 0:
                unit_escape_direction = [escape_direction[d] / escape_magnitude for d in range(2)]
                escape_speed = target_step_size * (1 + (scope - closest_distance) / scope)
                escape_step = [int(unit_escape_direction[d] * escape_speed) for d in range(2)]
                new_target_position = (target_position[0] + escape_step[0], target_position[1] + escape_step[1])
                
                # Ensure the new target position is within bounds and not on an obstacle
                if 0 <= new_target_position[0] < grid_size and 0 <= new_target_position[1] < grid_size and map_grid[new_target_position[0], new_target_position[1]] == 0:
                    target_position = new_target_position
                    target_path.append(target_position)
                else:
                    # 使用A*算法计算目标逃逸路径
                    escape_goal = (
                        random.randint(0, grid_size - 1),  # 随机生成一个远离机器人的点作为逃逸目标
                        random.randint(0, grid_size - 1)
                    )
                    while map_grid[escape_goal[0], escape_goal[1]] != 0:  # 确保逃逸目标无障碍物
                        escape_goal = (
                            random.randint(0, grid_size - 1),
                            random.randint(0, grid_size - 1)
                        )

                    path = a_star_search(map_grid, target_position, escape_goal)
                    if path:
                        next_escape_position = path[0]  # 获取路径中的下一步
                        target_position = next_escape_position
                        target_path.append(target_position)
                    else:
                        print("Target failed to escape.")

        
        # Update target position on map
        map_grid[:, :] = 0  # Clear previous target and obstacles
        for x in range(grid_size):
            for y in range(grid_size):
                if map_grid[x, y] == 1:  # Keep obstacles
                    continue
                elif map_grid[x, y] == 2:  # Keep robots
                    continue
                elif (x, y) == target_position:
                    map_grid[x, y] = 3  # Mark new target position
    if iteration >= max_iterations:
        print("达到最大迭代次数，终止算法")
        print(f"机器人位置: {robots_positions}")
        print(f"目标位置: {target_position}")
    # break

    return robots_positions, paths, target_position, target_path,iteration

def main1():
    #random_seed = 16
    random_seed = random.randint(1,100)
    # 设置随机种子以保证结果可重复
    random.seed(random_seed)
    np.random.seed(random_seed)
    grid_size = 300
    
    num_robots = 5
    
    step_size = 2
    target_step_size = 0.8
    
    robot_ractangle = (1,1,50,50)
    # position1 = [(10, 15), (21, 10), (35, 33), (14, 2), (22, 3)] #集中分布与左上角
    # position2 = [(5, 1), (5, 80), (5, 160), (5, 210), (5, 280)] #分散分布与顶部
    # position3 = [(100, 0), (100, 80), (5, 160), (5, 210), (5, 280)]
    grid_1 = [100,200,300]
    grid_2 = [400,600,800,1000]
    grid_test = [300]
    iteration_all = []
    for grid_size in grid_test:
        #每次搜索三轮，一轮5次
        # success_rates = []
        num_obstacles = int(grid_size/20)
        obstacle_size = int(grid_size/20)
        max_iterations = int(grid_size * math.sqrt(2))
        stop_step = step_size
        scope = grid_size* 0.1
        
        for j in range(3):
            iteration_to_plot = []
            
            for i in range(5):
                # 生成地图
                map_grid = generate_map(grid_size, num_obstacles, obstacle_size)
                #print('map generate')
                
                # 放置机器人和目标
                robots_positions, target_position = place_robots_and_target(map_grid.copy(), num_robots,robot_ractangle)
                #print('robot place')
                
                # 记录初始机器人位置
                initial_robots_positions = robots_positions[:]
                
                # 运行GWO算法
                final_positions, paths, final_target_position, target_path, iteration = gwo_algorithm(map_grid.copy(), robots_positions, target_position, max_iterations, step_size, target_step_size, scope,stop_step)
                
                # 更新地图上的机器人位置
                for pos in initial_robots_positions:
                    map_grid[pos] = 0
                for pos in final_positions:
                    map_grid[pos] = 2
                #print('gwo running')
                # 绘制最终地图及路径
                plot_map(map_grid, initial_robots_positions, final_positions, target_path, paths)
                #time.sleep(1)
                iteration_to_plot.append(iteration)
                # print(f'i={i+1}')
                
            #把每次的迭代次数收集起来，输出本轮迭代平均值
            for i in range(len(iteration_to_plot)):
                iteration_all.append(iteration_to_plot[i])
            print(f'time of {j+1} avg iteration is {sum(iteration_to_plot) / len(iteration_to_plot)}')
            
           
            # print(iteration_to_plot)
            
            # #输出成功率，并存储        -*
            # iteration_success_rate , success_rates = success_rate_output(iteration_to_plot,success_rates)
            # print(f"time of {j+1} success rate is {iteration_success_rate*100}%")
        
        
        # #在当前环境下，取三次仿真实验所得的平均值，降低误差
        # avg_success_rates = sum(success_rates)/len(success_rates)    
        # print(f'the avg success rate = {avg_success_rates*100}% when grid_size = {grid_size}')     
        avg_iterations_all = sum(iteration_all) / len(iteration_all)
        print(f'the avg iteration for all = {avg_iterations_all} when grid_size = {grid_size}')  
        
        # 创建新的Figure对象用于本次绘图
        # fig = plt.figure()    
        # 设置横坐标，用元素的索引作为横坐标
    x = range(len(iteration_all))


           # 绘制散点图
    plt.scatter(x, iteration_all)
    plt.title(f'the iteration of {j+1} s plot' )
    plt.show()
    
    #绘制柱状图
    categories = ['A', 'B', 'C']
    values = [23, 45, 56, 78]
    # 创建柱状图
    plt.bar(categories, values, color='skyblue')
    
def main2():

    # 设置随机种子以保证结果可重复
    random_seed = random.randint(1, 100)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    grid_sizes = [100, 200, 300,400,500,600]
    grid_test = [300]
    num_robots = 5
    step_size = 2
    target_step_size = 0.8
    
    max_iterations = 1000  # 设定最大迭代次数
    stop_step = step_size
    
    # scope = 0.1 * max(grid_sizes)
    scope = 10
    # 存储每个网格大小的平均迭代次数
    avg_iterations = []

    for grid_size in grid_test:
        iteration_results = []
        robot_rectangle = (1, 1, grid_size-2, grid_size-2)
        # 对每个网格大小生成三次地图
        for _ in range(1):
            num_obstacles = int(grid_size / 20)
            obstacle_size = int(grid_size / 20)
            
            iteration_to_plot = []
            
            for _ in range(5):
                # 生成地图
                map_grid = generate_map(grid_size, num_obstacles, obstacle_size)
                
                # 放置机器人和目标
                robots_positions, target_position = place_robots_and_target(map_grid.copy(), num_robots, robot_rectangle)
                
                # 记录初始机器人位置
                initial_robots_positions = robots_positions[:]
                
                # 运行GWO算法
                final_positions, paths, final_target_position, target_path, iteration = gwo_algorithm(
                    map_grid.copy(), robots_positions, target_position, max_iterations, step_size, target_step_size, scope, stop_step
                )
                
                # 更新地图上的机器人位置
                for pos in initial_robots_positions:
                    map_grid[pos] = 0
                for pos in final_positions:
                    map_grid[pos] = 2
                
                # 记录本次迭代次数
                iteration_to_plot.append(iteration)
            
            # 计算本次地图生成的平均迭代次数
            avg_iteration = sum(iteration_to_plot) / len(iteration_to_plot)
            iteration_results.append(avg_iteration)
        
        # 计算每个网格大小的平均迭代次数
        avg_iterations.append(sum(iteration_results) / len(iteration_results))
    
    # 绘制折线图
    plt.plot(grid_test, avg_iterations, marker='x', linestyle='-', color='red')
    plt.title('Average Iterations vs Grid Size')
    plt.xlabel('Grid Size')
    plt.ylabel('Average Iterations')
    plt.grid(True)
    plt.show()

def main3():
    
    # 定义每个CASE的描述
    case_descriptions = {
       1: "底部集中分布",
       2: "底部分散分布",
       3: "区域内随机分散分布",
       4: "区域内随机集中分布",
       5: "正对角线分布",
       6: "反对角线分布",
       7: "过区域中心点平行X轴分布",
       8: "过中心点平行Y轴分布"
   }
    # 设置随机种子以保证结果可重复
    random_seed = random.randint(1, 100)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    grid_sizes = [100, 200, 300, 400, 500, 600]
    grid_test = [300]
    num_robots = 5
    step_size = 2
    target_step_size = 0.8
    
    max_iterations = 1000  # 设定最大迭代次数
    stop_step = step_size
    
    # scope = 0.1 * max(grid_sizes)
    scope = 10
    
    # 存储每个网格大小的平均迭代次数
    avg_iterations_by_case = {case: [] for case in range(1, 9)}

    for grid_size in grid_test:
        robot_rectangle = (1, 1, grid_size-2, grid_size-2)
        # 对每个网格大小生成三次地图
        for _ in range(1):
            num_obstacles = int(grid_size / 20)
            obstacle_size = int(grid_size / 20)
            
            for case in range(1, 9):  # 按顺序从1到8轮流实验
                iteration_to_plot = []
                
                for _ in range(5):
                    # 生成地图
                    map_grid = generate_map(grid_size, num_obstacles, obstacle_size)
                    
                    # 放置机器人和目标
                    robots_positions, target_position = place_robots_and_target(map_grid.copy(), num_robots, robot_rectangle)
                    
                    # 记录初始机器人位置
                    initial_robots_positions = robots_positions[:]
                    
                    # 运行GWO算法
                    final_positions, paths, final_target_position, target_path, iteration = gwo_algorithm(
                        map_grid.copy(), robots_positions, target_position, max_iterations, step_size, target_step_size, scope, stop_step
                    )
                    
                    # 更新地图上的机器人位置
                    for pos in initial_robots_positions:
                        map_grid[pos] = 0
                    for pos in final_positions:
                        map_grid[pos] = 2
                    
                    # 记录本次迭代次数
                    iteration_to_plot.append(iteration)
                
                # 计算本次地图生成的平均迭代次数
                avg_iteration = sum(iteration_to_plot) / len(iteration_to_plot)
                avg_iterations_by_case[case].append(avg_iteration)
        
        # 计算每个网格大小的平均迭代次数
        for case in range(1, 9):
            avg_iterations_by_case[case] = sum(avg_iterations_by_case[case]) / len(avg_iterations_by_case[case])
    
    # 绘制折线图
    cases = list(avg_iterations_by_case.keys())
    avg_iterations = list(avg_iterations_by_case.values())
    
    plt.figure(figsize=(12, 8))
    
   # plt.figure(figsize=(12, 8))
    for i, case in enumerate(cases):
        plt.plot([grid_size], [avg_iterations[i]], marker='o', label=f'Case {case} - {case_descriptions[case]}')
    plt.title('Average Iterations vs Grid Size for Each Case')
    plt.xlabel('Grid Size')
    plt.ylabel('Average Iterations')
    plt.xticks(grid_test)
    plt.legend()
    plt.grid(True)
    plt.show()

def main4():
    random_seed = random.randint(1, 100)
    random.seed(random_seed)
    np.random.seed(random_seed)

    grid_size = 300
    num_robots = 5
    step_size = 2
    target_step_size = 0.8

    max_iterations = 300
    stop_step = step_size
    scope = 10

    avg_iterations_by_case = {case: [] for case in range(1, 9)}
    robot_rectangle = (1, 1, grid_size-2, grid_size-2)
    num_obstacles = int(grid_size / 20)
    obstacle_size = int(grid_size / 20)

    for case in range(1, 9):
        print(f'case {case}')
        iteration_to_plot = []

        for _ in range(5):
            map_grid = generate_map(grid_size, num_obstacles, obstacle_size)
            robots_positions, target_position = place_robots_and_target(
                map_grid.copy(), num_robots, robot_rectangle, case
            )
            
            try:
                final_positions, paths, final_target_position, target_path, iteration = gwo_algorithm(
                    map_grid.copy(), robots_positions, target_position, max_iterations, step_size, target_step_size, scope, stop_step
                )
                iteration_to_plot.append(iteration)
            except Exception as e:
                print(f"Error in Case {case}: {e}")
                iteration_to_plot.append(max_iterations)

        avg_iteration = sum(iteration_to_plot) / len(iteration_to_plot)
        avg_iterations_by_case[case].append(avg_iteration)

    final_avg_iterations_by_case = {}
    for case in range(1, 9):
        if avg_iterations_by_case[case]:
            final_avg_iterations_by_case[case] = sum(avg_iterations_by_case[case]) / len(avg_iterations_by_case[case])
        else:
            final_avg_iterations_by_case[case] = None

    for case, avg_iter in final_avg_iterations_by_case.items():
        if avg_iter is not None:
            print(f"Average iterations for Case {case}: {avg_iter:.2f}")
        else:
            print(f"Average iterations for Case {case}: No data")

    cases = list(final_avg_iterations_by_case.keys())
    avg_iterations = [final_avg_iterations_by_case[case] for case in cases if final_avg_iterations_by_case[case] is not None]

    plt.figure(figsize=(12, 8))
    plt.plot(cases, avg_iterations, marker='o', linestyle='-', color='red')

    plt.title('Average Iterations vs Distribution Case on a Fixed 300x300 Grid')
    plt.xlabel('Distribution Case')
    plt.ylabel('Average Iterations')
    plt.xticks(cases)
    plt.grid(True)
    plt.show()

def main():
    # 设置随机种子以保证结果可重复
    random_seed = random.randint(1, 100)
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(f"Random seed used: {random_seed}")  # 打印随机种子，方便调试和结果复现

    # 参数配置
    grid_size = 300
    num_robots = 5
    step_size = 2
    target_step_size = 0.8
    max_iterations = 300  # 最大迭代次数
    stop_step = step_size
    scope = 10

    # 定义机器人和障碍物相关参数
    robot_rectangle = (1, 1, grid_size - 2, grid_size - 2)
    num_obstacles = int(grid_size / 20)
    obstacle_size = int(grid_size / 20)

    # 存储每个CASE的平均迭代次数（仅支持1到6）
    avg_iterations_by_case = {case: None for case in range(1, 7)}

    # 遍历每种CASE
    for case in range(1, 7):
        print(f'time of {case}\n')
        iteration_to_plot = []

        # 重复实验5次
        for _ in range(5):
            try:
                # 生成地图
                map_grid = generate_map(grid_size, num_obstacles, obstacle_size)

                # 放置机器人和目标
                robots_positions, target_position = place_robots_and_target(
                    map_grid.copy(), num_robots, robot_rectangle, case
                )

                # 记录初始机器人位置
                initial_robots_positions = robots_positions[:]

                # 运行灰狼优化算法 (GWO)
                final_positions, paths, final_target_position, target_path, iteration = gwo_algorithm(
                    map_grid.copy(), robots_positions, target_position,
                    max_iterations, step_size, target_step_size, scope, stop_step
                )

                # 检查结果是否合理
                if not final_positions or iteration >= max_iterations:
                    print(f"Warning: Algorithm did not converge in case {case}")
                    continue

                # 更新地图上的机器人位置
                for pos in initial_robots_positions:
                    if 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size:
                        map_grid[pos] = 0  # 清除旧位置
                for pos in final_positions:
                    if 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size:
                        map_grid[pos] = 2  # 标记新位置

                # 记录当前实验的迭代次数
                iteration_to_plot.append(iteration)

            except Exception as e:
                print(f"Error occurred in case {case}: {e}")
                continue

        # 计算当前CASE的平均迭代次数
        if iteration_to_plot:
            avg_iterations_by_case[case] = sum(iteration_to_plot) / len(iteration_to_plot)

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

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("手动终止")
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_minute = elapsed_time / 60
    print(f"程序算时间: {elapsed_time:.10f} 秒")
    print(f'程序算时间: {elapsed_time_minute:.10f} 分')



#测试成功率
def test(iteration_to_plot):
    temp_all = len(iteration_to_plot)
    count_success = 0
    for iteration in iteration_to_plot:
        if iteration != -1:
            count_success += 1
    success_rate = count_success/temp_all
    #返回成功率
    return success_rate

#记录并输出成功率
def success_rate_output(iteration_to_plot,success_rates):
    iteration_success_rate = test(iteration_to_plot)
    success_rates.append(iteration_success_rate)
    #print(success_rates)
    return iteration_success_rate,success_rates

#迭代次数的测试
def test_iteration(iteration_all):
    pass







