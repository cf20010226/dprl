import sys
import gym
from gym import spaces
import airsim
from configparser import NoOptionError
import torch as th
import numpy as np
import math
import cv2
from collections import deque
import time
import random
import msgpackrpc
from .dynamics.multirotor_fusion import MultirotorDynamicsFuse
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from scipy.spatial.transform import Rotation

np.random.seed(0)
th.manual_seed(0)
th.cuda.manual_seed(0)


class AirsimGymEnv(gym.Env, QtCore.QThread):
    # ! ---------------------initialize pyqt-------------------------------------
    # PyQt signals for real-time visualization
    action_signal = pyqtSignal(int, np.ndarray)
    state_signal = pyqtSignal(int, np.ndarray)
    attitude_signal = pyqtSignal(int, np.ndarray, np.ndarray)
    component_signal = pyqtSignal(int, list)
    reward_signal = pyqtSignal(int, float, float)
    pose_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    # ! ---------------------initialization-------------------------------------
    def __init__(self) -> None:
        super().__init__()

        # Set NumPy and PyTorch print options for better readability
        np.set_printoptions(formatter={'float': '{: 4.2f}'.format}, suppress=True)
        th.set_printoptions(profile="short", sci_mode=False, linewidth=1000)
        print("init airsim-gym-env.")

        # Initialize position tracking
        self.previous_position = [0, 0, 0]
        self.reward_components = []
        self.min_distance_to_obstacles = 5

        # Buffers to determine successful hover state
        self.success_flag = False
        self.flag_buffer = deque(maxlen=20)
        self.trajectory_list = []

        # Initialize action stacks for reward computation
        self.action_stack = np.zeros((3, 4), dtype=float)

        # Metrics for performance and curriculum learning
        self.success_rate = 0
        self.total_episodes = 0
        self.successful_episodes = 0
        self.success_buffer = deque(maxlen=100)

        # --- [配置] 难度与障碍物数量映射 ---
        self.difficulty_level = 1
        self.difficulty_map = {
            1: 0,  # Level 1: 0个 (刚开始一定是空的，这很正常)
            2: 10,  # Level 2: 10个
            3: 15,  # Level 3: 20个
            4: 20,  # Level 4: 25个
            5: 25  # Level 5: 25个
        }
        self.current_obstacle_num = self.difficulty_map[self.difficulty_level]
        # 2. [新增] 每一级的最大平均耗时要求 (秒)
        # 只有当：成功率达标 且 平均耗时 <= 阈值 时，才允许升级
        self.time_threshold_map = {
            1: 12.0,
            2: 12.0,
            3: 13.0,
            4: 14.0,
            5: 15.0
        }

        # 3. [新增] 记录成功回合耗时的 Buffer
        self.success_time_buffer = deque(maxlen=100)

        # 4. [配置] 仿真步长 (假设 1 step = 0.1s，如果你的配置不同请修改这里)
        self.step_interval = 0.1
        # Parameters for obstacle spawning
        self.inner_radius = 10
        self.outer_radius = 50
        self.cylinder_radius = 2.5
        self.tree_types = [f"tree0{i}" for i in range(1, 4)]
        self.obstacle_type = "Shape_Cube"
        self.existing_positions = []
        self.obstacle_names = []

    def set_config(self, cfg):
        """
        Load configuration and initialize the environment.
        """
        # Read config options
        self.cfg = cfg
        self.env_name = cfg.get('options', 'env_name')
        self.dynamic_name = cfg.get('options', 'dynamic_name')
        self.perception_type = cfg.get('options', 'perception')
        self.privileged_info = cfg.get('options', 'privileged_info')

        print('Environment: ', self.env_name, "Dynamics: ", self.dynamic_name,
              'Perception: ', self.perception_type)

        # Initialize dynamic model
        if self.dynamic_name == 'MultirotorFuse':
            self.dynamic_model = MultirotorDynamicsFuse(cfg)
        else:
            raise Exception("Invalid dynamic_name!", self.dynamic_name)

        # Course learning options
        self.use_cl = cfg.getboolean('options', 'use_course_learning')
        self.generate_obstacles = cfg.getboolean('options', 'generate_obstacles')

        # Configure environment-specific parameters
        if self.env_name == 'Hover':
            start_position = [0, 0, 5]
            boundary = 80
            self.goal_distance = 2
            self.obstacle_range = [[0, 30], [50, 70]]
            self.work_space_x = [start_position[0] - boundary, start_position[0] + boundary]
            self.work_space_y = [start_position[1] - boundary, start_position[1] + boundary]
            self.work_space_z = [0.2, 15]
            self.max_episode_steps = 100
            self.dynamic_model.set_start(obstacle_range=self.obstacle_range, work_space_x=self.work_space_x,
                                         work_space_y=self.work_space_y)
            self.dynamic_model.set_goal(goal_distance=self.goal_distance)
            self.fly_range = 20
            self.hover_radius = 0.1

        elif self.env_name == 'Nav':
            start_position = [0, 0, 5]
            boundary = 80
            self.goal_distance = 65
            self.obstacle_range = [[20, 60]]
            self.work_space_x = [start_position[0] - boundary, start_position[0] + boundary]
            self.work_space_y = [start_position[1] - boundary, start_position[1] + boundary]
            self.work_space_z = [1, 15]
            self.max_episode_steps = 500
            self.dynamic_model.set_start(obstacle_range=self.obstacle_range, work_space_x=self.work_space_x,
                                         work_space_y=self.work_space_y)
            self.dynamic_model.set_goal(goal_distance=self.goal_distance)
            self.fly_range = boundary

        else:
            raise Exception("Invalid env_name!", self.env_name)

        # Set state-related attributes
        self.client = self.dynamic_model.client
        self.state_feature_length = self.dynamic_model.state_feature_length
        self.cnn_feature_length = self.cfg.getint('options', 'cnn_feature_num')

        # Initialize episode statistics
        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = 0
        self.previous_direction = 0

        # Load parameters
        self.crash_distance = cfg.getfloat('multirotor_new', 'crash_distance')
        self.accept_radius = cfg.getfloat('multirotor_new', 'accept_radius')
        self.max_depth_meters = cfg.getint('environment', 'max_depth_meters')
        self.screen_height = cfg.getint('environment', 'screen_height')
        self.screen_width = cfg.getint('environment', 'screen_width')

        # --- [逻辑修正] 难度初始化 ---
        if not self.use_cl:
            # 推理模式：直接满级
            print(">>> [Config] use_course_learning is False. Setting Max Difficulty (Level 5).")
            self.difficulty_level = 5
        else:
            # 训练模式：从 Level 1 开始
            self.difficulty_level = 1

        # 立即更新障碍物数量
        self.current_obstacle_num = self.difficulty_map[self.difficulty_level]

        # Generate obstacles
        if self.generate_obstacles:
            if self.env_name == 'Hover':
                self.obstacle_positions = self.generate_obstacle_positions(self.obstacle_range, min_distance=12,
                                                                           num_obstacles=30)
                self.set_all_obstacles()
            elif self.env_name == 'Nav':
                # 这里会根据上面的 current_obstacle_num 生成
                print(
                    f">>> [Init] Generating obstacles for Level {self.difficulty_level} (Count: {self.current_obstacle_num})")
                self.obstacle_positions = self.generate_obstacle_positions(self.obstacle_range, min_distance=10,
                                                                           num_obstacles=self.current_obstacle_num)
                self.set_all_obstacles()

        # Configure noise
        self.gaussian_std = cfg.getfloat('noise', 'gaussian_std')
        self.salt_pepper_prob = cfg.getfloat('noise', 'salt_pepper_prob')
        self.motion_blur_kernel_size = cfg.getint('noise', 'motion_blur_kernel_size')
        self.state_feature_noise_std = cfg.getfloat('noise', 'state_feature_noise_std')

        # Action and Obs space
        self.action_space = self.dynamic_model.action_space
        if self.perception_type == 'depth_noise':
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self.screen_height, self.screen_width, 3),
                                                dtype=np.uint8)
        elif self.perception_type == 'depth':
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self.screen_height, self.screen_width, 2),
                                                dtype=np.uint8)

        # Reward type
        try:
            self.reward_type = cfg.get('options', 'reward_type')
            print('Reward type: ', self.reward_type)
        except NoOptionError:
            self.reward_type = None

    # ! ---------------------key function-------------------------------------
    def reset(self):
        """
        Reset the environment for a new episode.
        """
        # ---------------------------------------------------------------------
        # 1. 课程学习逻辑 (关键修改：增加延时和同步)
        # ---------------------------------------------------------------------
        if self.use_cl and self.env_name == 'Nav':
            # 如果当前场景里的障碍物数量 与 当前难度要求不符，说明升级了，需要重置地图
            if len(self.obstacle_names) != self.current_obstacle_num:
                print(
                    f"\n[Reset] >>> DETECTED LEVEL CHANGE! Updating Map to Level {self.difficulty_level} ({self.current_obstacle_num} obstacles) <<<")

                # 1. 删除旧的
                self.delete_all_obstacles()

                # 2. [关键] 强制睡0.5秒，等待AirSim彻底清除物体，否则新生成的可能会重名冲突导致生成失败
                time.sleep(0.5)

                # 3. 清空列表
                self.obstacle_names = []

                # 4. 生成新的 (种子由 generate_obstacle_positions 内部根据 difficulty_level 设定)
                self.obstacle_positions = self.generate_obstacle_positions(
                    self.obstacle_range,
                    min_distance=10,
                    num_obstacles=self.current_obstacle_num
                )

                # 5. 放置
                self.set_all_obstacles()
                print(f"[Reset] >>> Map Update Complete. Current obstacles: {len(self.obstacle_names)}")

        # ---------------------------------------------------------------------
        # 2. 重置无人机
        # ---------------------------------------------------------------------
        self.dynamic_model.set_start(self.obstacle_range, self.work_space_x, self.work_space_y)
        self.dynamic_model.set_goal(self.goal_distance)
        self.dynamic_model.reset()

        # ---------------------------------------------------------------------
        # 3. 可视化优化
        # ---------------------------------------------------------------------
        self.client.simFlushPersistentMarkers()

        sp = self.dynamic_model.start_position  # [x, y, z]
        gp = self.dynamic_model.goal_position  # [x, y, z]

        pole_bottom = 2.0
        pole_top = -10.0

        p1_start = airsim.Vector3r(sp[0], sp[1], pole_bottom)
        p1_end = airsim.Vector3r(sp[0], sp[1], pole_top)
        p2_start = airsim.Vector3r(gp[0], gp[1], pole_bottom)
        p2_end = airsim.Vector3r(gp[0], gp[1], pole_top)

        try:
            self.client.simPlotLineList([p1_start, p1_end], color_rgba=[0.0, 0.0, 1.0, 1.0], thickness=20.0,
                                        is_persistent=True)
            self.client.simPlotLineList([p2_start, p2_end], color_rgba=[1.0, 0.0, 0.0, 1.0], thickness=20.0,
                                        is_persistent=True)
            self.client.simPlotLineList([p1_end, p2_end], color_rgba=[0.0, 1.0, 0.0, 0.5], thickness=10.0,
                                        is_persistent=True)
        except Exception as e:
            print(f"Viz Error: {e}")

        # ---------------------------------------------------------------------
        # 4. 重置计数器
        # ---------------------------------------------------------------------
        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.dynamic_model.goal_distance = self.dynamic_model.get_distance_to_goal_2d()
        self.previous_distance_from_des_point = self.dynamic_model.goal_distance
        self.previous_direction = 0
        self.trajectory_list = []

        obs = self.get_obs()
        return obs

    def step(self, action):
        # ... (前半部分保持不变，直到计算出 done 和 info) ...
        # Update action stack
        self.action_stack = np.roll(self.action_stack, shift=1, axis=0)
        self.action_stack[0] = action
        self.dynamic_model.set_action(action)
        self.trajectory_list.append(self.dynamic_model.get_position())
        obs = self.get_obs()

        # Determine status
        if self.env_name == 'Hover':
            # ... (Hover 逻辑不变) ...
            reached_goal = self.is_reached_goal()
            self.flag_buffer.append(reached_goal)
            self.success_flag = len(self.flag_buffer) == self.flag_buffer.maxlen and all(self.flag_buffer)
            done = self.is_done()
            info = {'is_success': self.success_flag, 'is_crash': self.is_crashed(),
                    'is_not_in_workspace': self.is_not_inside_fly_range(), 'step_num': self.step_num}
        elif self.env_name == 'Nav':
            done = self.is_done()
            info = {
                'is_success': self.is_in_desired_pose(),
                'is_crash': self.is_crashed(),
                'is_not_in_workspace': self.is_not_inside_fly_range(),
                'step_num': self.step_num
            }

        # ---------------------------------------------------------------------
        # [修改] 增加耗时记录与多重过滤升级逻辑
        # ---------------------------------------------------------------------
        if done:
            print(info)
            # 1. 记录胜负
            is_win = 1 if info['is_success'] else 0
            self.success_buffer.append(is_win)
            if info['is_success']:
                self.successful_episodes += 1

                # 2. [新增] 如果成功，记录本局耗时 (秒)
                # 计算公式：步数 * 步长
                episode_time = self.step_num * self.step_interval
                self.success_time_buffer.append(episode_time)

            self.total_episodes += 1
            self.update_success_rate()

            # --- 升级判定逻辑 ---
            target_rate = 0.8  # 目标胜率 80%
            min_episodes = 20  # 最少场次
            target_time = self.time_threshold_map.get(self.difficulty_level, 999.0)  # 获取当前难度的限时

            # 计算当前成功的平均耗时
            if len(self.success_time_buffer) > 0:
                avg_success_time = sum(self.success_time_buffer) / len(self.success_time_buffer)
            else:
                avg_success_time = 999.0  # 如果还没赢过，设为无限大

            # [Debug] 打印详细状态，让你知道卡在哪里
            print(f"[Check L{self.difficulty_level}] Rate: {self.success_rate:.2f}/{target_rate} | "
                  f"AvgTime: {avg_success_time:.1f}s/{target_time}s | "
                  f"Buffer: {len(self.success_buffer)}/{min_episodes}")

            # 判定条件：必须同时满足 1.课程模式 2.胜率达标 3.场次足够 4.平均耗时达标
            if (self.use_cl and
                    self.success_rate >= target_rate and
                    len(self.success_buffer) >= min_episodes and
                    avg_success_time <= target_time):

                print(f"Debug: >>> 所有指标达成 (Rate={self.success_rate:.2f}, Time={avg_success_time:.1f}s)，升级！ <<<")
                self.increase_difficulty()

            elif self.use_cl and self.success_rate >= target_rate and len(self.success_buffer) >= min_episodes:
                # 如果胜率够了但时间不够，打印提示
                print(f"Debug: 胜率已达标，但飞得太慢了！(当前: {avg_success_time:.1f}s > 目标: {target_time}s)")

        # ... (后面的 Reward 计算逻辑保持不变) ...
        # Compute reward
        if self.reward_type == 'reward_hover':
            reward, reward_list = self.compute_reward_hover(done, self.action_stack)
        elif self.reward_type == 'reward_distance':
            reward, reward_list = self.compute_reward_distance(done, self.action_stack)
        elif self.reward_type == 'reward_doge':
            reward, reward_list = self.compute_reward_doge(done, self.action_stack)

        self.cumulated_episode_reward += reward
        self.set_pyqt_signal_multirotor(action, reward, reward_list)
        self.step_num += 1
        self.total_step += 1

        return obs, reward, done, info

    # ! ---------------------curriculum related-------------------------------------
    def update_success_rate(self):
        """
        Update the success rate based on recent episodes.
        """
        if len(self.success_buffer) > 0:
            self.success_rate = sum(self.success_buffer) / len(self.success_buffer)
        else:
            self.success_rate = 0

    def increase_difficulty(self):
        if self.difficulty_level < 5:
            print(f"\n========================================================")
            print(f">>> [Curriculum] LEVEL UP! {self.difficulty_level} -> {self.difficulty_level + 1} <<<")
            print(f"========================================================")

            self.difficulty_level += 1
            self.current_obstacle_num = self.difficulty_map[self.difficulty_level]

            # [重要] 清空所有统计 Buffer
            self.success_buffer.clear()
            self.success_time_buffer.clear()  # <--- [新增] 清空耗时记录

            self.success_rate = 0
            self.successful_episodes = 0

            print(f">>> [Curriculum] Stats & Timers reset. Waiting for reset().")
        else:
            print(">>> [Curriculum] Max Level Reached!")

    # ! ---------------------generate obstacles-------------------------------------
    def generate_obstacle_positions(self, obstacle_range, min_distance=10, num_obstacles=30):
        """
        Generate obstacle positions.
        """
        # --- [核心修改] 使用 difficulty_level 作为种子 ---
        # 这样 Level 2 的地图永远是一样的，Level 3 也是固定的，但 2 和 3 不一样。
        # 如果是推理模式 (Level 5)，地图也是固定的。
        seed_value = self.difficulty_level
        print(f"[MapGen] Using fixed seed based on difficulty: {seed_value}")

        random.seed(seed_value)
        np.random.seed(seed_value)

        obstacle_positions = []
        # 双环生成逻辑
        if len(obstacle_range) == 2:
            while len(obstacle_positions) < num_obstacles:
                chosen_ring = random.choice([0, 1])
                radius = random.uniform(obstacle_range[chosen_ring][0], obstacle_range[chosen_ring][1])
                angle = random.uniform(0, 2 * math.pi)
                x, y, z = radius * math.cos(angle), radius * math.sin(angle), 2

                valid_position = all(
                    math.sqrt((x - pos.x_val) ** 2 + (y - pos.y_val) ** 2) >= min_distance
                    for pos in obstacle_positions
                )
                if valid_position:
                    obstacle_positions.append(airsim.Vector3r(x, y, z))
        # 单环生成逻辑
        else:
            while len(obstacle_positions) < num_obstacles:
                radius = random.uniform(obstacle_range[0][0], obstacle_range[0][1])
                angle = random.uniform(0, 2 * math.pi)
                x, y, z = radius * math.cos(angle), radius * math.sin(angle), 2
                valid_position = all(
                    math.sqrt((x - pos.x_val) ** 2 + (y - pos.y_val) ** 2) >= min_distance
                    for pos in obstacle_positions
                )
                if valid_position:
                    obstacle_positions.append(airsim.Vector3r(x, y, z))

        self.obstacle_positions = obstacle_positions
        return obstacle_positions

    def set_all_obstacles(self):
        """
        Spawn obstacles in AirSim.
        """
        obstacle_positions = self.obstacle_positions
        if len(obstacle_positions) == 0:
            print("[Spawn] No obstacles to spawn (Level 1?).")
            return

        i = 0
        while i < len(obstacle_positions):
            # 分批生成，防止拥堵
            for _ in range(10):
                if i >= len(obstacle_positions):
                    break
                position = obstacle_positions[i]
                pose = airsim.Pose(position, airsim.Quaternionr(0, 0, 0, 1))

                obstacle_type = random.choices(
                    ["Shape_Cylinder", "Shape_Cube", "tree"],
                    weights=[0.3, 0.3, 0.4],
                    k=1
                )[0]

                if obstacle_type == "Shape_Cylinder":
                    obstacle_name = f"{obstacle_type}_Obstacle_{i + 1}"
                    scale = airsim.Vector3r(3, 3, 12)
                    segmentation_id = 10
                elif obstacle_type == "Shape_Cube":
                    obstacle_name = f"{obstacle_type}_Obstacle_{i + 1}"
                    scale = airsim.Vector3r(3, 3, 12)
                    segmentation_id = 15
                elif obstacle_type == "tree":
                    selected_tree = random.choice(self.tree_types)
                    obstacle_name = f"{selected_tree}_Obstacle_{i + 1}"
                    scale = airsim.Vector3r(1, 1, random.uniform(1.2, 1.8))
                    segmentation_id = 20
                else:
                    continue

                self.obstacle_names.append(obstacle_name)

                try:
                    self.dynamic_model.client.simSpawnObject(
                        obstacle_name,
                        selected_tree if obstacle_type == "tree" else obstacle_type,
                        pose,
                        scale
                    )
                    self.dynamic_model.client.simSetSegmentationObjectID(obstacle_name, segmentation_id, True)
                    # print(f"Spawned {obstacle_name} at {position}") # 太多了可以注释掉
                except Exception as e:
                    print(f"Failed to spawn {obstacle_name}: {e}")
                i += 1

            # 这里的 sleep 可以保留，防止一次发太多请求卡死
            time.sleep(0.2)

        print(f"[Spawn] Successfully spawned {len(self.obstacle_names)} obstacles.")

    def delete_all_obstacles(self):
        """
        Delete all spawned obstacles.
        """
        if not self.obstacle_names:
            return

        print(f"[Delete] Removing {len(self.obstacle_names)} obstacles...")
        for obstacle_id in self.obstacle_names:
            self.dynamic_model.client.simDestroyObject(obstacle_id)
        # 不在这里清空 list，在调用处清空，更安全

    # ! ---------------------auxiliary methods-------------------------------------
    def get_min_distance_to_obstacles(self):
        drone_position = self.dynamic_model.get_position()
        drone_x, drone_y, drone_z = drone_position
        min_distance = float('inf')

        # 如果没有障碍物（Level 1），返回无穷大
        if not self.obstacle_positions:
            return 999.0

        for obstacle in self.obstacle_positions:
            distance = math.sqrt((drone_x - obstacle.x_val) ** 2 + (drone_y - obstacle.y_val) ** 2)
            distance_to_surface = max(0, distance - self.cylinder_radius)
            if distance_to_surface < min_distance:
                min_distance = distance_to_surface
        return min_distance

    # ! ---------------------get obs-------------------------------------
    def get_obs(self):
        if self.perception_type == 'depth_noise':
            obs = self.get_obs_depth_noise()
        elif self.perception_type == 'depth':
            obs = self.get_obs_image()
        return obs

    def get_obs_image(self):
        image = self.get_depth_image("FrontDepthCamera")
        image_resize = cv2.resize(image, (self.screen_width, self.screen_height))
        self.min_distance_to_obstacles = image.min()
        image_scaled = np.clip(image_resize, 0, self.max_depth_meters) / self.max_depth_meters * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        state_feature_array = np.zeros((self.screen_height, self.screen_width))
        state_feature = self.dynamic_model._get_state_feature()
        state_feature_array[0, 0:self.state_feature_length] = state_feature

        image_with_state = np.array([image_uint8, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2).swapaxes(0, 1)

        self.feature_all = image_with_state
        return image_with_state

    def get_obs_depth_noise(self):
        image = self.get_depth_image("FrontDepthCamera")
        image_resize = cv2.resize(image, (self.screen_width, self.screen_height))

        # 障碍物距离逻辑优化
        if self.generate_obstacles and self.obstacle_positions:
            self.min_distance_to_obstacles = self.get_min_distance_to_obstacles()
        else:
            # 如果没有生成障碍物（Level 1），直接用图像最小值，或者给个大数
            self.min_distance_to_obstacles = image.min() if image.size > 0 else 100.0

        image_scaled = np.clip(image_resize, 0, self.max_depth_meters) / self.max_depth_meters * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        # Salt-and-pepper noise
        salt_pepper_image = image_uint8.copy()
        num_salt = np.ceil(self.salt_pepper_prob * image_uint8.size * 0.5)
        num_pepper = np.ceil(self.salt_pepper_prob * image_uint8.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_uint8.shape]
        salt_pepper_image[coords[0], coords[1]] = 255
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_uint8.shape]
        salt_pepper_image[coords[0], coords[1]] = 0

        # Gaussian noise
        gaussian_noise = np.random.normal(0, self.gaussian_std, salt_pepper_image.shape)
        gaussian_noise = np.clip(gaussian_noise, -30, 30).astype(np.uint8)
        noisy_image = cv2.add(salt_pepper_image, gaussian_noise)
        noisy_image = np.clip(noisy_image, 0, 255)

        # Motion blur
        kernel_motion_blur = np.zeros((self.motion_blur_kernel_size, self.motion_blur_kernel_size))
        kernel_motion_blur[int((self.motion_blur_kernel_size - 1) / 2), :] = np.ones(self.motion_blur_kernel_size)
        kernel_motion_blur /= self.motion_blur_kernel_size
        blurred_image = cv2.filter2D(noisy_image, -1, kernel_motion_blur)

        # State features and noise
        state_feature_array = np.zeros((self.screen_height, self.screen_width))
        state_feature = self.dynamic_model._get_state_feature()
        state_feature_noise = np.random.normal(0, self.state_feature_noise_std, state_feature.shape)
        state_feature_noise = np.clip(state_feature_noise, -5, 5)
        state_feature_with_noise = np.clip(state_feature + state_feature_noise, 0, 255)

        state_feature_array[0, 0:self.state_feature_length] = state_feature
        state_feature_array[0, self.state_feature_length:2 * self.state_feature_length] = state_feature_with_noise

        image_with_state = np.array([image_uint8, blurred_image, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2).swapaxes(0, 1)

        self.feature_all = image_with_state
        return image_with_state

    # ! ---------------------image utils-------------------------------------
    def get_depth_image(self, camera_name):
        while True:
            try:
                responses = self.client.simGetImages([
                    airsim.ImageRequest(camera_name, airsim.ImageType.DepthVis, True)
                ])

                if responses[0].width != 0:
                    depth_img = airsim.list_to_2d_float_array(
                        responses[0].image_data_float, responses[0].width, responses[0].height)
                    depth_meter = depth_img * 100
                    return depth_meter
                else:
                    print("get_image_fail...")
                    time.sleep(0.1)

            except Exception as e:
                print(f"Error getting image: {e}. Retrying...")
                time.sleep(0.1)

    # (Other image functions preserved but omitted for brevity if unused, add back if needed)

    # ! ---------------------calculate rewards-------------------------------------
    def compute_reward_hover(self, done, action_stack):
        reward = 0
        reward_success = 2
        reward_crash = -5
        reward_outside = -5
        reward_list = []
        if not done:
            if self.success_flag:
                reward = reward_success
                reward_list = [0, 0, 0]
            elif self.is_reached_goal():
                reward = 2
                reward_list = [0, 0, 0]
            else:
                distance_now = self.get_distance_to_goal_3d()
                reward_distance = self.previous_distance_from_des_point - distance_now
                reward_distance = np.clip(reward_distance, -1, 1)
                self.previous_distance_from_des_point = distance_now

                reward_proximity = (0.5 - distance_now) * 2 if distance_now < 0.5 else 0

                curr_roll, curr_pitch, curr_yaw = self.dynamic_model.get_attitude()
                current_quaternion = Rotation.from_euler('xyz', [curr_roll, curr_pitch, curr_yaw]).as_quat()
                goal_quaternion = self.dynamic_model.goal_orientation
                quaternion_error = Rotation.from_quat(goal_quaternion).inv() * Rotation.from_quat(current_quaternion)
                relative_error = quaternion_error.magnitude() / 2
                punishment_orientation = np.clip(relative_error, 0, 1)

                reward = 10.0 * reward_distance
                reward_list = [3.0 * reward_distance, -0.3 * punishment_orientation, reward_proximity]
                reward = np.clip(reward, -1, 1)
        else:
            reward_list = [0, 0, 0]
            reward = reward_success if self.success_flag else reward_outside
        return reward, reward_list

    def compute_reward_distance(self, done, action_stack):
        reward = 0
        reward_reach = 10
        reward_crash = -5
        reward_outside = -5
        action = action_stack[0]
        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = self.previous_distance_from_des_point - distance_now
            reward_distance = np.clip(reward_distance, -1, 1)
            self.previous_distance_from_des_point = distance_now

            direction_now = self.dynamic_model._get_relative_yaw()
            self.previous_direction = direction_now

            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x, y, z = current_pose
            x_g, y_g, z_g = goal_pose
            punishment_xy = np.clip(self.getDis(x, y, 0, 0, x_g, y_g) / 10, 0, 1)
            punishment_z = 2 * abs(np.clip((z - z_g) / 5, -1, 1))
            punishment_pose = punishment_xy + punishment_z

            error_now = self.dynamic_model._get_vector_angle()
            punishment_angle = abs(np.clip(error_now / (2 * math.pi), -1, 1))

            if self.min_distance_to_obstacles < 4:
                punishment_obs = 1 - np.clip((self.min_distance_to_obstacles - self.crash_distance) / 3, 0, 1)
            else:
                punishment_obs = 0

            reward = 5.0 * reward_distance - 0.3 * punishment_pose - punishment_angle - punishment_obs
            reward = np.clip(reward / 3, -1, 1)
            reward_list = [5.0 * reward_distance, -0.3 * punishment_pose, - punishment_obs]
        else:
            reward_list = [0, 0, 0]
            if self.step_num < 50:
                reward = -10
            elif self.is_in_desired_pose():
                reward = reward_reach
            elif self.is_not_inside_fly_range():
                reward = reward_outside
            else:
                reward = reward_crash
        return float(reward), reward_list

    def compute_reward_doge(self, done, action_stack):
        reward = 0
        reward_reach = 10
        reward_crash = -5
        reward_outside = -5
        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = self.previous_distance_from_des_point - distance_now
            reward_distance = np.clip(reward_distance, -1, 1)
            self.previous_distance_from_des_point = distance_now

            if self.min_distance_to_obstacles < 4:
                punishment_obs = 1 - np.clip((self.min_distance_to_obstacles - self.crash_distance) / 3, 0, 1)
            else:
                punishment_obs = 0

            error_now = self.dynamic_model._get_vector_angle()
            punishment_angle = abs(np.clip(error_now / (2 * math.pi), -1, 1))

            reward = 5.0 * reward_distance - punishment_obs - punishment_angle
            reward = np.clip(reward / 3, -1, 1)
            reward_list = [5.0 * reward_distance, - punishment_obs, - punishment_angle]
        else:
            reward_list = [0, 0, 0]
            if self.step_num < 50:
                reward = -10
            elif self.is_in_desired_pose():
                reward = reward_reach
            elif self.is_not_inside_fly_range():
                reward = reward_outside
            else:
                reward = reward_crash
        return float(reward), reward_list

    # ! ---------------------done check-------------------------------------
    def is_done(self):
        if self.env_name == 'Hover':
            return (self.is_not_inside_fly_range() or
                    self.is_crashed() or
                    self.step_num >= self.max_episode_steps)
        elif self.env_name == 'Nav':
            return (self.is_not_inside_fly_range() or
                    self.is_crashed() or
                    self.is_in_desired_pose() or
                    self.step_num >= self.max_episode_steps)
        return False

    def is_in_desired_pose(self):
        current_pos = self.dynamic_model.get_position()
        goal_pos = self.dynamic_model.goal_position
        dist_xy = math.sqrt((current_pos[0] - goal_pos[0]) ** 2 + (current_pos[1] - goal_pos[1]) ** 2)
        dist_z = abs(current_pos[2] - goal_pos[2])
        if dist_xy < self.accept_radius and dist_z < 5.0:
            return True
        return False

    def is_not_inside_fly_range(self):
        current_position = self.dynamic_model.get_position()
        distance = math.sqrt(
            (current_position[0] - self.dynamic_model.start_position[0]) ** 2 +
            (current_position[1] - self.dynamic_model.start_position[1]) ** 2
        )
        if distance > self.fly_range or current_position[2] < self.work_space_z[0] or \
                current_position[2] > self.work_space_z[1]:
            return True
        return False

    def is_crashed(self):
        collision_info = self.client.simGetCollisionInfo("SimpleFlight")
        return collision_info.has_collided

    def is_reached_goal(self):
        current_position = self.dynamic_model.get_position()
        distance = math.sqrt(
            (current_position[0] - self.dynamic_model.goal_position[0]) ** 2 +
            (current_position[1] - self.dynamic_model.goal_position[1]) ** 2 +
            (current_position[2] - self.dynamic_model.goal_position[2]) ** 2
        )
        return distance < self.hover_radius

    # ! ----------- useful functions-------------------------------------------
    def get_distance_to_goal_3d(self):
        current_pose = self.dynamic_model.get_position()
        goal_pose = self.dynamic_model.goal_position
        return math.sqrt(sum((c - g) ** 2 for c, g in zip(current_pose, goal_pose)))

    def getDis(self, pointX, pointY, lineX1, lineY1, lineX2, lineY2):
        a = lineY2 - lineY1
        b = lineX1 - lineX2
        c = lineX2 * lineY1 - lineX1 * lineY2
        return (math.fabs(a * pointX + b * pointY + c)) / (math.pow(a * a + b * b, 0.5))

    def set_pyqt_signal_multirotor(self, action, reward, reward_list):
        step = int(self.total_step)
        state = self.dynamic_model.state_raw
        self.action_signal.emit(step, action)
        self.state_signal.emit(step, state)
        self.attitude_signal.emit(step, np.asarray(self.dynamic_model.get_attitude()),
                                  np.asarray(self.dynamic_model.get_attitude_cmd()))
        self.reward_signal.emit(step, reward, self.cumulated_episode_reward)
        self.component_signal.emit(step, reward_list)
        self.pose_signal.emit(np.asarray(self.dynamic_model.goal_position),
                              np.asarray(self.dynamic_model.start_position),
                              np.asarray(self.dynamic_model.get_position()),
                              np.asarray(self.trajectory_list))