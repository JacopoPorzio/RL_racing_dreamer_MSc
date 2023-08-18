import math
import random
from dataclasses import dataclass
from typing import Dict, Any, List

import gymnasium
import numpy as np
import pybullet as p
from gymnasium import logger

from racecar_gym.bullet import util
from racecar_gym.bullet.configs import MapConfig
from racecar_gym.bullet.positioning import AutomaticGridStrategy, RandomPositioningStrategy, \
    RandomPositioningWithinBallStrategy
from racecar_gym.core import world
from racecar_gym.core.agent import Agent
from racecar_gym.core.definitions import Pose
from racecar_gym.core.gridmaps import GridMap
import matplotlib.pyplot as plt


class World(world.World):
    FLOOR_ID = 0
    WALLS_ID = 1
    FINISH_ID = 2

    @dataclass
    class Config:
        name: str
        sdf: str
        map_config: MapConfig
        rendering: bool
        time_step: float
        gravity: float

    def __init__(self, config: Config, agents: List[Agent]):
        self._config = config
        self._map_id = None
        self._time = 0.0
        self._agents = agents
        self._state = dict([(a.id, {}) for a in agents])
        self._objects = {}
        self._starting_grid = np.load(config.map_config.starting_grid)['data']
        self._maps = dict([
            (name, GridMap(
                grid_map=np.load(config.map_config.maps)[data],
                origin=self._config.map_config.origin,
                resolution=self._config.map_config.resolution
            ))
            for name, data
            in [
                ('progress', 'norm_distance_from_start'),
                ('obstacle', 'norm_distance_to_obstacle'),
                ('occupancy', 'drivable_area')
            ]
        ])

        # self.angle_map()
        # self.curvature_maps = self.curvature_map(self._maps)
        self._maps.update({'angle': self.angle_map()})
        curvature_map, k_max, k_min = self.curvature_map(self._maps)
        self._maps.update({'curvature': curvature_map})
        self._maps['curvature'].k_max = k_max
        self._maps['curvature'].k_min = k_min
        self._maps['progress'].reduced_map = self._maps['progress'].map*((self._maps['occupancy'].map * self._maps['obstacle'].map) >= 0.9)
        self._state['maps'] = self._maps
        self._tmp_occupancy_map = None      # used for `random_ball` sampling
        self._progress_center = None        # used for `random_ball` sampling
        self._trajectory = []

    def init(self) -> None:
        if self._config.rendering:
            id = -1  # p.connect(p.SHARED_MEMORY)
            if id < 0:
                p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self._load_scene(self._config.sdf)
        p.setTimeStep(self._config.time_step)
        p.setGravity(0, 0, self._config.gravity)

    def reset(self):
        p.setTimeStep(self._config.time_step)
        p.setGravity(0, 0, self._config.gravity)
        p.stepSimulation()
        self._time = 0.0
        self._state = dict([(a.id, {}) for a in self._agents])

    def _load_scene(self, sdf_file: str):
        ids = p.loadSDF(sdf_file)
        objects = dict([(p.getBodyInfo(i)[1].decode('ascii'), i) for i in ids])
        self._objects = objects

    def get_starting_position(self, agent: Agent, mode: str) -> Pose:
        start_index = list(map(lambda agent: agent.id, self._agents)).index(agent.id)
        if mode == 'grid':
            strategy = AutomaticGridStrategy(obstacle_map=self._maps['obstacle'], number_of_agents=len(self._agents))
        elif mode == 'random':
            strategy = RandomPositioningStrategy(progress_map=self._maps['progress'],
                                                 obstacle_map=self._maps['obstacle'], alternate_direction=False)
        elif mode == 'random_bidirectional':
            strategy = RandomPositioningStrategy(progress_map=self._maps['progress'],
                                                 obstacle_map=self._maps['obstacle'], alternate_direction=True)
        elif mode == 'random_ball':
            progress_radius = 0.05
            min_distance_to_wall = 0.9
            progress_map = self._maps['progress'].map
            obstacle_map = self._maps['obstacle'].map
            if start_index == 0:    # on first agent, compute a fixed interval for sampling and copy occupancy map
                progresses = progress_map[obstacle_map > min_distance_to_wall]                                  # center has enough distance from the walls
                progresses = progresses[(progresses > progress_radius) & (progresses < (1-progress_radius))]    # center+-radius in [0,1]
                self._progress_center = np.random.choice(progresses)
                self._tmp_occupancy_map = self._maps['occupancy'].map.copy()
            strategy = RandomPositioningWithinBallStrategy(progress_map=self._maps['progress'],
                                                           obstacle_map=self._maps['obstacle'],
                                                           drivable_map=self._tmp_occupancy_map,
                                                           progress_center=self._progress_center,
                                                           progress_radius=progress_radius,
                                                           min_distance_to_obstacle=min_distance_to_wall)
        else:
            raise NotImplementedError(mode)
        position, orientation = strategy.get_pose(agent_index=start_index)
        if mode == 'random_ball':  # mark surrounding pixels as occupied
            px, py = self._maps['obstacle'].to_pixel(position)
            neigh_sz = int(1.0 / self._maps['obstacle'].resolution)  # mark 1 meter around the car
            self._tmp_occupancy_map[px - neigh_sz:px + neigh_sz, py - neigh_sz:py + neigh_sz] = False
        return position, orientation

    def update(self):
        p.stepSimulation()
        self._time += self._config.time_step

    def state(self) -> Dict[str, Any]:
        for agent in self._agents:
            self._update_race_info(agent=agent)

        self._update_ranks()

        return self._state

    def space(self) -> gymnasium.Space:
        return gymnasium.spaces.Dict({
            'time': gymnasium.spaces.Box(low=0, high=math.inf, shape=(1,))
        })

    def _update_race_info(self, agent):
        contact_points = set([c[2] for c in p.getContactPoints(agent.vehicle_id)])
        progress_map = self._maps['progress']
        obstacle_map = self._maps['obstacle']
        angle_map = self._maps['angle']
        pose = util.get_pose(id=agent.vehicle_id)
        if pose is None:
            logger.warn('Could not obtain pose.')
            self._state[agent.id]['pose'] = np.append((0, 0, 0), (0, 0, 0))
        else:
            self._state[agent.id]['pose'] = pose
        collision_with_wall = False
        opponent_collisions = []
        opponents = dict([(a.vehicle_id, a.id) for a in self._agents])
        for contact in contact_points:
            if self._objects['walls'] == contact:
                collision_with_wall = True
            elif contact in opponents:
                opponent_collisions.append(opponents[contact])

        self._state[agent.id]['wall_collision'] = collision_with_wall
        self._state[agent.id]['opponent_collisions'] = opponent_collisions
        velocity = util.get_velocity(id=agent.vehicle_id)
        wheels_velocity = util.get_wheels_velocity(id=agent.vehicle_id)

        if 'velocity' in self._state[agent.id]:
            previous_velocity = self._state[agent.id]['velocity']
            self._state[agent.id]['acceleration'] = (velocity - previous_velocity) / self._config.time_step
        else:
            self._state[agent.id]['acceleration'] = velocity / self._config.time_step

        pose = self._state[agent.id]['pose']
        progress = progress_map.get_value(position=(pose[0], pose[1], 0))
        dist_obstacle = obstacle_map.get_value(position=(pose[0], pose[1], 0))
        track_angle = angle_map.get_value(position=(pose[0], pose[1], 0))
        curvature = self.curvature_lookahead(pose, progress)

        self._state[agent.id]['velocity'] = velocity
        self._state[agent.id]['wheels_velocity'] = wheels_velocity
        self._state[agent.id]['progress'] = progress
        self._state[agent.id]['obstacle'] = dist_obstacle
        self._state[agent.id]['angle'] = track_angle
        self._state[agent.id]['curvature'] = curvature
        self._state[agent.id]['time'] = self._time

        progress = self._state[agent.id]['progress']
        checkpoints = 1.0 / float(self._config.map_config.checkpoints)
        checkpoint = int(progress / checkpoints)

        if 'checkpoint' in self._state[agent.id]:
            last_checkpoint = self._state[agent.id]['checkpoint']
            if last_checkpoint + 1 == checkpoint:
                self._state[agent.id]['checkpoint'] = checkpoint
                self._state[agent.id]['wrong_way'] = False
            elif last_checkpoint - 1 == checkpoint:
                self._state[agent.id]['wrong_way'] = True
            elif last_checkpoint == self._config.map_config.checkpoints and checkpoint == 0:
                self._state[agent.id]['lap'] += 1
                self._state[agent.id]['checkpoint'] = checkpoint
                self._state[agent.id]['wrong_way'] = False
            elif last_checkpoint == 0 and checkpoint == self._config.map_config.checkpoints:
                self._state[agent.id]['wrong_way'] = True
        else:
            self._state[agent.id]['checkpoint'] = checkpoint
            self._state[agent.id]['lap'] = 1
            self._state[agent.id]['wrong_way'] = False

    def _update_ranks(self):

        agents = [
            (agent_id, self._state[agent_id]['lap'], self._state[agent_id]['progress'])
            for agent_id
            in map(lambda a: a.id, self._agents)
        ]

        ranked = [item[0] for item in sorted(agents, key=lambda item: (item[1], item[2]), reverse=True)]

        for agent in self._agents:
            rank = ranked.index(agent.id) + 1
            self._state[agent.id]['rank'] = rank

    def render(self, agent_id: str, mode: str, width=640, height=480) -> np.ndarray:
        agent = list(filter(lambda a: a.id == agent_id, self._agents))
        assert len(agent) == 1
        agent = agent[0]
        if mode == 'follow':
            return util.follow_agent(agent=agent, width=width, height=height)
        elif mode == 'birds_eye':
            return util.birds_eye(agent=agent, width=width, height=height)

    def seed(self, seed: int = None):
        if self is None:
            seed = 0
        np.random.seed(seed)
        random.seed(seed)

    def curvature_lookahead(self, position, progress):
        import time
        lookahead_progress = 5
        delta_p = 0.002
        eps = 0.0001
        poseX = position[0]
        poseY = position[1]
        curr_curvature = self._maps['curvature'].get_value(position=(poseX, poseY, 0))
        curvature = []
        curvature.append(curr_curvature)
        # c_points = []  # They serve only for plot.

        for i in range(lookahead_progress):
            dp = (i + 1) * delta_p
            prg = progress + dp
            # what happens when prg -> 1?
            low_bound = prg - eps
            up_bound = prg + eps
            idx = np.argwhere(np.logical_and(self._maps['curvature'].centerline_parametrization['p'] > low_bound,
                                             self._maps['curvature'].centerline_parametrization['p'] <= up_bound))
            """""
            if len(idx) > 1:
                idx = idx[0]
                # or
                # idx = idx[-1]
            """""
            if len(idx) > 0:
                idx = idx[0]
                proj_curv = self._maps['curvature'].map[
                            self._maps['curvature'].centerline_parametrization['row'][idx],
                            self._maps['curvature'].centerline_parametrization['column'][idx]
                                                       ]
                # appnd = np.squeeze(
                #         np.array([
                #                   self._maps['curvature'].centerline_parametrization['column'][idx],
                #                   self._maps['curvature'].centerline_parametrization['row'][idx]
                #                   ])
                #                   )
            else:
                proj_curv = 0
                # Hacky exception handling: I know from track geometry that the curvature is 0,
                # when progress is close to 0 and 1.
                # idx = [], when progress is (up_bound > 1 or low_bound > 1).
                # More elegant alternative is:
                # centerline_parametrization['p'].append(centerline_parametrization['p'] + 1)
                # centerline_parametrization['row'].append(centerline_parametrization['row'])
                # centerline_parametrization['column'].append(centerline_parametrization['column'])
                # Nevertheless, this isn't enough, because I've manually tested a case
                # where low_bound > 1 and up_bound > 1, I've subtracted 1 to both of them and idx = [].
                # I could only use `interp` to have denser p, row and column, also losing
                # the meaning of row and column as pixels, because they stop being `int` and become `float`:
                # I would no more be able to enter the matrix self._maps['curvature'].map!
                # This solution is fine.

            curvature.append(proj_curv)
            # c_points.append(appnd)

        # fig101 = plt.figure(110)
        # px = int(2000 - (poseY + 50) / 0.05)
        # py = int((poseX + 50) / 0.05)
        # c_points = np.array(c_points)
        # plt.imshow(self._maps['occupancy'].map)
        # plt.scatter(py, px)
        # plt.scatter(c_points[:, 0], c_points[:, 1])
        # fig101.show()

        curvature = np.array(curvature, dtype='float32')

        return curvature

    def unwrap_angles(self, angles):
        unwrapped_angles = np.copy(angles)
        threshold = np.pi  # Adjust the threshold based on your data

        for i in range(1, len(angles)):
            diff = angles[i] - angles[i - 1]
            if np.abs(diff) > threshold:
                unwrapped_angles[i:] -= 2 * np.pi * np.sign(diff)

        return unwrapped_angles

    def rewrap_angles(self, angles, range_start=-np.pi, range_end=np.pi):
        wrapped_angles = (angles - range_start) % (range_end - range_start) + range_start
        return wrapped_angles

    def angle_map(self):
        # Now duplicate, then one external function thinning the image!
        import cv2
        from scipy.interpolate import interp1d
        img = self._maps['occupancy'].map
        # img = cv2.resize(np.uint8(img), [6000, 6000])
        img = np.uint8(img)
        centerline_img = cv2.ximgproc.thinning(img * 255)

        # Find contours of the centerline image.
        # contours, _ = cv2.findContours(centerline_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, _ = cv2.findContours(centerline_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Sort contours by size and choose one
        sel_cont = 'inner'  # difference is very small and can be eliminated tweaking `div` and `shift`
        if sel_cont == 'outer':
            contour = max(contours, key=cv2.contourArea)
        elif sel_cont == 'inner':
            contour = min(contours, key=cv2.contourArea)
        else:
            raise Exception('Error, not implemented!')

        # Using interpolation to increase the number of points isn't that smart:
        # at the end, the GridMap will refer poseX and poseY to pixels inside the 2000x2000 matrix.
        # We need an under-sampling, as we have too many points.
        contour = contour[::3, :, :]  # 5 is nice, 4 is good!

        # Centerline from pixel to meters.
        origin_x, origin_y = self._maps['occupancy']._origin[0], self._maps['occupancy']._origin[1]
        resolution = self._maps['occupancy']._resolution
        x_c = []
        y_c = []
        p_v = []
        for px, py in contour.squeeze():
            x = -(origin_y - (px - self._maps['occupancy']._height) * resolution)
            y = -(py * resolution + origin_x)
            x_c.append(x)
            y_c.append(y)
            p_v.append(self._maps['progress'].map[py, px])
        x_c = np.array(x_c)
        y_c = np.array(y_c)
        p_v = np.array(p_v)

        # offset = 3
        # diffx = []
        # diffy = []
        # for i in range(len(x_c) - offset):
        #     diffx.append(x_c[i + offset] - x_c[i])
        #     diffy.append(y_c[i + offset] - y_c[i])
        # diffx = np.array(diffx)
        # diffy = np.array(diffy)

        diffx = np.diff(x_c)
        diffy = np.diff(y_c)
        angles = np.arctan2(diffy, diffx)
        # angles = self.remove_outliers(angles, 0.8, 4)  # 2 seems nice, 4 looks good as well
        # Results with up-scaling:
        # nice with 2, even better with 4, close to perfection with 6, very good 8
        # angles = self.moving_average(angles, 4)

        # Unwrap, process, re-wrap the unwrapped and processed elements
        angles_2 = self.unwrap_angles(angles)
        angles_2 = np.array(angles_2)
        angles_3 = angles_2
        angles_3 = self.remove_outliers(angles_3, 0.8, 4)
        angles_3 = self.moving_average(angles_3, 2)
        angles = self.rewrap_angles(angles_3)
        # End - unwrap, process, re-wrap the unwrapped and processed elements

        angles[0] = angles[1]

        # figurona = plt.figure()
        # vmin = min(angles)*180/np.pi
        # vmax = max(angles)*180/np.pi
        # plt.imshow(img)
        # sc = plt.scatter(contour[:-1, 0, 0], contour[:-1, 0, 1], c=angles*180/np.pi, cmap='rainbow', vmin=vmin, vmax=vmax, s=10)
        # # sc = plt.scatter(contour[:-1-(offset-1), 0, 0], contour[:-1-(offset-1), 0, 1], c=angles * 180 / np.pi, cmap='rainbow', vmin=vmin, vmax=vmax, s=10)
        # cbar = plt.colorbar(sc)
        # cbar.set_label('Angle')
        # plt.tight_layout()
        # figurona.show()

        # figu = plt.figure(2)
        # plt.plot(p_v[:-1], angles*180/np.pi)
        # figu.show()

        angle_map = self._maps['occupancy'].map.astype('float64')
        track_indices = np.argwhere(self._maps['occupancy'].map > 0)
        a_p_row = contour[:-1, 0, 0]
        a_p_col = contour[:-1, 0, 1]

        for i in range(len(track_indices)):
            row = track_indices[i, 1]
            col = track_indices[i, 0]

            dist = ((a_p_row - row) ** 2 + (a_p_col - col) ** 2) ** (1 / 2)
            idx = np.argmin(dist)

            angle_map[col, row] = angles[idx]

        # figurina = plt.figure(3)
        # plt.imshow(angle_map)
        # figurina.show()

        a_map = GridMap(angle_map, resolution=self._maps['progress']._resolution,origin=self._maps['progress']._origin)
        # Let's now refine it a bit and do the same reading!

        # print('Ciao')
        return a_map

    def remove_outliers(self, data, threshold, num_neighbors):
        cleaned_data = np.copy(data)
        num_elements = len(data)

        for i in range(num_elements):
            if i < num_neighbors // 2 or i >= num_elements - num_neighbors // 2:
                # Skip elements with not enough neighbors
                continue

            neighbors = data[i - num_neighbors // 2: i + num_neighbors // 2 + 1]
            median = np.median(neighbors)
            mad = np.median(np.abs(neighbors - median))

            if np.abs(data[i] - median) > threshold * mad:
                cleaned_data[i] = median

        return cleaned_data

    def moving_average(self, data, window_size):
        smoothed_data = np.zeros_like(data)
        num_elements = len(data)

        for i in range(num_elements):
            if i < window_size // 2 or i >= num_elements - window_size // 2:
                # If there are not enough elements on one side, use available data
                smoothed_data[i] = np.mean(
                    data[max(0, i - window_size // 2):min(num_elements, i + window_size // 2 + 1)])
            else:
                # Compute the moving average using the full window
                smoothed_data[i] = np.mean(data[i - window_size // 2:i + window_size // 2 + 1])

        return smoothed_data

    def curvature_map(self, maps):
        solid_map = maps['occupancy'].map

        k, k_points, contour = self.curvature_fitter(solid_map, 1000, 21)  # 20 works good
        curvature_map = solid_map.astype('float64')
        track_indices = np.argwhere(solid_map > 0)
        k_p_row = k_points[:, 0]
        k_p_col = k_points[:, 1]

        for i in range(len(track_indices)):
            row = track_indices[i, 1]
            col = track_indices[i, 0]

            dist = ((k_p_row - row)**2 + (k_p_col - col)**2)**(1/2)
            idx = np.argmin(dist)

            curvature_map[col, row] = k[idx]

        # f10 = plt.figure(10)
        # plt.imshow(curvature_map)
        # f10.show()

        c_map = GridMap(curvature_map, resolution=self._maps['progress']._resolution, origin=self._maps['progress']._origin)

        # Centerline parametrization.
        # The centerline is parametrized as F(p) = (x(p), y(p)).
        x = contour[:, 0, 0]
        y = contour[:, 0, 1]
        p_vec = [self._maps['progress'].map[y[i], x[i]] for i in range(len(x))]
        p_vec = np.array(p_vec)

        c_map.centerline_parametrization = {'row': y, 'column': x, 'p': p_vec}

        max_curvature = max(k)
        min_curvature = min(k)

        return c_map, max_curvature, min_curvature

    def curvature_fitter(self, img, div, shift):
        import cv2
        from scipy.interpolate import interp1d
        from scipy.spatial import distance

        def shift_pairs(perimeter, shift):
            newls = np.concatenate((perimeter[-shift:], perimeter, perimeter[:shift]))
            return [(newls[i - shift], newls[i], newls[i + shift]) for i in range(1 + shift, len(newls) - shift)]

        # def shift_pairs(perimeter, shift):
        #     pairs = []
        #     length = len(perimeter)
        #     for i in range(length):
        #         pair = [perimeter[i], perimeter[(i + shift) % length]]
        #         pairs.append(pair)
        #     pairs.pop()
        #     return pairs

        def align_points_with_contour(pts, contour):
            closest_idx = np.argmin(distance.cdist(pts, contour), axis=1)
            return contour[closest_idx]

        def find_minimum_enclosing_circle(contour):
            center, radius = cv2.minEnclosingCircle(np.float32(contour))
            return center, radius

        def curvature_measure(img, div, shift):
            # img = cv2.resize(np.uint8(img), [6000, 6000])  # not resizing is ideal atm.
            img = np.uint8(img)
            centerline_img = cv2.ximgproc.thinning(img * 255)

            # Find contours of the centerline image.
            # contours, _ = cv2.findContours(centerline_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours, _ = cv2.findContours(centerline_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # Sort contours by size and choose one
            sel_cont = 'inner'  # difference is very small and can be eliminated tweaking `div` and `shift`
            if sel_cont == 'outer':
                contour = max(contours, key=cv2.contourArea)
            elif sel_cont == 'inner':
                contour = min(contours, key=cv2.contourArea)
            else:
                raise Exception('Error, not implemented!')

            # This could be an idea.
            # div = len(contour)
            # shift = int(div*20/1000)  # Tweak the parameteres of the proportion

            t = np.linspace(0, 1, len(contour))
            # interp = interp1d(t, contour.squeeze(), kind='linear', fill_value='extrapolate', axis=0)
            # interp = interp1d(t, contour.squeeze(), kind='quadratic', fill_value='extrapolate', axis=0)
            interp = interp1d(t, contour.squeeze(), kind='cubic', fill_value='extrapolate', axis=0)
            sub = np.linspace(0, 1, div + 1)
            sampled_pts = interp(sub)
            aligned_pts = align_points_with_contour(sampled_pts, contour.squeeze())

            paired_pts = shift_pairs(aligned_pts, shift)
            kappas = []
            for pts in paired_pts:
                center, radius = find_minimum_enclosing_circle(pts)
                kappa = 1 / radius if radius > 0 else 0
                kappas.append(kappa)

            # f2 = plt.figure(2)
            # vmin, vmax = np.min(kappas), np.max(kappas)
            # plt.gca().set_aspect('equal')
            # plt.imshow(img, cmap='gray')
            # sc = plt.scatter(aligned_pts[:-1, 0], aligned_pts[:-1, 1], c=kappas, cmap='rainbow', vmin=vmin, vmax=vmax, s=10)  # Exclude the last point
            # cbar = plt.colorbar(sc)
            # cbar.set_label('Curvature')
            # plt.tight_layout()
            # f2.show()

            return kappas, aligned_pts[:-1, :], contour

        k, k_points, contour = curvature_measure(img, div, shift)

        return k, k_points, contour
