from .task import Task
import numpy as np
import pybullet
import yaml
import time

def wrap_angles(angles, range_start=-np.pi, range_end=np.pi):
    wrapped_angles = (angles - range_start) % (range_end - range_start) + range_start
    return wrapped_angles


class DreamingTask(Task):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool,
                 delta_progress: float = 0.0, collision_reward: float = 0.0,
                 frame_reward: float = 0.0, progress_reward: float = 100.0, n_min_rays_termination=1080):
        self._time_limit = time_limit
        self._laps = laps
        self._terminate_on_collision = terminate_on_collision
        self._n_min_rays_termination = n_min_rays_termination
        # self._last_state = None
        # reward params
        self._delta_progress = delta_progress
        self._progress_reward = progress_reward
        self._collision_reward = collision_reward
        self._frame_reward = frame_reward
        # MOD vector containing the weights
        # self._weight_vector = [2.8, 2.8, 0.01, 0.06, 0.05, 0.10, 0.01]
        # MOD  progress, collision, slip, steer, throttle, angle_diff, bias
        self._weight_dict = {
                              'rcp': 2.8,   # progress  # 2.8, I've tried 3.1
                              'rwc': 2.8,   # collision
                              'rts': 0.01,  # slip
                              'rds': 0.06,  # steer  # 0.06, I've tried 0.07
                              'rdt': 0.05,  # throttle  # 0.05, I've tried 0.06
                              'rad': 0.1,   # angle_diff
                              'bias': 0.01  # bias
                            }
        with open('/home/jacopo/Scrivania/Thesis-work/a0-algorithms/Dreamer-SA/logs/experiments'
                  + '/' + f'task_weights_{time.time()}.yml', 'w') as outfile:
            yaml.dump(self._weight_dict, outfile, default_flow_style=False)

        self._tire_names = ['left_front', 'right_front', 'left_rear', 'right_rear']
        self._slip_vel_tol = 0.1  # MOD
        self._track_length = 465.7  # MOD: only for Barcelona
        self._last_stored_progress = None
        self._last_stored_action = None

    def reward(self, agent_id, state, action) -> float:
        """""
        agent_state = state[agent_id]
        progress = agent_state['lap'] + agent_state['progress']
        if self._last_stored_progress is None:
            self._last_stored_progress = progress
        delta = abs(progress - self._last_stored_progress)
        if delta > .5:  # the agent is crossing the starting line in the wrong direction
            delta = (1 - progress) + self._last_stored_progress
        reward = self._frame_reward
        if self._check_collision(agent_state):
            reward += self._collision_reward
        reward += delta * self._progress_reward
        self._last_stored_progress = progress
        """""

        # Let's write:
        # Course progress
        # Wall collision penalty
        # Tire-slip penalty
        # lin_tire, ang_tire = self.get_wheels_velocity(id=2)
        # print('Linear: ', lin_tire['left_rear'])
        # print('Angular:', ang_tire['left_rear'])
        # print('Angular*R:', ang_tire['left_rear']*0.05)  # debug

        # prev_state = self._last_state
        lin_tire, ang_tire = state[agent_id]['wheels_velocity']
        lon_car_velocity = state[agent_id]['velocity'][0]
        wheel_spin_velocity = np.array([ang_tire[name] for name in self._tire_names])  # just a placeholder
        lon_tire_velocity = np.array([lin_tire[name][0] for name in self._tire_names])
        lat_tire_velocity = np.array([lin_tire[name][1] for name in self._tire_names])
        slip_angles = -np.arctan(lat_tire_velocity/abs(lon_tire_velocity))
        tire_radius = 0.05

        # rcp = self._track_length*(state[agent_id]['progress'] - state[agent_id]['previous_progress'])

        progress = state[agent_id]['lap'] + state[agent_id]['progress']
        if self._last_stored_progress is None:
            self._last_stored_progress = progress
        rcp = self._track_length*(progress - self._last_stored_progress)
        # print('PROGRESS: ', progress - self._last_stored_progress)

        rwc = - self._check_collision(state[agent_id]) * (lon_car_velocity ** 2)
        if lon_car_velocity < 0.05:
            # We should try to discourage also the static collisions.
            rwc -= self._check_collision(state[agent_id])

        rts = 0
        if abs(lon_car_velocity) > self._slip_vel_tol:
            for i in range(4):
                TSR = tire_radius*wheel_spin_velocity[i]/lon_car_velocity - 1
                rts -= min(1, abs(TSR))*abs(slip_angles[i])

        if self._last_stored_action is None:
            self._last_stored_action = action
        rds = - abs(action[0] - self._last_stored_action[0])**2  # Is linear better?
        rdt = - abs(action[1] - self._last_stored_action[1])**2  # Is linear better?

        ad = state[agent_id]['pose'][5] - state[agent_id]['angle']
        rad = - abs(wrap_angles(ad))**2  # Non-linear effect to exclude small differences.

        # reward = (rcp * self._weight_vector[0] + rwc * self._weight_vector[1]
        #           + rts * self._weight_vector[2] + rds * self._weight_vector[3]
        #           + rdt * self._weight_vector[4] + rad * self._weight_vector[5]
        #           - self._weight_vector[6])  # last one is just a bias for time passing

        reward = (rcp * self._weight_dict['rcp'] + rwc * self._weight_dict['rwc']
                  + rts * self._weight_dict['rts'] + rds * self._weight_dict['rds']
                  + rdt * self._weight_dict['rdt'] + rad * self._weight_dict['rad']
                  - self._weight_dict['bias'])  # last one is just a bias for time passing

        # self._last_state[agent_id] = state[agent_id].copy()

        # """""
        if state[agent_id]['wrong_way']:
            print('WRONG WAY')
            reward -= 200
        # """""  # Strongly negative reward when wrong_way

        self._last_stored_progress = progress
        self._last_stored_action = action
        return reward

    def done(self, agent_id, state) -> bool:
        agent_state = state[agent_id]
        if self._terminate_on_collision and self._check_collision(agent_state):
            print('COLLISION: TERMINATE')
            return True

        # """""
        if state[agent_id]['wrong_way']:
            print('WRONG WAY: TERMINATE')
            return True
        # """""  # Terminate episode when wrong_way

        return agent_state['lap'] > self._laps or self._time_limit < agent_state['time']

    def _check_collision(self, agent_state):
        safe_margin = 0.25
        collision = agent_state['wall_collision'] or len(agent_state['opponent_collisions']) > 0
        if 'observations' in agent_state and 'lidar' in agent_state['observations']:
            n_min_rays = sum(np.where(agent_state['observations']['lidar'] <= safe_margin, 1, 0))
            return n_min_rays > self._n_min_rays_termination or collision
        return collision

    def reset(self):
        self._last_stored_progress = None
        self._last_stored_action = None
        # return

    # def get_wheels_velocity(self, id):
    #     """""
    #     left_front = pybullet.getLinkState(id, 1, computeLinkVelocity=1)
    #     right_front = pybullet.getLinkState(id, 3, computeLinkVelocity=1)
    #     left_rear = pybullet.getLinkState(id, 12, computeLinkVelocity=1)
    #     right_rear = pybullet.getLinkState(id, 14, computeLinkVelocity=1)
    #     """""
    #
    #     tire_dict = {
    #         'left_front': pybullet.getLinkState(id, 1, computeLinkVelocity=1),
    #         'right_front': pybullet.getLinkState(id, 3, computeLinkVelocity=1),
    #         'left_rear': pybullet.getLinkState(id, 12, computeLinkVelocity=1),
    #         'right_rear': pybullet.getLinkState(id, 14, computeLinkVelocity=1)
    #     }
    #
    #     names = ['left_front', 'right_front', 'left_rear', 'right_rear']
    #
    #     linear_dict = {}
    #     angular_dict = {}
    #
    #     for name in names:
    #         linear, angular = tire_dict[name][-2:]
    #         # position, orientation = tire_dict[name][0:2]
    #         # position, orientation = tire_dict[name][0:2]
    #         position, orientation = tire_dict[name][4:6]
    #         # print(orientation)
    #         rotation = pybullet.getMatrixFromQuaternion(orientation)
    #         rotation = np.reshape(rotation, (-1, 3)).transpose()
    #         linear = rotation.dot(linear)
    #         linear = np.array([np.sqrt(linear[0]**2 + linear[1]**2), linear[2]])  # HAND-CRAFTED SOLUTION
    #         angular = rotation.dot(angular)
    #         angular = np.array(abs(angular[2]))  # HAND-CRAFTED SOLUTION
    #         linear_dict.update({name: linear})
    #         angular_dict.update({name: angular})
    #
    #     return linear_dict, angular_dict

