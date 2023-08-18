from typing import Dict, Any, Tuple, Optional, SupportsFloat, Union
import gymnasium
import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame

from racecar_gym.envs.scenarios import SingleAgentScenario


class SingleAgentRaceEnv(gymnasium.Env):

    metadata = {
        'render_modes': ['human', 'rgb_array_follow', 'rgb_array_birds_eye', 'rgb_array_lidar']
    }

    def __init__(self, scenario: str, render_mode: str = 'human', render_options: Optional[Dict[str, Any]] = None):
        scenario = SingleAgentScenario.from_spec(scenario, rendering=render_mode == 'human')
        self._scenario = scenario
        self._initialized = False
        self._render_mode = render_mode
        self._render_options = render_options or {}
        self.action_space = scenario.agent.action_space
        self._terminal_judge_start = 5000  # MOD
        self._min_velocity = 0.05  # 0.28  # 1 (km/h)
        # self._low_velocity_buffer = 0
        self._max_low_velocity = 5000  # MOD
        self._cyclic_buffer_size = 1500
        self._cyclic_buffer = np.zeros(self._cyclic_buffer_size)
        self.time_step = 0

    @property
    def observation_space(self):
        space = self._scenario.agent.observation_space
        space.spaces['time'] = gymnasium.spaces.Box(low=0, high=1, shape=())
        return space

    @property
    def scenario(self):
        return self._scenario

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        assert self._initialized, 'Reset before calling step'
        observation, info = self._scenario.agent.step(action=action)
        self._scenario.world.update()
        state = self._scenario.world.state()
        state[self._scenario.agent.id].update({'discount': 1})
        observation['time'] = np.array(state[self._scenario.agent.id]['time'], dtype=np.float32)
        state[self._scenario.agent.id]['lidar_complete'] = self.lidar_merge(observation['lidar'],
                                                                            observation['lidar_rear'])
        state[self._scenario.agent.id]['distance_from_other_cars'] = [0, 0, 0]  # Zero-padding for SA.
        done = self._scenario.agent.done(state)
        reward = self._scenario.agent.reward(state, action)
        # state[self._scenario.agent.id].update({'previous_progress': state[self._scenario.agent.id]['progress'].copy()})  # after reward calculation!

        """""
        if observation['velocity'][0] < self._termination_limit_progress:
            self._low_velocity_buffer += 1
            if (self.time_step > self._terminal_judge_start) and (self._low_velocity_buffer >= self._max_low_velocity):
                done = True
                reward -= 100
        """""  # OLD BUFFER

        if abs(observation['velocity'][0]) < self._min_velocity:
            idx = np.mod(self.time_step, self._cyclic_buffer_size)
            self._cyclic_buffer[idx] = 1
            if sum(self._cyclic_buffer) == self._cyclic_buffer_size:
                print('Still for too much time')
                done = True
                reward -= 550

        """""  
        if self._terminal_judge_start < self.time_step:  # Episode terminates if the progress of agent is small
            # I don't like this solution: if the car moves for a number of step == self._terminal_judge_start
            # and stops for just one after those -> (v < v_lim) -> (done = True) -> sort of unfair!
            if observation['velocity'][0] < self._termination_limit_progress:
                done = True
        """""

        """""
        if state[self._scenario.agent.id]['wrong_way']:
            print('WRONG WAY: TERMINATE')
            done = True
        """""

        if done:
            state[self._scenario.agent.id]['discount'] = 0
        self.time_step += 1
        return observation, reward, done, False, state[self._scenario.agent.id]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.time_step = 0
        # self._low_velocity_buffer = 0
        self._cyclic_buffer[:] = 0
        if not self._initialized:
            self._scenario.world.init()
            self._initialized = True
        else:
            self._scenario.world.reset()
        if options is not None and 'mode' in options:
            mode = options['mode']
        else:
            mode = 'grid'
        obs = self._scenario.agent.reset(self._scenario.world.get_starting_position(self._scenario.agent, mode))
        self._scenario.world.update()
        state = self._scenario.world.state()
        # state[self._scenario.agent.id].update({'previous_progress': state[self._scenario.agent.id]['progress']})
        state[self._scenario.agent.id].update({'discount': 1})
        state[self._scenario.agent.id]['lidar_complete'] = self.lidar_merge(obs['lidar'], obs['lidar_rear'])
        state[self._scenario.agent.id]['distance_from_other_cars'] = [0, 0, 0]  # Zero-padding for SA.
        obs['time'] = np.array(state[self._scenario.agent.id]['time'], dtype=np.float32)
        # starting_state = obs.copy()
        # starting_state.update(state[self._scenario.agent.id].copy())
        # starting_state = {self._scenario.agent.id: starting_state.copy()}
        self._scenario.agent.task.reset()
        return obs, state[self._scenario.agent.id]

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        if self._render_mode == 'human':
            return None
        else:
            mode = self._render_mode.replace('rgb_array_', '')
            return self._scenario.world.render(mode=mode, agent_id=self._scenario.agent.id, **self._render_options)

    def lidar_merge(self, front_lidar, rear_lidar):
        split_idx = int(len(rear_lidar) / 2)  # len(rear_lidar) is always even by choice.
        rl1 = rear_lidar[:split_idx]
        rl2 = rear_lidar[split_idx:]
        merged_lidar = np.concatenate((rl2, front_lidar, rl1))
        return merged_lidar
