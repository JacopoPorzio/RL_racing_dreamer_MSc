import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile
import re

tf.config.run_functions_eagerly(run_eagerly=True)

# Save the actions' and velocity's graphs.
# load_dir = '/home/jacopo/Scrivania/Thesis-work/a0-algorithms/Dreamer-SA/logs/evaluations/sixth-complete-bis/trajectories'  # trajectories folder
# extract_dir_master = load_dir + '/' + 'unzipped'
# eval_data = []
# for file_name in os.listdir(load_dir):
#     if file_name != 'unzipped':
#         file_path = load_dir + '/' + file_name
#         with ZipFile(file_path, 'r') as f:
#             extract_dir = extract_dir_master + '/' + file_name[:-4]
#             f.extractall(path=extract_dir)  # members
#         time_path = extract_dir + '/' + 'time.npy'
#         times = np.load(time_path, allow_pickle=True)
#         action_path = extract_dir + '/' + 'actions.npy'
#         actions = np.load(action_path, allow_pickle=True)
#         steering = [actions[0]['steering']] + [actions[i+1].numpy()[0] for i in range(len(actions)-1)]
#         motor = [actions[0]['motor']] + [actions[i + 1].numpy()[1] for i in range(len(actions) - 1)]
#         step_idx = range(len(actions))
#         ep_num = int(re.findall(r'\d+', file_name)[0])
#         plt.figure(ep_num, figsize=(18.5, 10.5))
#         plt.title(f'Episode number {ep_num}')
#         plt.subplot(211)
#         # plt.plot(step_idx, steering)
#         plt.plot(times, steering)
#         # plt.xlabel('Step indices')
#         plt.xlabel('Time')
#         plt.ylabel('Steering actuation')
#         plt.grid()
#         plt.subplot(212)
#         # plt.plot(step_idx, motor)
#         plt.plot(times, motor)
#         # plt.xlabel('Step indices')
#         plt.xlabel('Time')
#         plt.ylabel('Throttle actuation')
#         plt.grid()
#         plt.show(block=False)
#         # plt.savefig('foo.png')
#         figure_path = extract_dir + '/' + 'actions_graph.png'
#         plt.savefig(figure_path)
#
#         plt.figure(ep_num+100, figsize=(18.5, 10.5))
#         plt.title(f'Episode number {ep_num}')
#         velocity_path = extract_dir + '/' + 'velocity.npy'
#         velocity = np.load(velocity_path, allow_pickle=True)
#         plt.plot(times, velocity)
#         plt.xlabel('Time')
#         plt.ylabel('Longitudinal velocity')
#         plt.grid()
#         plt.show(block=False)
#         figure_vel_path = extract_dir + '/' + 'lon_velocity.svg'
#         plt.savefig(figure_vel_path)


# Save the curvature's, actions' and velocity's graphs.
load_dir = '/home/jacopo/Scrivania/Thesis-work/a0-algorithms/Dreamer-SA/logs/evaluations/sixth-complete-bis/episodes'  # trajectories folder
extract_dir_master = load_dir + '/' + 'unzipped'
eval_data = []
for file_name in os.listdir(load_dir):
    if file_name != 'unzipped':
        file_path = load_dir + '/' + file_name
        with ZipFile(file_path, 'r') as f:
            extract_dir = extract_dir_master + '/' + file_name[:-4]
            f.extractall(path=extract_dir)  # members
        # Extract `time`, `actions` and `curvature`.
        time_path = extract_dir + '/' + 'time.npy'
        times = np.load(time_path, allow_pickle=True)
        action_path = extract_dir + '/' + 'action.npy'
        actions = np.load(action_path, allow_pickle=True)
        curvature_path = extract_dir + '/' + 'curvature.npy'
        curvatures = np.load(curvature_path, allow_pickle=True)
        curvature = curvatures[:, 0]
        steering = [actions[0]['steering']] + [actions[i+1].numpy()[0] for i in range(len(actions)-1)]
        motor = [actions[0]['motor']] + [actions[i + 1].numpy()[1] for i in range(len(actions) - 1)]
        step_idx = range(len(actions))
        ep_num = int(re.findall(r'\d+', file_name)[0])

        # Calculate rate of change of `curvature`.
        dk = np.diff(curvature)
        dt = np.diff(times)
        dkdt = dk/dt

        # Plot `steering` vs `curvature`.
        plt.figure(ep_num + 200, figsize=(18.5, 10.5))
        plt.subplot(211)
        plt.title(f'Episode number {ep_num}', fontweight='bold')
        plt.plot(times, steering)
        plt.xlabel('Time')
        plt.ylabel('Steering actuation')
        plt.grid()
        plt.subplot(212)
        plt.plot(times, curvature)
        plt.xlabel('Time')
        plt.ylabel('Current curvature')
        plt.grid()
        plt.show(block=False)
        figure_path = extract_dir + '/' + 'steer_vs_curvature_graph.svg'
        plt.savefig(figure_path)

        # Plot `steering` vs rate of change of `curvature`.
        plt.figure(ep_num + 300, figsize=(18.5, 10.5))
        plt.subplot(211)
        plt.title(f'Episode number {ep_num}', fontweight='bold')
        plt.plot(times[:-1], steering[:-1])
        plt.xlabel('Time')
        plt.ylabel('Steering actuation')
        plt.grid()
        plt.subplot(212)
        plt.plot(times[:-1], dkdt)
        plt.xlabel('Time')
        plt.ylabel('Rate of change of curvature')
        plt.grid()
        plt.show(block=False)
        figure_path = extract_dir + '/' + 'steer_vs_change_rate_curvature_graph.svg'
        plt.savefig(figure_path)

        # Plot `longitudinal_velocity` vs `curvature`.
        velocity_path = extract_dir + '/' + 'velocity.npy'
        velocities = np.load(velocity_path, allow_pickle=True)
        velocity = velocities[:, 0]
        plt.figure(ep_num + 400, figsize=(18.5, 10.5))
        plt.subplot(211)
        plt.title(f'Episode number {ep_num}', fontweight='bold')
        plt.plot(times, velocity)
        plt.xlabel('Time')
        plt.ylabel('Longitudinal velocity')
        plt.grid()
        plt.subplot(212)
        plt.plot(times, curvature)
        plt.xlabel('Time')
        plt.ylabel('Current curvature')
        plt.grid()
        plt.show(block=False)
        figure_vel_path = extract_dir + '/' + 'lon_velocity_vs_curvature.svg'
        plt.savefig(figure_vel_path)

plt.pause(200)

