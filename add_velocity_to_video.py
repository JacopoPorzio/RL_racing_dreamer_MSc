import imageio
import numpy as np
import cv2
import copy
import tensorflow as tf

tf.config.run_functions_eagerly(run_eagerly=True)

video_directory = '/home/jacopo/Scrivania/Thesis-work/Dreamer-SA/logs/evaluations/fourth-complete-run-MOD-AR4-FPS25/videos'
un_traj_directory = '/home/jacopo/Scrivania/Thesis-work/Dreamer-SA/logs/evaluations/fourth-complete-run-MOD-AR4-FPS25/trajectories/unzipped'
delta_y = 30
# add throttle and steer on video: move up text

for i in range(10):
	idx = i + 1
	video_name = video_directory + f'/birds_eye-A_{idx}_barcelona.mp4'
	vid = imageio.get_reader(video_name, 'ffmpeg')
	vid_with_text_writer = imageio.get_writer(video_name[:-4] + '_text' + video_name[-4:], fps=25)
	vid_with_text = []
	velocity = np.load(un_traj_directory + f'/trajectory_{idx}_barcelona/velocity.npy', allow_pickle=True)
	actions = np.load(un_traj_directory + f'/trajectory_{idx}_barcelona/actions.npy', allow_pickle=True)
	steer = [actions[0]['steering']] + [actions[i+1].numpy()[0] for i in range(len(actions)-1)]
	throttle = [actions[0]['motor']] + [actions[i + 1].numpy()[1] for i in range(len(actions) - 1)]
	
	for j, frame in enumerate(vid):
		# vel_text = f'Velocity = {velocity[j]} (m/s)'
		over_text = [
					f'Velocity = {velocity[j]} (m/s)',
					f'Throttle = {throttle[j]}',
					f'Steer = {steer[j]}'
					]
		frame_to_mod = copy.copy(frame)
		for k in range(len(over_text)):
			frame_with_text = cv2.putText(
										frame_to_mod,
										# vel_text,
										over_text[k],
										# (100, 100),
										(10, 50 + k*delta_y),
										cv2.FONT_HERSHEY_SIMPLEX,
										.3,  # .4,
										(0, 0, 0)
										)
			frame_to_mod = copy.copy(frame_with_text)
		vid_with_text.append(frame_with_text)

	for j, frame in enumerate(vid_with_text):
		vid_with_text_writer.append_data(vid_with_text[j])

	print(f'Video number {idx} has been saved.')

