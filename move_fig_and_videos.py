import os
import shutil

master = '/home/jacopo/Scrivania/Thesis-work/Dreamer-SA/logs/evaluations/fourth-complete-run-MOD-AR4-FPS25'
os.mkdir(master + '/' + 'video_with_figures')

src_trajectories =  master + '/' + 'trajectories/unzipped'
src_videos = master + '/' + 'videos'

for i in range(10):
	idx = i+1
	os.mkdir(master + '/' + f'video_with_figures/Ep{idx}')

for i in range(10):
	idx = i+1
	# src = '/home/jacopo/Scrivania/Thesis-work/Dreamer-SA/logs/evaluations/second-complete-run/trajectories/unzipped/'
	src_t = src_trajectories + '/' + f'trajectory_{idx}_barcelona/actions_graph.png'
	src_vel = src_trajectories + '/' + f'trajectory_{idx}_barcelona/lon_velocity.png'
	src_v = src_videos + '/' + f'birds_eye-A_{idx}_barcelona_text.mp4'
	dest = master + '/' + f'video_with_figures/Ep{idx}'
	
	shutil.copy(src_t, dest)
	shutil.copy(src_vel, dest)
	shutil.copy(src_v, dest)

