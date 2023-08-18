import argparse
import collections
import functools
import json
import os
import pathlib
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.mixed_precision as prec
from tensorflow_probability import distributions as tfd
import models
# import tools
import wrappers
import copy
import gymnasium as gym
import racecar_gym.envs.gym_api
import math
from typing import Dict, Tuple
from scipy.signal import medfilt
from evaluations_module.callbacks import save_eval_videos, save_trajectory, summarize_eval_episode, save_episodes
from evaluations_module.wrappers import Render, Collect
from evaluations_module import tools


tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['MUJOCO_GL'] = 'egl'
sys.path.append(str(pathlib.Path(__file__).parent))


def define_config():
    config = tools.AttrDict()
    # General.
    config.logdir = pathlib.Path('./logs/experiments/PAST/fourth-complete-run')  # #####
    config.seed = 0
    config.steps = 5e8  # 5e6
    config.eval_every = 1e4  # TEST: 400
    config.log_every = 1e3  # TEST: 100
    config.log_scalars = True
    config.gpu_growth = True
    config.precision = 32  # 16  --> old, 16 gives inf and/or nan as grad and losses
    config.obs_type = {1: 'image', 2: 'lidar', 3: 'complete_low_dim'}[3]  # 1
    config.image_sensor = None
    config.log_images = False
    if config.obs_type in ['image', 'lidar', 'complete_low_dim']:
        config.log_images = True
    # Environment.
    config.task = 'racecar_gym'
    config.track = 'barcelona'  # Just for logging purposes, not that important.
    config.envs = 1
    config.parallel = 'none'
    config.action_repeat = 4
    config.time_limit = 100000  # 6000  # 15000  # 1000 originally, but very fast!
    config.prefill_agent = {1: 'random', 2: 'follow_the_gap'}[2]
    config.prefill = 30000  # 5000 originally, but very fast!
    config.eval_noise = 0.0
    config.clip_rewards = 'none'
    # The names of the observations are used iff obs_type == 'complete_low_dim'
    config.obs_names = [
                        'lidar_complete',  # 'lidar',
                        'velocity',
                        'acceleration',
                        'steer_old',
                        'wall_collision',
                        'curvature',
                        'distance_from_other_cars',
                        'progress'
                        ]  # add angle?
    # Model.
    config.deter_size = 200
    config.stoch_size = 30
    config.num_units = 400
    config.dense_act = 'elu'
    config.cnn_act = 'relu'
    # config.lidar_decoder_depth = 128  ######### UNUSED: MANUALLY AS FOR NOW
    config.cnn_depth = 32
    config.cnn_depth = 32
    config.pcont = True  # it was False
    config.free_nats = 3.0
    config.kl_scale = 1.0
    config.pcont_scale = 10.0
    config.weight_decay = 0.0
    config.weight_decay_pattern = r'.*'
    # Training.
    config.batch_size = 50
    config.batch_length = 50
    if config.obs_type == 'image':
        config.train_every = 1000
    elif config.obs_type == 'lidar':
        config.train_every = 1000  # Danijar suggested 200
    elif config.obs_type == 'complete_low_dim':
        config.train_every = 200
    config.train_steps = 100
    config.pretrain = 100
    config.model_lr = 6e-4
    config.value_lr = 8e-5
    config.actor_lr = 8e-5
    config.grad_clip = 100.0
    config.dataset_balance = False
    # Behavior.
    config.discount = 0.99
    config.disclam = 0.95
    config.horizon = 15
    config.action_dist = 'tanh_normal'
    config.action_init_std = 5.0
    config.expl = 'additive_gaussian'
    config.expl_amount = 0.3
    config.expl_decay = 0.0
    config.expl_min = 0.0
    return config


class Dreamer(tools.Module):

    def __init__(self, config, datadir, obsspace, actspace, writer):
        self._c = config
        self._obsspace = obsspace
        self._image_sensor = self._c.image_sensor
        self._actspace = actspace
        self._actdim = actspace.n if hasattr(actspace, 'n') else len(actspace)  # actspace.shape[0]
        self._writer = writer
        self._random = np.random.RandomState(config.seed)
        with tf.device('cpu:0'):
            self._step = tf.Variable(count_steps(datadir, config), dtype=tf.int64)  # tf object to count steps
        self._should_pretrain = tools.Once()
        self._should_train = tools.Every(config.train_every)
        self._should_log = tools.Every(config.log_every)
        self._last_log = None
        self._last_time = time.time()
        self._metrics = collections.defaultdict(tf.metrics.Mean)
        self._metrics['expl_amount']  # Create variable for checkpoint.
        self._float = prec.global_policy().compute_dtype
        self._strategy = tf.distribute.MirroredStrategy()
        # self._strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        self._step_from_call = 0
        with self._strategy.scope():
            self._dataset = iter(self._strategy.experimental_distribute_dataset(
                load_dataset(datadir, self._c)))
            self._build_model()

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step.numpy().item()  # actual number assignment from object
        tf.summary.experimental.set_step(step)
        if state is not None and reset:
            mask = tf.cast(1 - reset, self._float)[:, None]
            state = tf.nest.map_structure(lambda x: x * mask, state)
        action, state = self.policy(obs, state, training)
        # Sanity checks
        # a_list = [[0.0, 1.0]]
        # action = tf.stack(a_list)
        # print('Action to apply: ', action)

        return action, state

    @tf.function
    def policy(self, obs, state, training):
        if state is None:
            # latent = self._dynamics.initial(len(obs['image']))
            # action = tf.zeros((len(obs['image']), self._actdim), self._float)
            if self._c.obs_type == 'image':
                # This serves for the policy: the length of the observation type
                # is used, because we can have more than one environment!
                # We are no more considering training batches!
                # latent = self._dynamics.initial(len(obs['image']))
                # action = tf.zeros((len(obs['image']), self._actdim), self._float)
                # New: 'image' isn't an observation field!
                latent = self._dynamics.initial(len(obs[self._image_sensor]))
                action = tf.zeros((len(obs[self._image_sensor]), self._actdim), self._float)
            elif self._c.obs_type == 'lidar':
                latent = self._dynamics.initial(len(obs['lidar']))
                action = tf.zeros((len(obs['lidar']), self._actdim), self._float)
            else:
                # For the same reason, we just take a generic observation!
                generic_obs = self._c.obs_names[0]
                latent = self._dynamics.initial(len(obs[generic_obs]))
                action = tf.zeros((len(obs[generic_obs]), self._actdim), self._float)
            # we could probably substitute this big chunk with just the last condition...
        else:
            latent, action = state
        embed = self._encode(preprocess(obs, self._c))
        latent, _ = self._dynamics.obs_step(latent, action, embed)
        feat = self._dynamics.get_feat(latent)
        if training:
            action = self._actor(feat).sample()
        else:
            action = self._actor(feat).mode()
        action = self._exploration(action, training)
        state = (latent, action)
        return action, state

    def load(self, filename):
        super().load(filename)
        self._should_pretrain()

    @tf.function()
    def train(self, data, log_images=False):
        self._strategy.run(self._train, args=(data, log_images))

    def _train(self, data, log_images):
        # tf.print('Log images in train? ', log_images)  # DEBUG
        with tf.GradientTape() as model_tape:
            # Dynamics learning: only on data!
            embed = self._encode(data)
            post, prior = self._dynamics.observe(embed, data['action'])
            feat = self._dynamics.get_feat(post)
            image_pred = self._decode(feat)  # not real images if obs_type != 'image'
            reward_pred = self._reward(feat)
            likes = tools.AttrDict()
            if self._c.obs_type == 'image':
                # likes.image = tf.reduce_mean(image_pred.log_prob(data['image']))  # () = fn( (50,50) )
                # New: 'image' isn't an obs field
                likes.image = tf.reduce_mean(image_pred.log_prob(data[self._image_sensor]))
            elif self._c.obs_type == 'lidar':
                likes.lidar = tf.reduce_mean(image_pred.log_prob(data['lidar']))
            elif self._c.obs_type == 'complete_low_dim':
                # Process the data to extract and concatenate the observations
                obj_to_concat = [
                                 tf.reshape(data[name], (int(self._c.batch_size/self._strategy.num_replicas_in_sync),  # Account for distributed training.
                                                         self._c.batch_length, 1))
                                 if tf.shape(data[name]).shape == 2
                                 else data[name]
                                 for name in self._c.obs_names
                                 ]
                processed_data = tf.concat(obj_to_concat, -1)
                # processed_data = tf.concat([data[name] for name in self._c.obs_names], -1)  # OLD: NO GOOD
                likes.complete_low_dim = tf.reduce_mean(image_pred.log_prob(processed_data))
            likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))  # () = fn( (50,50) )
            if self._c.pcont:
                pcont_pred = self._pcont(feat)
                pcont_target = self._c.discount * data['discount']
                likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
                likes.pcont *= self._c.pcont_scale
            prior_dist = self._dynamics.get_dist(prior)
            post_dist = self._dynamics.get_dist(post)
            div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            div = tf.maximum(div, self._c.free_nats)
            model_loss = self._c.kl_scale * div - sum(likes.values())
            # I want to minimize the KL divergence, but maximize the log-likelihood:
            # I want to maximize, because I want my forecasts to be as close as possible to reality!
            model_loss /= float(self._strategy.num_replicas_in_sync)

        with tf.GradientTape() as actor_tape:
            # Behaviour learning: actor
            imag_feat = self._imagine_ahead(post)
            reward = self._reward(imag_feat).mode()
            if self._c.pcont:
                pcont = self._pcont(imag_feat).mean()
            else:
                pcont = self._c.discount * tf.ones_like(reward)
            value = self._value(imag_feat).mode()
            returns = tools.lambda_return(
                reward[:-1], value[:-1], pcont[:-1],
                bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
            discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
                [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
            actor_loss = -tf.reduce_mean(discount * returns)
            actor_loss /= float(self._strategy.num_replicas_in_sync)

        with tf.GradientTape() as value_tape:
            # Behaviour learning: critic
            value_pred = self._value(imag_feat)[:-1]
            target = tf.stop_gradient(returns)
            value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))
            value_loss /= float(self._strategy.num_replicas_in_sync)

        model_norm = self._model_opt(model_tape, model_loss)
        actor_norm = self._actor_opt(actor_tape, actor_loss)
        value_norm = self._value_opt(value_tape, value_loss)

        if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
            if self._c.log_scalars:
                self._scalar_summaries(
                    data, feat, prior_dist, post_dist, likes, div,
                    model_loss, value_loss, actor_loss, model_norm, value_norm,
                    actor_norm)
            if log_images:  # tf.equal(log_images, True):
                # tf.print('Inside _train: step ', tf.summary.experimental.get_step())  # DEBUG
                self._image_summaries(data, embed, image_pred)

    def load(self, filename):
        super().load(filename)
        self._should_pretrain()

    def _build_model(self):
        acts = dict(
            elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
            leaky_relu=tf.nn.leaky_relu)
        cnn_act = acts[self._c.cnn_act]
        act = acts[self._c.dense_act]

        # # # # Different types of observation! # # # #
        if self._c.obs_type == 'image':
            self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act, self._image_sensor)
            self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act, (64, 64, 3))  # MOD HERE THE RESOLUTION OF THE IMAGE: START WITH THESE
        elif self._c.obs_type == 'lidar':
            # self._encode = models.IdentityEncoder()
            self._encode = models.MLPEncoder(350, 400, 4)  # We encode 1080 into (x, _, _)
            self._decode = models.LidarDistanceDecoder(400, self._obsspace['lidar'].shape, 4)
        elif self._c.obs_type == 'complete_low_dim':
            # self._encode = models.IdentityEncoder()
            self._encode = models.CompleteEncoder(480, 600, 4, self._c.obs_names)  # We encode 1080+N into (x, _, _, _)
            # OLD: 360, 400, 4
            decoder_shape = (sum([self._obsspace[name].shape[0] for name in self._c.obs_names]),)
            self._decode = models.CompleteDecoder(400, decoder_shape, 4)
            # OLD 400,

        self._dynamics = models.RSSM(self._c.stoch_size, self._c.deter_size, self._c.deter_size)
        self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
        if self._c.pcont:
            self._pcont = models.DenseDecoder((), 3, self._c.num_units, 'binary', act=act)
        self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
        self._actor = models.ActionDecoder(
            self._actdim, 4, self._c.num_units, self._c.action_dist,
            init_std=self._c.action_init_std, act=act)
        model_modules = [self._encode, self._dynamics, self._decode, self._reward]
        if self._c.pcont:
            model_modules.append(self._pcont)
        Optimizer = functools.partial(
            tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
            wdpattern=self._c.weight_decay_pattern)
        self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
        self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
        self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
        self.train(next(self._dataset))
        # Do a train step to initialize all variables, including optimizer statistics.
        # Ideally, we would use batch size zero, but that doesn't work in multi-GPU mode.

    def _exploration(self, action, training):
        if training:
            amount = self._c.expl_amount
            if self._c.expl_decay:
                amount *= 0.5 ** (tf.cast(self._step, tf.float32) / self._c.expl_decay)
            if self._c.expl_min:
                amount = tf.maximum(self._c.expl_min, amount)
            self._metrics['expl_amount'].update_state(amount)
        elif self._c.eval_noise:
            amount = self._c.eval_noise
        else:
            return action
        if self._c.expl == 'additive_gaussian':
            return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
        if self._c.expl == 'completely_random':
            return tf.random.uniform(action.shape, -1, 1)
        if self._c.expl == 'epsilon_greedy':
            indices = tfd.Categorical(0 * action).sample()
            return tf.where(
                tf.random.uniform(action.shape[:1], 0, 1) < amount,
                tf.one_hot(indices, action.shape[-1], dtype=self._float),
                action)
        raise NotImplementedError(self._c.expl)

    def _imagine_ahead(self, post):
        if self._c.pcont:  # Last step could be terminal.
            post = {k: v[:, :-1] for k, v in post.items()}
        flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in post.items()}
        policy = lambda state: self._actor(tf.stop_gradient(self._dynamics.get_feat(state))).sample()
        # This 'static_scan' cycles, starting from the posteriors obtained from the batch,
        # in order to get a number of trajectories, which are self._c.horizon long,
        # which contain the imagined states
        states = tools.static_scan(
                                   lambda prev, _: self._dynamics.img_step(prev, policy(prev)),
                                   tf.range(self._c.horizon), start
                                  )
        imag_feat = self._dynamics.get_feat(states)
        return imag_feat

    def _scalar_summaries(
          self, data, feat, prior_dist, post_dist, likes, div,
          model_loss, value_loss, actor_loss, model_norm, value_norm, actor_norm):
        self._metrics['model_grad_norm'].update_state(model_norm)
        self._metrics['value_grad_norm'].update_state(value_norm)
        self._metrics['actor_grad_norm'].update_state(actor_norm)
        self._metrics['prior_ent'].update_state(prior_dist.entropy())
        self._metrics['post_ent'].update_state(post_dist.entropy())
        for name, logprob in likes.items():
            self._metrics[name + '_loss'].update_state(-logprob)
        self._metrics['div'].update_state(div)
        self._metrics['model_loss'].update_state(model_loss)
        self._metrics['value_loss'].update_state(value_loss)
        self._metrics['actor_loss'].update_state(actor_loss)
        self._metrics['action_ent'].update_state(self._actor(feat).entropy())

    def _image_summaries(self, data, embed, image_pred, ep_idx=None):
        # for reference: pag.5 of Dreamer paper
        # tf.print('Self step inside image summaries: ', self._step)  # DEBUG
        # tf.print('Summarizing image: step ', tf.summary.experimental.get_step())  # DEBUG
        summary_length = 5  # nr step observed before dreaming
        recon = image_pred.mode()
        init, _ = self._dynamics.observe(embed[:, :summary_length], data['action'][:, :summary_length])
        init = {k: v[:, -1] for k, v in init.items()}
        prior = self._dynamics.imagine(data['action'][:, summary_length:], init)
        openl = self._decode(self._dynamics.get_feat(prior)).mode()
        model = tf.concat([recon[:, :summary_length] + 0.5, openl + 0.5], 1)
        if self._c.obs_type == 'image':
            truth = data[self._image_sensor] + 0.5
            error = (model - truth + 1) / 2
        elif self._c.obs_type == 'lidar':
            truth = data['lidar'] + 0.5
            truth = tools.lidar_to_image(truth)
            model = tools.lidar_to_image(model)
            error = model - truth
        else:
            truth = data['lidar_complete'] + 0.5
            truth = tools.lidar_to_image(truth)
            model = tools.lidar_to_image(model[:, :, 0:1438])
            error = model - truth
        openl = tf.concat([truth, model, error], 2)
        tools.graph_summary(self._writer, ep_idx, tools.video_summary, 'agent/openl', openl)

    def _write_summaries(self):
        step = int(self._step.numpy())
        metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
        if self._last_log is not None:
            duration = time.time() - self._last_time
            self._last_time += duration
            metrics.append(('fps', (step - self._last_log) / duration))
        self._last_log = step
        [m.reset_states() for m in self._metrics.values()]
        with (self._c.logdir / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
        [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
        print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
        self._writer.flush()


### ### End of Dreamer class ### ###

def preprocess(obs, config):
    # used inside Dreamer only
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    with tf.device('cpu:0'):
        if config.obs_type == 'image':
            # TODO: insert these normalizing values in config
            # normalizing and centering around zero
            # obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
            # New: 'image' isn't an observation field
            obs[config.image_sensor] = tf.cast(obs[config.image_sensor], dtype) / 255.0 - 0.5
        if config.obs_type == 'lidar':
            # TODO: insert these normalizing values in config
            # normalizing and centering around zero
            # TBN: the normalizing values to use are contained inside ./models/vehicles/[...]car/[...]car.yaml
            obs['lidar'] = tf.cast(obs['lidar'], dtype) / 25.0 - 0.5
        if config.obs_type == 'complete_low_dim':
            # TODO: insert these normalizing values in config
            # normalizing and centering around zero
            # TBN: the normalizing values to use are contained inside ./models/vehicles/[...]car/[...]car.yaml
            # obs = tf.cast(obs, dtype)
            obs['lidar_complete'] = tf.cast(obs['lidar_complete'], dtype) / 25.0 - 0.5
            obs['velocity'] = tf.cast(obs['velocity'], dtype) / 14.0 * 0.5
            # obs['acceleration'] = sigmoid(tf.cast(obs['acceleration'], dtype)) - 0.5  # acceleration has no bound
            obs['acceleration'] = tanh(tf.cast(obs['acceleration'], dtype))*0.5  # acceleration has no bound, but it can be negative, should be similar to sigmoid
            obs['steer_old'] = tf.cast(obs['steer_old'], dtype) * 0.5
            obs['wall_collision'] = tf.cast(obs['wall_collision'], dtype) - 0.5
            obs['curvature'] = tf.cast(obs['curvature'], dtype) - 0.5  # Already normalized.
            obs['progress'] = tf.cast(obs['progress'], dtype)/(5 + 1) - 0.5  # i.e. max_laps + 1
            # obs['distance_from_other_cars'] = sigmoid(tf.cast(obs['distance_from_other_cars'], dtype)) - 0.5
            obs['distance_from_other_cars'] = tf.cast(obs['distance_from_other_cars'], dtype) * 0  # Simpler.
            # Why multiplying times 0.5? Because those observations can be negative as well:
            # by dividing them by their max value, we obtain a number between -1 and 1.
            # Since the other observations are inside the interval [-0.5; 0.5], we add *0.5.

        # The following rows are usable iff <env.wrappers.RewardObs> is added to make_env.
        # Nevertheless, this identity doesn't make any sense...
        # clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
        # obs['reward'] = clip_rewards(obs['reward'])
    return obs


def sigmoid(x):
    return 1/(1 + tf.math.exp(-x))


def tanh(x):
    return tf.math.tanh(x)


def count_steps(datadir, config):
    # used inside Dreamer only
    return tools.count_episodes(datadir)[1] * config.action_repeat


def load_dataset(directory, config):
    episode = next(tools.load_episodes(directory, 1))
    types = {k: v.dtype for k, v in episode.items()}
    shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
    generator = lambda: tools.load_episodes(
      directory, config.train_steps, config.batch_length,
      config.dataset_balance)
    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.map(functools.partial(preprocess, config=config))
    dataset = dataset.prefetch(10)
    return dataset


def summarize_episode(episode, config, datadir, writer, prefix, step_counter):  # ##############################
    # Change directory
    episodes, steps = tools.count_episodes(datadir)
    length = (len(episode['reward']) - 1) * config.action_repeat
    ret = sum(episode['reward'])
    print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
    metrics = [
               (f'{prefix}/return', float(ret)),
               (f'{prefix}/length', len(episode['reward']) - 1),
               (f'episodes', episodes)
              ]
    # step = count_steps(datadir, config)  # Step doesn't obviously change!
    step = step_counter.count_step()
    tf.summary.experimental.set_step(step)
    with (config.logdir / 'metrics.jsonl').open('a') as f:
        f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
    with writer.as_default():  # Env might run in a different thread.
        tf.summary.experimental.set_step(step)
        [tf.summary.scalar('sim/' + k, v) for k, v in metrics]
        if prefix == 'test':
            print('Summarizing episode: step ', step)
            print('Sanity check step: ', tf.summary.experimental.get_step())
            if config.obs_type == 'image':
                # tools.video_summary(f'sim/{prefix}/video', episode['image'][None])
                # New: 'image' isn't an observation field
                tools.video_summary(f'sim/{prefix}/video', episode[config.image_sensor][None])
            elif config.obs_type == 'lidar':
                video = tools.lidar_to_image(episode['lidar'][None])  # lidar_complete
                proto_tensor_video = tf.make_tensor_proto(video)
                video = tf.make_ndarray(proto_tensor_video)
                tools.video_summary(f'sim/{prefix}/video', video)
            elif config.obs_type == 'complete_low_dim':
                video = tools.lidar_to_image(episode['lidar_complete'][None])
                proto_tensor_video = tf.make_tensor_proto(video)
                video = tf.make_ndarray(proto_tensor_video)
                tools.video_summary(f'sim/{prefix}/video', video)


def make_base_env(config):
    if config.obs_type == 'image':
        env = gym.make(
                       id='SingleAgentRaceEnv-v0',
                       scenario='/home/jacopo/Scrivania/Thesis-work/Dreamer-SA/scenarios/barcelona_dreamer_IMAGE.yml',
                       render_mode='rgb_array_birds_eye',  # human
                       # render_options=dict(width=128, height=128)  # (width=320, height=240)
                       )
    elif config.obs_type == 'lidar':
        env = gym.make(
                       id='SingleAgentRaceEnv-v0',
                       scenario='/home/jacopo/Scrivania/Thesis-work/Dreamer-SA/scenarios/barcelona_dreamer_LIDAR.yml',
                       render_mode='rgb_array_birds_eye',  # human
                       # render_options=dict(width=128, height=128)  # (width=320, height=240)
                      )
    else:
        env = gym.make(
                       id='SingleAgentRaceEnv-v0',
                       scenario='/home/jacopo/Scrivania/Thesis-work/Dreamer-SA/scenarios/barcelona_dreamer_COMPLETE.yml',
                       render_mode='rgb_array_birds_eye',  # human
                       render_options=dict(width=640, height=480)  # (width=320, height=240)
                      )
        env = wrappers.ObsAndStateMixWrapper(env)
    return env


def wrap_env(base_env, config, writer, outdir):
    track = config.track
    env = wrappers.ActionRepeat(base_env, config.action_repeat)
    env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat, config.obs_type)

    render_callbacks = []
    videodir = outdir / 'videos'
    render_callbacks.append(lambda videos: save_eval_videos(videos, videodir, track))
    env = Render(env, render_callbacks, follow_view=False)
    callbacks = []
    callbacks.append(lambda ep: save_episodes(outdir, [ep]))  # eval_episode_dir
    trajdir = outdir
    callbacks.append(lambda episodes: save_trajectory(episodes, trajdir, track))
    eval_episode_dir = outdir
    callbacks.append(lambda episodes: summarize_eval_episode(episodes, eval_episode_dir, writer, f'{track}'))
    env = Collect(env, callbacks)  # wrappers.Collect(env, callbacks)  # This should work as well.

    return env


def imagination_summary(agent, ep_dir, ep_idx, config):
    # Move to evaluation_module/tools.py?
    sorted_episodes = sorted(ep_dir.glob('*npz'), key=lambda f: int(f.name.split('.')[0][8:]))
    selected_episode_path = sorted_episodes[-1]
    # How `data` must be organized:
    # shape == [batch_size, batch_length, actual obs/action dimension]
    with selected_episode_path.open('rb') as file_path:
        selected_episode = np.load(file_path, allow_pickle=True)
        data = {name: np.array([selected_episode[name]]) for name in config.obs_names}
        action_v = selected_episode['action']
    first_actions = np.array([[action_v[0]['steering'], action_v[0]['motor']]])
    last_actions = action_v[1:]
    last_actions_np = np.array([last_actions[k].numpy() for k in range(len(last_actions))])
    action_v = np.concatenate((first_actions, last_actions_np), 0)
    data.update({'action': np.array([action_v], dtype='float32')})
    data = preprocess(data, config)
    embed = agent._encode(data)
    post, prior = agent._dynamics.observe(embed, data['action'])
    feat = agent._dynamics.get_feat(post)
    image_pred = agent._decode(feat)
    agent._image_summaries(data, embed, image_pred, ep_idx)


def save_gifs_from_event(path_in, tag, path_out):
    # Move to evaluation_module/tools.py?
    import imageio
    assert(os.path.isdir(path_out))

    for e in tf.compat.v1.train.summary_iterator(path_in):
        for v in e.summary.value:
            if v.tag == tag:
                tf_img = tf.io.decode_image(v.image.encoded_image_string)
                np_img = tf_img.numpy()
                imageio.mimsave(path_out / f'Episode_{e.step}.gif', np_img, fps=25)


def main(config):
    if config.gpu_growth:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        prec.set_global_policy(prec.Policy('mixed_float16'))
    config.steps = int(config.steps)
    # config.logdir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_dir = config.logdir / 'best_model'
    print('Logdir', config.logdir)

    # Create environments.
    datadir = config.logdir / 'episodes'
    writer_dir = pathlib.Path('./logs/evaluations/fourth-complete-run-MOD-AR4-FPS25')
    writer_dir.mkdir(parents=True, exist_ok=True)
    # writer = tf.summary.create_file_writer(str(config.logdir), max_queue=1000, flush_millis=20000)
    writer = tf.summary.create_file_writer(str(writer_dir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    base_env = make_base_env(config)
    actspace = base_env.action_space
    obsspace = base_env.observation_space
    test_env = [wrap_env(base_env, config, writer, writer_dir)]
    if config.obs_type == 'complete_low_dim':
        obsspace.spaces.update({'steer_old': gym.spaces.Box(low=-1, high=1, shape=(1,), dtype='float64')})
        obsspace.spaces.update({'wall_collision': gym.spaces.Box(low=0, high=1, shape=(1,), dtype='float64')})
        curv_length = len(base_env.scenario.world._lookahead_progress) + 1
        obsspace.spaces.update({'curvature': gym.spaces.Box(low=0, high=1, shape=(curv_length,), dtype='float64')})
        obsspace.spaces.update({'distance_from_other_cars': gym.spaces.Box(low=0, high=1, shape=(3,), dtype='float64')})
        lidar_complete_shape = obsspace.spaces['lidar'].shape[0] + obsspace.spaces['lidar_rear'].shape[0]
        obsspace.spaces.update(
                                {'lidar_complete': gym.spaces.Box(low=0, high=1, shape=(lidar_complete_shape,),
                                 dtype='float64')}
                               )
        agent_max_laps = base_env._env.env.env.scenario.agent.task._laps
        obsspace.spaces.update({'progress': gym.spaces.Box(low=0, high=agent_max_laps + 1, shape=(1,), dtype='float64')})
    camera_sensor_list = [match for match in list(obsspace.keys()) if 'camera' in match]
    if camera_sensor_list:
        config.image_sensor = camera_sensor_list[0]
    else:
        config.image_sensor = None

    # Load the agent.
    agent = Dreamer(config, datadir, obsspace, actspace, writer)
    print('Building complete.')
    checkpoints = sorted(best_checkpoint_dir.glob('*pkl'), key=lambda f: int(f.name.split('.')[0][16:]))
    if len(checkpoints):
        try:
            agent.load(checkpoints[-1])
            print('Load checkpoint.')
        except:
            raise Exception(f"the resume of checkpoint {checkpoints[-1]} failed")
        print('Checkpoint loading complete.')

    eval_step = 0

    while eval_step < 10:
        _, ep_reward = tools.simulate(functools.partial(agent, training=False), test_env, episodes=1)
        print(f'The reward of the episode {eval_step+1} is: ', ep_reward)  # ep_reward.numpy()?
        # This RL agent... Dreams! It's important to assess the quality of its imagination.
        imagination_summary(agent, writer_dir / 'episodes', eval_step + 1, config)
        eval_step += 1

    gifs_dir = writer_dir / 'saved_gifs'
    gifs_dir.mkdir(parents=True, exist_ok=True)
    event_path_list = [filename for filename in writer_dir.glob('*.v2')]
    # Given the nature of the experiment, there must be only one event file.
    if len(event_path_list) > 1:
        raise Exception('There must be only one event file regarding evaluation.')
    event_path = event_path_list[0]
    save_gifs_from_event(str(event_path), 'agent/openl/gif', gifs_dir)


class StepCounter:
    def __init__(self):
        self._step = 0

    def count_step(self):
        self._step += 1
        return self._step


if __name__ == '__main__':
    try:
        import colored_traceback
        colored_traceback.add_hook()
    except ImportError:
        pass
    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
    main(parser.parse_args())
