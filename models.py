import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl
import tensorflow.keras.mixed_precision as prec
import tools


class RSSM(tools.Module):

  def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
    super().__init__()
    self._activation = act
    self._stoch_size = stoch
    self._deter_size = deter
    self._hidden_size = hidden
    self._cell = tfkl.GRUCell(self._deter_size)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    return dict(
        mean=tf.zeros([batch_size, self._stoch_size], dtype),
        std=tf.zeros([batch_size, self._stoch_size], dtype),
        stoch=tf.zeros([batch_size, self._stoch_size], dtype),
        deter=self._cell.get_initial_state(None, batch_size, dtype))

  @tf.function
  def observe(self, embed, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    embed = tf.transpose(embed, perm=[1, 0, 2])
    action = tf.transpose(action, perm=[1, 0, 2])
    post, prior = tools.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (action, embed), (state, state))
    post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()}
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = tf.transpose(action, [1, 0, 2])
    prior = tools.static_scan(self.img_step, action, state)
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    return tf.concat([state['stoch'], state['deter']], -1)

  def get_dist(self, state):
    return tfd.MultivariateNormalDiag(state['mean'], state['std'])

  @tf.function
  def obs_step(self, prev_state, prev_action, embed):
    prior = self.img_step(prev_state, prev_action)
    x = tf.concat([prior['deter'], embed], -1)
    # x = tf.concat([prior['deter'], tf.cast(embed, dtype='float16')], -1)  # with config.precision == 16 (actually, not necessary)
    x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action):
    x = tf.concat([prev_state['stoch'], prev_action], -1)
    x = self.get('img1', tfkl.Dense, self._hidden_size, self._activation)(x)
    # x = tf.cast(x, dtype='float16')  # with config.precision == 16 (actually, not necessary)
    x, deter = self._cell(x, prev_state['deter'])
    x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
    return prior


class ConvEncoder(tools.Module):

    def __init__(self, depth=32, act=tf.nn.relu, img_sensor='low_res_camera'):
        super().__init__()
        self._act = act
        self._depth = depth
        self._img_sensor = img_sensor

    def __call__(self, obs):
        kwargs = dict(strides=2, activation=self._act)
        # x = obs['image']
        # New: 'image' isn't an observation field
        x = obs[self._img_sensor]
        x = tf.reshape(x, (-1,) + tuple(x.shape[-3:]))
        # x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
        x = self.get('h1', tfkl.Conv2D, 1 * self._depth, 4, **kwargs)(x)
        x = self.get('h2', tfkl.Conv2D, 2 * self._depth, 4, **kwargs)(x)
        x = self.get('h3', tfkl.Conv2D, 4 * self._depth, 4, **kwargs)(x)
        x = self.get('h4', tfkl.Conv2D, 8 * self._depth, 4, **kwargs)(x)
        # shape = tf.concat([tf.shape(obs['image'])[:-3], [32 * self._depth]], 0)
        shape = tf.concat([tf.shape(obs[self._img_sensor])[:-3], [32 * self._depth]], 0)
        return tf.reshape(x, shape)


class ConvDecoder(tools.Module):

    def __init__(self, depth=32, act=tf.nn.relu, shape=(64, 64, 3)):
        super().__init__()
        self._act = act
        self._depth = depth
        self._shape = shape

    def __call__(self, features):
        kwargs = dict(strides=2, activation=self._act)
        x = self.get('h1', tfkl.Dense, 32 * self._depth, None)(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
        x = self.get('h2', tfkl.Conv2DTranspose, 4 * self._depth, 5, **kwargs)(x)
        x = self.get('h3', tfkl.Conv2DTranspose, 2 * self._depth, 5, **kwargs)(x)
        x = self.get('h4', tfkl.Conv2DTranspose, 1 * self._depth, 6, **kwargs)(x)
        x = self.get('h5', tfkl.Conv2DTranspose, self._shape[-1], 6, strides=2)(x)
        mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
        return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))


class DenseDecoder(tools.Module):

    def __init__(self, shape, layers, units, dist='normal', act=tf.nn.elu):
        self._shape = shape
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act

    def __call__(self, features):
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
        x = self.get(f'hout', tfkl.Dense, np.prod(self._shape))(x)
        x = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
        if self._dist == 'normal':
            return tfd.Independent(tfd.Normal(x, 1), len(self._shape))
        if self._dist == 'binary':
            return tfd.Independent(tfd.Bernoulli(x), len(self._shape))
        raise NotImplementedError(self._dist)


class ActionDecoder(tools.Module):

    def __init__(self, size, layers, units, dist='tanh_normal', act=tf.nn.elu,
                min_std=1e-4, init_std=5, mean_scale=5):
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    def __call__(self, features):
        raw_init_std = np.log(np.exp(self._init_std) - 1)
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
        if self._dist == 'tanh_normal':
            # https://www.desmos.com/calculator/rcmcf5jwe7
            x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
            mean, std = tf.split(x, 2, -1)
            mean = self._mean_scale * tf.tanh(mean / self._mean_scale)
            std = tf.nn.softplus(std + raw_init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
            dist = tfd.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == 'onehot':
            x = self.get(f'hout', tfkl.Dense, self._size)(x)
            dist = tools.OneHotDist(x)
        else:
            raise NotImplementedError(dist)
        return dist


# # # Low-dimension feature observations # # #
## LiDAR case ##


class MLPEncoder(tools.Module):
    # Not used by racing_dreamer's authors
    # It could be useful if we don't limit to LiDAR obs...
    def __init__(self, encoded_dim, depth, num_layers, act=tf.nn.relu):
        super().__init__()
        self._name = "MLP_encoder"
        self._act = act
        self._depth = depth
        self._layers = num_layers
        self._encoded_dim = encoded_dim

    def __call__(self, obs):
        if type(obs) == dict:
            lidar = obs['lidar']
        else:
            lidar = obs
        if len(lidar.shape) > 2:
            x = tf.reshape(lidar, shape=(-1, *lidar.shape[2:], 1))
        else:
            x = lidar
        x = self.get('flat', tfkl.Flatten)(x)
        for idx in range(self._layers):
            x = self.get(f'dense{idx+1}', tfkl.Dense, units=self._depth, activation=self._act)(x)
        x = self.get(f'dense{idx + 2}', tfkl.Dense, units=self._encoded_dim)(x)
        # x = self.get('dense1', tfkl.Dense, units=4 * self._depth, activation=self._act)(x)
        # x = self.get('dense2', tfkl.Dense, units=2 * self._depth, activation=self._act)(x)
        # x = self.get('dense3', tfkl.Dense, units=self._encoded_dim)(x)
        shape = (*lidar.shape[:-1], *x.shape[1:])
        return tf.reshape(x, shape=shape)


class IdentityEncoder(tools.Module):
    # Original comment by racing_dreamer's authors
    # This is a dummy encoder created for working with Lidar observations.
    # The size of the lidar scan is 1080, so we pass it directly without any compression.
    # In this way, the algorithm's structure is the same for all the observations.
    def __init__(self):
        super().__init__()
        self._name = "identity_encoder"

    def __call__(self, obs):
        if type(obs) == dict:
            lidar = obs['lidar']
        else:
            lidar = obs
        return lidar


class LidarDistanceDecoder(tools.Module):
    def __init__(self, depth, shape, num_layers, act=tf.nn.relu):
        super().__init__()
        self._name = "lidar_decoder"
        self._act = act
        self._shape = shape
        self._depth = depth
        self._layers = num_layers

    def __call__(self, features):
        # Non-unitary variance solution
        """""
        # note: features = tf.concat([state['stoch'], state['deter']], -1)])
        x = tf.reshape(features, shape=(-1, *features.shape[2:]))  # (2500, featdim) --> featdim = 200 + 30
        x = self.get('dense1', tfkl.Dense, units=2 * self._depth, activation=None)(x)  # (2500, 2*depth)
        x = self.get('dense2', tfkl.Dense, units=4 * self._depth, activation=self._act)(x)   # (2500, 4*depth)
        params = tfpl.IndependentNormal.params_size(self._shape[0])
        x = self.get('params', tfkl.Dense, units=params, activation=tf.nn.leaky_relu)(x)
        x = self.get('dist', tfpl.IndependentNormal, event_shape=self._shape[0])(x)
        dist_old = tfd.BatchReshape(x, batch_shape=features.shape[:2])
        """""

        # Alternative: unitary variance
        x = tf.reshape(features, shape=(-1, *features.shape[2:]))
        for idx in range(self._layers):
            x = self.get(f'dense{idx+1}', tfkl.Dense, units=self._depth, activation=self._act)(x)
        x = self.get(f'dense{idx + 2}', tfkl.Dense, units=self._shape[0])(x)
        mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
        dist = tfd.Independent(tfd.Normal(mean, 1), len(self._shape))
        return dist


# # # Low-dimension feature observations # # #
## Complete observation case ##


class CompleteEncoder(tools.Module):
    def __init__(self, encoded_dim, depth, num_layers, obs_names, act=tf.nn.relu):
        super().__init__()
        self._name = "complete_encoder"
        self._act = act
        self._depth = depth
        self._layers = num_layers
        self._encoded_dim = encoded_dim
        self._obs_names = obs_names

    def __call__(self, raw_obs):
        if type(raw_obs) == dict:
            # PAST
            # [list(raw_obs[name].shape)[0], list(raw_obs[name].shape)[1], 1]
            # obj_to_concat = [
            #     tf.reshape(raw_obs[name], tools.flatten([list(raw_obs[name].shape), [1]])) if tf.shape(raw_obs[name]).shape == 2
            #      else raw_obs[name] for name in self._obs_names]
            # obj_to_concat = [tf.reshape(raw_obs[name], (50, 50, 1)) if tf.shape(raw_obs[name]).shape == 2 else raw_obs[name] for name in self._obs_names]
            # obs = tf.concat([raw_obs[name] for name in self._obs_names], -1)  # NO GOOD

            max_shape = max([len(list(raw_obs[name].shape)) for name in self._obs_names])
            obj_to_concat = [
                             tf.reshape(raw_obs[name], tools.flatten([list(raw_obs[name].shape), [1]]))
                             if tuple(tf.shape(raw_obs[name]).shape)[0] < max_shape
                             else raw_obs[name]
                             for name in self._obs_names
                             ]
            obs = tf.concat(obj_to_concat, -1)
        else:
            obs = raw_obs
        if len(obs.shape) > 2:
            x = tf.reshape(obs, shape=(-1, *obs.shape[2:], 1))
        else:
            x = obs
        x = self.get('flat', tfkl.Flatten)(x)
        for idx in range(self._layers):
            x = self.get(f'dense{idx+1}', tfkl.Dense, units=self._depth, activation=self._act)(x)
        x = self.get(f'dense{idx + 2}', tfkl.Dense, units=self._encoded_dim)(x)
        # x = self.get('dense1', tfkl.Dense, units=4 * self._depth, activation=self._act)(x)
        # x = self.get('dense2', tfkl.Dense, units=2 * self._depth, activation=self._act)(x)
        # x = self.get('dense3', tfkl.Dense, units=self._encoded_dim)(x)
        shape = (*obs.shape[:-1], *x.shape[1:])
        return tf.reshape(x, shape=shape)


class CompleteDecoder(tools.Module):
    # take inspiration from this, in order to build the "CompleteLowDimFeaturesDecoder"
    # Shape is crucial ---> TBN: they use convolutional decoder!
    # Thus, cloning and modifying "LidarDistanceDecoder" is probably better
    def __init__(self, depth, shape, num_layers, act=tf.nn.relu):
        # it reconstructs the occupancy map of the surrounding area as binary img of size (64,64,1)
        super().__init__()
        self._name = "complete_decoder"
        self._act = act
        self._depth = depth
        self._shape = shape
        self._layers = num_layers

    def __call__(self, features):
        # Non-unitary variance solution
        """""
        # note: features = tf.concat([state['stoch'], state['deter']], -1)])
        x = tf.reshape(features, shape=(-1, *features.shape[2:]))  # (2500, featdim) --> featdim = 200 + 30
        x = self.get('dense1', tfkl.Dense, units=2 * self._depth, activation=None)(x)  # (2500, 2*depth)
        x = self.get('dense2', tfkl.Dense, units=4 * self._depth, activation=self._act)(x)   # (2500, 4*depth)
        params = tfpl.IndependentNormal.params_size(self._shape[0])
        x = self.get('params', tfkl.Dense, units=params, activation=tf.nn.leaky_relu)(x)
        x = self.get('dist', tfpl.IndependentNormal, event_shape=self._shape[0])(x)
        dist_old = tfd.BatchReshape(x, batch_shape=features.shape[:2])
        """""

        # Alternative: unitary variance
        x = tf.reshape(features, shape=(-1, *features.shape[2:]))
        for idx in range(self._layers):
            x = self.get(f'dense{idx+1}', tfkl.Dense, units=self._depth, activation=self._act)(x)
        x = self.get(f'dense{idx + 2}', tfkl.Dense, units=self._shape[0])(x)
        mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
        dist = tfd.Independent(tfd.Normal(mean, 1), len(self._shape))
        return dist
