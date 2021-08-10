import os
import numpy as np
import h5py
import gym
from gym import spaces
from gym.utils import seeding
from skimage.transform import resize

T = 8
pred_status = 2
max_status = 20.
max_random_status = 11
n_action = 3
n_status = 2
n_labels = 4
n_fv = 4
action_map = ['none', 'eat', 'run']
status_map = ['hunger', 'sickness']
vision_map = ['none', 'pred', 'prey', 'rprey']

state_transition_table = [[(0.3, 0.35, 0.35*0.65, 0.35*0.35),   # (none, no op) -> (none, pred, prey, rprey)
                           (0.3, 0.35, 0.35*0.65, 0.35*0.35),   # (none, eat) -> (none, pred, prey, rprey)
                           (0.3, 0.25, 0.45*0.65, 0.45*0.35),   # (none, run) -> (none, pred, prey, rprey)
                           ],
                          [(0.45, 0.2, 0.35*0.65, 0.35*0.35),   # pred
                           (0.45, 0.2, 0.35*0.65, 0.35*0.35),
                           (0.55, 0.1, 0.35*0.65, 0.35*0.35),
                           ],
                          [(0.25, 0.35, 0.4*0.65, 0.4*0.35),    # prey
                           (0.45, 0.35, 0.2*0.65, 0.2*0.35),
                           (0.45, 0.35, 0.2*0.65, 0.2*0.35),
                           ],
                          [(0.25, 0.35, 0.4*0.65, 0.4*0.35),    # rprey
                           (0.45, 0.35, 0.2*0.65, 0.2*0.35),
                           (0.45, 0.35, 0.2*0.65, 0.2*0.35),
                           ]]


status_table = [[(1, -1),   # (none, no op) -> (hunger, sickness)
                 (1, 1),    # (none, eat) -> (hunger, sickness)
                 (3, 2),    # (none, run) -> (hunger, sickness)
                 ],
                [(1, -1),   # pred
                 (-20, 1),
                 (3, 2),
                 ],
                [(1, -1),   # prey
                 (-20, 1),
                 (3, 2),
                 ],
                [(1, -1),   # rprey
                 (-20, 5),
                 (3, 2),
                 ]]


class EnvSurvive(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dataset, vision_shape, seed=0):
        super(EnvSurvive, self).__init__()

        # reinitialize
        self.n_action = n_action
        self.n_status = n_status
        self.n_labels = n_labels
        self.pred_status = pred_status
        self.vision_shape = None  # initialized in load_data
        self.max_status = max_status
        self.cur_t = None
        self.T = T
        self.ep_status = None  # status tracking in single ep
        self.ep_va = None  # vision and action tracking in single ep
        self.ep_h = None  # hidden tracking in single ep
        # vision input
        self.cur_vision = None  # current vision input to agent
        self.cur_label = None  # current label
        # status input
        self.cur_status = None  # current status of agent
        self.death_rew = - 5.0
        # vision dataset
        self.dataset = dataset
        self.vision_shape = vision_shape
        self.vision_size = np.prod(self.vision_shape)
        # define observation space and action space
        self.vision_index = 2  # we give vision number (what class, what index) instead of image
        self.ob_dim = self.vision_index + n_status
        self.ob_dim_true = self.vision_size + n_status
        self.observation_space = spaces.Box(low=0.0, high=np.inf,
                                            shape=(self.ob_dim,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(n_action)
        self.action_space = spaces.Discrete(n_action)
        # environment viewer
        self.viewer = None
        self.seed(seed)
        # for easy size calculation
        self.a1 = self.vision_size + self.n_status
        self.b1 = self.a1 + self.pred_status
        self.i1 = self.vision_size + self.n_action
        self.a2 = self.vision_index + self.n_status
        self.b2 = self.a2 + self.pred_status
        self.i2 = self.vision_index + self.n_action

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        random_label = self.np_random.choice(a=n_labels, size=None)
        random_status = self.np_random.randint(low=0, high=max_random_status, size=2, dtype=np.int8)
        random_image, random_label_ = self.data_from_dataset(random_label)
        assert random_label == random_label_

        self.cur_label = random_label
        self.cur_status = random_status
        self.cur_vision = random_image

        obs = np.concatenate((self.cur_vision.flatten(),
                              self.cur_status / self.max_status), axis=0).astype(np.float32)  # 8,

        return obs

    def step(self, action, agent_info=None):
        # Update status
        status_tuple = status_table[self.cur_label][action]
        status_np = np.array(status_tuple, dtype=np.int8)
        self.cur_status = self.cur_status + status_np
        self.cur_status = np.where(self.cur_status >= 0, self.cur_status, 0)  # minus values reset to 0

        # Check termination condition
        #####################
        # death by predator #
        #####################
        # if (pred, no op), terminate with prob 0.6
        if self.cur_label == 1 and action == 0:
            if self.np_random.uniform(low=0.0, high=1.0, size=None) < 0.6:
                return np.zeros((self.ob_dim,), np.float32), self.death_rew, True, {'death_sign': 1}
        # if (pred, eat), terminate with prob 0.7
        if self.cur_label == 1 and action == 1:
            if self.np_random.uniform(low=0.0, high=1.0, size=None) < 0.7:
                return np.zeros((self.ob_dim,), np.float32), self.death_rew, True, {'death_sign': 1}

        #####################
        #  death by status  #
        #############rue########
        death_prob = [0.] * 15 + [0.05, 0.1, 0.2, 0.4, 0.7, 1.0] + [1.] * 10  # margin for > 20
        # with death prob of hunger, terminate
        p = self.np_random.uniform(low=0.0, high=1.0, size=2)
        if p[0] < death_prob[self.cur_status[0]]:
            return np.zeros((self.ob_dim,), np.float32), self.death_rew, True, {'death_sign': 2}
        # with death prob of sickness, terminate
        if p[1] < death_prob[self.cur_status[1]]:
            return np.zeros((self.ob_dim,), np.float32), self.death_rew, True, {'death_sign': 3}

        # Update vision
        state_transition_prob = state_transition_table[self.cur_label][action]
        next_label = self.np_random.choice(a=n_labels, size=None, p=state_transition_prob)
        next_image, next_label_ = self.data_from_dataset(next_label)
        assert next_label == next_label_

        self.cur_label = next_label
        self.cur_vision = next_image

        obs = np.concatenate((self.cur_vision.flatten(),
                              self.cur_status / self.max_status), axis=0).astype(np.float32)  # 8,

        return obs, 1.0, False, {'death_sign': 0}

    def render(self, mode='human', action=None):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer(maxwidth=10000)

        height = 256
        width = 256
        num_channels = 3

        # vision
        # notice that cur_vision is (28, 28, 1)
        resized_vision = resize(self.cur_vision[..., 0], (height, width))
        resized_vision = (resized_vision * 255).astype(np.uint8)
        resized_vision = np.stack((resized_vision,) * 3, axis=-1)

        # more info on rendering
        button_size = 30
        interval = 10
        separate_line_thickness = 3
        num_pixel_per_status = 8

        canvas_width = max(n_action, n_status) * (button_size + interval) + interval
        separate_line_1 = height - (2 * interval + button_size +
                                    num_pixel_per_status * int(max_status) + 3 * separate_line_thickness)
        separate_line_2 = height - (2 * interval + button_size + separate_line_thickness)

        # draw on canvas
        canvas = np.full((height, canvas_width, num_channels), fill_value=255, dtype=np.uint8)
        canvas[separate_line_2 - separate_line_thickness:separate_line_2 + separate_line_thickness] = 100
        canvas[separate_line_1 - separate_line_thickness:separate_line_1 + separate_line_thickness] = 100

        # draw current status
        for i in range(n_status):
            y = separate_line_2 - separate_line_thickness
            x = 2 * interval + i * 4 * interval
            canvas[y - num_pixel_per_status * self.cur_status[i]:y, x:x + 10] = [0, 255, 0]

        # draw action buttons
        for i in range(n_action):
            y = separate_line_2 + interval + separate_line_thickness
            x = interval + i * (button_size + interval)
            if i == action:
                canvas[y:y + button_size, x:x + button_size] = [0, 0, 255]
            else:
                canvas[y:y + button_size, x:x + button_size] = [255, 0, 0]

        total_vision = np.concatenate((resized_vision, canvas), axis=1)
        self.viewer.imshow(total_vision)

    def data_from_dataset(self, index):
        img_set = self.dataset[vision_map[index] + '_image']
        lab_set = self.dataset[vision_map[index] + '_label']
        rand_idx = self.np_random.choice(a=len(img_set), size=None)
        return np.array([index, rand_idx]), lab_set[rand_idx]

    def obs_to_image_obs(self, obs):
        new_obs_dim = self.a1
        img_obs = np.zeros(shape=(new_obs_dim,), dtype=np.float32)
        img_obs[:self.vision_size] = self.dataset[vision_map[int(obs[0])] + '_image'][int(obs[1])].flatten()
        img_obs[self.vision_size:self.a1] = obs[self.vision_index:self.a2]
        return img_obs


def load_data(path):
    dataset = {}
    for vision in vision_map:
        fpath = os.path.join(path, vision + '.hdf5')
        data_dict = get_dataset(fpath)
        dataset[vision + '_image'] = data_dict['image']
        dataset[vision + '_label'] = data_dict['label']
    img = dataset['none_image'][0]
    return dataset, img.shape


def get_keys(h5file):
    keys = []
    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)
    h5file.visititems(visitor)
    return keys


def get_dataset(h5path):
    dataset_file = h5py.File(h5path, 'r')
    data_dict = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
    dataset_file.close()
    return data_dict










