import os
import gzip
import pickle
import numpy as np
import h5py

# environment specifications
env_specs = dict(none=[0, 6], pred=[8, 1], prey=[3, 7], rprey=[2, 4])
img_map = ['none', 'pred', 'prey', 'rprey']
path = './env_survive/mnist'
num_train_img = 1000

# load mnist dataset
with gzip.open(os.path.join(path, 'mnist.pkl.gz'), 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    img_shape = (28, 28)
    x_train, y_train = train_set
    x_train = x_train.reshape(len(x_train), img_shape[0], img_shape[1])
    print(x_train.shape, np.min(x_train), np.max(x_train), x_train.dtype)

path = path + '/perm3'
# generate images for survival environment
for i in range(len(img_map)):
    train_data_img = []
    train_data_lab = []
    test_data_img = []
    test_data_lab = []
    len_set = [0] * 10

    for j in range(len(x_train)):
        img = x_train[j]
        lab = y_train[j]
        if lab in env_specs[img_map[i]] and len_set[lab] < num_train_img:
            train_data_img.append(img[..., None])
            train_data_lab.append([i])
            len_set[lab] += 1
        elif lab in env_specs[img_map[i]]:
            test_data_img.append(img[..., None])
            test_data_lab.append([i])

    train_data_img = np.asarray(train_data_img, dtype=np.float32)
    train_data_lab = np.asarray(train_data_lab, dtype=np.uint8)
    test_data_img = np.asarray(test_data_img, dtype=np.float32)
    test_data_lab = np.asarray(test_data_lab, dtype=np.uint8)
    print(train_data_img.shape, train_data_lab.shape, test_data_img.shape, test_data_lab.shape)
    f1 = h5py.File(os.path.join(path, img_map[i] + '.hdf5'), 'w')
    f2 = h5py.File(os.path.join(path, img_map[i] +'_test.hdf5'), 'w')
    f1['image'] = train_data_img
    f1['label'] = train_data_lab
    f2['image'] = test_data_img
    f2['label'] = test_data_lab
    f1.close()
    f2.close()

