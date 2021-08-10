# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
import os
import gzip
import pickle
import numpy as np
import h5py

# environment specifications
env_specs = dict(none=[2], pred=[3], prey=[4], rprey=[6])
img_map = ['none', 'pred', 'prey', 'rprey']
path = './env_survive/cifar10/cifar-10-batches-py'


def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    # test data
    # cifar_test_data_dict
    # 'batch_label': 'testing batch 1 of 1'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names


train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = load_cifar_10_data(path)
print("Train data: ", train_data.shape)
print("Train filenames: ", train_filenames.shape)
print("Train labels: ", train_labels.shape)
print("Test data: ", test_data.shape)
print("Test filenames: ", test_filenames.shape)
print("Test labels: ", test_labels.shape)
print("Label names: ", label_names.shape)
print(train_data.dtype, np.max(train_data), np.min(train_data))
x_train = train_data.astype(np.float32) / 255.
x_test = test_data.astype(np.float32) / 255.
y_train = train_labels.astype(np.uint8)
y_test = test_labels.astype(np.uint8)
print(x_train.dtype, np.max(x_train), np.min(x_train))
print(x_test.dtype, np.max(x_test), np.min(x_test))

path = './env_survive/cifar10'
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
        if lab in env_specs[img_map[i]]:
            train_data_img.append(img[...])
            train_data_lab.append([i])
    for j in range(len(x_test)):
        img = x_test[j]
        lab = y_test[j]
        if lab in env_specs[img_map[i]]:
            test_data_img.append(img[...])
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

