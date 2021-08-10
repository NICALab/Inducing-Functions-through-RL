import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from collections import defaultdict
import seaborn as sns
import pandas as pd

task_default_list = ['task_b_2021',
                     'task_b_vision_only_2021',
                     'task_b_sequence_ext_use_pred_20',
                     'task_b_sequence_ext_use_pred_60',
                     'task_b_sequence_ext_use_pred_80',
                     'task_b_random_ext_8',
                     'task_b_random_ext_10',
                     'task_b_random_ext_12',
                     'task_b_sequence_ext_use_pred_perm1',
                     'task_b_sequence_ext_use_pred_perm2',
                     'task_b_sequence_ext_use_pred_perm3',
                     'cifar10_2021',
                     'cifar10_sequence_ext_use_pred_2021',
                     'cifar10_vision_only_2021'
                     ]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_one(exp_names, csv_slices, feature, env_name):
    # plot features for every experiments
    fig = plt.figure(figsize=(8, 8))
    fig.canvas.set_window_title(feature)
    for csv_slice in csv_slices:
        plt.plot(moving_average(csv_slice[feature].to_numpy(), 100))
    plt.legend(exp_names)
    plt.title(env_name, fontsize=17)
    plt.xlabel("iteration", fontsize=15)
    plt.xticks(fontsize=13)
    plt.ylabel(feature, fontsize=15)
    plt.yticks(fontsize=13)

    # make dataframe for multiple experiments
    task_list = []
    for task in task_default_list:
        if any(task in s for s in exp_names):
            task_list.append(task)
    num_df = len(task_list)
    df = []
    for i in range(num_df):
        feature_vals_list = []
        feature_vals_len_list = []
        print(i)
        for j, exp_name in enumerate(exp_names):
            if task_list[i] in exp_name:
                print(task_list[i], exp_name)
                csv_slice = csv_slices[j]
                feature_vals = moving_average(csv_slice[feature].to_numpy(), 100)
                max_len = min([2500, len(feature_vals)])
                feature_vals_list.append(feature_vals[:max_len])
                feature_vals_len_list.append(range(max_len))

        feature_vals_array = np.concatenate(feature_vals_list, axis=0)
        feature_vals_len_array = np.concatenate(feature_vals_len_list, axis=0)
        df_i = pd.DataFrame({'iteration': feature_vals_len_array,
                             task_list[i]: feature_vals_array})
        df.append(df_i)

    fig = plt.figure(figsize=(8, 8))
    fig.canvas.set_window_title(feature)
    for i in range(num_df):
        sns.lineplot(data=df[i], x='iteration', y=task_list[i])
    plt.legend(task_list)
    plt.title(env_name, fontsize=17)
    plt.xlabel("iteration", fontsize=15)
    plt.xticks(fontsize=13)
    plt.ylabel(feature, fontsize=15)
    plt.yticks(fontsize=13)




def plot_data(args):
    path = args.file
    features = args.f
    style = args.s

    plt.style.use(style)
    features = features[0].split(",")

    for feature in features:
        path = path.rstrip('/').rstrip('\\')
        env_name = path.split('/')[-1]
        method = env_name.split('-')[0]
        env_name = env_name.replace(method + '-', '')
        csv_paths = glob.glob(f"{path}/**/progress.csv")
        exp_names = [csv_path.split("/")[-2] for csv_path in csv_paths]

        assert len(csv_paths) > 0, "There is no csv files"

        csv_slices = []
        for csv_path in csv_paths:
            csv = pd.read_csv(csv_path)
            csv_slices.append(csv.loc[:, [feature]])
            del csv

        plot_one(exp_names, csv_slices, feature, env_name)
    plt.show()


if __name__ == "__main__":
    # To run, refer README.md
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the task directory')
    parser.add_argument('--f', type=str, nargs='+',
                        help='List of features to plot')
    parser.add_argument('--s', type=str, default='ggplot',
                        help='Style of plots, Look at (https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html)')
    args = parser.parse_args()
    plot_data(args)