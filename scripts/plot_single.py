import pandas as pd
import matplotlib.pyplot as plt
import glob
import argparse


def plot_one(exp_name, csv_slices, feature, env_name):
    fig = plt.figure(figsize=(8, 8))
    fig.canvas.set_window_title(exp_name)
    for i, csv_slice in enumerate(csv_slices):
        plt.plot(csv_slice[feature[i]].to_numpy())
    plt.legend(feature)
    plt.title(env_name, fontsize=17)
    plt.xlabel("iteration", fontsize=15)
    plt.xticks(fontsize=13)
    if 'Average Returns' in feature[0]:
        y_lab = 'evaluation/Average Returns'
    elif 'death' in feature[0]:
        y_lab = 'evaluation/death_sign'
    else:
        y_lab = None
    plt.ylabel(y_lab, fontsize=15)
    plt.yticks(fontsize=13)


def plot_data(args):
    path = args.file
    features = args.f
    style = args.s
    plt.style.use(style)
    features = features[0].split("|")
    feat_dict = {}
    for i, feat in enumerate(features):
        feat_dict[i] = feat.split(",")

    path = path.rstrip('/').rstrip('\\')
    env_name = path.split('/')[-1]
    csv_path = glob.glob(f"{path}/progress.csv")
    assert len(csv_path) == 1
    csv_path = csv_path[0]
    exp_name = csv_path.split("/")[-2]
    for feat_key in feat_dict.keys():
        feat = feat_dict[feat_key]
        csv_slices = []
        for single_feat in feat:
            csv = pd.read_csv(csv_path)
            csv_slices.append(csv.loc[:, [single_feat]])
            del csv
        plot_one(exp_name, csv_slices, feat, env_name)
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