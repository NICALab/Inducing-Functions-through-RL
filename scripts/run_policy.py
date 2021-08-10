import os
import copy
import numpy as np
import argparse
import torch
import uuid
import scipy.io as scio
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from rlkit.core import logger
from rlkit.torch.pytorch_util import set_gpu_mode
import torch.nn.functional as F

filename = str(uuid.uuid4())


def rolloutM0(env, agent, max_path_length=np.inf):
    if hasattr(env, 'obs_to_image_obs'):
        preprocess_obs_for_policy_fn = env.obs_to_image_obs
    else:
        preprocess_obs_for_policy_fn = lambda x: x

    raw_obs = []
    raw_next_obs = []
    full_obs = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    full_next_obs = []
    path_length = 0
    agent.reset()
    o = env.reset()
    full_o = {'v': env.cur_label, 's': env.cur_status}
    while path_length < max_path_length:
        raw_obs.append(o)
        full_obs.append(full_o)
        a, agent_info = agent.get_action(preprocess_obs_for_policy_fn(o))
        next_o, r, d, env_info = env.step(copy.deepcopy(a), agent_info)
        next_full_o = {'v': env.cur_label, 's': env.cur_status}
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        raw_next_obs.append(next_o)
        full_next_obs.append(next_full_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        full_o = next_full_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    return dict(
        observations=raw_obs,
        actions=actions,
        rewards=rewards,
        next_observations=raw_next_obs,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        full_observations=full_obs,
        full_next_observations=full_next_obs,
    )


def rollout(env, agent, max_path_length=np.inf):
    raw_obs = []
    raw_next_obs = []
    full_obs = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    full_next_obs = []
    path_length = 0
    agent.reset()
    o = env.reset()
    full_o = {'v': env.cur_label, 's': env.cur_status}
    while path_length < max_path_length:
        raw_obs.append(o)
        full_obs.append(full_o)
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(copy.deepcopy(a))
        next_full_o = {'v': env.cur_label, 's': env.cur_status}
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        raw_next_obs.append(next_o)
        full_next_obs.append(next_full_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        full_o = next_full_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    return dict(
        observations=raw_obs,
        actions=actions,
        rewards=rewards,
        next_observations=raw_next_obs,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        full_observations=full_obs,
        full_next_observations=full_next_obs,
    )


def write_log(n, fpath, path, env):
    # save log file as .txt and .mat
    vision_map = ['None', 'Pred', 'Prey', 'RPrey']
    action_map = ['No op', 'Eat', 'Run']
    death_map = ['Success', 'Predator', 'Hunger', 'Sickness']
    log_path = fpath + '/test_policy_log_' + str(n + 1)
    log_txt_path = log_path + '.txt'
    log_mat_path = log_path + '.mat'
    mat_np_array = []
    file = open(log_txt_path, 'w+')
    file.write('%dth Episodes starts!!!' % (n + 1))
    file.write('\n' + '%s\t %s\t %s\t %s\t' % ('Vision', 'Hunger', 'Sick', 'Action'))

    # log
    for j in range(len(path['observations'])):
        full_o = path['full_observations'][j]
        a = path['actions'][j][0]
        file.write('\n' + '%s\t %s\t %s\t %s\t' %
                   (vision_map[full_o['v']], full_o['s'][0], full_o['s'][1], action_map[a]))
        mat_np_array.append([env.cur_label, env.cur_status[0], env.cur_status[1], a])
    death_sign = path['env_infos'][-1]['death_sign']
    death_sign = death_map[death_sign]
    ep_ret = np.sum(path['rewards'])
    ep_len = len(path['observations'])
    file.write('\n' + death_sign + ' killed the agent')
    file.write('\n' + 'Episode %d \t EpRet %.1f \t EpLen %d' % (n + 1, ep_ret, ep_len))
    file.write('\nEpisodes Ends...')
    file.close()
    mat_np_array = np.array(mat_np_array, dtype=np.int8)
    scio.savemat(log_mat_path, dict(death_certificate=mat_np_array, death_sign=death_sign))

    if not hasattr(env, 'sequential_update'):
        return ep_ret, None, None

    # track status
    track_path = fpath + '/test_policy_track_' + str(n+1)
    log_track_path = track_path + '.png'
    log_txt_path = track_path + '.txt'
    true_hunger = []
    true_sick = []
    hidden_hunger = []
    hidden_sick = []
    max_status = env.max_status
    for j in range(len(path['observations'])):
        true_val = path['full_observations'][j]['s']  # hunger and sickness
        true_hunger.append(true_val[0] / max_status)
        true_sick.append(true_val[1] / max_status)
        hidden_val = path['agent_infos'][j]['hidden']
        hidden_hunger.append(hidden_val[0])
        hidden_sick.append(hidden_val[1])
    true_status = [true_hunger, true_sick]
    hidden_status = [hidden_hunger, hidden_sick]

    status_map = ['hunger', 'sickness']
    t = np.arange(0, len(true_hunger))
    fig, ax = plt.subplots(nrows=2, ncols=1)
    for i in range(2):
        ax1 = ax[i]
        color = 'tab:red'
        ax1.set_xlabel('time step')
        ax1.set_ylabel('true ' + status_map[i], color=color)
        ax1.plot(t, true_status[i], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0.0, 1.0)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('pred ' + status_map[i], color=color)  # we already handled the x-label with ax1
        ax2.plot(t, hidden_status[i], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0.0, 1.0)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(log_track_path)
    plt.close()

    file = open(log_txt_path, 'w+')
    file.write('%dth Episodes' % (n + 1))
    file.write('\n' + '%s\t %s\t' % ('Hunger', 'Sick'))
    for j in range(20):
        for i in range(20):
            file.write('\n' + '%s\t %s\t' % (j, i))
            for k in range(len(true_hunger)):
                if j == true_status[0][k] and i == true_status[1][k]:
                    file.write('%.2f\t %.2f\t' % (hidden_status[0][k], hidden_status[1][k]))
    file.close()

    return ep_ret, true_status, hidden_status


def analyze_memory_component_corr(fpath, true_status_list, hidden_status_list, bptt):
    from scipy.stats import pearsonr
    hunger_corr_list = []
    sick_corr_list = []
    for i, true_hidden_status in enumerate(zip(true_status_list, hidden_status_list)):
        true_status, hidden_status = true_hidden_status
        true_hunger, true_sick = true_status
        hidden_hunger, hidden_sick = hidden_status
        hunger_corr, _ = pearsonr(true_hunger, hidden_hunger)
        sick_corr, _ = pearsonr(true_sick, hidden_sick)
        hunger_corr_list.append(hunger_corr)
        sick_corr_list.append(sick_corr)

        if len(true_hunger) > 100:
            status_map = ['hunger', 'sickness']
            title_map = ['Hunger Prediction (Score: {0:.4f})'.format(hunger_corr),
                         'Sickness Prediction (Score: {0:.4f})'.format(sick_corr)]
            t = np.arange(0, len(true_hunger))
            fig, ax = plt.subplots(nrows=2, ncols=1)
            for j in range(2):
                ax1 = ax[j]
                # ax1.set_title(title_map[j])
                color = 'black'
                # ax1.set_xlabel('time step')
                # ax1.set_ylabel('true ' + status_map[j], color=color)
                ax1.plot(t, true_status[j], color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.set_ylim(0.0, 1.0)
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                color = 'tab:blue'
                # ax2.set_ylabel('pred ' + status_map[j], color=color)  # we already handled the x-label with ax1
                ax2.plot(t, hidden_status[j], color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylim(0.0, 1.0)
                fig.tight_layout()  # otherwise the right y-label is slightly clipped

            log_track_path = fpath + '/test_policy_track_learned_' + str(i + 1) + '.svg'
            plt.savefig(log_track_path)
            plt.close()
    return hunger_corr_list, sick_corr_list


def analyze_memory_component(fpath, true_status_list, hidden_status_list, bptt):

    # gather episodes bigger than length 100
    episode_num_list = []
    good_true_status_list = []
    good_hidden_status_list = []
    for i, true_hidden_status in enumerate(zip(true_status_list, hidden_status_list)):
        true_status, hidden_status = true_hidden_status
        true_hunger, true_sick = true_status
        hidden_hunger, hidden_sick = hidden_status
        if len(true_hunger) > 100:
            good_true_status_list += list(zip(true_hunger[bptt:], true_sick[bptt:]))
            good_hidden_status_list += list(zip(hidden_hunger[bptt:], hidden_sick[bptt:]))
            episode_num_list.append(i)

    # train regressor
    good_true_status_array = np.array(good_true_status_list)
    good_hidden_status_array = np.array(good_hidden_status_list)
    from sklearn.linear_model import LinearRegression
    X_hunger = good_hidden_status_array[:, 0:1]
    y_hunger = good_true_status_array[:, 0:1]
    reg_hunger = LinearRegression().fit(X_hunger, y_hunger)
    X_sick = good_hidden_status_array[:, 1:2]
    y_sick = good_true_status_array[:, 1:2]
    reg_sick = LinearRegression().fit(X_sick, y_sick)
    print('regression score hunger: %.4f, sickenss: %.4f' % (reg_hunger.score(X_hunger, y_hunger),
                                                             reg_sick.score(X_sick, y_sick)))

    for ep_num in episode_num_list:
        true_status = true_status_list[ep_num]
        hidden_status = hidden_status_list[ep_num]
        true_hunger, true_sick = true_status
        hidden_hunger, hidden_sick = hidden_status
        good_true_status_list = list(zip(true_hunger[bptt:], true_sick[bptt:]))
        good_hidden_status_list = list(zip(hidden_hunger[bptt:], hidden_sick[bptt:]))

        good_true_status_array = np.array(good_true_status_list)
        good_hidden_status_array = np.array(good_hidden_status_list)
        X_hunger = good_hidden_status_array[:, 0:1]
        y_hunger = good_true_status_array[:, 0:1]
        X_sick = good_hidden_status_array[:, 1:2]
        y_sick = good_true_status_array[:, 1:2]

        hidden_hunger, hidden_sick = hidden_status
        hidden_status_reshape = np.array(list(zip(hidden_hunger[bptt:], hidden_sick[bptt:])))
        pred_hunger = reg_hunger.predict(hidden_status_reshape[:, 0:1]).tolist()
        pred_sick = reg_sick.predict(hidden_status_reshape[:, 1:2]).tolist()
        pred_status = [pred_hunger, pred_sick]
        status_map = ['hunger', 'sickness']
        title_map = ['Hunger Prediction (Score: {0:.4f})'.format(reg_hunger.score(X_hunger, y_hunger)),
                     'Sickness Prediction (Score: {0:.4f})'.format(reg_sick.score(X_sick, y_sick))]
        t = np.arange(0, len(pred_hunger))
        fig, ax = plt.subplots(nrows=2, ncols=1)
        for j in range(2):
            ax1 = ax[j]
            ax1.set_title(title_map[j])
            color = 'tab:red'
            ax1.set_xlabel('time step')
            ax1.set_ylabel('true ' + status_map[j], color=color)
            ax1.plot(t, true_status[j], color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_ylim(0.0, 1.0)
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel('pred ' + status_map[j], color=color)  # we already handled the x-label with ax1
            ax2.plot(t, pred_status[j], color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(0.0, 1.0)
            fig.tight_layout()  # otherwise the right y-label is slightly clipped

        log_track_path = fpath + '/test_policy_track_learned_' + str(ep_num + 1) + '.png'
        plt.savefig(log_track_path)
        plt.close()
    '''    
    for i, true_hidden_status in enumerate(zip(true_status_list, hidden_status_list)):
        true_status, hidden_status = true_hidden_status
        true_hunger, true_sick = true_status
        hidden_hunger, hidden_sick = hidden_status
        if len(true_hunger) > 60:
            good_true_status_list = list(zip(true_hunger[bptt:], true_sick[bptt:]))
            good_hidden_status_list = list(zip(hidden_hunger[bptt:], hidden_sick[bptt:]))

            good_true_status_array = np.array(good_true_status_list)
            good_hidden_status_array = np.array(good_hidden_status_list)
            from sklearn.linear_model import LinearRegression
            X_hunger = good_hidden_status_array[:, 0:1]
            y_hunger = good_true_status_array[:, 0:1]
            reg_hunger = LinearRegression().fit(X_hunger, y_hunger)
            X_sick = good_hidden_status_array[:, 1:2]
            y_sick = good_true_status_array[:, 1:2]
            reg_sick = LinearRegression().fit(X_sick, y_sick)

            ep_num = i
            hidden_hunger, hidden_sick = hidden_status
            hidden_status_reshape = np.array(list(zip(hidden_hunger[bptt:], hidden_sick[bptt:])))
            pred_hunger = reg_hunger.predict(hidden_status_reshape[:, 0:1]).tolist()
            pred_sick = reg_sick.predict(hidden_status_reshape[:, 1:2]).tolist()
            pred_status = [pred_hunger, pred_sick]
            status_map = ['hunger', 'sickness']
            title_map = ['Hunger Prediction (Score: {0:.2f})'.format(reg_hunger.score(X_hunger, y_hunger)),
                         'Sickness Prediction (Score: {0:.2f})'.format(reg_sick.score(X_sick, y_sick))]
            t = np.arange(0, len(pred_hunger))
            fig, ax = plt.subplots(nrows=2, ncols=1)
            for j in range(2):
                ax1 = ax[j]
                ax1.set_title(title_map[j])
                color = 'tab:red'
                ax1.set_xlabel('time step')
                ax1.set_ylabel('true ' + status_map[j], color=color)
                ax1.plot(t, true_status[j], color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.set_ylim(0.0, 1.0)
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                color = 'tab:blue'
                ax2.set_ylabel('pred ' + status_map[j], color=color)  # we already handled the x-label with ax1
                ax2.plot(t, pred_status[j], color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylim(0.0, 1.0)
                fig.tight_layout()  # otherwise the right y-label is slightly clipped

            log_track_path = fpath + '/test_policy_track_learned_' + str(ep_num + 1) + '.png'
            plt.savefig(log_track_path)
            plt.close()
    '''


def analyze_corner_case(fpath, env, policy):
    vision_map = ['None', 'Pred', 'Prey', 'RPrey']
    action_map = ['No op', 'Eat', 'Run']
    corner_path = fpath + '/corner_case.txt'
    file = open(corner_path, 'w+')
    for i in range(env.n_labels):
        file.write(vision_map[i] + '\n')
        file.write('%s\t %s\n' % ('Hunger', 'Sick'))
        v = np.zeros((env.n_labels,), np.float32)
        v[i] = 1
        for j in range(int(env.max_status)):
            for k in range(int(env.max_status)):
                s = np.array([j / env.max_status, k / env.max_status], np.float32)
                o = np.concatenate((v, s), axis=0)
                a, agent_info = policy.get_action(o)
                file.write('%s\t %s\t %s\n' % (j, k, action_map[a]))
    file.close()


def extract_features(policy, fpath, data_path):
    vision_map = ['none', 'pred', 'prey', 'rprey']
    # vision_map = ['pred', 'prey', 'rprey']
    online_result = {'images': [], 'labels': []}
    test_result = {'images': [], 'labels': []}
    q_net = policy.qf
    from env_survive.env_survive import get_dataset
    import rlkit.torch.pytorch_util as ptu
    for vision in vision_map:
        online_data_path = data_path + vision + '.hdf5'
        test_data_path = data_path + vision + '_test.hdf5'
        online_data_dict = get_dataset(online_data_path)
        test_data_dict = get_dataset(test_data_path)
        online_images = online_data_dict['image']
        online_labels = online_data_dict['label']
        test_images = test_data_dict['image']
        test_labels = test_data_dict['label']
        online_images = online_images.reshape(online_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)
        online_features = ptu.get_numpy(q_net.get_fv(ptu.from_numpy(online_images).float()))
        test_features = ptu.get_numpy(q_net.get_fv(ptu.from_numpy(test_images).float()))
        online_result['images'].append(online_features)
        online_result['labels'].append(online_labels)
        test_result['images'].append(test_features)
        test_result['labels'].append(test_labels)

    online_result['images'] = np.concatenate(online_result['images'], axis=0)  # 8000x4
    online_result['labels'] = np.concatenate(online_result['labels'], axis=0)  # 8000x1
    test_result['images'] = np.concatenate(test_result['images'], axis=0)  # Bx4
    test_result['labels'] = np.concatenate(test_result['labels'], axis=0)  # Bx1

    scio.savemat(fpath + '/online_result.mat', dict(features=online_result['images'], labels=online_result['labels']))

    return online_result, test_result


def plot_pca_tsne(fpath, online_result, do_normalize=False):
    # plot pca and tsne of clustered encoded images
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    from sklearn.manifold import TSNE
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    fig_path = fpath + '/online_data_'
    if do_normalize:
        X = normalize(online_result['images'], norm='l2')
    else:
        X = online_result['images']

    y = online_result['labels'].squeeze()
    vision_map = [('none', 0), ('pred', 1), ('prey', 2), ('rprey', 3)]
    colors = ['navy', 'turquoise', 'darkorange', 'red']
    # vision_map = [('pred', 1), ('prey', 2), ('rprey', 3)]

    '''
    # pca 2d
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)
    plt.figure()
    lw = 2
    for color, (name, label) in zip(colors, vision_map):
        plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], color=color, alpha=.8, lw=lw,
                    label=name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA 2d')
    plt.show()
    plt.close()

    # lda 2d
    lda = LinearDiscriminantAnalysis(n_components=3)
    X_lda = lda.fit(X, y).transform(X)
    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange', 'red']
    lw = 2
    for color, (name, label) in zip(colors, vision_map):
        plt.scatter(X_lda[y == label, 0], X_lda[y == label, 1], X_lda[y == label, 2],
                    color=color, alpha=.8, lw=lw,
                    label=name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('lda 2d')
    plt.show()
    plt.close()

    # tsne 2d
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X)
    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange', 'red']
    lw = 2
    for color, (name, label) in zip(colors, vision_map):
        plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1], color=color, alpha=.8, lw=lw,
                    label=name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('tSNE 2d')
    plt.show()
    plt.close()
    '''

    # pca 3d
    fig = plt.figure(1, figsize=(5, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .9, 1], elev=30, azim=180)
    plt.cla()
    pca = PCA(n_components=3)
    pca.fit(X)
    X_pca = pca.transform(X)

    for (name, label), color in zip(vision_map, colors):
        '''
        ax.text3D(X_pca[y == label, 0].mean(),
                  X_pca[y == label, 1].mean(),
                  X_pca[y == label, 2].mean(),
                  "",
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        '''

        ax.scatter(X_pca[y == label, 0], X_pca[y == label, 1], X_pca[y == label, 2], color=color)
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        #ax.set_xticks([])
        #ax.set_yticks([])
        #ax.set_zticks([])
    #ax.legend(loc='upper left', labels=[name[0] for name in vision_map], shadow=False, scatterpoints=1)
    plt.savefig(fig_path + 'pca.svg')
    plt.show()
    plt.close()

    '''
    # lda 3d
    fig = plt.figure(1, figsize=(10, 10))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    lda = LinearDiscriminantAnalysis(n_components=3)
    pca.fit(X)
    X_lda = lda.fit(X, y).transform(X)

    for (name, label), color in zip(vision_map, colors):
        ax.text3D(X_lda[y == label, 0].mean(),
                  X_lda[y == label, 1].mean() + 1.5,
                  X_lda[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        ax.scatter(X_lda[y == label, 0], X_lda[y == label, 1], X_lda[y == label, 2], color=color)
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
    # plt.savefig(fig_path + 'pca.png')
    plt.show()
    plt.close()

    # tsne 3d
    fig = plt.figure(1, figsize=(10, 10))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    tsne = TSNE(n_components=3, random_state=0)
    X_tsne = tsne.fit_transform(X)

    for name, label in vision_map:
        ax.text3D(X_tsne[y == label, 0].mean(),
                  X_tsne[y == label, 1].mean() + 1.5,
                  X_tsne[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y, cmap=plt.cm.nipy_spectral,
               edgecolor='k')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    #plt.savefig(fig_path + 'tsne.png')
    plt.show()
    plt.close()
    '''


def check_linearity(online_result, test_result):
    # accuracy of linear SVM
    from sklearn.svm import LinearSVC
    X_train = online_result['images']
    y_train = online_result['labels'].squeeze()
    X_test = test_result['images']
    y_test = test_result['labels'].squeeze()
    model = LinearSVC(C=1, max_iter=1000000)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print('training accuracy: %.2f (%d/%d)' % (train_acc * 100, len(X_train) * train_acc, len(X_train)))
    print('test accuracy: %.2f (%d/%d)' % (test_acc * 100, len(X_test) * test_acc, len(X_test)))

'''
python -m scripts.run_policy ./data/task-b/task_b_2021_03_16_13_53_40_0000--s-0/itr_4000.pkl
python -m scripts.run_policy ./data/task-b-sequence-ext-use-pred/task_b_sequence_ext_use_pred_2021_03_21_02_30_33_0002--s-60/itr_2500.pkl
'''
def simulate_policy(args):
    data = torch.load(args.file)
    fpath = '/'.join(args.file.split('/')[:-1]) + '/log_dir'
    if not os.path.exists(fpath):
        os.mkdir(fpath)

    policy = data['evaluation/policy']
    env = data['evaluation/env']
    print("\npolicy loaded...")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()

    # if not env.raw_vision:
    #     # analyze corner case
    #     print("\nanalyzing corner cases...")
    #     analyze_corner_case(fpath, env, policy)
    # else:
    #     # visualize clustering and check linearity with linear svm
    print("\nvisualizing clustering and checking linearity...")
    #online_result, test_result = extract_features(policy, fpath, data_path='./env_survive/cifar10/')
    online_result, test_result = extract_features(policy, fpath, data_path='./env_survive/mnist/')
    #online_result, test_result = extract_features(policy, fpath, data_path='./env_survive/mnist/perm2/')
    #plot_pca_tsne(fpath, online_result)
    check_linearity(online_result, test_result)

    # write log
    print("\nwriting log...")
    ep_ret_list = []
    true_status_list = []
    hidden_status_list = []
    for n in range(200):
        path = rolloutM0(
            env,
            policy,
            max_path_length=args.H,
        )
        ep_ret, true_status, hidden_status = write_log(n, fpath, path, env)
        ep_ret_list.append(ep_ret)
        true_status_list.append(true_status)
        hidden_status_list.append(hidden_status)
    print('return avg: %.1f, std: %.1f, max: %d, min: %d' %
          (np.mean(ep_ret_list), np.std(ep_ret_list), np.max(ep_ret_list), np.min(ep_ret_list)))

    if hasattr(env, 'sequential_update'):
        # analyze memory component
        print("\nanalyzing memory component...")
        hunger_corr_list, sick_corr_list = analyze_memory_component_corr(fpath, true_status_list, hidden_status_list, bptt=0)
        print('correlation hunger: %.4f, sickenss: %.4f' % (np.mean(hunger_corr_list), np.mean(sick_corr_list)))
        # analyze_memory_component(fpath, true_status_list, hidden_status_list, bptt=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=500,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_false')
    args = parser.parse_args()

    simulate_policy(args)
