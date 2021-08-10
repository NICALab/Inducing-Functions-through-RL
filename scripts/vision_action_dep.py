#sub_path = './data/vision_action_decouple_purpose/task_b_sequence_ext_use_pred_2021_03_21_02_30_33_0002--s-60/log_dir/'
sub_path = './data/vision_action_decouple_purpose/task_b_2021_03_16_13_53_40_0000--s-0/log_dir/'
#sub_path = './data/vision_action_decouple_purpose/task_b_vision_only_2021_03_17_22_57_43_0001--s-0/log_dir/'

none = [0] * 3
pred = [0] * 3
prey = [0] * 3
rprey = [0] * 3
death_sign = [0] * 3

for i in range(200):
    path = sub_path + 'test_policy_log_' + str(i+1) + '.txt'
    f = open(path, 'r')
    for j in range(2):
        lines = f.readline()
    while True:
        lines = f.readline()
        lines = lines.split()

        # check death_sign
        if lines[0] == 'Predator':
            death_sign[0] += 1
            break
        elif lines[0] == 'Hunger':
            death_sign[1] += 1
            break
        elif lines[0] == 'Sickness':
            death_sign[2] += 1
            break

        if int(lines[1]) <= 12 and int(lines[2]) >= 15:
            if lines[3] == 'No':
                action = 0
            elif lines[3] == 'Run':
                action = 1
            elif lines[3] == 'Eat':
                action = 2

            if lines[0] == 'None':
                none[action] += 1
            elif lines[0] == 'Pred':
                pred[action] += 1
            elif lines[0] == 'Prey':
                prey[action] += 1
            elif lines[0] == 'RPrey':
                rprey[action] += 1

print(none, pred, prey, rprey, death_sign)
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from matplotlib import rc
import pandas as pd

r = [0, 1, 2, 3]
# From raw value to percentage
none = none / np.sum(none)
pred = pred / np.sum(pred)
prey = prey / np.sum(prey)
rprey = rprey / np.sum(rprey)
greenBars = [none[0], pred[0], prey[0], rprey[0]]
orangeBars = [none[1], pred[1], prey[1], rprey[1]]
blueBars = [none[2], pred[2], prey[2], rprey[2]]

# plot
barWidth = 0.85
names = ('none', 'pred', 'prey', 'rprey')
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
# Create blue Bars
plt.bar(r, blueBars, bottom=[i + j for i, j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white',
        width=barWidth)

# Custom x axis
plt.xticks(r, names)
#plt.xlabel("group")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels=['stay', 'run', 'eat'], shadow=False, scatterpoints=1)
plt.ylim(0, 1)
# Show graphic
plt.savefig(sub_path+'vision_action_decouple.svg')
plt.show()
plt.close()


