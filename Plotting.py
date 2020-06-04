import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


# ************* Plotting ************************
path = 'out/'
scores = ['Q3', 'Q7', 'Q8']

models = ['NN', 'RF', 'DSSP', 'STR']

for score in scores:
    accuracy = []
    for model in models:
        file = model + '_' + score
        if not (score == 'Q7' and (model == 'DSSP' or model == 'STR')):
            metric = pd.read_csv(path + file + '.csv', sep='\t', header=0,index_col=0)
            accuracy.append(np.asarray(metric.loc['acc']))

    plt.rc('font', size=10)
    fontdict = {'fontsize': 11}

    ind = np.arange(len(accuracy[0]))  # the x locations for the groups
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots()

    if score !='Q7':
        rects1 = ax.bar(ind - width / 2, accuracy[0], width / 4, label='Neural Network')
        rects2 = ax.bar(ind - width / 6, accuracy[1], width / 4, label='Random Forest')
        rects3 = ax.bar(ind + width / 6, accuracy[2], width / 4, label='DSSP')
        rects4 = ax.bar(ind + width / 2, accuracy[3], width / 4, label='STRIDE')
    else:
        rects1 = ax.bar(ind - width / 4, accuracy[0], width / 2, label='Neural Network')
        rects2 = ax.bar(ind + width / 4, accuracy[1], width / 2, label='Random Forest')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Predicted classes')
    ax.set_title('Comparison of different models for multiclass prediction ({})'.format(score))
    ax.set_xticks(ind)
    ax.set_xticklabels(metric.columns.tolist())
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    fig.tight_layout()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    #plt.show()
    plt.savefig('plots/Accuracy_per_class_{}.png'.format(score))


SOV_refine_Q3 = [0.549, 0.572, 0.891, 0.874]
SOV_refine_Q7 = [0.640, 0.600, 0.000, 0.000]
SOV_refine_Q8 = [0.703, 0.711, 0.960, 0.762]

ind = np.arange(4)
width = 0.5  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(ind - width / 4, SOV_refine_Q3, width / 5, label='Q3')
rects2 = ax.bar(ind, SOV_refine_Q7, width / 4, label='Q7')
rects3 = ax.bar(ind + width / 4, SOV_refine_Q8, width / 5, label='Q8')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('SOV Score')
ax.set_title('Comparison of SOV score for different models')
ax.set_xticks(ind)
ax.set_xticklabels(models)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
fig.tight_layout()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
#plt.show()
plt.savefig('plots/SOV_Scores.png')
