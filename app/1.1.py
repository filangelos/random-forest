# EXECUTION TIME: 7s

# Python 3 ImportError
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import typing

import src as ya

# prettify plots
plt.rcParams['figure.figsize'] = [3.0, 3.0]
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

b_sns, g_sns, r_sns, p_sns, y_sns, l_sns = sns.color_palette("muted")

np.random.seed(0)

# fetch data
data_train, data_query = ya.data.getData('Toy_Spiral')
N, D = data_train.shape

###########################################################################
# Fraction vs Uniqueness plot
###########################################################################
num_frac = 100
mean = np.empty(num_frac)
std = np.empty(num_frac)

num_obs = 10

fractions_set = np.linspace(0.01, 1, num_frac)

for i, fraction in enumerate(fractions_set):
    # store uniqueness observations
    batch = np.empty(num_obs)
    for j in range(num_obs):
        # index
        idx = np.random.choice(range(N), int(N*fraction), True)
        # cardinality of subset
        batch[j] = 100 * len(np.unique(idx)) / len(idx)
    mean[i] = batch.mean()
    std[i] = batch.std()

ya.visualise.plot_x_mean_std(100*fractions_set,
                             mean=mean,
                             std=std,
                             title='Unique Elements vs Fraction',
                             xlabel='Fraction [%]',
                             ylabel='Uniqueness [%]',
                             legend=True,
                             savefig_path='1.1/uniqueness_vs_fraction')

###########################################################################
# Bagging
###########################################################################
Params = typing.NamedTuple('Params', replace=bool, fraction=float)

params = [Params(True, 1.0),
          Params(True, 1-1/np.exp(1)),
          Params(True, 0.5),
          Params(True, 0.75),
          Params(False, 1.0),
          Params(False, 1-1/np.exp(1)),
          # 100% fraction with replacement
          Params(True, 1.0),
          Params(True, 1.0),
          Params(True, 1.0),
          Params(True, 1.0)]

for j, (replace, fraction) in enumerate(params):
    # index
    idx = np.random.choice(range(N), int(N*fraction), replace)
    # cardinality of subset
    card = 100 * len(np.unique(idx)) / len(idx)
    # plot title
    title = 'Fraction: %.2f%%, Uniqueness %.2f%%\n' % (
        fraction*100, card)
    title += '[With Replacement]' if replace else '[Without Replacement]'
    # color map
    cmap = {0: y_sns, 1: b_sns, 2: g_sns, 3: r_sns}
    # create new figure
    fig = plt.figure()
    # add main axes
    ax_main = fig.add_axes([0.1, 0.1, 1.0, 1.0])
    # scatter plot
    ax_main.scatter(data_train[idx, 0],
                    data_train[idx, 1],
                    c=list(map(lambda l: cmap[l],
                               data_train[idx, 2])), alpha=1.0)
    ax_main.set_title(title)
    # add secondary axes
    ax_sec = fig.add_axes([0.85, 0.85, 0.2, 0.2])
    bars, bins = ya.util.histc(data_train[idx, 2], return_bins=True)
    norm_bars = bars / np.sum(bars)
    ax_sec.bar(bins, norm_bars, color=[b_sns, g_sns, r_sns])
    ax_sec.set_xlim([0.5, 3.5])
    ax_sec.set_ylim([0, np.max(norm_bars)*1.05])
    ax_sec.set_yticks([0.33])
    ax_sec.set_xticks([])
    ax_sec.set_xlabel('Class\nRepresentation', fontdict={'fontsize': 5})
    # save figure to file
    savefig_path = '1.1/subset_%i' % j
    fig.savefig('assets/%s.pdf' % savefig_path, format='pdf', dpi=300,
                transparent=True, bbox_inches='tight', pad_inches=0.01)
