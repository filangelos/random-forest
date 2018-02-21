# Python 3 ImportError
import sys
sys.path.append('.')

import numpy as np
import typing

import src as ya

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
          Params(False, 1-1/np.exp(1))]

for j, (replace, fraction) in enumerate(params):
    # index
    idx = np.random.choice(range(N), int(N*fraction), replace)
    # cardinality of subset
    card = 100 * len(np.unique(idx)) / len(idx)
    # plot title
    title = 'Fraction: %.2f%%, Uniqueness %.2f%%\n' % (
        fraction*100, card)
    title += '[With Replacement]' if replace else '[Without Replacement]'
    # visualise
    ya.visualise.plot_toydata(data=data_train[idx, :],
                              new_figure=True,
                              title=title,
                              savefig_path='1.1/subset_%i' % j)
