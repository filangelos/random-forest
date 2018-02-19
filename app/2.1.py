# Python 3 ImportError
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import typing
import time

from sklearn.ensemble import RandomForestClassifier

import src as ya
from src.struct import ForestParams

# prettify plots
plt.rcParams['figure.figsize'] = [5.0, 5.0]
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

b_sns, g_sns, r_sns, p_sns, y_sns, l_sns = sns.color_palette("muted")

np.random.seed(0)

# fetch data
data_train, data_query = ya.data.getData('Toy_Spiral')
N, D = data_train.shape

###########################################################################
# Grow a Forest
###########################################################################

forest = ya.tree.growForest(
    data_train, ForestParams(5, 3, 'IG', 50, 'axis-aligned', 5))

# test points for classification
test_points = np.array([[-.5, -.7], [.4, .3], [-.7, .4], [.5, -.5]])

# distribution plots


# label prediction
labels = forest.predict(test_points)

###########################################################################
# Validation of Hyperparameters
###########################################################################

X_train, y_train = data_train[:, :-1], data_train[:, -1]

# random forest classifier
forest = RandomForestClassifier(n_estimators=20,
                                criterion='entropy',
                                min_samples_split=5,
                                max_depth=5)

forest.fit(X_train, y_train)

# data range
r = [-1.5, 1.5]
# split function
xx, yy = np.meshgrid(np.linspace(*r, 200), np.linspace(*r, 200))

Z = forest.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# decision boundary line
plt.contour(xx, yy, Z, linewidths=0.8, colors='k')
# decision surfaces
plt.contourf(xx,
             yy,
             Z,
             cmap=plt.cm.jet.from_list(
                 'contourf', [b_sns, g_sns, r_sns], 3),
             alpha=0.4)
# plot toy data
ya.visualise.plot_toydata(data_train)

plt.show()
