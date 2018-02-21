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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import src as ya

# prettify plots
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

b_sns, g_sns, r_sns, p_sns, y_sns, l_sns = sns.color_palette("muted")

np.random.seed(13)

# fetch data
data_train, data_query = ya.data.getCaltech(
    num_descriptors=10000, num_features=128, pickle_load=True)

X_train, y_train = data_train[:, :-1], data_train[:, -1]
X_test, y_test = data_query[:, :-1], data_query[:, -1]

clf = RandomForestClassifier(
    n_estimators=50, max_depth=10).fit(X_train, y_train)

print(clf.score(X_test, y_test))

sns.heatmap(confusion_matrix(y_test, clf.predict(X_test)))
plt.show()
