# EXECUTION TIME: 38m

# Python 3 ImportError
import sys
sys.path.append('.')

import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import src as ya

# prettify plots
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

b_sns, g_sns, r_sns, p_sns, y_sns, l_sns = sns.color_palette("muted")

np.random.seed(1)

###########################################################################
# Visualize Raw & SIFT Training/Testing Samples from Caltech_101
###########################################################################

# set all hyperparameters to small values to speed codebook generation
# since only interested in images generated at folder `assets/3.1/examples`
data_train, data_query = ya.data.getCaltech(codebook="random-forest",
                                            num_descriptors=100000,
                                            pickle_load=False,
                                            pickle_dump=False,
                                            num_features=10)

X_train, y_train = data_train[:, :-1], data_train[:, -1]
X_test, y_test = data_query[:, :-1], data_query[:, -1]

###########################################################################
# Validation of Hyperparameters
###########################################################################

grid_params = {'n_estimators': [10, 20, 50, 100, 250, 500, 1000],
               'max_depth': [2, 5, 7, 11],
               'min_samples_split': [5, 10, 20, 50],
               'min_impurity_decrease': [0.00, 0.01, 0.02, 0.05, 0.1]
               }

try:
    # fetch GridSearchCV object from `tmp` folder
    search = pickle.load(open('tmp/models/3.3/search.pkl', 'rb'))
except Exception:
    # Cross-Validation Container
    # WARNING: execution time ~50 minutes
    search = GridSearchCV(RandomForestClassifier(),
                          param_grid=grid_params, cv=10).fit(X_train, y_train)
    # cache GridSearchCV object to `tmp` folder
    pickle.dump(search, open('tmp/models/3.3/search.pkl', 'wb'))

# Best Parameters
best_params_ = search.best_params_
print(best_params_)

# Best Estimator
clf = search.best_estimator_

print(clf.score(X_test, y_test))

sns.heatmap(confusion_matrix(y_test, clf.predict(X_test)))
plt.show()
