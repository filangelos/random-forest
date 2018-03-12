# EXECUTION TIME:

# Python 3 ImportError
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

import src as ya
from src.struct import ForestParams

# prettify plots
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

b_sns, g_sns, r_sns, p_sns, y_sns, l_sns = sns.color_palette("muted")

np.random.seed(13)

# fetch data
data_train, data_query = ya.data.getData('Toy_Spiral')
N, D = data_train.shape

# Supervised Data
X_train, y_train = data_train[:, :-1], data_train[:, -1]

###########################################################################
# Validation of Hyperparameters
###########################################################################

grid_params = {'max_depth': np.arange(2, 26, 2),
               'n_estimators': np.arange(1, 15, 1),
               'min_samples_split': np.arange(5, 30, 5),
               'min_impurity_decrease': np.arange(0, 0.11, 0.03),
               'max_features': [1, 2],
               }

# Cross-Validation Container
search = GridSearchCV(RandomForestClassifier(),
                      param_grid=grid_params, cv=10).fit(X_train, y_train)

# cache GridSearchCV object to `tmp` folder
pickle.dump(search, open('tmp/models/2c/search.pkl', 'wb'))

# Best Parameters
best_params_ = search.best_params_

print(best_params_)
