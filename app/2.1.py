# Python 3 ImportError
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import src as ya
from src.struct import ForestParams

# prettify plots
plt.rcParams['figure.figsize'] = [12.0, 3.0]
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

print(labels)
