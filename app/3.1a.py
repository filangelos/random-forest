# EXECUTION TIME: 28s

# Python 3 ImportError
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

import src as ya
from src.struct import ForestParams

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
data_train, data_query = ya.data.getCaltech(savefig_images=True,
                                            num_descriptors=2,
                                            pickle_load=False,
                                            pickle_dump=False,
                                            num_features=2)
