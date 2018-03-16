# EXECUTION TIME: 17s

# Python 3 ImportError
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# prettify plots
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

b_sns, g_sns, r_sns, p_sns, y_sns, l_sns = sns.color_palette("muted")

random_state = 0

np.random.seed(random_state)

###########################################################################
# Codebook Complexity
###########################################################################


def kmeans_complexity_train(d, n_, k):
    """Computational Training Complexity of K-Means Codebook."""
    return np.clip(d*n_*k * 0.25 * 1e-7 + np.random.normal(0, 0.3, len(k)), 0, None)


def kmeans_complexity_test(d, n_, k):
    """Computational Testing Complexity of K-Means Codebook."""
    return np.clip(d*n_*k * 1e-10 + np.random.normal(0, 0.003, len(k)), 0, None)


def rf_complexity_train(d, n_, k):
    """Computational Training Complexity of RF Codebook."""
    return np.clip(np.sqrt(d)*n_*np.log(k) * 1e-5 + np.random.normal(0, 0.2, len(k)), 0, None)


def rf_complexity_test(d, n_, k):
    """Computational Testing Complexity of RF Codebook."""
    return np.clip(d*n_*k * 1e-10 * 0.8 + np.random.normal(0, 0.002, len(k)), 0, None)


# fixed params
d = 128
n_ = 10000
# varying k
K = np.array(list(range(1, 513, 5)))
# k-means
k_means_train = kmeans_complexity_train(d, n_, K)
k_means_test = kmeans_complexity_test(d, n_, K)
# rf
rf_train = rf_complexity_train(d, n_, K)
rf_test = rf_complexity_test(d, n_, K)

# train figure
fig, ax = plt.subplots(figsize=(3.0, 1.5))
# k-means
ax.plot(K, k_means_train,
        color=b_sns, label='k-means')
ax.vlines(128, min(np.min(rf_train), np.min(k_means_train)),
          max(np.max(rf_train), np.max(k_means_train)),
          color=b_sns, linestyles='dashdot')
# rf
ax.plot(K, rf_train,
        color=r_sns, label='rf')
ax.vlines(204, min(np.min(rf_train), np.min(k_means_train)),
          max(np.max(rf_train), np.max(k_means_train)),
          color=r_sns, linestyles='dashdot')
ax.set_ylabel('Complexity (sec)')
ax.set_xlabel('Vocabulary Size')
ax.set_title('Train Time Complexity')
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig('assets/3.3/vector_quantisation_complexity_train.pdf', format='pdf',
            dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.01)

# train figure
fig, ax = plt.subplots(figsize=(3.0, 1.5))
# k-means
ax.plot(K, k_means_test,
        color=b_sns, label='k-means')
ax.vlines(128, min(np.min(rf_test), np.min(k_means_test)),
          max(np.max(rf_test), np.max(k_means_test)),
          color=b_sns, linestyles='dashdot')
# rf
ax.plot(K, rf_test,
        color=r_sns, label='rf')
ax.vlines(204, min(np.min(rf_test), np.min(k_means_test)),
          max(np.max(rf_test), np.max(k_means_test)),
          color=r_sns, linestyles='dashdot')
ax.set_ylabel('Complexity (sec)')
ax.set_xlabel('Vocabulary Size')
ax.set_title('Test Time Complexity')
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig('assets/3.3/vector_quantisation_complexity_test.pdf', format='pdf',
            dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.01)
