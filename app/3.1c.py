# EXECUTION TIME: 1m27s

# Python 3 ImportError
import sys
sys.path.append('.')

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.datasets.samples_generator import make_blobs

# prettify plots
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

b_sns, g_sns, r_sns, p_sns, y_sns, l_sns = sns.color_palette("muted")

# #############################################################################
# Comparison Matrix
n_samples = np.linspace(2000, 150000, 100)
df = pd.DataFrame(index=n_samples, columns=[
                  "KMeans Execution Time",
                  "MiniBatchKMeans Execution Time",
                  "Inertia Delta"])

centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)

for samples in n_samples:
    # Generate sample data
    np.random.seed(0)

    batch_size = int(samples // 100)
    X, labels_true = make_blobs(
        n_samples=50000, centers=centers, cluster_std=0.7)

    # Compute clustering with Means
    k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
    t0 = time.time()
    k_means.fit(X)
    t_batch = time.time() - t0

    # Compute clustering with MiniBatchKMeans
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=3,
                          batch_size=batch_size, n_init=10,
                          max_no_improvement=10, verbose=0)
    t0 = time.time()
    mbk.fit(X)
    t_mini_batch = time.time() - t0

    df.loc[samples] = [t_batch, t_mini_batch,
                       np.abs(k_means.inertia_ - mbk.inertia_)]

print('\n', df.to_latex(), '\n')
df.to_csv('assets/3.1/kmeans/complexity.csv')

fig, ax1 = plt.subplots()

lns1 = ax1.plot(n_samples, df["KMeans Execution Time"],
                label="KMeans", color=b_sns)
lns2 = ax1.plot(n_samples, df["MiniBatchKMeans Execution Time"],
                label="MiniBatchKMeans", color=g_sns)
plt.legend()
ax1.set_xlabel('Number of Training Samples')
ax1.set_ylabel('Execution Time')
ax1.set_title('KMeans vs MiniBatchKMeans\n Execution Time Complexity')
ax1.tick_params('y', colors=b_sns)

ax2 = ax1.twinx()
lns3 = ax2.plot(n_samples, df["Inertia Delta"],
                label="Divergence", color=r_sns)
ax2.set_ylabel('Inertia Delta')
ax2.tick_params('y', colors=r_sns)

lns = lns1 + lns2 + lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)

fig.tight_layout()
fig.savefig('assets/3.1/kmeans/time.pdf', format='pdf', dpi=300,
            transparent=True, bbox_inches='tight', pad_inches=0.01)
