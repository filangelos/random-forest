import numpy as np
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

import src as ya

# prettify plots
plt.rcParams['figure.figsize'] = [3.0, 3.0]
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

b_sns, g_sns, r_sns, p_sns, y_sns, l_sns = sns.color_palette("muted")


def visualise_splitfunc(idx_best,
                        data,
                        dim,
                        t,
                        ig_best,
                        iterations,
                        predictor,
                        learner,
                        savefig_path=None):
    """Draw the split line."""
    # subplots
    plt.close('all')
    fig, axes = plt.subplots(ncols=4, figsize=(12.0, 3.0))

    # data range
    r = [-1.5, 1.5]
    # split function
    xx, yy = np.meshgrid(np.linspace(*r, 200), np.linspace(*r, 200))
    Z = predictor(np.c_[xx.ravel(), yy.ravel()], dim, t).reshape(xx.shape)
    # decision boundary line
    axes[0].contour(xx, yy, Z, linewidths=0.8, colors='k')
    # decision surfaces
    axes[0].contourf(xx,
                     yy,
                     Z,
                     cmap=plt.cm.jet.from_list(
                         'contourf', [p_sns, y_sns], 2),
                     alpha=0.4)
    # spiral data
    axes[0].plot(data[~idx_best, 0],
                 data[~idx_best, 1],
                 marker='*',
                 markeredgecolor='k',
                 markersize=10,
                 linestyle='None')
    axes[0].plot(data[idx_best, 0],
                 data[idx_best, 1],
                 marker='+',
                 markeredgecolor='k',
                 markersize=10,
                 linestyle='None')
    axes[0].plot(data[data[:, -1] == 1, 0],
                 data[data[:, -1] == 1, 1],
                 marker='o',
                 markerfacecolor=b_sns,
                 markeredgecolor='k',
                 linestyle='None')
    axes[0].plot(data[data[:, -1] == 2, 0],
                 data[data[:, -1] == 2, 1],
                 marker='o',
                 markerfacecolor=g_sns,
                 markeredgecolor='k',
                 linestyle='None')
    axes[0].plot(data[data[:, -1] == 3, 0],
                 data[data[:, -1] == 3, 1],
                 marker='o',
                 markerfacecolor=r_sns,
                 markeredgecolor='k',
                 linestyle='None')

    axes[0].set_xlim(r)
    axes[0].set_ylim(r)
    axes[0].set_xticks(np.linspace(*r, 5))
    axes[0].set_yticks(np.linspace(*r, 5))
    if iterations == -1:
        axes_0_title = '$\\mathbf{%s}$\nIG = %.4f' % (
            learner.capitalize(), ig_best)
    else:
        axes_0_title = '$\\mathbf{%s}$\nTrial %i, IG = %.4f' % (
            learner.capitalize(), iterations, ig_best)
    axes[0].set_title(axes_0_title)

    # parent node
    labels = data[:, -1]
    bars, bins = ya.util.histc(labels, return_bins=True)
    axes[1].bar(bins, bars, color=[b_sns, g_sns, r_sns])
    axes[1].set_xlim([0.5, 3.5])
    axes[1].set_ylim([0, np.max(bars)*1.05])
    axes[1].set_title('Class histogram of\n$\\mathbf{Parent}$ node')

    # left child
    labels_left = data[idx_best, -1]
    bars_left = ya.util.histc(labels_left, bins)
    axes[2].bar(bins, bars_left, color=[b_sns, g_sns, r_sns])
    axes[2].set_xlim([0.5, 3.5])
    axes[2].set_ylim([0, np.max(bars)*1.05])
    axes[2].set_title('Class histogram of\n$\\mathbf{Left\\ Child}$ node')
    # background color
    axes[2].set_facecolor((*p_sns, .4))

    # right child
    labels_right = data[~idx_best, -1]
    bars_right = ya.util.histc(labels_right, bins)
    axes[3].bar(bins, bars_right, color=[b_sns, g_sns, r_sns])
    axes[3].set_xlim([0.5, 3.5])
    axes[3].set_ylim([0, np.max(bars)*1.05])
    axes[3].set_title('Class histogram of\n$\\mathbf{Right\\ Child}$ node')
    # background color
    axes[3].set_facecolor((*y_sns, .4))

    plt.tight_layout()

    # save figure to file
    if isinstance(savefig_path, str):
        fig.savefig('assets/%s.pdf' % savefig_path, format='pdf', dpi=300,
                    transparent=False, bbox_inches='tight', pad_inches=0.01)
    else:
        plt.draw()
        fig.waitforbuttonpress()


def plot_x_mean_std(x: np.ndarray,
                    mean: np.ndarray,
                    std: np.ndarray,
                    title: str=None,
                    xlabel: str=None,
                    ylabel: str=None,
                    new_figure: bool=False,
                    legend: bool=False,
                    show: bool=False,
                    savefig_path: str=None) -> None:
    """Plot `(x, mean±std)`.

    Parameters
    ----------
    x: numpy.ndarray
        x-axis data
    mean: numpy.ndarray
        y-axis values
    std: numpy.ndarray
        Standard deviation of y-axis values
    title: str
        Figure Title
    xlabel: str
        Figure x-axis label
    ylabel: str
        Figure y-axis label
    new_figure: bool
        Create new `plt.figure()`
    legend: bool
        Flag to call `plt.legend()`
    show: bool
        Flag to call `plt.show()`
    savefig_path: str
        `plt.savefig()` path
    """
    if new_figure:
        plt.figure(figsize=(6.0, 2.0))
    # scatter plot
    plt.plot(x, mean, color=r_sns)
    plt.fill_between(x, mean-std, mean+std, color=b_sns, alpha=0.5)
    # figure title
    if title is not None:
        plt.title(title)
    # figure xlabel
    if xlabel is not None:
        plt.xlabel(xlabel)
    # figure ylabel
    if ylabel is not None:
        plt.ylabel(ylabel)
    if legend:
        plt.legend(['mean', 'std'])
    # sns.despine()
    # save figure to file
    if isinstance(savefig_path, str):
        plt.savefig('assets/%s.pdf' % savefig_path, format='pdf', dpi=300,
                    transparent=True, bbox_inches='tight', pad_inches=0.01)
    # show figure
    if show:
        plt.show()


def plot_confusion_matrix(cm, savefig,
                          classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    sns.set_style({"xtick.direction": "out", "ytick.direction": "out"})
    if classes is None:
        classes = list(range(cm.shape[0]))
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    fmt = 'd' if not normalize else '.2f'
    sns.heatmap(cm, cmap=cmap,
                annot=True, fmt=fmt, ax=ax)
    ax.set_title(title)
    # tick_marks = np.arange(len(classes))
    # ax.set_xticks(tick_marks)
    # ax.set_xticklabels(classes)
    # ax.set_yticks(tick_marks)
    # ax.set_yticklabels(classes)
    ax.tick_params(axis=u'both', which=u'both', length=0)

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    fig.tight_layout()
    fig.savefig('assets/%s.pdf' % savefig, format='pdf', dpi=300,
                transparent=False, bbox_inches='tight', pad_inches=0.02)
