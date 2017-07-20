import math
import matplotlib.pyplot as plt
import numpy as np

import util

distribution_names = ['high exponential', 'segments', 'low linear']

def plot_runtime():
    # 10 runs on real:
    # gaussian 13.528346545714886s with sample standard deviation 0.6546116197874118
    # baseline 5.61081526670605s with sample standard deviation 0.6207300805105078

    # 10 more runs on real

    # 10 runs on synthetic:
    # gaussian average 11.311129187862388s with sample standard deviation 1.2197720987320115
    # baseline average 10.373363356303889s with sample standard deviation 0.8547104563218956
    title = "runtime"

    proposedMeans = [13.528346545714886, 10.117791139532347, 11.311129187862388]
    proposedStds = [0.6546116197874118, 0.28226746046907586, 1.2197720987320115]
    baselineMeans = [5.61081526670605, 7.868961964687332, 10.373363356303889]
    baselineStds = [0.6207300805105078, 0.21140644188581317, 0.8547104563218956]

    group_names = ['real\n(1797 samples,\n1000 iterations)', 'synthetic\n(10000 samples,\n1000 iterations)',
                   'synthetic\n(100000 samples,\n100 iterations)']

    ylabel = 'runtime for all iterations in s'
    xlabel = 'dataset'

    bar_plot(baselineMeans, baselineStds, proposedMeans, proposedStds, group_names, xlabel, ylabel, title=title)


def plot_distributions():
    title = "distributions"

    proposedMeans = [0.715, 0.91, 0.955]
    proposedStds = [0.155804364509, 0.07, 0.0589491306128]
    baselineMeans = [0.735, 0.85, 0.925]
    baselineStds = [0.131434394281, 0.0866025403784, 0.0622494979899]

    ylabel = 'share of relevant features selected'
    xlabel = 'relevance distribution'

    bar_plot(baselineMeans, baselineStds, proposedMeans, proposedStds, distribution_names, xlabel, ylabel, title=title)


def bar_plot(baselineMeans, baselineStds, proposedMeans, proposedStds, group_names, xlabel, ylabel, title):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    group_x_locations = np.arange(len(proposedMeans))
    bar_width = 0.2
    colors = plt.cm.rainbow(np.linspace(0, 1, 2))
    alpha = .6
    error_linewidth = 2
    rects2 = ax.bar(group_x_locations + bar_width, proposedMeans, bar_width,
                    color=colors[0],
                    alpha=alpha,
                    yerr=proposedStds,
                    error_kw=dict(elinewidth=error_linewidth, ecolor=colors[0]))
    rects1 = ax.bar(group_x_locations, baselineMeans, bar_width,
                    color=colors[1],
                    alpha=alpha,
                    yerr=baselineStds,
                    error_kw=dict(elinewidth=error_linewidth, ecolor=colors[1]))
    ax.set_xlim(-bar_width, len(group_x_locations) + bar_width)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(group_x_locations + bar_width)
    xtickNames = ax.set_xticklabels(group_names)
    plt.setp(xtickNames, rotation=0, fontsize=10)
    ax.legend((rects1[0], rects2[0]), ('baseline', 'proposed'))
    plt.tight_layout()
    plt.savefig(str(util.timestamp_directory / ".." / f'{title}.pdf'))


if __name__ == '__main__':
    plot_distributions()
