import matplotlib.pyplot as plt
import numpy as np

import util

fig = plt.figure(figsize=(6, 4))

ax = fig.add_subplot(111)

# 10 runs on real:
# gaussian 13.528346545714886s with sample standard deviation 0.6546116197874118
# baseline 5.61081526670605s with sample standard deviation 0.6207300805105078

# 10 more runs on real

# 10 runs on synthetic:
# gaussian average 11.311129187862388s with sample standard deviation 1.2197720987320115
# baseline average 10.373363356303889s with sample standard deviation 0.8547104563218956
title = "runtime"

proposedMeans = [13.528346545714886, 10.117791139532347, 11.311129187862388]
proposedStd = [0.6546116197874118, 0.28226746046907586, 1.2197720987320115]

baselineMeans = [5.61081526670605, 7.868961964687332, 10.373363356303889]
baselineStd = [0.6207300805105078, 0.21140644188581317, 0.8547104563218956]

group_names = ['real\n(1797 samples,\n1000 iterations)', 'synthetic\n(10000 samples,\n1000 iterations)',
               'synthetic\n(100000 samples,\n100 iterations)']

ind = np.arange(len(proposedMeans))  # the x locations for the groups
width = 0.2  # the width of the bars

colors = plt.cm.rainbow(np.linspace(0, 1, 2))

alpha = .6
rects2 = ax.bar(ind + width, proposedMeans, width,
                color=colors[0],
                alpha=alpha,
                yerr=proposedStd,
                error_kw=dict(elinewidth=2, ecolor=colors[0]))

rects1 = ax.bar(ind, baselineMeans, width,
                color=colors[1],
                alpha=alpha,
                yerr=baselineStd,
                error_kw=dict(elinewidth=2, ecolor=colors[1]))

ax.set_xlim(-width, len(ind) + width)
ax.set_ylabel('runtime for all iterations in s')
ax.set_xlabel('dataset')

ax.set_xticks(ind + width)
xtickNames = ax.set_xticklabels(group_names)
plt.setp(xtickNames, rotation=0, fontsize=10)

ax.legend((rects1[0], rects2[0]), ('baseline', 'proposed'))

plt.tight_layout()

plt.savefig(str(util.timestamp_directory / ".." / f'{title}.pdf'))
