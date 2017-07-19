import matplotlib.pyplot as plt
import numpy as np

import util

fig = plt.figure(figsize=(5,4))

ax = fig.add_subplot(111)

# 10 runs on real:
# gaussian 13.528346545714886s with sample standard deviation 0.6546116197874118
# baseline 5.61081526670605s with sample standard deviation 0.6207300805105078
#
# 10 runs on synthetic:
# gaussian average 11.311129187862388s with sample standard deviation 1.2197720987320115
# baseline average 10.373363356303889s with sample standard deviation 0.8547104563218956

gaussianMeans = [13.528346545714886, 11.311129187862388]
gaussianStd = [0.6546116197874118, 1.2197720987320115]

baselineMeans = [5.61081526670605, 10.373363356303889]
baselineStd = [0.6207300805105078, 0.8547104563218956]

ind = np.arange(len(gaussianMeans))  # the x locations for the groups
width = 0.2  # the width of the bars

rects2 = ax.bar(ind + width, gaussianMeans, width,
                color='blue',
                alpha=.7,
                yerr=gaussianStd,
                error_kw=dict(elinewidth=2, ecolor='blue'))

rects1 = ax.bar(ind, baselineMeans, width,
                color='red',
                alpha=.7,
                yerr=baselineStd,
                error_kw=dict(elinewidth=2, ecolor='red'))

ax.set_xlim(-width, len(ind) + width)
ax.set_ylabel('runtime for all iterations in s')
ax.set_xlabel('dataset')
xTickMarks = ['real\n(1797 samples,\n1000 iterations)', 'synthetic\n(100000 samples,\n100 iterations)']
ax.set_xticks(ind + width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=45, fontsize=10)

ax.legend((rects1[0], rects2[0]), ('baseline', 'proposed'))

plt.tight_layout()

plt.savefig(str(util.timestamp_directory / ".." / "runtime.pdf"))
