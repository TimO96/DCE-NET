import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')

rootdir = '/scratch/tottens/DCE_MRI/github_code/params/'
SNRs = [7, 10, 20, 30, 40, 60, 80, 100]
sns.set()


def make_plot(method_scores):
    params = ['$K_{ep}$', '$v_e$', '$v_p$', 'dt']

    for i, param in enumerate(params):
        all_values_mean = []
        all_values_std = []
        for k, (scores, label) in enumerate(method_scores):
            values_mean = []
            values_std = []
            for j in range(len(SNRs)):
                values_std.append(scores[j][i][0])
                values_mean.append(scores[j][i][1])

            all_values_mean.append((np.abs(values_mean), label))
            all_values_std.append((values_std, label))

        for (values, label) in all_values_mean:
            plt.plot(SNRs, values, label=label)

        plt.xlabel('SNR')
        plt.ylabel('Absolute Mean Error')
        plt.title(param)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

        for (values, label) in all_values_std:
            plt.plot(SNRs, values, label=label)

        plt.xlabel('SNR')
        plt.ylabel('Standard Deviation Error')
        plt.title(param)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()


method_scores = []
for method in sorted(os.listdir(rootdir), reverse=True):
    print(method)
    scores = np.load(rootdir+method)
    method_scores.append((scores, method.replace('.npy', '')))

make_plot(method_scores)
