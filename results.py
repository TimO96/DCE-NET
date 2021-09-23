import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')

rootdir = '/scratch/tottens/DCE_MRI/github_code/params/synthetic_data_plots/'
SNRs = [7, 10, 20, 30, 40, 60, 80, 100]
sns.set()


def make_plot(method_scores):
    params = ['$K_{ep}$', '$v_e$', '$v_p$', 'dt']

    for i, param in enumerate(params):
        all_values_mean = []
        all_values_std = []
        for k, (scores, label) in enumerate(method_scores):
            if k%5 == 0:
                values_mean = [[] for _ in range(len(SNRs))]
                values_std = [[] for _ in range(len(SNRs))]
            
            if label == 'LSQ':
                values_mean = []
                values_std = []

            for j in range(len(SNRs)):
                if label!='LSQ':
                    values_mean[j].append(scores[j][i][1])
                    values_std[j].append(scores[j][i][0])
                else:
                    values_mean.append(scores[j][i][1])
                    values_std.append(scores[j][i][0])
            
            if k%5==4:
                values_mean_mean = [np.median(np.abs(scores)) for scores in values_mean]
                values_mean_std = [np.std(np.abs(scores)) for scores in values_mean]
                values_std_mean = [np.median(scores) for scores in values_std]
                values_std_std = [np.std(scores) for scores in values_std]

                all_values_mean.append((values_mean_mean, values_mean_std, label))
                all_values_std.append((values_std_mean, values_std_std, label))

            if label=='LSQ':
                all_values_mean.append((np.abs(values_mean), label))
                all_values_std.append((values_std, label))

                print(all_values_mean)
        
        for values in all_values_mean:
            if values[-1] == 'LSQ':
                plt.plot(SNRs, values[0], label=values[1])
            else:
                #plt.errorbar(SNRs, values[0], yerr=values[1], label=values[2])
                plt.plot(SNRs, values[0], yerr=values[1], label=values[2])

        plt.xlabel('SNR')
        plt.ylabel('Absolute Mean Error')
        plt.title(param)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

        for values in all_values_std:
            if values[-1] == 'LSQ':
                plt.plot(SNRs, values[0], label=values[1])
            else:
                plt.errorbar(SNRs, values[0], yerr=values[1], label=values[2])

        plt.xlabel('SNR')
        plt.ylabel('Standard Deviation Error')
        plt.title(param)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()


method_scores = []
for method in sorted(os.listdir(rootdir)):
    print(method)
    scores = np.load(rootdir+method)
    if 'gru' in method:
        method = 'GRU'
    elif 'lstm' in method:
        method = 'LSTM'
    elif 'linear' in method:
        method = 'FCN'
    else:
        method = 'LSQ'
    
    #method_scores.append((scores, method.replace('.npy', '')))
    method_scores.append((scores, method))

make_plot(method_scores)
