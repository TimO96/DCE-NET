import matplotlib.pyplot as plt
import numpy as np
import pickle
import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')

mask_to_slices = pickle.load(open('mask_to_slice_MIPA.p', "rb"))
slices = mask_to_slices['CR05_1']
path_mask = '/home/tottens/scratch/DCE_MRI/data/CR05_1/kEp-label.nii.gz'
mask = np.moveaxis(nib.load(path_mask).get_fdata(), 2, 0)

params = pickle.load(open('params/patient_data_convlin_[160, 160, 160]_0.001_8_True.pt', 'rb'))
params_linear = pickle.load(open('params/patient_data_gru_[128, 4]_0.001_256.p', 'rb'))
params_lsq = pickle.load(open('params/param_maps_patients_lsq.p', 'rb'))
C1s, hcts = pickle.load(open('all_patient_data_MIPA_ip.p', "rb"))

bad_points = np.mean(C1s, axis=3) < 1e-3
params[0][bad_points] = 0
params[2][bad_points] = 0
params[3][bad_points] = 0

params_lsq[:, 0][bad_points] = 0
params_lsq[:, 2][bad_points] = 0
params_lsq[:, 3][bad_points] = 0

for i, slice in enumerate(slices):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    ax1 = axs[0, 0].imshow(np.rot90(params[0][slice]*params[2][slice], 1, (0, 1)), cmap='jet', vmin=0,  vmax=3)
    axs[0, 0].set_ylabel('$K_{trans}$', fontsize=12, rotation=0, labelpad=30)
    axs[0, 0].set_title('ConvLin Dual Path', fontsize=12, pad=20)
    axs[0, 0].get_xaxis().set_visible(False)
    axs[0, 0].get_yaxis().set_ticks([])
    ax2 = axs[1, 0].imshow(np.rot90(params[2][slice], 1, (0, 1)), cmap='jet', vmin=0, vmax=1)
    axs[1, 0].set_ylabel('$v_e$', fontsize=12, rotation=0, labelpad=30)
    axs[1, 0].get_xaxis().set_visible(False)
    axs[1, 0].get_yaxis().set_ticks([])
    ax3 = axs[2, 0].imshow(np.rot90(params[3][slice], 1, (0, 1)), cmap='jet', vmin=0, vmax=0.1)
    axs[2, 0].set_ylabel('$v_p$', fontsize=12, rotation=0, labelpad=30)
    axs[2, 0].get_xaxis().set_visible(False)
    axs[2, 0].get_yaxis().set_ticks([])
    fig.colorbar(ax1, ax=axs[0, 0])
    fig.colorbar(ax2, ax=axs[1, 0])
    fig.colorbar(ax3, ax=axs[2, 0])
    ax4 = axs[0, 1].imshow(np.rot90(params_lsq[slice][0]*params_lsq[slice][2], 1, (0, 1)), cmap='jet', vmin=0,  vmax=3)
    axs[0, 1].set_title('NLLS', fontsize=12, pad=20)
    axs[0, 1].get_xaxis().set_visible(False)
    axs[0, 1].get_yaxis().set_visible(False)
    ax5 = axs[1, 1].imshow(np.rot90(params_lsq[slice][2], 1, (0, 1)), cmap='jet', vmin=0, vmax=1)
    axs[1, 1].get_xaxis().set_visible(False)
    axs[1, 1].get_yaxis().set_visible(False)
    ax6 = axs[2, 1].imshow(np.rot90(params_lsq[slice][3], 1, (0, 1)), cmap='jet', vmin=0, vmax=0.1)
    axs[2, 1].get_xaxis().set_visible(False)
    axs[2, 1].get_yaxis().set_visible(False)
    fig.colorbar(ax4, ax=axs[0, 1])
    fig.colorbar(ax5, ax=axs[1, 1])
    fig.colorbar(ax6, ax=axs[2, 1])
    ax7 = axs[0, 2].imshow(np.rot90(params_linear[0].reshape(-1, 160, 160)[slice]*params_linear[2].reshape(-1, 160, 160)[slice], 1, (0, 1)), cmap='jet', vmin=0,  vmax=3)
    axs[0, 2].get_xaxis().set_visible(False)
    axs[0, 2].get_yaxis().set_visible(False)
    axs[0, 2].set_title('GRU', fontsize=12, pad=20)
    ax8 = axs[1, 2].imshow(np.rot90(params_linear[2].reshape(-1, 160, 160)[slice], 1, (0, 1)), cmap='jet', vmin=0, vmax=1)
    axs[1, 2].get_xaxis().set_visible(False)
    axs[1, 2].get_yaxis().set_visible(False)
    ax9 = axs[2, 2].imshow(np.rot90(params_linear[3].reshape(-1, 160, 160)[slice], 1, (0, 1)), cmap='jet', vmin=0, vmax=0.1)
    axs[2, 2].get_xaxis().set_visible(False)
    axs[2, 2].get_yaxis().set_visible(False)
    fig.colorbar(ax7, ax=axs[0, 2])
    fig.colorbar(ax8, ax=axs[1, 2])
    fig.colorbar(ax9, ax=axs[2, 2])
    plt.show()
