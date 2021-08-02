import pydcemri.Matts_DCE as classic
import pydcemri.dcemri as dce
from matplotlib import pyplot as plt
import hyperparams
import numpy as np
hp=hyperparams.Hyperparams()

T1 = 1
R1 = 1/T1
Kep=1
Ve=0.1
Vp=0.01
rep1 = hp.acquisition.rep1 - 1
rep2 = hp.acquisition.rep2 - 1
hp.acquisition.timing = np.array(range(0, rep2 + rep1)) * hp.acquisition.time / 60
hp.acquisition.FAlist = np.append(np.tile(hp.acquisition.FA1, rep1), np.tile(hp.acquisition.FA2, rep2))

fig, axs = plt.subplots(2,2)
for Kep in [0.01, 0.1, 1, 2]:
    rep1 = hp.acquisition.rep1 - 1
    C = classic.Cosine4AIF_ExtKety(hp.acquisition.timing, hp.aif.aif, Kep, (60 + rep1 * hp.acquisition.time) / 60,
                                                  Ve, Vp)
    R1eff = dce.con_to_R1eff(C, R1, hp.acquisition.r1)
    X_dw = dce.r1eff_to_dce(R1eff, hp.acquisition.TR, hp.acquisition.FAlist)
    axs[0,0].plot(hp.acquisition.timing,X_dw)
Kep=0.5
for Ve in [0.001, 0.01, 0.1, 0.3]:
    rep1 = hp.acquisition.rep1 - 1
    C = classic.Cosine4AIF_ExtKety(hp.acquisition.timing, hp.aif.aif, Kep, (60 + rep1 * hp.acquisition.time) / 60,
                                                  Ve, Vp)
    R1eff = dce.con_to_R1eff(C, R1, hp.acquisition.r1)
    X_dw = dce.r1eff_to_dce(R1eff, hp.acquisition.TR, hp.acquisition.FAlist)
    axs[1,0].plot(hp.acquisition.timing,X_dw)
Ve=0.1
for Vp in [0.001, 0.05, 0.1, 0.3]:
    rep1 = hp.acquisition.rep1 - 1
    C = classic.Cosine4AIF_ExtKety(hp.acquisition.timing, hp.aif.aif, Kep, (60 + rep1 * hp.acquisition.time) / 60,
                                                  Ve, Vp)
    R1eff = dce.con_to_R1eff(C, R1, hp.acquisition.r1)
    X_dw = dce.r1eff_to_dce(R1eff, hp.acquisition.TR, hp.acquisition.FAlist)
    axs[1,1].plot(hp.acquisition.timing,X_dw)
Vp = 0.01
for T1 in [1, 2, 3, 4]:
    R1 = 1/T1
    rep1 = hp.acquisition.rep1 - 1
    C = classic.Cosine4AIF_ExtKety(hp.acquisition.timing, hp.aif.aif, Kep, (60 + rep1 * hp.acquisition.time) / 60,
                                                  Ve, Vp)
    R1eff = dce.con_to_R1eff(C, R1, hp.acquisition.r1)
    X_dw = dce.r1eff_to_dce(R1eff, hp.acquisition.TR, hp.acquisition.FAlist)
    axs[0,1].plot(hp.acquisition.timing,X_dw)

plt.show()