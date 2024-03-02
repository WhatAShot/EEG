# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import mne
from mne.io import concatenate_raws, read_raw_edf
import pickle

path = "/srv/local/data/EEG100"
# find all edf files
files = []
for r, d, f in os.walk(path):
    for file in f:
        if '.edf' in file:
            files.append(os.path.join(r, file))
# print(len(files))


def mantages(Fp1, F3, C3, P3, O1, Fp2, F4, C4, P4, O2, T3, Cz, T4):
    c1 = Fp1 - F3
    c2 = F3 - C3
    c3 = C3 - P3
    c4 = P3 - O1
    c5 = Fp2 - F4
    c6 = F4 - C4
    c7 = C4 - P4
    c8 = P4 - O2
    c9 = T3 - C3
    c10 = C3 - Cz
    c11 = Cz - C4
    c12 = C4 - T4
    return np.array([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12])

# %%
def clean_eeg(eeg_data):

    input_EEG = []
    sfreq_list = []
    channel_list = []

    for _, d in enumerate(eeg_data):

        raw = read_raw_edf(d, preload=True, verbose=False)
        raw.set_eeg_reference(ref_channels='average', projection=True)
        raw.filter(l_freq=0.5, h_freq=40)

        sfreq_list.append(raw.info['sfreq'])
        channel_list.append(raw.info['ch_names'])
        
        raw_data = raw.get_data()

        input_EEG.append(raw_data)
    
    return input_EEG, sfreq_list, channel_list

X_eeg, freq, chal_names = clean_eeg(files)

# %%
required_channels = ["Fp1", "F3", "C3", "P3", "O1", "Fp2", "F4", "C4", "P4", "O2", "T3", "Cz", "T4"]
exclusion_eeg_id = []

for i in range(len(chal_names)):
    # check if all the elements of required_channels in chal_names[i]
    if all(elem in chal_names[i] for elem in required_channels):
        continue
    else:
        print("not all the required channels are in the chal_names")
        exclusion_eeg_id.append(i)

print(exclusion_eeg_id)
# %%

for i in exclusion_eeg_id:
    X_eeg.pop(i)
    freq.pop(i)
    chal_names.pop(i)

mantage_eeg = []

for i in range(len(X_eeg)):
    out = mantages(X_eeg[i][chal_names[i].index("Fp1")], X_eeg[i][chal_names[i].index("F3")], 
             X_eeg[i][chal_names[i].index("C3")], X_eeg[i][chal_names[i].index("P3")], 
             X_eeg[i][chal_names[i].index("O1")], X_eeg[i][chal_names[i].index("Fp2")], 
             X_eeg[i][chal_names[i].index("F4")], X_eeg[i][chal_names[i].index("C4")], 
             X_eeg[i][chal_names[i].index("P4")], X_eeg[i][chal_names[i].index("O2")], 
             X_eeg[i][chal_names[i].index("T3")], X_eeg[i][chal_names[i].index("Cz")], 
             X_eeg[i][chal_names[i].index("T4")])
    
    mantage_eeg.append(out)

for f in freq:
    if f != 200.0:
        resize = int(X_eeg[i].shape[1] * 200.0 / f)
        mantage_eeg[i] = np.resize(mantage_eeg[i], (12, resize))


# %%
begin = 2000
end = 4000
plt.figure(figsize=(10, 5))
for i in range(12):
    # compute the 5% and 95% of the data value
    data = mantage_eeg[1][i][begin:end]
    p_5 = np.percentile(data, 5)
    p_95 = np.percentile(data, 95)
    # limit the data value to -0.05 to 0.05
    data = np.clip(data, p_5, p_95)
    plt.plot(data + i * 0.0002)
plt.show()
# %%
