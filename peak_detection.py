import numpy as np
from numpy.testing import assert_allclose
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import ModelCheckpoint
!pip install plotly --upgrade  # make sure your plotly is up to date

import plotly.express as px
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt


#!pip install plotly --upgrade  # make sure your plotly is up to date

import plotly.express as px
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

spectra = loadmat('/content/drive/MyDrive/Spectra_3s_25c.mat')
#more_spectra = loadmat('/content/drive/MyDrive/5_more_spectra.mat')
spectra.keys()

new_model = load_model('/content/drive/MyDrive/model_complete.h5')

all_channels = pd.DataFrame(np.transpose(np.transpose(spectra['Dm_total_save'])[0]))
modes = [x[0]*np.shape(spectra['Dm_total_save'])[0] for x in spectra['omega_total_save']]

labeled_modes=[]
for i in range(3):
  print("ITERATION: "+str(i))
  amp=[]
  length = np.shape(spectra['Dm_total_save'])[0]
  modes = [x[i]*length for x in spectra['omega_total_save']]
  seeker = 0
  modes = np.sort(modes)
  for j in range(length):
    all_amp = [0]
    while seeker < len(modes) and modes[seeker] <= j+100:
        # print(np.exp(np.square((j-modes[seeker])/100)))
      all_amp.append(1/(np.exp(np.square((j-modes[seeker])/10))))
      seeker += 1
    while seeker < len(modes) and seeker >= 0 and modes[seeker] >= j-100:
      # print(modes[seeker])
      seeker -= 1
    seeker = max(seeker,0)
    amp.append(max(all_amp))
    if j %100000 ==0: print(j)
  for i in range(5):
    labeled_modes.extend(amp)
  print(len(labeled_modes))


# training set generation
x = []
for i in range(3):
  all_channels = pd.DataFrame(np.transpose(np.transpose(spectra['Dm_total_save'])[i]))
  x.extend(np.array(all_channels[[0,1,2,3,4]]))
  x.extend(np.array(all_channels[[5,6,7,8,9]]))
  x.extend(np.array(all_channels[[10,11,12,13,14]]))
  x.extend(np.array(all_channels[[15,16,17,18,19]]))
  x.extend(np.array(all_channels[[20,21,22,23,24]]))

 !pip freeze | grep plotly

pd.options.plotting.backend = "plotly" 

def show_spectra(view_range,spect,modes, spectra_sample=True):
  # take subset of plot
  t_modes = np.sort(modes)
  temp_modes = pd.Series(t_modes)[(pd.Series(t_modes) < view_range[1])]
  temp_modes = pd.Series(temp_modes)[(pd.Series(temp_modes) > view_range[0])]
  fig = []
  if spectra_sample:
    fig = pd.Series(spect).iloc[range(view_range[0],view_range[1])].plot(kind='line')
  else:
    fig = pd.Series(spect).plot(template='plotly_dark', kind='line')
    temp_modes -= view_range[0]
  for mode in temp_modes:
    fig.add_vline(x=mode, line_width=1, line_color="green") #requires plotly 4.12 and above
  fig.show()

 modes = [x[0]*length for x in spectra['omega_total_save']]
show_spectra((int(500000), int(501000)), labeled_modes, modes)


def stop():
  spectra =  loadmat('Spectra.mat')#['Dm_total_save']

  labeled_modes=[]
  for i in range(1):
    print("ITERATION: "+str(i))
    amp=[]
    length = np.shape(spectra['Dm_total_save'])[0]
    modes = [x[i]*length for x in spectra['omega_total_save']]
    seeker = 0
    modes = np.sort(modes)
    for j in range(length):
      all_amp = [0]
      while seeker < len(modes) and modes[seeker] <= j+100:
          # print(np.exp(np.square((j-modes[seeker])/100)))
        all_amp.append(1/(np.exp(np.square((j-modes[seeker])/10))))
        seeker += 1
      while seeker < len(modes) and seeker >= 0 and modes[seeker] >= j-100:
        # print(modes[seeker])
        seeker -= 1
      seeker = max(seeker,0)
      amp.append(max(all_amp))
      if j %100000 ==0: print(j)
    labeled_modes.extend(amp)
    print(len(labeled_modes))

#y = labeled_modes
#x = spectra['Dm_total_save']
#window_len = 1000
#reshaped_y = np.reshape(y, (int(len(y)/window_len), window_len))
#reshaped_x = np.reshape(np.array(x), (int(len(x)/window_len), window_len, 5))

y = labeled_modes
window_len = 1000
reshaped_y = np.reshape(y, (int(len(y)/window_len), window_len))
reshaped_x = np.reshape(np.array(x), (int(len(x)/window_len), window_len, 5))

first_spectra = reshaped_x[0:1000]

y_pred = new_model.predict(first_spectra)

np.reshape(y_pred, (1, 1000000))

#modes = [x[0]*length for x in spectra['omega_total_save']]
show_spectra((int(500000), int(501000)), np.reshape(y_pred, (1, 1000000))[0], modes)

show_spectra((int(500000), int(501000)), y, modes)

import seaborn as sns
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

reshaped_y_pred = np.reshape(y_pred, (1, 1000000))[0]
#pd.Series(np.reshape(reshaped_y, (1, 1000000))[0][0:10000]).plot()
#pd.Series(savgol_filter(y_pred, 51, 3)[0:10000]).plot()
predicted_modes = find_peaks(reshaped_y_pred, height=0.65)[0]
#print(len(modes))
#sns.lineplot(data = pd.Series(np.reshape(reshaped_y, (1, 1000000))[0][0:10000]))

modes = [x[0]*length for x in spectra['omega_total_save']]

peaks_detected_count = []
peaks_detected = []
for s in range(10):
  window_peaks = [i for i in predicted_modes if (i >= s * 100000) and (i < (s+1) * 100000)]#find_peaks(smooth_density[s*100000 : (s+1)*100000], prominence=density_prominence)
  peaks_detected_count.append(len(window_peaks))
  window_peaks = [i for i in modes if (i >= s * 100000) and (i < (s+1) * 100000)]
  peaks_detected.append(len(window_peaks))
  #peaks_detected.extend(window_peaks+s*100000)
print("number of peaks detected:", str(np.sum(peaks_detected_count)))

modes = np.array(modes)
modes.sort()