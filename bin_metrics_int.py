# -*- coding: utf-8 -*-
"""Bin Metrics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TSET7SJWo94UruMIGOm2TAJYcuSGjLJS
"""

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

"""# Bin Metrics"""

import plotly.graph_objects as go
from scipy import integrate

#args: 
#predicted modes: list, actual modes: list, [default] number of bins: int 
#returns: None  
def bin_plot(predicted_modes, modes, num_bins=20):
  peaks_detected_count = []
  peaks_detected = []

  for s in range(num_bins):
   window_peaks = [i for i in predicted_modes if (i >= s * 1000000/num_bins) and (i < (s+1) * 1000000/num_bins)]
   peaks_detected_count.append(len(window_peaks))
   window_peaks = [i for i in modes if (i >= s * 1000000/num_bins) and (i < (s+1) * 1000000/num_bins)]
   peaks_detected.append(len(window_peaks))

  x_bins=list(range(num_bins))
  fig = go.Figure()
  # actual mode distribution by bin
  fig.add_trace(go.Scatter(x=x_bins, y=peaks_detected,
                    mode='lines+markers',
                    name='Actual values'))
  # predicted mode distribution by bin
  fig.add_trace(go.Scatter(x=x_bins, y=peaks_detected_count,
                    mode='lines+markers',
                    name='Predicted values'))
  fig.update_layout(title='Number of Modes by Bins of Normalized Hertz',
                  xaxis_title='Bin Number',
                   yaxis_title='Number of Modes')
  fig.show()

bin_plot(predicted_modes, modes)
bin_plot(predicted_modes, modes, 15)

#args: 
#predicted modes: list, actual modes: list, [default] number of bins: int 
#returns: None  
def bin_plot_int(predicted_modes, modes, num_bins=20):
  peaks_detected_count = []
  peaks_detected = []

  # creating predicted and actual bins for mode counts
  for s in range(num_bins):
   window_peaks = [i for i in predicted_modes if (i >= s * 1000000/num_bins) and (i < (s+1) * 1000000/num_bins)]
   peaks_detected_count.append(len(window_peaks))
   window_peaks = [i for i in modes if (i >= s * 1000000/num_bins) and (i < (s+1) * 1000000/num_bins)]
   peaks_detected.append(len(window_peaks))

  # creating the actual and predicted integrals 
  x_bins=list(range(num_bins))
  y_int = integrate.cumtrapz(peaks_detected_count, x_bins, initial=0)
  y_int_actual = integrate.cumtrapz(peaks_detected, x_bins, initial=0)

  fig = go.Figure()

  fig.add_trace(go.Scatter(x=x_bins, y=y_int_actual,
                    mode='lines+markers',
                    name='Actual values'))
  
  fig.add_trace(go.Scatter(x=x_bins, y=y_int,
                    mode='lines+markers',
                    name='Predicted values'))
  
  fig.update_layout(title='Number of Modes by Bins of Normalized Hertz',
                  xaxis_title='Bin Number',
                   yaxis_title='Cumulative Number of Modes')
  fig.show()

  

bin_plot_int(predicted_modes, modes, 10)
bin_plot_int(predicted_modes, modes, 30)

#args: 
#predicted modes: list, actual modes: list, [default] number of bins: int 
#returns: dict 
def bin_metrics(predicted_modes, modes, num_bins=20):
  peaks_detected_count = []
  peaks_detected = []
  for s in range(num_bins):
    window_peaks = [i for i in predicted_modes if (i >= s * 1000000/num_bins) and (i < (s+1) * 1000000/num_bins)]
    peaks_detected_count.append(len(window_peaks))
    window_peaks = [i for i in modes if (i >= s * 1000000/num_bins) and (i < (s+1) * 1000000/num_bins)]
    peaks_detected.append(len(window_peaks))

  diff = []
  total_missed = 0 
  pred_direction = 0
  for i in range(num_bins):
    diff.append(abs(peaks_detected_count[i] - peaks_detected[i]))
    total_missed += abs(peaks_detected_count[i] - peaks_detected[i])
    pred_direction += peaks_detected_count[i] - peaks_detected[i]

  avg_missed = sum(diff)/num_bins 
  avg_direction = int(pred_direction)/num_bins

  # put diff metrics in dictionary 
  result_dict = {'average_missed': avg_missed, 'total_missed': total_missed, 'error_direction': avg_direction}

  print("Each bin on average misses %f modes" % avg_missed)
  print("This model missed a total of %d modes" % total_missed)

  if pred_direction > 0:
    print("On average, each bin tends to overpredict the number of modes by %f" % avg_direction)
  else:
    print("On average, each bin tends to underpredict the number of modes by %f" % avg_direction)
  return result_dict

bin_metrics(predicted_modes, modes)
bin_metrics(predicted_modes, modes, 30)

