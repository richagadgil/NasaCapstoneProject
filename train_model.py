import pandas as pd
import numpy as np
from scipy.io import loadmat
import sys
sys.path.insert(0, '../NasaCapstoneProject')
from feature_set_gen import get_feature_set
from model import train

import tensorflow as tf

if not tf.test.is_built_with_cuda(): # check cuda enabled
    exit('CUDA not enabled!!')

if len(tf.config.list_physical_devices()) < 2:
    exit('Hardware acceleration not available!!') # check gpu is available

args = sys.argv
print(args)
if len(args) < 5:
    exit('format must be train_model.py [spectra data] [model weights path] [number of channels] [length of model input] -s [start_spectra-end_spectra]')
spectra = loadmat(args[1])
if args[5] == '-s':
    start = int(args[6].split('-')[0])
    end = int(args[6].split('-')[1])
    spec = spectra['spectra']['fft'][0][0][:,:,start:end]
    modes = spectra['spectra']['freqLiq'][0][0][:,start:end]
else:
    spec = spectra['spectra']['fft'][0][0]
    modes = spectra['spectra']['freqLiq'][0][0]

spec = np.abs(spec)
X, y = get_feature_set(spec, modes, width = int(args[3]), length = int(args[4]),verbose=True)

print('feature set generated.')
print('training model...')
model = train(X, y,int(args[4]),int(args[3]), lr=3e-4, epochs=250, filepath=args[2])