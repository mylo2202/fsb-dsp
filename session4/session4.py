# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from scipy.datasets import electrocardiogram
ecg = electrocardiogram()
ecg

# %%
type(ecg)

# %%
ecg.shape, ecg.mean(), ecg.std()

# %%
ecg.size

# %%
import numpy as np
import matplotlib.pyplot as plt

fs = 360
time = np.arange(ecg.size) / fs
plt.plot(time, ecg)
plt.xlabel("time in s")
plt.ylabel("ECG in mV")
plt.xlim(9, 10.2)
plt.ylim(-1, 1.5)
plt.show()

# %%
plt.plot(time, ecg)
plt.xlabel("time in s")
plt.ylabel("ECG in mV")
plt.xlim(46.5, 50)
plt.ylim(-2, 1.5)
plt.show()

# %%
plt.plot(time, ecg)
plt.xlabel("time in s")
plt.ylabel("ECG in mV")
plt.xlim(207, 215)
plt.ylim(-2, 3.5)
plt.show()

# %%
import sounddevice
from scipy.io.wavfile import  write
fs= 44100
second = int(input("enter time duration in second:"))
print("recording...")
record_voice= sounddevice.rec(int(second * fs), samplerate=fs, channels=1)
sounddevice.wait()
write("output.wav", fs, record_voice)
print("recording complete")

# %%
sounddevice.query_devices()

# %%
import sounddevice as sd 
import soundfile as sf 

filename = 'output.wav'

data,fs = sf.read(filename, dtype='float32')
sd.play(data, fs)
status = sd.wait()

# %%
from IPython.display import Audio
Audio(data, rate=fs, autoplay=True)

# %%
type(data)

# %%
data.shape

# %%
fs = 44100
time = np.arange(data.size) / fs
plt.plot(time, data)
plt.xlabel("time in s")
plt.ylabel("amplitude")
plt.xlim(0.5, 0.6)
plt.ylim(-1.5, 1.5)
plt.show()

# %%
import numpy as np
import cmath
import math
import matplotlib.pyplot as plt

# %%
cmath.sqrt(-1)

# %%
z = 2 + 3j
print(z)

# %%
print(np.real(z))
print(np.imag(z))
print(np.abs(z))
print(np.angle(z))

# %%
# plot the complex number
plt.plot(np.real(z),np.imag(z), 'ks')

# make plot look nicer
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.plot([-5,5], [0,0],'k')
plt.plot([0,0], [-5,5],'k')
plt.xlabel('real axis')
plt.ylabel('imag axis')
plt.show()

# %%
mag = np.abs(z)
ang = np.angle(z)

plt.polar([0, ang], [0, mag], 'r')
plt.show()

# %%
p = [1, 0, 0, 1]
np.roots(p)
