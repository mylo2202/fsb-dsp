# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv (3.14.4)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Digital Signal Processing - Session 6

# %% [markdown]
# ## Implement the DFT and FFT in Python

# %%
import numpy as np

def dft(x):

    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# Example signal
x = np.array([2, 1])

# Compute DFT
X_manual = dft(x)

print(X_manual)

# %%
import numpy as np

# Example signal
x = np.array([2, 1])

# Compute FFT
X_fft = np.fft.fft(x)
print(X_fft)

# %%
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np
from math import pi

plt.close('all')

plt.rcParams['figure.figsize']=[16,16]
plt.rcParams.update({'font.size':18})

fs = 1200 # Sampling Frequency
t = np.arange(0,0.5,1/fs) # Time axis in sec.
f1 = 120 # frequency of the sine wave is 20Hz or discrete frequency = 2pi/6
f2 = 50 # frequency of the sine wave is 20Hz or discrete frequency = 2pi/3
x = 0.3*np.sin(2*pi*f1*t) + 0.2*np.sin(2*pi*f2*t)
x = x + 0.5*np.random.randn(len(t))

plt.subplot(2,1,1)
plt.plot(t,x)
plt.title('Sinusoidal Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# %%
# Generate freqeuncy axis
n = np.size(t)
# We just need half of the samples in frequency domain since the signal is real-valued
fr = (fs/2)*np.linspace(0,1,int(n/2))
# Compute FFT of x
X = fft(x)
# Magnitude of the FFT
X_mag = 2/n * np.abs(X[0:np.size(fr)])

plt.subplot(2,1,2)
plt.plot(fr, X_mag)
plt.title('Magnitude of the Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()
