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

# %% [markdown] id="5ae23ca9"
# ## Digital Signal Processing - Session 4

# %% [markdown] id="S1mVZyvn8-B_"
# ### 1. Electrocardiogram (ECG) Data Loading and Basic Properties
#
# Here, we load the `electrocardiogram` dataset from `scipy.datasets`. The `ecg` variable will hold the signal data.

# %% id="748f1166"
from scipy.datasets import electrocardiogram
ecg = electrocardiogram()
ecg

# %% [markdown] id="281777a4"
# First, we load a sample electrocardiogram (ECG) signal from `scipy.datasets` and inspect its properties. The `electrocardiogram()` function returns a 1D NumPy array representing the ECG signal.

# %% id="8353bb96"
type(ecg)

# %% [markdown] id="9aa6c0e5"
# We check the data type of the loaded `ecg` signal, which is typically a NumPy array.

# %% id="17dd4b87"
ecg.shape, ecg.mean(), ecg.std()

# %% [markdown] id="ff97fd06"
# This cell calculates and displays the `shape` (dimensions), `mean` (average value), and `std` (standard deviation) of the ECG signal. These are fundamental statistical properties.

# %% id="7658b472"
ecg.size

# %% [markdown] id="4854e402"
# We check the total number of samples (elements) in the `ecg` array using `.size`.

# %% id="3afd9270"
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

# %% [markdown] id="5b765e94"
# This code block imports `numpy` for numerical operations and `matplotlib.pyplot` for plotting. It then calculates a time vector based on the sampling frequency (`fs = 360 Hz`) and plots a specific segment of the ECG signal (from 9 to 10.2 seconds).

# %% [markdown] id="e73158b2"
# ### 2. Visualizing the ECG Signal
#
# We use `matplotlib` to plot sections of the ECG signal over time. The `fs` (sampling frequency) is used to convert sample indices to time in seconds.

# %% id="58dbd3c9"
plt.plot(time, ecg)
plt.xlabel("time in s")
plt.ylabel("ECG in mV")
plt.xlim(46.5, 50)
plt.ylim(-2, 1.5)
plt.show()

# %% [markdown] id="739247c2"
# This plot shows another segment of the ECG signal, from 46.5 to 50 seconds, allowing for examination of different parts of the waveform.

# %% id="653bc6ab"
plt.plot(time, ecg)
plt.xlabel("time in s")
plt.ylabel("ECG in mV")
plt.xlim(207, 215)
plt.ylim(-2, 3.5)
plt.show()

# %% [markdown] id="f0228b1c"
# This plot visualizes a segment of the ECG signal from 207 to 215 seconds, with an adjusted y-axis range to better display the amplitude variations in this section.

# %% id="ed63844e"
import sounddevice
from scipy.io.wavfile import write
fs= 44100
second = int(input("enter time duration in second:"))
print("recording...")
record_voice= sounddevice.rec(int(second * fs), samplerate=fs, channels=1)
sounddevice.wait()
write("output.wav", fs, record_voice)
print("recording complete")

# %% [markdown] id="973582e5"
# This cell uses the `sounddevice` library to record audio. It prompts the user for a duration in seconds, records audio at a sampling rate of `44100 Hz`, and saves it as `output.wav` using `scipy.io.wavfile.write`.

# %% [markdown] id="03428146"
# ### 3. Audio Recording and Playback
#
# This section demonstrates how to record audio from the microphone and play it back using the `sounddevice` and `soundfile` libraries.

# %% id="b14835d8"
sounddevice.query_devices()

# %% [markdown] id="28c171e9"
# This command queries and lists the available audio devices on the system, which can be useful for debugging or selecting specific input/output devices.

# %% id="c7796c8c"
import sounddevice as sd
import soundfile as sf

filename = 'output.wav'

data,fs = sf.read(filename, dtype='float32')
sd.play(data, fs)
status = sd.wait()

# %% [markdown] id="5cd2615d"
# This code reads the recorded `output.wav` file using `soundfile` and then plays it back using `sounddevice.play()`. The `status = sd.wait()` ensures the script waits until playback is complete.

# %% id="910af002"
from IPython.display import Audio
Audio(data, rate=fs, autoplay=True)

# %% [markdown] id="d501a695"
# This cell provides an alternative way to play the audio directly within the Jupyter/Colab environment using `IPython.display.Audio`, which embeds an audio player.

# %% id="1f158a0e"
type(data)

# %% [markdown] id="d2b8618d"
# We check the data type of the audio data loaded from the `.wav` file.

# %% id="0096bd17"
data.shape

# %% [markdown] id="defbce96"
# This displays the `shape` of the audio data array, indicating its dimensions (e.g., number of samples).

# %% id="70c5efce"
fs = 44100
time = np.arange(data.size) / fs
plt.plot(time, data)
plt.xlabel("time in s")
plt.ylabel("amplitude")
plt.xlim(0.5, 0.6)
plt.ylim(-1.5, 1.5)
plt.show()

# %% [markdown] id="46ab16cc"
# This cell plots a small segment of the recorded audio signal over time, similar to the ECG plotting. It helps visualize the waveform of the recorded sound.

# %% id="8330733c"
import numpy as np
import cmath
import math
import matplotlib.pyplot as plt

# %% [markdown] id="30026326"
# We import the necessary libraries: `numpy` for numerical operations, `cmath` for complex math functions, `math` for standard math functions, and `matplotlib.pyplot` for plotting.

# %% [markdown] id="f228ebc6"
# ### 4. Complex Numbers in Python
#
# This section introduces complex number operations and their visualization using `numpy`, `cmath`, and `matplotlib`.

# %% id="39bae581"
cmath.sqrt(-1)

# %% [markdown] id="dbdb234e"
# This demonstrates the use of `cmath.sqrt()` to calculate the square root of a negative number, resulting in a complex number.

# %% id="f0e2681c"
z = 2 + 3j
print(z)

# %% [markdown] id="a5e17732"
# A complex number `z` is defined and printed. In Python, `j` is used to denote the imaginary unit.

# %% id="e7491c38"
print(np.real(z))
print(np.imag(z))
print(np.abs(z))
print(np.angle(z))

# %% [markdown] id="f5f61e9f"
# This cell extracts and prints the `real` part, `imaginary` part, `magnitude` (absolute value), and `angle` (phase) of the complex number `z` using `numpy` functions.

# %% id="e2897f0a"
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

# %% [markdown] id="8f26fbae"
# This code plots the complex number `z` on a Cartesian (real-imaginary) plane. The `plt.plot` function plots the point, and `plt.xlim`, `plt.ylim`, `plt.plot([lines])` are used to set up the axes and make the plot visually appealing.

# %% id="76c1f841"
mag = np.abs(z)
ang = np.angle(z)

plt.polar([0, ang], [0, mag], 'r')
plt.show()

# %% [markdown] id="6e4fcc00"
# This cell visualizes the complex number `z` in polar coordinates. `plt.polar()` is used to draw a line from the origin to the point defined by its angle (`ang`) and magnitude (`mag`).

# %% id="a4b693d6"
p = [1, 0, 0, 1]
np.roots(p)

# %% [markdown] id="c5ff62f0"
# This cell finds the roots of the polynomial defined by the coefficients `p = [1, 0, 0, 1]`, which represents the polynomial $x^3 + 1 = 0$. `np.roots()` returns the roots, which can be real or complex.
