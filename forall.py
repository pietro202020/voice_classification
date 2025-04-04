import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def estimate_formants(data, samplerate, min_frequency=300, max_frequency=3500):
    """
    Estimate the first formant (F1) using spectral peak detection.

    :param data: The audio signal (1D array).
    :param samplerate: The sample rate of the audio.
    :param min_frequency: The minimum frequency to search for formants (usually around 300 Hz).
    :param max_frequency: The maximum frequency to search for formants (usually around 3500 Hz).
    :return: The first formant frequency (F1).
    """
    N = len(data)
    spectrum = np.abs(fft(data))[:N // 2]
    freqs = np.linspace(0, samplerate / 2, len(spectrum))
    valid_range = (freqs >= min_frequency) & (freqs <= max_frequency)
    valid_freqs = freqs[valid_range]
    valid_spectrum = spectrum[valid_range]
    peak_index = np.argmax(valid_spectrum)
    return valid_freqs[peak_index]


def analyze_gender(filepath):
    samplerate, data = wavfile.read(filepath)
    if data.ndim > 1:
        data = data[:, 0]
    data = data / np.max(np.abs(data))
    n = len(data)
    window = np.hanning(n)
    windowed_data = data * window
    fft_data = fft(windowed_data)
    fft_magnitude = np.abs(fft_data)
    log_magnitude = np.log(fft_magnitude + 1e-10)  # Avoid log(0)
    cepstrum = np.abs(ifft(log_magnitude))

    # Frequency range for human voice
    min_f0 = 80  # Lower limit for male voice
    max_f0 = 250  # Upper limit for female voice
    min_quefrency = 1 / max_f0
    max_quefrency = 1 / min_f0

    quefrency = np.linspace(0, n / samplerate, n)

    valid_idx = (quefrency >= min_quefrency) & (quefrency <= max_quefrency)
    valid_cepstrum = cepstrum[valid_idx]
    valid_quefrency = quefrency[valid_idx]

    # Find the dominant quefrency
    dominant_quefrency = valid_quefrency[np.argmax(valid_cepstrum)]
    f0 = 1 / dominant_quefrency

    window_size = 15
    peak_indices = np.argsort(valid_cepstrum)[-window_size:]
    f0_candidates = 1 / valid_quefrency[peak_indices]
    average_f0 = np.mean(f0_candidates)

    # Find F1 using formant estimation
    f1 = estimate_formants(data, samplerate)

    # Classify gender based on f0 approximation
    if (80 <= average_f0 < 160) and (f1<800):
        return 'M', average_f0, f0, f1
    elif (185 <= average_f0 <= 250) and (f1>300):
        return 'K', average_f0, f0, f1
    else:
        if f1 < 500:
            return 'M', average_f0, f0, f1
        else:
            return 'K', average_f0, f0, f1

train_folder = "train"
ground_truths = []
predictions = []

for filename in sorted(os.listdir(train_folder)):
    if filename.endswith(".wav"):
        filepath = os.path.join(train_folder, filename)
        prediction, av_f0, ac_f0, f1 = analyze_gender(filepath)
        ground_truth = filename.split('_')[-1][0]
        ground_truths.append(ground_truth)
        predictions.append(prediction)
        if(not (prediction==ground_truth)):
            print(f"{filename} {prediction} {av_f0} {ac_f0} {f1}")

# Generate the confusion matrix
labels = ['M', 'K']  # Classes: Male (M), Female (K)
cm = confusion_matrix(ground_truths, predictions, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', values_format='d')
plt.title("Gender Classification Confusion Matrix")
plt.show()