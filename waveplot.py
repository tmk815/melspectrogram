import librosa.display
import numpy as np
import matplotlib.pyplot as plt

filename = 'Haunting_song_of_humpback_whales.wav'
y, sr = librosa.load("sound.mp3")
# trim silent edges
whale_song, _ = librosa.effects.trim(y)
librosa.display.waveplot(whale_song, sr=sr)

plt.show()

n_fft = 2048
D = np.abs(librosa.stft(whale_song[:n_fft], n_fft=n_fft, hop_length=n_fft + 1))
plt.plot(D)
plt.show()

hop_length = 512

n_mels = 128
mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

S = librosa.feature.melspectrogram(whale_song, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')

# 逆だと真っ白になる
plt.savefig("figure.png")
plt.show()
