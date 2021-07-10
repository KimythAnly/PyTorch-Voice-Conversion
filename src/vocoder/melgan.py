import torch
import numpy as np
import soundfile as sf
import librosa
import librosa.effects
from librosa.core import load

from .base_vocoder import BaseVocoder


class Vocoder(BaseVocoder):

    def __init__(self, device):
        super().__init__(device)
        self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
        # Config for melspectrogram 
        self.n_fft = 1024
        self.fmin = 0
        self.fmax = 11025
        self.n_mels = 80
        self.sample_rate = 22050
        self.hop_length = 256
        self.win_length = 1024
        # Build mel_basis
        self.mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

    def wav2mel(self, wav, trim=None, save=''):
        y, sr = load(wav, sr=self.sample_rate)
        if trim:
            y, _ = librosa.effects.trim(y, top_db=trim)
        y = np.clip(y, -1.0, 1.0)
        D = np.abs(librosa.stft(y, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.win_length)**2)
        D = np.sqrt(D)
        S = np.dot(self.mel_basis, D)
        log_S = np.log10(S)
        return log_S

    def _wav2mel(self, wav, save=''):
        device = self.device
        with torch.no_grad():
            if isinstance(wav, str):
                wav, sr = load(wav, sr=self.sample_rate)
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav).float()
            mel = self.vocoder(wav.to(device).view(1, -1))
        if save:
            np.save(save, wav.cpu().numpy())
        return mel

    def mel2wav(self, mel, save=''):
        device = self.device
        with torch.no_grad():
            if isinstance(mel, torch.Tensor):
                mel = mel.squeeze()
                mel = mel[None].to(device).float()
            else:
                mel = torch.from_numpy(mel[None]).to(device).float()
            y = self.vocoder.inverse(mel).cpu().numpy().flatten()
        if save:
            sf.write(file=save, data=y, samplerate=self.sr)
        return y
    
    def mel2plt(self, mel):
        if isinstance(mel, torch.Tensor):
            mel = mel.cpu().detach().numpy()
        plt = mel/5 + 1
        return plt
