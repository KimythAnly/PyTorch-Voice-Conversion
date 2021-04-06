from abc import ABC, abstractmethod

class BaseVocoder(ABC):
    def __init__(self, device):
        self.vocoder = None
        self.device = device
        self.n_mels = 80
        self.sr = 22050

    @abstractmethod
    def mel2wav(self, mel, save=''):
        pass

    @abstractmethod
    def wav2mel(self, wav, save=''):
        pass

    @abstractmethod
    def mel2plt(self, mel, save=''):
        pass