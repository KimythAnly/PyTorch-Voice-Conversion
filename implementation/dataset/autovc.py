import os
import re
from glob import glob
import logging
import json

import numpy as np
from torch.utils.data import Dataset as BaseDataset

from .util import split_data
from ..util.transform import segment, resample, random_scale, normalize_f0

logger = logging.getLogger(__name__)


def get_dataset(train_dir, seg_len, n_speaker=None, n_data_per_speaker=None, max_data=None):
    split, speaker2id = split_data(
        Data, train_dir, seg_len, n_speaker, n_data_per_speaker, max_data, default_feature='mel')
    dataset = {
        'train': Dataset(split['train'], seg_len=seg_len, speaker2id=speaker2id),
        'val': Dataset(split['val'], seg_len=seg_len, speaker2id=speaker2id),
    }
    return dataset


class Data():
    def __init__(self, train_dir, basename):
        self.speaker = basename.split('_')[0]
        self.mel_path = os.path.join(train_dir, 'mel', basename+'.npy')
        self.f0_path = os.path.join(train_dir, 'f0', basename+'.npy')
        # Attributes
        self.mel = None
        self.f0 = None
        
        emb_path = os.path.join(train_dir, 'resemblyzer', basename+'.npy')
        self.emb = np.load(emb_path)

    def valid(self):
        f0 = np.load(self.f0_path)
        return os.path.isfile(self.mel_path) and not (f0==0).all()

    def get(self):
        if not self.mel:
            self.mel = np.load(self.mel_path).squeeze()
            f0 = np.load(self.f0_path)
            self.f0 = resample(f0, self.mel.shape[-1])
        return self.mel, self.f0, self.emb


class Dataset(BaseDataset):
    def __init__(self, data, seg_len, speaker2id):
        super().__init__()
        self.data = data
        self.seg_len = seg_len
        self.speaker2id = speaker2id
        # self.set_speaker_emb()

    def __len__(self):
        return len(self.data)
    
    def set_speaker_emb(self):
        embs = [0 for i in self.speaker2id]
        counts = [0 for i in self.speaker2id]

        for d in self.data:
            sid = self.speaker2id[d.speaker]
            embs[sid] += d.emb
            counts[sid] += 1
        
        for i, emb in enumerate(embs):
            if counts[i] != 0:
                embs[i] = emb / counts[i]
        
        for d in self.data:
            sid = self.speaker2id[d.speaker]
            d.emb = embs[sid]
        
    def __getitem__(self, index):
        feat = self.data[index]

        speaker = feat.speaker
        # mel: (80, T)
        # f0 : (T, )
        mel, f0, emb = feat.get()

        f0[f0 < 1e-4] = 0
        f = len(np.trim_zeros(f0, trim='f'))
        b = len(np.trim_zeros(f0, trim='b'))
        start = f0.shape[0] - f
        end = b - f0.shape[0]
        if end == 0:
            mel, f0 = mel[..., start:], f0[start:]
        else:
            mel, f0 = mel[..., start:end], f0[start:end]

        mel, r = segment(mel, return_r=True, seg_len=self.seg_len, axis=1)
        f0 = segment(f0, r=r, seg_len=self.seg_len, axis=0)

        meta = {
            'sid': self.speaker2id[speaker],
            'mel': mel,
            'emb': emb,
        }
        return meta
