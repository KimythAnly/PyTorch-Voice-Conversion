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


def get_dataset(root_dir, seg_len, n_speaker=None, n_data_per_speaker=None, max_data=None):
    split, speaker2id = split_data(
        Data, root_dir, seg_len, n_speaker, n_data_per_speaker, max_data, default_feature='mel_trim')
    dataset = {
        'train': Dataset(split['train'], seg_len=seg_len, speaker2id=speaker2id),
        'val': Dataset(split['val'], seg_len=seg_len, speaker2id=speaker2id),
    }
    return dataset


class Data():
    def __init__(self, train_dir, basename):
        self.speaker = basename.split('_')[0]
        self.mel_path = os.path.join(train_dir, 'mel_trim', basename+'.npy')
        # Attributes
        self.mel = None
        
        emb_path = os.path.join(train_dir, 'resemblyzer', basename+'.npy')
        self.emb = np.load(emb_path)

    def valid(self):
        return os.path.isfile(self.mel_path)

    def get(self):
        if not self.mel:
            self.mel = np.load(self.mel_path).squeeze()
        return self.mel, self.emb


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
        mel, emb = feat.get()

        mel = segment(mel, seg_len=self.seg_len, axis=1)

        meta = {
            'sid': self.speaker2id[speaker],
            'mel': mel,
            'emb': emb,
        }
        return meta
