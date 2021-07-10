import os
import logging

import numpy as np
from torch.utils.data import Dataset as BaseDataset

from .util import split_data
from ..util.transform import segment, resample, random_scale, normalize_f0

logger = logging.getLogger(__name__)


def get_dataset(
    feature_dir,
    seg_len,
    n_speaker=None,
    n_data_per_speaker=None,
    max_data=None,
    use_unseen=False,
):
    split, speaker2id = split_data(
        Data, feature_dir, n_speaker,
        n_data_per_speaker, max_data,
        default_feature='mel-trim', use_unseen=use_unseen)
    dataset = {
        'train': Dataset(split['train'], seg_len=seg_len, speaker2id=speaker2id),
        'val': Dataset(split['val'], seg_len=seg_len, speaker2id=speaker2id),
    }
    return dataset


class Data():
    def __init__(self, feature_dir, basename):
        self.speaker = basename.split('_')[0]
        self.mel_path = os.path.join(feature_dir, 'mel-trim', basename+'.npy')
        # Attributes
        self.mel = None

    def valid(self):
        return os.path.isfile(self.mel_path)

    def get(self):
        if not self.mel:
            self.mel = np.load(self.mel_path).squeeze()
        return self.mel


class Dataset(BaseDataset):
    def __init__(self, data, seg_len, speaker2id):
        super().__init__()
        self.data = data
        self.seg_len = seg_len
        self.speaker2id = speaker2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat = self.data[index]

        speaker = feat.speaker
        # mel: (80, T)
        mel = feat.get()

        if self.seg_len:
            mel = segment(mel, seg_len=self.seg_len, axis=1)

        meta = {
            'sid': self.speaker2id[speaker],
            'mel': mel,
        }

        return meta
