import os
import re
from glob import glob
import logging
import json

import numpy as np
from torch.utils.data import Dataset as BaseDataset

from ..util.transform import segment, resample, random_scale

logger = logging.getLogger(__name__)

def get_voiced_segment(f0):
    ret = f0.copy().astype(int)
    ret[ret!=0] = 1
    return ret

def get_seg_matrix(f0):
    voiced_segment = get_voiced_segment(f0)
    z = voiced_segment.copy()
    c = 0
    m = np.zeros((f0.shape[1], f0.shape[1]))
    for i in range(1, m.shape[1]-1):
        # condition:
        #   - voiced at time t, t+1
        #   - unvoiced at time t-1
        t = np.bitwise_and(voiced_segment[0, i]==1, voiced_segment[0, i+1]==1)
        c += np.bitwise_and(t, voiced_segment[0, i-1]==0)
        z[0, i] = c
    z[voiced_segment==0] = 0
    for i in range(1, z.shape[1]):
        m[i, :] = z == z[0, i]
    return m.astype(np.float32)


class Data():
    def __init__(self, train_dir, basename):
        self.speaker = basename.split('_')[0]
        self.mel = os.path.join(train_dir, 'mel', basename+'.npy')
        self.gop = os.path.join(train_dir, 'gop', basename+'.json')

    def valid(self):
        return os.path.isfile(self.mel) and os.path.isfile(self.gop)

class Dataset(BaseDataset):
    def __init__(self, train_dir, seg_len, phone_dict, max_data=None):
        super().__init__()
        if isinstance(train_dir, str):
            train_dir = [train_dir]

        self.data = []
        self.seg_len = seg_len
        self.phone_dict = phone_dict
        
        for d in train_dir:
            indexes = glob(os.path.join(d, 'mel', '*.npy'))
            for index in indexes:
                basename = os.path.basename(index).split('.')[0]
                data = Data(d, basename)
                if data.valid():
                    self.data.append(data)
    
        if max_data:
            self.data = [d for i, d in zip(range(max_data), self.data)]
                        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat = self.data[index]

        speaker = feat.speaker
        mel = np.load(feat.mel)
        gop = json.load(open(feat.gop, 'r'))
        for i in range(len(gop)):
            gop[i][0] = self.phone_dict[gop[i][0]]
            gop[i][1] = int(gop[i][1])

        # gop = resample(gop, mel.shape[1], axis=1)
        start = int(gop[0][2][0]*22050) // 256
        end = int(gop[-1][2][1]*22050) // 256


        mel = mel[..., start:end]
        mel, r = segment(mel, return_r=True, seg_len=self.seg_len)

        pos = random_scale(mel)

        meta = {
            'speaker': speaker,
            'mel': mel,
            'pos': pos,
        }
        return meta