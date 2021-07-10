import numpy as np
import cv2
from einops import reduce
from scipy.signal import resample as _resample
from matplotlib import cm


def normalize_f0(x):
    """
    Source:
        https://github.com/auspicious3000/SpeechSplit/blob/master/utils.py
    """
    x = np.log(1 + x)
    voiced = x != 0
    mean = x[voiced].mean()
    std = x[voiced].std()
    x[voiced] = (x[voiced]-mean) / std / 4.0
    x[voiced] = np.clip(x[voiced], -1, 1)
    x[voiced] = (x[voiced]+1) / 2.0
    return x


def resample(x, to_size, mode='interpolate', axis=0):
    if mode == 'interpolate':
        x = np.swapaxes(x, 0, axis)
        new_len = np.arange(0, len(x), len(x)/to_size)
        old_len = np.arange(0, len(x))
        x = np.interp(new_len, old_len, x)[:to_size]
        return np.swapaxes(x, 0, axis)
    elif mode == 'cv':
        return cv2.resize(x, to_size, interpolation=cv2.INTER_AREA)
    else:
        return _resample(x, to_size)


def viridis(x, flip=False):
    fn = cm.get_cmap('viridis')
    if flip:
        x = x[::-1].copy()
    return fn(x)


def trim(x, eps=1e-4):
    s = einops.reduce(x, 'c t -> t', 'sum')
    s[s < eps] = 0
    f = len(np.trim_zeros(s, trim='f'))
    b = len(np.trim_zeros(s, trim='b'))
    end = b - x.shape[axis]
    if end == 0:
        return x[len(x) - f:]
    else:
        return x[len(x) - f: end]


def pad(x, seg_len, axis=1, mode='reflect'):
    padlen = seg_len - x.shape[axis]
    if padlen <= 0:
        return x
    npad = [(0, 0)] * x.ndim
    npad[axis] = (0, padlen)
    y = np.pad(x, pad_width=npad, mode=mode)
    return y


def segment(x, seg_len=128, r=None, return_r=False, axis=1):
    if x.shape[axis] < seg_len:
        y = pad(x, seg_len, axis=axis)
    elif x.shape[axis] == seg_len:
        y = x
    else:
        if r is None:
            r = np.random.randint(x.shape[axis] - seg_len)
        y = np.swapaxes(x, 0, axis)[r:r+seg_len, ...]
        y = np.swapaxes(y, 0, axis)
    if return_r:
        return y, r
    else:
        return y


def resize(x, dim):
    return cv2.resize(x, dim, interpolation=cv2.INTER_AREA)


def random_scale(mel, allow_flip=False, r=None, return_r=False, axis=1):
    mel = mel.swapaxes(0, axis)
    if r is None:
        r = np.random.random(3)

    rate = r[0]*0.4 + 0.3  # 0.3-0.7
    trans_from = int(mel.shape[0] * rate)

    rate = r[1]*0.6 + 0.7  # 0.7-1.3
    left_len = int(trans_from * rate)
    right_len = mel.shape[0] - left_len
    mel[:left_len] = resample(
        mel[:trans_from], (mel.shape[1], left_len), mode='cv')
    mel[left_len:] = resample(
        mel[trans_from:], (mel.shape[1], right_len), mode='cv')

    if r[2] > 0.5 and allow_flip:
        ret = mel[::-1].copy()
    else:
        ret = mel

    if return_r:
        return ret, r
    else:
        return ret

# def random_scale(mel, allow_flip=False, r=None, return_r=False, axis=2):
#     if r is None:
#         r = np.random.random(3)
#     rate = r[0] * 0.6 + 0.7 # 0.7-1.3
#     dim = (int(mel.shape[axis] * rate), mel.shape[0])
#     r_mel = resize(mel, dim)

#     rate = r[1] * 0.4 + 0.3 # 0.3-0.7
#     trans_point = int(dim[0] * rate)
#     dim = (mel.shape[1]-trans_point, mel.shape[0])
#     if r_mel.shape[1] < mel.shape[1]:
#         r_mel = pad(r_mel, mel.shape[1])
#     # r_mel[:,trans_point:mel.shape[1]] = cv2.resize(r_mel[:,trans_point:], dim, interpolation=cv2.INTER_AREA)
#     r_mel[:,trans_point:mel.shape[1]] = resize(r_mel[:,trans_point:], dim)
#     if r[2] > 0.5 and allow_flip:
#         ret = r_mel[:,:mel.shape[1]][:,::-1].copy()
#     else:
#         ret = r_mel[:,:mel.shape[1]]
#     if return_r:
#         return ret, r
#     else:
#         return ret
