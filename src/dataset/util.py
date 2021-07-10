import os
from glob import glob


def split_data(
    data_type,
    train_dir,
    n_speaker=None,
    n_data_per_speaker=None,
    max_data=None,
    default_feature='mel',
    n_data_per_speaker_val=100,
    use_unseen=False,
):
    if isinstance(train_dir, str):
        train_dir = [train_dir]

    all_data = []
    speakers = set()

    for d in train_dir:
        indexes = glob(os.path.join(d, default_feature, '*.npy'))
        for index in indexes:
            basename = os.path.basename(index).split('.')[0]
            data = data_type(d, basename)
            if data.valid():
                all_data.append(data)
                speakers.add(data.speaker)

    speakers = sorted(list(speakers))

    speaker2id = {}
    n_data_per_speaker_counter = {}
    n_data_per_speaker_counter_val = {}
    for i, speaker in enumerate(speakers):
        speaker2id[speaker] = i
        n_data_per_speaker_counter[speaker] = 0
        n_data_per_speaker_counter_val[speaker] = 0

    if n_speaker:
        split = {
            'train': [],
            'val': [],
        }
        for data in all_data:
            if data.speaker in speakers[:n_speaker]:
                if n_data_per_speaker:
                    if n_data_per_speaker_counter[data.speaker] < n_data_per_speaker:
                        split['train'].append(data)
                        n_data_per_speaker_counter[data.speaker] += 1
                    else:
                        split['val'].append(data)
                        n_data_per_speaker_counter_val[data.speaker] += 1
                else:
                    split['train'].append(data)
            else:
                if use_unseen:
                    if data.speaker not in n_data_per_speaker_counter_val:
                        n_data_per_speaker_counter_val[data.speaker] = 0
                    if n_data_per_speaker_counter_val[data.speaker] < n_data_per_speaker_val:
                        split['val'].append(data)
                        n_data_per_speaker_counter_val[data.speaker] += 1

    else:
        split = {
            'train': all_data,
            'val': [],
        }
    if max_data:
        split['train'] = [d for i, d in zip(range(max_data), split['train'])]
        split['val'] = [d for i, d in zip(range(max_data), split['val'])]

    return split, speaker2id


def create_inference_data(
    data_type,
    wav_list,
):
    data_list = []
    for wav in wav_list:
        data = data_type(wav)
        data_list.append(data)
    return data_list
