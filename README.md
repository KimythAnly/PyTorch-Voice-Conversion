# PyTorch-Voice-Conversion
Deep models for Voice Conversion.

## Preprocess

- Create a csv file for preprocessing. See `data/example_csv/vctk.csv`
```bash
python preprocess.py gen_csv_vctk -i ~/VCTK-Corpus/ -o data/example_csv/vctk.csv
```

- Preprocess using a csv file. See `data/example_csv/vctk.csv`
```bash
python preprocess.py mel_trim -i data/example_csv/vctk.csv -o data/feature/vctk
```

- The output directory structure
```
data/feature/vctk
└── mel
    ├── p225_p225_001.npy
    ├── p225_p225_002.npy
    ├── p225_p225_003.npy
    ...
```

## Train
```bash
python train.py vqvc_plus --config config/train_vqvc_plus.yaml -t data/feature/vctk -d en_mel --save-steps 10000
```
