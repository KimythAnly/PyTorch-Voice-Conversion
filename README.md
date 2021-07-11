# PyTorch-Voice-Conversion
Deep models for Voice Conversion.

## Preprocess

- Create a csv file for preprocessing. See `data/example_csv/vctk.csv`
```bash
python gencsv.py vctk -i ~/VCTK-Corpus/ -o data/example_csv/vctk.csv
```

- Preprocess using a csv file. See `data/example_csv/vctk.csv`
```bash
python preprocess.py mel-trim -i data/example_csv/vctk.csv -o data/feature/vctk
```

- The output directory structure
```
data/feature/vctk
└── mel-trim
    ├── p225_p225_001.npy
    ├── p225_p225_002.npy
    ├── p225_p225_003.npy
    ...
```

## Train
Edit the config file and then run the script
```bash
python train.py --config config/AGAIN-VC.yaml
```

## Inference
```bash
python inference.py --config config/AGAIN-VC.yaml -s <source_wav> -t <target_wav> -o <output_wav_or_dir>
```
