# PyTorch-Voice-Conversion
Implementations of Voice Conversion models.

| Method | Paper | Official Code | State |
| -- | -- | -- | -- |
| AdaIN-VC | [Chou et al.](https://arxiv.org/abs/1904.05742) | [GitHub](https://github.com/jjery2243542/adaptive_voice_conversion) |  |
| AGAIN-VC | [Chen et al.](https://arxiv.org/abs/2011.00316) | [GitHub](https://github.com/kimythanly/again-vc) | done |
| AutoVC | [Qian et al.](https://arxiv.org/abs/1905.05879) | [GitHub](https://github.com/auspicious3000/autovc) |  |
| VQVC+ | [Wu et al.](https://arxiv.org/abs/2006.04154) | [GitHub](https://github.com/ericwudayi/SkipVQVC) |  |


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
