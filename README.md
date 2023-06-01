# Local Implicit Normalizing Flow for Arbitrary-Scale Image Super-Resolution [CVPR 2023]

This is the official repository of the following paper:

**Local Implicit Normalizing Flow for Arbitrary-Scale Image Super-Resolution (LINF)**<br>
[Jie-En Yao*](https://scholar.google.com/citations?user=4mk_dZwAAAAJ&hl=zh-TW), [Li-Yuan Tsao*](https://liyuantsao.github.io/), [Yi-Chen Lo](https://scholar.google.com/citations?user=EPYQ48sAAAAJ&hl=zh-TW), [Roy Tseng](https://scholar.google.com/citations?user=uKgYlYYAAAAJ&hl=zh-TW), [Chia-Che Chang](https://scholar.google.com/citations?user=FK1RcpoAAAAJ&hl=zh-TW), [Chun-Yi Lee](https://scholar.google.com/citations?user=5mYNdo0AAAAJ&hl=zh-TW)

[[arxiv](https://arxiv.org/abs/2303.05156)] [[Video](https://www.youtube.com/watch?v=kB2sm_k8P6I)]

If you are interested in our work, **you can access [ElsaLab](http://elsalab.ai/) for more and feel free to contact us.**

<img src="https://i.imgur.com/gDY5gMI.jpg" width="600pt" height="400pt">

---

## Setup & Preparation
### Environment setup
```bash
pip install torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Data preparation
1. SR benchmark datasets (Set5, Set14, BSD100, Urban100)
You can find the download link in the repo  [jbhuang0604/SelfExSR](https://github.com/jbhuang0604/SelfExSR)
2. [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
You should download ```DIV2K/DIV2K_train_HR``` and  ```DIV2K/DIV2K_valid_HR``` with their corresponding downsampled versions for training and validation.
3. [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)

### Checkpoints
<!-- Models for arbitrary-scale SR (patch size 1x1):
| Model | Download |
|:-----:|:--------:|
| EDSR-baseline-LINF | [Google drive](https://drive.google.com/file/d/1TX1TL6Af2Cu679rDBJ7-U-5LH--q3lEc/view?usp=sharing) |
|RDN-LINF|[Google Drive](https://drive.google.com/file/d/1Bfak8fLc71WIoHtMDx4rbZUwjCnGUZSW/view?usp=sharing)|
|SwinIR-LINF|[Google Drive](https://drive.google.com/file/d/1k6E6WwxDIA5TcOJcwG25xzvex85IIzUK/view?usp=sharing)|

Model for generative SR (patch size 3x3):
| Model | Download |
|:-----:|:--------:|
| RRDB-LINF (3x3 patch) | [Google drive](https://drive.google.com/file/d/1o2YBiJ5zkvO_udotBeDo72eNdmlEeDll/view?usp=sharing) | -->

| Model | Download |
|:-----:|:--------:|
| EDSR-baseline-LINF | [Google drive](https://drive.google.com/file/d/1TX1TL6Af2Cu679rDBJ7-U-5LH--q3lEc/view?usp=sharing) |
|RDN-LINF|[Google Drive](https://drive.google.com/file/d/1Bfak8fLc71WIoHtMDx4rbZUwjCnGUZSW/view?usp=sharing)|
|SwinIR-LINF|[Google Drive](https://drive.google.com/file/d/1k6E6WwxDIA5TcOJcwG25xzvex85IIzUK/view?usp=sharing)|
| RRDB-LINF (3x3 patch) | [Google drive](https://drive.google.com/file/d/1o2YBiJ5zkvO_udotBeDo72eNdmlEeDll/view?usp=sharing) |


---

## Training
### Preliminary
1. You should modify the `root_path` in the config files to the path of the datasets. 
2. For stage 1, you can use the config files located in `configs/train-div2k`. For stage 2, the config files are located in `configs/fine-tune`.
3. If you want to resume training, please specify the the path of your checkpoint in the `resume` argument in the config file.
4. The checkpoints will be automatically saved in `./save/<EXP_NAME>`. 

### Launch your experiments
To launch your experiments, you can use the following command:
```bash
python train.py --config <CONFIG_PATH> --gpu <GPU_ID(s)> --name <EXP_NAME> --patch <PATCH_SIZE>
```

### Stage 1

```bash
# EDSR
python train.py --config configs/train-div2k/train_edsr-flow.yaml --gpu 0 --name edsr
# RDN
python train.py --config configs/train-div2k/train_rdn-flow.yaml --gpu 0 --name rdn
# SwinIR
python train.py --config configs/train-div2k/train_swinir-flow.yaml --gpu 0 --name swinir
# RRDB patch DF2K
python train.py --config configs/train-div2k/train_rrdb-flow-DF2K.yaml --gpu 0 --patch 3 --name rrdb
```

### Stage 2 (Fine-tuning)
```bash
# EDSR
python train.py --config configs/fine-tune/fine-tune_edsr-flow.yaml --gpu 0 --name edsr_finetune
# RDN
python train.py --config configs/fine-tune/fine-tune_rdn-flow.yaml --gpu 0 --name rdn_finetune
# SwinIR
python train.py --config configs/fine-tune/fine-tune_swinir-flow.yaml --gpu 0 --name swinir_finetune
# RRDB patch DF2K
python train.py --config configs/fine-tune/fine-tune_rrdb-flow-DF2K.yaml --gpu 0 --patch 3 --name rrdb_finetune
```
---

## Evaluation
### Preliminary
1. You should modify the `root_path` in the config files to the path of the datasets.
2. You can store visualization results using the arguments `--sample <NUM_SAMPLES>` and `--name <VIS_RESULTS_NAME>`. The generated images will be automatically saved in `./sample/<VIS_RESULTS_NAME>`.
3. Other arguments<br>
`--patch`: Whether the model is a patch-based model (patch size > 1).<br>
`--detail`: Print the results of SSIM and LPIPS.<br>
`--randomness`: Run five experiments and report the mean results.<br>
`--temperature`: Set the sampling temperature<br>

### Launch the evaluation
To launch your evaluation, you can use the following command:

1. For benchmark datasets (EDSR-baseline-LINF, RDN-LINF, SwinIR-LINF)
```bash
# benchmark deterministic
sh scripts/test-benchmark-ours-t0.sh <MODEL_PATH> <GPU_ID>
# benchmark random sample
sh scripts/test-benchmark-ours-t.sh <MODEL_PATH> <GPU_ID> <TEMPERATURE>
```
2. For DIV2K dataset (RRDB-LINF 3x3 patch)
```bash
# RRDB patch div2k deterministic
python test.py --config configs/test/test-fast-div2k-4.yaml --model <MODEL_PATH> --gpu <GPU_ID> --detail --temperature 0.0 --patch
# RRDB patch div2k random sample
python test.py --config configs/test/test-fast-div2k-4.yaml --model <MODEL_PATH> --gpu <GPU_ID> --detail --randomness --temperature <TEMPERATURE> --patch
```

### Note
1. For the evaluation of generative SR, we didn't use the same border-shaving method as in the arbitrary-scale SR evaluation since generative SR works didn't evaluate with that. You can modify the line 142 of `utils.py` from `shave = scale + 6` to `shave = scale` in order to get the same scores reported in our paper (with only a negligible difference on PSNR). 

---
## Citation
If you find our work helpful for your research, we would greatly appreciate your assistance in sharing it with the community and citing it using the following BibTex. Thank you for supporting our research.

```
@inproceedings{yao2023local,
      title     = {Local Implicit Normalizing Flow for Arbitrary-Scale Image Super-Resolution},
      author    = {Jie-En Yao and Li-Yuan Tsao and Yi-Chen Lo and Roy Tseng and Chia-Che Chang and Chun-Yi Lee},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year      = {2023},
}
```

---

## Acknowledgements

Our code was built on [LIIF](https://github.com/yinboc/liif) and [LTE](https://github.com/jaewon-lee-b/lte). We would like to express our gratitude to the authors for generously sharing their code and contributing to the community.
