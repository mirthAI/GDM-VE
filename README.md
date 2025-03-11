# GDM-VE
Official PyTorch implementation of: 

[GDM: Geodesic Diffusion Models for Medical Image-to-Image Generation](https://arxiv.org/abs/2503.00745)

Diffusion models transform an unknown data distribution into a Gaussian prior by progressively adding noise until the data become indistinguishable from pure noise. This stochastic process traces a path in probability space, evolving from the original data distribution (considered as a Gaussian with near-zero variance) to an isotropic Gaussian. The denoiser then learns to reverse this process, generating high-quality samples from random Gaussian noise. However, standard diffusion models, such as the Denoising Diffusion Probabilistic Model (DDPM), do not ensure a geodesic (i.e., shortest) path in probability space. This inefficiency necessitates the use of many intermediate time steps, leading to high computational costs in training and sampling. To address this limitation, we propose the Geodesic Diffusion Model (GDM), which defines a geodesic path under the Fisher-Rao metric with a variance-exploding noise scheduler. This formulation transforms the data distribution into a Gaussian prior with minimal energy, significantly improving the efficiency of diffusion models. We trained GDM by continuously sampling time steps from 0 to 1 and using as few as 15 evenly spaced time steps for model sampling. Experimental results show that GDM achieved state-of-the-art performance while reducing training time by 50× compared to DDPM and 10× compared to Fast-DDPM, with 66× faster sampling than DDPM and a similar sampling speed to Fast-DDPM.


## Requirements
* Python==3.10.6
* torch==1.12.1
* torchvision==0.13.1
* numpy 
* opencv-python
* tqdm
* tensorboard
* tensorboardX
* scikit-image
* medpy
* pillow
* scipy
* **Note:** To install the list of packages, kindly use the command `pip install -r requirements.txt`


## Datasets
In the paper, we train and evaluate our model on two image-to-image generation tasks:

* LDCT-and-Projection-data dataset for single-condition image-to-image translation (denoising task).
* Prostate-MRI-US-Biopsy dataset for multiple-condition image-to-image translation (super-resolution task).

The processed dataset can be accessed here: https://drive.google.com/file/d/1sdN59XYFQHtMcp0ACFr8iQonLpAPccS5/view?usp=drive_link


## Usage
### 1. Git clone or download the codes.

### 2. Download model pre-trained weights
We provide the model weights trained on the above two datasets, which you can access at this link: https://drive.google.com/drive/folders/1-la4Rp3CeSBAWcTfybCF-Mag5QoEBiXt?usp=sharing.

### 3. Prepare data
You can download the dataset from the link we provide or the official public dataset website, and then put the dataset folder in the main directory of the project. The directory structure should be like this:

```bash
│
├── data
│	├── LD_FD_CT_train
│	├── LD_FD_CT_test
│	├── PMUB-train
│	└── PMUB-test
│
├── other folders
│
└── other files
```


### 4. Running the Experiments
* Training a GDM model

```
sh GDM.sh --config {DATASET}.yml --dataset {DATASET_NAME} --exp {PROJECT_PATH} --doc {MODEL_NAME}
```

* Sampling from a GDM model

```
sh GDM.sh --config {DATASET}.yml --dataset {DATASET_NAME} --exp {PROJECT_PATH} --doc {MODEL_NAME} --sample --fid --timesteps {STEPS}
```

For the image denoising task, the `{DATASET}.yml` and `DATASET_NAME` should be `ldfd.yml` and `LDFDCT`. For the super-resolution task, the `{DATASET}.yml` and `DATASET_NAME` should be `pmub.yml` and `PMUB`. `STEPS` controls the number of steps from the pure noise distribution to the image distribution during sampling, with 15 steps used by default.

If you want to use our pre-trained model for sampling, you can put the downloaded pre-trained weights in `pretrained_GDM/logs/{TASK_NAME}/ckpt_45000.pth`, and then run the sampling with the following command:

```
sh GDM.sh --config {DATASET}.yml --dataset {DATASET_NAME} --exp pretrained_GDM --doc {TASK_NAME} --sample --fid --timesteps {STEPS}
```

where `TASK_NAME` is `denoising` or `super-resolution` respectively.


## References
The code is mainly adapted from [DDIM](https://github.com/ermongroup/ddim).


## Citations and Acknowledgements
The code is only for research purposes. If you have any questions regarding how to use this code, feel free to contact Teng Zhang at zhangt@ufl.edu.

Kindly cite our paper as follows if you use our code or dataset.

```bibtex
@misc{zhang2025geodesicdiffusionmodelsmedical,
      title={Geodesic Diffusion Models for Medical Image-to-Image Generation}, 
      author={Teng Zhang and Hongxu Jiang and Kuang Gong and Wei Shao},
      year={2025},
      eprint={2503.00745},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2503.00745}, 
}
```
