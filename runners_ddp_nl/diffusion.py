##### Diffusion with DDP


import os
import logging
import time
import glob
import csv

import numpy as np
import math
import tqdm
import torch
import random
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry, calculate_psnr
from datasets import data_transform, inverse_data_transform
from datasets.pmub import PMUB
from datasets.LDFDCT import LDFDCT
from functions.ckpt_util import get_ckpt_path
from skimage.metrics import structural_similarity as ssim
import torchvision.utils as tvu
import torchvision
from PIL import Image


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.num_timesteps = self.config.diffusion.num_diffusion_timesteps

       
    
    # Training GDM for tasks that have only one condition: CT denoising.
    def sg_train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        
        if self.args.dataset=='LDFDCT':
            # LDFDCT for CT image denoising
            dataset = LDFDCT(self.config.data.train_dataroot, self.config.data.image_size, split='train')
            print('Start training your GDM model on LDFDCT dataset.')

        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            pin_memory=True,
            sampler=DistributedSampler(dataset)
        )

        model = Model(config)
        model = model.to(self.gpu_id)
        model = DDP(model, device_ids=[self.gpu_id])

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, x in enumerate(train_loader):
                n = x['LD'].size(0)
                model.train()
                step += 1

                x_img = x['LD'].to(self.gpu_id)
                x_gt = x['FD'].to(self.gpu_id)

                e = torch.randn_like(x_gt)     
                
                # sample t and compute sigma
                rnd_uniform = np.random.uniform(0, 1.001, n)
                sigma = 0.002 * ((80 / 0.002) ** rnd_uniform)                
                sigma = torch.from_numpy(sigma).to(self.gpu_id)
                
                # compute embedding
                embedding = rnd_uniform * 1000
                embedding = torch.from_numpy(embedding).to(self.gpu_id)
                
                if torch.distributed.get_rank() == 0:
                    print(f'sigma: {sigma}')
                    print(f'embedding: {embedding}')

                loss = loss_registry[config.model.type](model, x_img, x_gt, sigma, embedding, e)

                tb_logger.add_scalar("loss", loss, global_step=step)
                logging.info(f"GPU{self.gpu_id}, step: {step}, loss: {loss.item()}")

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                dist.barrier()

                if (step % self.config.training.snapshot_freq == 0 or step == 1):
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                    

    # Training GDM for tasks that have two conditions: multi image super-resolution.
    def sr_train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        dataset = PMUB(self.config.data.train_dataroot, self.config.data.image_size, split='train')
        print('Start training your GDM model on PMUB dataset.')
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True)

        model = Model(config)
        model = model.to(self.gpu_id)
        model = DDP(model, device_ids=[self.gpu_id])

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, x in enumerate(train_loader):
                n = x['BW'].size(0)
                model.train()
                step += 1

                x_bw = x['BW'].to(self.gpu_id)
                x_md = x['MD'].to(self.gpu_id)
                x_fw = x['FW'].to(self.gpu_id)

                e = torch.randn_like(x_md)
                
                # sample t and compute sigma
                rnd_uniform = np.random.uniform(0, 1.001, n)
                sigma = 0.002 * ((80 / 0.002) ** rnd_uniform)                
                sigma = torch.from_numpy(sigma).to(self.gpu_id)
                
                # compute embedding
                embedding = rnd_uniform * 1000
                embedding = torch.from_numpy(embedding).to(self.gpu_id)               
                
                if torch.distributed.get_rank() == 0:
                    print(f'sigma: {sigma}')
                    print(f'embedding: {embedding}')

                loss = loss_registry[config.model.type](model, x_bw, x_md, x_fw, sigma, embedding, e)

                tb_logger.add_scalar("loss", loss, global_step=step)
                logging.info(f"GPU{self.gpu_id}, step: {step}, loss: {loss.item()}")

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                dist.barrier()

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))


    
    # Training original DDPM for tasks that have only one condition: image translation and CT denoising.
    def sg_ddpm_train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        if self.args.dataset=='LDFDCT':
            # LDFDCT for CT image denoising
            dataset = LDFDCT(self.config.data.train_dataroot, self.config.data.image_size, split='train')
            print('Start training DDPM model on LDFDCT dataset.')
        elif self.args.dataset=='BRATS':
            # BRATS for brain image translation
            dataset = BRATS(self.config.data.train_dataroot, self.config.data.image_size, split='train')
            print('Start training DDPM model on BRATS dataset.')
            
        print('The number of involved time steps is {} out of 1000.'.format(self.args.timesteps))
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True)

        model = Model(config)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, x in enumerate(train_loader):
                n = x['LD'].size(0)
                model.train()
                step += 1

                x_img = x['LD'].to(self.device)
                x_gt = x['FD'].to(self.device)

                e = torch.randn_like(x_gt)
                b = self.betas

                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                loss = loss_registry[config.model.type](model, x_img, x_gt, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))


    # Training original DDPM for tasks that have two conditions: multi image super-resolution.
    def sr_ddpm_train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        dataset = PMUB(self.config.data.train_dataroot, self.config.data.image_size, split='train')
        print('Start training DDPM model on PMUB dataset.')
        print('The number of involved time steps is {} out of 1000.'.format(self.args.timesteps))
        
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True)

        model = Model(config)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        time_start = time.time()
        total_time = 0
        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, x in enumerate(train_loader):
                n = x['BW'].size(0)
                model.train()
                step += 1

                x_bw = x['BW'].to(self.device)
                x_md = x['MD'].to(self.device)
                x_fw = x['FW'].to(self.device)

                e = torch.randn_like(x_md)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x_bw, x_md, x_fw, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

               
    # Sampling for tasks that have two conditions: multi-image super-resolution.
    def sr_sample(self):
        ckpt_list = self.config.sampling.ckpt_id
        for ckpt_idx in ckpt_list:
            self.ckpt_idx = ckpt_idx
            print('Start inference on model of {} steps'.format(ckpt_idx))
            
            model = Model(self.config)
            model = model.to(self.gpu_id)
            model = DDP(model, device_ids=[self.gpu_id])

            if not self.args.use_pretrained:
                states_path = os.path.join(self.args.log_path, f"ckpt_{ckpt_idx}.pth")
                loc = f"cuda:{self.gpu_id}"
                states = torch.load(states_path, map_location=f"cuda:{self.gpu_id}")
                model.load_state_dict(states[0], strict=True)

                if self.config.model.ema:
                    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                    ema_helper.register(model)
                    ema_helper.load_state_dict(states[-1])
                    ema_helper.ema(model)
                else:
                    ema_helper = None
            else:
                # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
                if self.config.data.dataset == "CIFAR10":
                    name = "cifar10"
                elif self.config.data.dataset == "LSUN":
                    name = f"lsun_{self.config.data.category}"
                else:
                    raise ValueError
                ckpt = get_ckpt_path(f"ema_{name}")
                print("Loading checkpoint {}".format(ckpt))
                model.load_state_dict(torch.load(ckpt, map_location=f"cuda:{self.gpu_id}"))
                model.to(self.gpu_id)
                model = DDP(model, device_ids=[self.gpu_id])  
                
            model.eval()

            if self.args.fid:
                self.sr_sample_fid(model)
            elif self.args.interpolation:
                self.sr_sample_interpolation(model)
            elif self.args.sequence:
                self.sample_sequence(model)
            else:
                raise NotImplementedError("Sample procedeure not defined")


    # Sampling for tasks that have only one condition: image translation and CT denoising.
    def sg_sample(self):
        ckpt_list = self.config.sampling.ckpt_id
        for ckpt_idx in ckpt_list:
            self.ckpt_idx = ckpt_idx
            print('Start inference on model of {} steps'.format(ckpt_idx))
            
            model = Model(self.config)
            model = model.to(self.gpu_id)
            model = DDP(model, device_ids=[self.gpu_id])
        
            if not self.args.use_pretrained:
                states_path = os.path.join(self.args.log_path, f"ckpt_{ckpt_idx}.pth")
                loc = f"cuda:{self.gpu_id}"
                states = torch.load(states_path, map_location=f"cuda:{self.gpu_id}")
                model.load_state_dict(states[0], strict=True)

                if self.config.model.ema:
                    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                    ema_helper.register(model)
                    ema_helper.load_state_dict(states[-1])
                    ema_helper.ema(model)
                else:
                    ema_helper = None
            else:
                # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
                if self.config.data.dataset == "CIFAR10":
                    name = "cifar10"
                elif self.config.data.dataset == "LSUN":
                    name = f"lsun_{self.config.data.category}"
                else:
                    raise ValueError
                ckpt = get_ckpt_path(f"ema_{name}")
                print("Loading checkpoint {}".format(ckpt))
                model.load_state_dict(torch.load(ckpt, map_location=f"cuda:{self.gpu_id}"))
                model.to(self.gpu_id)
                model = DDP(model, device_ids=[self.gpu_id])  

            model.eval()

            if self.args.fid:
                self.sg_sample_fid(model)
            elif self.args.interpolation:
                self.sr_sample_interpolation(model)
            elif self.args.sequence:
                self.sample_sequence(model)
            else:
                raise NotImplementedError("Sample procedeure not defined")

                
    def sr_sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")

        sample_dataset = PMUB(self.config.data.sample_dataroot, self.config.data.image_size, split='calculate')
        print('Start sampling model on PMUB dataset.')
        print('The inference sample type is {}. The scheduler sampling type is {}. The number of involved time steps is {} out of 1000.'.format(self.args.sample_type, self.args.scheduler_type, self.args.timesteps))
        
        sample_loader = data.DataLoader(
            sample_dataset,
            batch_size=config.sampling_fid.batch_size,
            shuffle=False,
            sampler=DistributedSampler(sample_dataset)
            )

        with torch.no_grad():
            data_num = len(sample_dataset)
            print('The length of test set is:', data_num)
            
            
            if self.args.sample_type == "generalized":
                if self.args.scheduler_type == 'uniform':
                    t = np.linspace(1/self.args.timesteps, 1, self.args.timesteps)
                elif self.args.scheduler_type == 'non-uniform':
                    t = np.array([0.03, 0.14, 0.23, 0.28, 0.34, 0.41, 0.44, 0.54, 0.66, 0.76, 0.81, 0.84, 0.87, 0.90, 1.0]) # design non-uniform t you want
                else:
                    raise Exception("The scheduler type is either uniform or non-uniform.")   
                        
                t = torch.from_numpy(t)
                sigma = 0.002 * ((80 / 0.002) ** t)
                embedding = t * 1000          
                        
                        
            avg_psnr = 0.0
            avg_ssim = 0.0
            time_list = []
            psnr_list = []
            ssim_list = []

            for batch_idx, sample in tqdm.tqdm(enumerate(sample_loader), desc="Generating image samples for FID evaluation."):
                n = sample['BW'].shape[0]
                
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                ).to(self.gpu_id) * 80  # Scale by σmax  
                x_bw = sample['BW'].to(self.gpu_id)
                x_md = sample['MD'].to(self.gpu_id)
                x_fw = sample['FW'].to(self.gpu_id)
                case_name = sample['case_name']
                
                time_start = time.time()
                x = self.sr_sample_image(x, x_bw, x_fw, model, sigma, embedding, t)
                time_end = time.time()
                
                x = inverse_data_transform(config, x)
                x_md = inverse_data_transform(config, x_md)
                x_tensor = x
                x_md_tensor = x_md
                x_md = x_md.squeeze().float().cpu().numpy()
                x = x.squeeze().float().cpu().numpy()
                x_md = (x_md*255.0).round()
                x = (x*255.0).round()

                PSNR = 0.0 
                SSIM = 0.0
                for i in range(x.shape[0]):
                    psnr_temp = calculate_psnr(x[i,:,:], x_md[i,:,:])
                    ssim_temp = ssim(x_md[i,:,:], x[i,:,:], data_range=255)
                    PSNR += psnr_temp
                    SSIM += ssim_temp
                    psnr_list.append(psnr_temp)
                    ssim_list.append(ssim_temp)

                PSNR_print = PSNR/x.shape[0]
                SSIM_print = SSIM/x.shape[0]

                case_time = time_end-time_start
                time_list.append(case_time)

                avg_psnr += PSNR
                avg_ssim += SSIM
                logging.info('GPU{}, Case {}: PSNR {}, SSIM {}, time {}'.format(
                    int(os.environ["LOCAL_RANK"]), case_name[0], PSNR_print, SSIM_print, case_time))

                dist.barrier()  

                for i in range(0, n):
                    tvu.save_image(
                        x_tensor[i], os.path.join(self.args.image_folder, "{}_{}_{}_pt.png".format(self.ckpt_idx, self.gpu_id, img_id)) 
                    )
                    tvu.save_image(
                        x_md_tensor[i], os.path.join(self.args.image_folder, "{}_{}_{}_gt.png".format(self.ckpt_idx, self.gpu_id, img_id)) 
                    )
                    img_id += 1
                            
            psnr_sum = torch.tensor(np.array(avg_psnr), dtype=torch.float32).to(self.gpu_id)
            dist.all_reduce(psnr_sum, op=dist.ReduceOp.SUM)
            ssim_sum = torch.tensor(np.array(avg_ssim), dtype=torch.float32).to(self.gpu_id)
            dist.all_reduce(ssim_sum, op=dist.ReduceOp.SUM)

            avg_psnr = (psnr_sum / data_num).cpu().numpy()
            avg_ssim = (ssim_sum / data_num).cpu().numpy()
            avg_time = sum(time_list[1:-1]) / (len(time_list) - 2)
            logging.info('Average: PSNR {}, SSIM {}, time {}'.format(avg_psnr, avg_ssim, avg_time))


    def sg_sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")

        if self.args.dataset=='LDFDCT':
            # LDFDCT for CT image denoising
            sample_dataset = LDFDCT(self.config.data.sample_dataroot, self.config.data.image_size, split='calculate')
            print('Start sampling model on LDFDCT dataset.')

        print('The inference sample type is {}. The scheduler sampling type is {}. The number of involved steps is {}.'.format(self.args.sample_type, self.args.scheduler_type, self.args.timesteps))

        sample_loader = data.DataLoader(
            sample_dataset,
            batch_size=config.sampling_fid.batch_size,
            shuffle=False,
            sampler=DistributedSampler(sample_dataset)
            )

        with torch.no_grad():
            data_num = len(sample_dataset)
            print('The length of test set is:', data_num)
            
            if self.args.sample_type == "generalized":
                if self.args.scheduler_type == 'uniform':
                    t = np.linspace(1/self.args.timesteps, 1, self.args.timesteps)
                elif self.args.scheduler_type == 'non-uniform':
                    t = np.array([0.05, 0.13, 0.17, 0.21, 0.23, 0.29, 0.38, 0.47, 0.48, 0.56, 0.61, 0.62, 0.66, 0.69, 0.74, 0.77, 0.79, 0.86, 0.98, 1.0]) # define non-uniform t steps
                else:
                    raise Exception("The scheduler type is either uniform or non-uniform.")

                t = torch.from_numpy(t)
                sigma = 0.002 * ((80 / 0.002) ** t)
                embedding = t * 1000


            avg_psnr = 0.0
            avg_ssim = 0.0
            avg_dice = 0.0
            time_list = []
            psnr_list = []
            ssim_list = []

            for batch_idx, sample in tqdm.tqdm(enumerate(sample_loader), desc="Generating image samples for FID evaluation."):
                n = sample['LD'].shape[0]

                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                ).to(self.gpu_id) * 80  # Scale by σmax                    
                x_img = sample['LD'].to(self.gpu_id)
                x_gt = sample['FD'].to(self.gpu_id)
                case_name = sample['case_name']

                time_start = time.time()
                x = self.sg_sample_image(x, x_img, model, sigma, embedding, t)
                time_end = time.time()
                  
                    
                x = inverse_data_transform(config, x)
                x_gt = inverse_data_transform(config, x_gt)
                x_tensor = x
                x_gt_tensor = x_gt
                x_gt = x_gt.squeeze().float().cpu().numpy()
                x = x.squeeze().float().cpu().numpy()
                x_gt = x_gt * 255
                x = x * 255

                PSNR = 0.0
                SSIM = 0.0
                for i in range(x.shape[0]):
                    psnr_temp = calculate_psnr(x[i, :, :], x_gt[i, :, :])
                    ssim_temp = ssim(x_gt[i, :, :], x[i, :, :], data_range=255)
                    PSNR += psnr_temp
                    SSIM += ssim_temp
                    psnr_list.append(psnr_temp)
                    ssim_list.append(ssim_temp)

                PSNR_print = PSNR / x.shape[0]
                SSIM_print = SSIM / x.shape[0]
    
                case_time = time_end - time_start
                time_list.append(case_time)

                avg_psnr += PSNR
                avg_ssim += SSIM
                logging.info('GPU{}, Case {}: PSNR {}, SSIM {}, time {}'.format(
                    int(os.environ["LOCAL_RANK"]), case_name[0], PSNR_print, SSIM_print, case_time))

                dist.barrier()  

                for i in range(0, n):
                    tvu.save_image(
                        x_tensor[i], os.path.join(self.args.image_folder, "{}_{}_{}_pt.png".format(self.ckpt_idx, self.gpu_id, img_id)) 
                    )
                    tvu.save_image(
                        x_gt_tensor[i], os.path.join(self.args.image_folder, "{}_{}_{}_gt.png".format(self.ckpt_idx, self.gpu_id, img_id)) 
                    )
                    img_id += 1

    
            psnr_sum = torch.tensor(np.array(avg_psnr), dtype=torch.float32).to(self.gpu_id)
            dist.all_reduce(psnr_sum, op=dist.ReduceOp.SUM)
            ssim_sum = torch.tensor(np.array(avg_ssim), dtype=torch.float32).to(self.gpu_id)
            dist.all_reduce(ssim_sum, op=dist.ReduceOp.SUM)

            avg_psnr = (psnr_sum / data_num).cpu().numpy()
            avg_ssim = (ssim_sum / data_num).cpu().numpy()
            avg_time = sum(time_list[1:-1]) / (len(time_list) - 2)
            logging.info('Average: PSNR {}, SSIM {}, time {}'.format(avg_psnr, avg_ssim, avg_time))

            
                


    def sr_sample_image(self, x, x_bw, x_fw, model, sigma, embedding, t, last=True):
                
        from functions.denoising import sr_geodesic_fisherve_steps

        xs = sr_geodesic_fisherve_steps(x, x_bw, x_fw, embedding, t, model, sigma, eta=self.args.eta)
        x = xs

        if last:
            x = x[0][-1]

        return x

    def sg_sample_image(self, x, x_img, model, sigma, embedding, t, last=True):
                
        from functions.denoising import sg_geodesic_fisherve_steps

        xs = sg_geodesic_fisherve_steps(x, x_img, embedding, t, model, sigma, eta=self.args.eta)
        x = xs

        if last:
            x = x[0][-1]

        return x


    def test(self):       
        pass