#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
import random

import torchio as tio
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.create_datasets import create_dataset
from src.training import simple_train_step, simple_val_step
from src.metrics import Loss
from src.trainer import Trainer
from src.utilities import median_pool, ModelSaver, load_splits
from src.unet import UNet
from src.patch2loc import patch2loc


def denoising(identifier: str, training_dataloader: DataLoader = None, validation_dataloader: DataLoader = None,
              lr=0.001, depth=4,
              wf=7, n_input=4, noise_std=0.2, noise_res=16, use_patch2loc=False):
    device = torch.device("cuda")

    def noise(x):
        ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res), std=noise_std).to(x.device)

        ns = F.upsample_bilinear(ns, size=[240, 240])

        # Roll to randomly translate the generated noise.
        roll_x = random.choice(range(240))
        roll_y = random.choice(range(240))
        ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

        mask = x.sum(dim=1, keepdim=True) > 0.01
        ns *= mask  # Only apply the noise in the foreground.
        res = x + ns

        return res

    def get_scores(trainer, batch, median_f=True):
        batch = batch['vol'][tio.DATA].squeeze(0).permute(3, 0, 1, 2).to('cuda')
        slice_idxs = torch.linspace(0, batch.shape[0], batch.shape[0])
        slice_idxs = ((slice_idxs - batch.shape[0] // 2) / batch.shape[0]) * 100
        kwargs = {}
        kwargs['slice_idx'] = slice_idxs.unsqueeze(1)
        x = batch
        trainer.model = trainer.model.eval()
        with torch.no_grad():
            # Assume it's in batch shape
            clean = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1]).clone().to(trainer.device)
            mask = clean.sum(dim=1, keepdim=True) > 0.01

            # Erode the mask a bit to remove some of the reconstruction errors at the edges.
            mask = (F.avg_pool2d(mask.float(), kernel_size=5, stride=1, padding=2) > 0.95)

            res = trainer.model(clean, kwargs)

            err = ((clean - res) * mask).abs().mean(dim=1, keepdim=True)
            if median_f:
                err = median_pool(err, kernel_size=5, stride=1, padding=2)

        return err.cpu()

    def loss_f(trainer, batch, batch_results):
        y = batch
        mask = batch.sum(dim=1, keepdim=True) > 0.01
        return (torch.pow(batch_results - y, 2) * mask.float()).mean()

    def forward(trainer, batch, **kwargs):

        batch = noise(batch.clone())
        return trainer.model(batch, kwargs)

    patch2loc_model = None
    if use_patch2loc:
        patch2loc_model = patch2loc(position_conditional=True)
        patch2loc_model = torch.nn.DataParallel(patch2loc_model)
        patch2loc_model = patch2loc_model.to(device)
        patch2loc_model.load_state_dict(
            torch.load('/home/2063/resnet_data:brats_aug:True_beta:1_loss:beta_nll_target_dim:2_conditional:True.pth',
                       map_location=device))
        for param in patch2loc_model.parameters():
            param.requires_grad = False
        patch2loc_model.eval()
    model = UNet(in_channels=n_input, n_classes=n_input, norm="group", up_mode="upconv", depth=depth, wf=wf,
                 padding=True, patch2loc=patch2loc_model).to(device)

    train_step = partial(simple_train_step, forward=forward, loss_f=loss_f)
    val_step = partial(simple_val_step, forward=forward, loss_f=loss_f)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=0.00001)
    callback_dict = defaultdict(list)

    model_saver = ModelSaver(path=Path(__file__).resolve().parent.parent / "saved_models" / f"{identifier}.pt")
    model_saver.register(callback_dict)

    Loss(lambda batch_res, batch: loss_f(trainer, batch, batch_res)).register(callback_dict, log=True, tensorboard=True,
                                                                              train=True, val=True)

    trainer = Trainer(model=model,
                      train_dataloader=training_dataloader,
                      val_dataloader=validation_dataloader,
                      optimiser=optimiser,
                      train_step=train_step,
                      val_step=val_step,
                      callback_dict=callback_dict,
                      device=device,
                      identifier=identifier)

    trainer.noise = noise
    trainer.get_scores = get_scores
    trainer.reset_state()

    trainer.lr_scheduler = CosineAnnealingLR(optimizer=optimiser, T_max=100)

    def update_learning_rate(trainer):
        trainer.lr_scheduler.step()

    trainer.callback_dict["after_train_epoch"].append(update_learning_rate)

    return trainer


def train(trainin_path: str, evaluation_path: str, id: str = "model", noise_res: int = 16, noise_std: float = 0.2,
          seed: int = 0, batch_size: int = 16, split_num=1, use_patch2loc=False):
    split_file = f'split_{split_num}.pkl'  # Update with the path to the split file you want to load
    val_files = train_files = None
    if os.path.exists(trainin_path + '/' + split_file):
        split_info = load_splits(trainin_path + '/' + split_file)
        train_files = split_info['train']
        val_files = train_files[-int(.1 * len(train_files)):]
        train_files = train_files[:-int(.1 * len(train_files))]
    training_dataloader = create_dataset(trainin_path, True, batch_size, num_workers=1, image_files=train_files,
                                         abnormal_data=True, exclude_abnormal=True)
    eval_dataloader = create_dataset(evaluation_path, True, batch_size, num_workers=1, image_files=val_files,
                                     abnormal_data=True, exclude_abnormal=True)
    trainer = denoising(id, training_dataloader, eval_dataloader, lr=0.0001, depth=4,
                        wf=6, noise_std=noise_std, noise_res=noise_res, n_input=1, use_patch2loc=use_patch2loc)

    trainer.train(epoch_len=32, max_epochs=2100, val_epoch_len=32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--identifier", type=str, default="model", help="model name.")
    parser.add_argument("-nr", "--noise_res", type=float, default=16, help="noise resolution.")
    parser.add_argument("-ns", "--noise_std", type=float, default=0.2, help="noise magnitude.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed.")
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="model training batch size")
    parser.add_argument("-tp", "--training_path", type=str, default='/lustre/cniel/BraTS2021_Training_Data/heldout/',
                        help="training path")
    parser.add_argument("-ep", "--evaluation_path", type=str,
                        default='/lustre/cniel/BraTS2021_Training_Data/heldout/',
                        help="evaluation path.")

    parser.add_argument("-patch2loc", "--patch2loc", type=bool, default=False,
                        help="use patch2loc")
    parser.add_argument("-split_num", "--split_num", type=int, default=1,
                        help="which split number to use (if any)")
    args = parser.parse_args()

    train(
        trainin_path=args.training_path,
        evaluation_path=args.evaluation_path,
        id=args.identifier,
        noise_res=args.noise_res,
        noise_std=args.noise_std,
        seed=args.seed,
        split_num=args.split_num,
        batch_size=args.batch_size,
        use_patch2loc=args.patch2loc)
