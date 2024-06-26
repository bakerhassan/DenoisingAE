#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
import os
from math import ceil

import numpy as np
import torch
import torchio as tio
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from src.denoising import denoising
from src.data import BrainDataset
from src.cc_filter import connected_components_3d
from src.create_datasets import create_dataset
from src.utilities import load_splits


def eval_anomalies_batched(trainer, dataset, get_scores, batch_size=32, threshold=None,
                           return_dice=False, filter_cc=False):
    def dice(a, b):
        num = 2 * (a & b).sum()
        den = a.sum() + b.sum()

        den_float = den.float()
        den_float[den == 0] = float("nan")

        return num.float() / den_float

    sub_ap, dice_sub = [], []
    dice_thresholds = [x / 1000 for x in range(1000)] if threshold is None else [threshold]
    y_true_ = []
    y_pred_ = []
    error_maps = []
    labels = []
    counter = 0
    for batch in dataset:
        with torch.no_grad():
            anomaly_scores = get_scores(trainer, batch=batch)
        error_maps.append(anomaly_scores.cpu().numpy())
        y_ = batch['label'][tio.DATA].squeeze(0).permute(3, 0, 1, 2)
        labels.append(y_.cpu().numpy())
        y_ = (y_.reshape(-1) > 0.5)
        y_hat = anomaly_scores.reshape(-1)
        # Use half precision to save space in RAM. Want to evaluate the whole dataset at once.
        y_true_.append(y_.cpu().half())
        y_pred_.append(y_hat.cpu().half())

        sub_ap.append(average_precision_score(y_, y_hat))
        if return_dice and threshold is not None:
            dice_sub.append(dice(y_ > 0.5, y_hat > threshold).cpu().item())
        print("done with subject: ", counter)
        counter += 1
    tmp = {'error_maps': error_maps, 'labels': labels}
    np.save('/lustre/cniel/DAE_results', tmp)
    del tmp
    y_true_, y_pred_ = torch.cat(y_true_, dim=0).cpu(), torch.cat(y_pred_, dim=0).cpu()
    ap = average_precision_score(y_true_, y_pred_)
    if return_dice:
        sub_ap_cc, dice_sub_cc = [], []
        if threshold is None:
            with torch.no_grad():
                y_true_ = y_true_.to(trainer.device)
                y_pred_ = y_pred_.to(trainer.device)
                dices = [dice(y_true_ > 0.5, y_pred_ > x).cpu().item() for x in tqdm(dice_thresholds)]
            max_dice, threshold = max(zip(dices, dice_thresholds), key=lambda x: x[0])
        max_dice = dice(y_true_ > .5, y_pred_ > threshold).cpu().item()
        if filter_cc:
            # Now that we have the threshold we can do some filtering and recalculate the Dice
            i = 0
            y_true_ = []
            y_pred_ = []

            for batch in dataset:
                with torch.no_grad():
                    anomaly_scores = get_scores(trainer, batch=batch)
                y_ = (batch['label'][tio.DATA].squeeze(0).permute(3, 0, 1, 2).reshape(-1) > 0.5)
                # Do CC filtering:
                anomaly_scores_bin = anomaly_scores > threshold
                anomaly_scores_bin = connected_components_3d(anomaly_scores_bin.squeeze(dim=1)).unsqueeze(dim=1)

                y_hat = anomaly_scores_bin.reshape(-1)
                y_true_.append(y_.cpu())
                y_pred_.append(y_hat.cpu())
                sub_ap_cc.append(average_precision_score(y_, y_hat))
                dice_sub_cc.append(dice(y_ > 0.5, y_hat > threshold).cpu().item())
                i += y_.numel()
            y_true_, y_pred_ = torch.cat(y_true_, dim=0).cpu(), torch.cat(y_pred_, dim=0).cpu()
            with torch.no_grad():
                post_cc_max_dice = dice(y_true_ > .5, y_pred_ > threshold).cpu().item()

            return ap, max_dice, threshold, sub_ap, dice_sub, post_cc_max_dice, sub_ap_cc, dice_sub_cc
        return ap, max_dice, threshold, sub_ap, dice_sub
    return ap, sub_ap


def evaluate(testing_path: str, eval_testing_path: str, id: str = "model", split: str = "test", use_cc: bool = True,
             split_num=1,
             use_patch2loc=False):
    split_file = f'split_{split_num}.pkl'  # Update with the path to the split file you want to load
    val_files = test_files = None
    if os.path.exists(testing_path + '/' + split_file):
        split_info = load_splits(testing_path + '/' + split_file)
        val_files = split_info['val']
        test_files = split_info['test']
    testing_dataloader = create_dataset(testing_path, False, batch_size=1, num_workers=0, image_files=test_files,
                                        abnormal_data=True)
    eval_testing_dataloader = create_dataset(eval_testing_path, False, batch_size=1, num_workers=0,
                                             image_files=val_files, abnormal_data=True)
    trainer = denoising(id, None, None, lr=0.0001, depth=4,
                        wf=6, noise_std=0.2, noise_res=16,
                        n_input=1, use_patch2loc=use_patch2loc)  # Noise parameters don't matter during evaluation.

    trainer.load(id)

    results = eval_anomalies_batched(trainer, dataset=eval_testing_dataloader, get_scores=trainer.get_scores,
                                     return_dice=True,
                                     filter_cc=False)

    print(f"Done with validation!! Moving to Testing... Thresholds are: {results[2]}")

    results = eval_anomalies_batched(trainer, dataset=testing_dataloader, get_scores=trainer.get_scores,
                                     return_dice=True,
                                     threshold=results[2],
                                     filter_cc=use_cc)

    print(f"AP: {results[0]}")
    print(f"max Dice: {results[1]}")
    print(f"Optimal threshold: {results[2]}")
    print(f"Micro-AP:{np.array(results[3]).mean()} +- {np.array(results[3]).std()}")
    print(f"Micro-Dice:{np.array(results[4]).mean()} +- {np.array(results[4]).std()}")

    if use_cc:
        print(f"max Dice post CC: {results[5]}")
        print(f"Micro-AP Post CC:{np.array(results[6]).mean()} +- {np.array(results[6]).std()}")
        print(f"Micro-Dice PostCC :{np.array(results[7]).mean()} +- {np.array(results[7]).std()}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--identifier", default='healthy', type=str, help="identifier for model to evaluate")
    parser.add_argument("-s", "--split", default='test', type=str, help="'train', 'val' or 'test'")
    parser.add_argument("-cc", "--use_cc", required=False, type=bool, default=True,
                        help="Whether to use connected component filtering.")
    parser.add_argument("-te", "--eval_testing_path", type=str,
                        default='/lustre/cniel/BraTS2021_Training_Data/heldout/', help="eval testing path")
    parser.add_argument("-tp", "--testing_path", type=str, default='/lustre/cniel/BraTS2021_Training_Data/heldout',
                        help="testing path")
    parser.add_argument("-split_num", "--split_num", type=int, default=1,
                        help="which split number to use (if any)")
    parser.add_argument("-patch2loc", "--patch2loc", type=bool, default=False,
                        help="use patch2loc")
    args = parser.parse_args()

    evaluate(testing_path=args.testing_path,
             eval_testing_path=args.eval_testing_path,
             id=args.identifier,
             split=args.split,
             use_cc=args.use_cc,
             split_num=args.split_num,
             use_patch2loc=args.patch2loc)
