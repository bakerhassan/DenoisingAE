#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
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
    y_true_ = torch.zeros(240 * 240 * len(dataset), dtype=torch.half)
    y_pred_ = torch.zeros(240 * 240 * len(dataset), dtype=torch.half)
    i = 0
    for batch in dataset:
        with torch.no_grad():
            anomaly_scores = get_scores(trainer, batch=batch['vol'][tio.DATA].squeeze(-1))
        y_ = (batch['label'][tio.DATA].squeeze(-1).view(-1) > 0.5)
        y_hat = anomaly_scores.reshape(-1)
        # Use half precision to save space in RAM. Want to evaluate the whole dataset at once.
        y_true_[i:i + y_.numel()] = y_.half()
        y_pred_[i:i + y_hat.numel()] = y_hat.half()
        i += y_.numel()

        sub_ap.append(average_precision_score(y_, y_hat))
        if return_dice:
            dice_sub.append(dice(y_ > 0.5, y_hat > threshold).cpu().item())
    ap = average_precision_score(y_true_, y_pred_)
    if return_dice:
        with torch.no_grad():
            y_true_ = y_true_.to(trainer.device)
            y_pred_ = y_pred_.to(trainer.device)
            dices = [dice(y_true_ > 0.5, y_pred_ > x).cpu().item() for x in tqdm(dice_thresholds)]
        max_dice, threshold = max(zip(dices, dice_thresholds), key=lambda x: x[0])

        sub_ap_cc, dice_sub_cc = [], []
        if filter_cc:
            # Now that we have the threshold we can do some filtering and recalculate the Dice
            i = 0
            y_true_ = torch.zeros(240 * 240 * len(dataset), dtype=torch.bool)
            y_pred_ = torch.zeros(240 * 240 * len(dataset), dtype=torch.bool)

            for pd in dataset:
                with torch.no_grad():
                    anomaly_scores = get_scores(trainer, batch=batch['vol'][tio.DATA].squeeze(-1))
                y_ = (batch['label'][tio.DATA].squeeze(-1).view(-1) > 0.5)
                # Do CC filtering:
                anomaly_scores_bin = anomaly_scores > threshold
                anomaly_scores_bin = connected_components_3d(anomaly_scores_bin.squeeze(dim=1)).unsqueeze(dim=1)

                y_hat = anomaly_scores_bin.reshape(-1)
                y_true_[i:i + y_.numel()] = y_
                y_pred_[i:i + y_hat.numel()] = y_hat
                sub_ap.append(average_precision_score(y_, y_hat))
                dice_sub.append(dice(y_ > 0.5, y_hat > threshold).cpu().item())
                i += y_.numel()

            with torch.no_grad():
                y_true_ = y_true_.to(trainer.device)
                y_pred_ = y_pred_.to(trainer.device)
                post_cc_max_dice = dice(y_true_, y_pred_).cpu().item()

            return ap, max_dice, threshold, sub_ap, dice_sub, post_cc_max_dice, sub_ap_cc, dice_sub_cc
        return ap, max_dice, threshold, sub_ap, dice_sub
    return ap, sub_ap


def evaluate(testing_path: str, eval_testing_path: str, id: str = "model", split: str = "test", use_cc: bool = True):
    testing_dataloader = create_dataset(testing_path, False, batch_size=1, num_workers=1)
    eval_testing_dataloader = create_dataset(eval_testing_path, False, batch_size=1, num_workers=1)
    trainer = denoising(id, None, None, lr=0.0001, depth=4,
                        wf=6, noise_std=0.2, noise_res=16,
                        n_input=1)  # Noise parameters don't matter during evaluation.

    trainer.load(id)

    results = eval_anomalies_batched(trainer, dataset=eval_testing_dataloader, get_scores=trainer.get_scores,
                                     return_dice=True,
                                     filter_cc=False)

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
    parser.add_argument("-id", "--identifier",default='model', type=str, help="identifier for model to evaluate")
    parser.add_argument("-s", "--split",default='test',type=str, help="'train', 'val' or 'test'")
    parser.add_argument("-cc", "--use_cc", required=False, type=bool, default=True,
                        help="Whether to use connected component filtering.")
    parser.add_argument("-te", "--eval_testing_path", type=str,default='/lustre/cniel/BraTS2021_Training_Data/heldout/eval', help="eval testing path")
    parser.add_argument("-tp", "--testing_path", type=str,default='/lustre/cniel/BraTS2021_Training_Data/heldout', help="testing path")

    args = parser.parse_args()

    evaluate(testing_path=args.testing_path,
             eval_testing_path=args.eval_testing_path,
             id=args.identifier,
             split=args.split,
             use_cc=args.use_cc)
