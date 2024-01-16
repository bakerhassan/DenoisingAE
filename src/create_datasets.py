# Note the majority of this code was taken from: https://github.com/FinnBehrendt/Conditioned-Diffusion-Models-UAD/tree/main repo
import torchio as tio
import os
import torch
from torch.utils.data import DataLoader, Dataset
import SimpleITK as sitk


class vol2slice(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.counter = 0

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        subject = self.ds.__getitem__(index)
        low = 0
        high = subject['vol'].data.shape[-1]
        self.ind = torch.randint(low, high, size=[1])
        subject['ind'] = self.ind
        subject['vol'].data = subject['vol'].data[..., self.ind]
        subject['mask'].data = subject['mask'].data[..., self.ind]
        subject['label'].data = subject['label'].data[..., self.ind]
        slice_idx = ((self.ind - high // 2) / high) * 100
        return subject, slice_idx


def exclude_empty_slices(image, mask, slice_dim=-1):
    slices = []
    mask_slices = []
    if slice_dim == -1:
        for i in range(image.shape[slice_dim]):
            if (image[..., i] > .0001).float().mean() >= .05:
                slices.append(image[..., i])
                mask_slices.append(mask[..., i])
    else:
        raise NotImplementedError(f'slice_dim = {slice_dim} is not supported')
    return torch.stack(slices).permute((1, 2, 0)), torch.stack(mask_slices).permute((1, 2, 0))


def sitk_reader(path):
    image_nii = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if not 'mask' in str(path) and not 'seg' in str(path):  # only for volumes / scalar images
        image_nii = sitk.CurvatureFlow(image1=image_nii, timeStep=0.125, numberOfIterations=3)
    vol = sitk.GetArrayFromImage(image_nii).transpose(2, 1, 0)
    return vol, None


def exclude_abnomral_slices(image, mask, slice_dim=-1):
    no_abnormal_image = []
    mask_slices = []
    if slice_dim == -1:
        for i in range(image.shape[slice_dim]):
            if (mask[..., i] > 0).float().mean() < .001:
                no_abnormal_image.append(image[..., i])
                mask_slices.append(mask[..., i])
    else:
        raise NotImplementedError(f'slice_dim = {slice_dim} is not supported')
    return torch.stack(no_abnormal_image).permute((1, 2, 0)), torch.stack(mask_slices).permute((1, 2, 0))


def get_transform():
    return tio.Compose([
        tio.RescaleIntensity((0, 1), percentiles=(1, 99),
                             masking_method='mask')
    ])


def create_dataset(images_path: str, training: bool, batch_size: int, num_workers: int = 1):
    # Get a list of image files
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.nii.gz') and f.find('seg') == -1])[10:]

    # Get a list of corresponding mask files
    mask_files = sorted([f.replace('t1', 'seg') for f in image_files])
    subjects = []
    counter = 0
    for img_file, mask_file in zip(image_files, mask_files):
        # Read MRI images using tio
        sub = tio.ScalarImage(os.path.join(images_path, img_file), reader=sitk_reader)
        label = tio.LabelMap(os.path.join(images_path, mask_file))

        if training:
            image, label = exclude_abnomral_slices(sub.data[0].float(), label.data[0].float())
            image, label = exclude_empty_slices(image, label)
        else:
            image, label = exclude_empty_slices(sub.data[0].float(), label.data[0].float())
        image = image[None, ...]
        label = label[None, ...]
        brain_mask = (image > .001)
        subject_dict = {'vol': tio.ScalarImage(tensor=image), 'name': img_file,
                        'label': tio.LabelMap(tensor=label),
                        'mask': tio.LabelMap(tensor=brain_mask)}
        subject = tio.Subject(subject_dict)
        subjects.append(subject)
        counter += 1
    ds = tio.SubjectsDataset(subjects, transform=get_transform())
    if training:
        ds = vol2slice(ds)
    ds = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                    shuffle=training)

    return ds
