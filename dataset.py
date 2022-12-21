import nibabel as nib
import numpy as np
from nibabel.funcs import concat_images
from nilearn.masking import compute_brain_mask, apply_mask

import os

from nilearn.image import clean_img

class Dataset():
    """
    TODO
    """
    def __init__(self, directory, debug=False):
        '''
        TODO
        '''
        beta_maps_dir = os.path.join(directory, 'beta_maps/')
        mask_dir = os.path.join(directory, 'anatomy/')
        self.mask = compute_brain_mask(nib.load(os.path.join(mask_dir, 'mask.nii')), threshold=.3)

        self.beta_maps = []
        for file in sorted(os.listdir(beta_maps_dir)):
            if file.endswith('.nii.gz'):
                map = nib.load(os.path.join(beta_maps_dir, file))
                self.beta_maps.append(clean_img(map, standardize=False, ensure_finite=True))
        
        self.nb_subs_ = len(self.beta_maps)
        classes = ['caught', 'chase', 'checkpoint', 'close_enemy', 'protected_by_wall', 'vc_hit']
        self.nb_runs_per_sub_ = self.beta_maps[0].shape[-1] // len(classes)

        if debug:
            self.nb_subs_ = 5
            self.beta_maps = self.beta_maps[:self.nb_subs_]

        self.beta_maps = concat_images(self.beta_maps, axis=-1)
        self.labels = np.tile(classes, self.nb_runs_per_sub_*self.nb_subs_)

    
    def split_train_val(self, train_idx, validation_idx):
        '''
        TODO
        '''
        raw_data, affine = self.beta_maps.get_fdata(), self.beta_maps.affine
        train_raw_data, val_raw_data = raw_data[..., train_idx], raw_data[..., validation_idx]
        train_data, val_data = nib.Nifti1Image(train_raw_data, affine), nib.Nifti1Image(val_raw_data, affine)
        return train_data, val_data, self.labels[train_idx], self.labels[validation_idx]

    def get_beta_maps(self):
        '''
        Returns the beta maps of the dataset.
        '''
        return self.beta_maps
        
    def get_samples(self):
        '''
        Returns the samples of the dataset.
        Samples are the masked and flattened beta maps.
        '''
        samples = apply_mask(self.beta_maps, self.mask)
        return samples
    
    def get_labels(self):
        '''
        Returns the labels of the dataset.
        '''
        return self.labels

