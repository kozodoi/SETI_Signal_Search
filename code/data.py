import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import numpy as np
import pandas as pd

from utilities import *
from augmentations import get_augs



class ImageData(Dataset):
    
    '''
    Image dataset class
    '''
    
    def __init__(self, 
                 df, 
                 freqs,
                 stacking  = True,
                 difference = False,
                 normalize = False,
                 labeled   = True,
                 transform = None):
        self.df        = df
        self.freqs     = freqs
        self.stacking  = stacking
        self.difference = difference
        self.normalize = normalize
        self.labeled   = labeled
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        # import image
        file_path = self.df.loc[idx, 'file_path']
        if '_off' in file_path:
            image = np.load(file_path.replace('_off', ''))[[f for f in range(6) if f not in self.freqs]] # (n, 273, 256)
        else: 
            image = np.load(file_path)[self.freqs] # (n, 273, 256)
        if image is None:
            raise FileNotFoundError(file_path)
                        
        # preprocess image
        image = image.astype(np.float32)
        if self.normalize == 'layer' or '_old' in file_path:
            image = (image - np.mean(image, axis = (1, 2), keepdims = True)) / np.std(image, axis = (1, 2), keepdims = True)
        
        # stack if needed
        if self.stacking:
            image = np.vstack(image).T # (256, freqs * 273)
        else:
            image = image.T # (256, 273, freqs)
            
        # min-max normalization 
        if self.normalize == 'max':
            image -= np.min(image) 
            image /= np.max(image)
            
        # anti-image preparation
        if self.difference:
            anti_image = np.load(file_path)[[f for f in range(6) if f not in self.freqs]]
            anti_image = anti_image.astype(np.float32)
            if self.normalize:
                anti_image = (anti_image - np.mean(anti_image, axis = (1, 2), keepdims = True)) / np.std(anti_image, axis = (1, 2), keepdims = True)
            if self.stacking:
                anti_image = np.vstack(anti_image).T 
            else:
                anti_image = anti_image.T
            image = image - anti_image
                
        # augmentations
        if self.transform:
            image = self.transform(image = image)['image']
        else:
            image = image[np.newaxis, :, :]
            image = torch.from_numpy(image).float()

        # output
        if self.labeled:
            label = torch.tensor(self.df.loc[idx, 'target']).float()
            return image, label            
        return image            

    

def get_data(df, df_old, df_off, fold, CFG, accelerator, silent = False, debug = None):
    
    '''
    Get training and validation data
    '''

    # load splits
    df_train = df.loc[df.fold != fold].reset_index(drop = True)
    df_valid = df.loc[df.fold == fold].reset_index(drop = True)
    if not silent:
        accelerator.print('- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)))
        
    # add old data
    if df_old is not None:
        df_old_tmp = df_old.loc[df_old.fold != fold].reset_index(drop = True)
        df_train   = pd.concat([df_train, df_old_tmp], axis = 0).reset_index(drop = True)
        if not silent:
            accelerator.print('- adding old labeled images...')
            accelerator.print('- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)))
            
    # add off data
    if df_off is not None:
        df_off_tmp = df_off.loc[df_off.fold != fold].reset_index(drop = True)
        df_train  = pd.concat([df_train, df_off_tmp], axis = 0).reset_index(drop = True)
        if not silent:
            accelerator.print('- adding off images with zero target...')
            accelerator.print('- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)))

    # subset for debug mode
    if debug is None:
        debig = CFG['debug']
    if debug:
        df_train = df_train.sample(CFG['batch_size'] * 10, random_state = CFG['seed']).reset_index(drop = True)
        df_valid = df_valid.sample(CFG['batch_size'] * 10, random_state = CFG['seed']).reset_index(drop = True)
        accelerator.print('- subsetting data for debug mode...')
        accelerator.print('- no. images: train - {}, valid - {}'.format(len(df_train), len(df_valid)))
    
    return df_train, df_valid



def get_loaders(df_train, df_valid, CFG, accelerator, labeled = True, silent = False):
    
    '''
    Get training and validation dataloaders
    '''

    ##### DATASETS
        
    # augmentations
    train_augs, valid_augs = get_augs(CFG, CFG['image_size'], CFG['p_aug'])
    
    # datasets
    train_dataset = ImageData(df        = df_train, 
                              freqs     = CFG['freqs'],
                              stacking  = CFG['stacking'],
                              difference = CFG['difference'],
                              normalize = CFG['normalize'],
                              transform = train_augs,
                              labeled   = labeled)
    valid_dataset = ImageData(df        = df_valid, 
                              freqs     = CFG['freqs'],
                              stacking  = CFG['stacking'],
                              difference = CFG['difference'],
                              normalize = CFG['normalize'],
                              transform = valid_augs,
                              labeled   = labeled)

        
    ##### DATA LOADERS
    
    # data loaders
    train_loader = DataLoader(dataset        = train_dataset, 
                              batch_size     = CFG['batch_size'], 
                              shuffle          = True,
                              num_workers    = CFG['cpu_workers'],
                              drop_last      = False, 
                              worker_init_fn = worker_init_fn,
                              pin_memory     = False)
    valid_loader = DataLoader(dataset     = valid_dataset, 
                              batch_size  = CFG['valid_batch_size'], 
                              shuffle       = False,
                              num_workers = CFG['cpu_workers'],
                              drop_last   = False,
                              pin_memory  = False)
    
    # feedback
    if not silent:
        accelerator.print('- image size: {}, p(augment): {}'.format(CFG['image_size'], CFG['p_aug']))
        accelerator.print('-' * 55)
    
    return train_loader, valid_loader