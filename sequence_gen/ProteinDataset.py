import torch
from torch.utils.data import Dataset
from transformProtein import transformProtein
import numpy as np
import pickle

class ProteinDataset(Dataset):

    def __init__(self, pklpath, firstAAidx, transformFull=None, transformPartial=None, transformNone=None, evalTransform=None):
#        with open(pklpath, 'rb') as handle:
#            self.data_chunk = pickle.load(handle)
        self.data_chunk = pklpath
        
        self.transformFull = transformFull
        self.transformPartial = transformPartial
        self.transformNone = transformNone
        self.evalTransform = evalTransform
        self.firstAAidx = firstAAidx
        
        self.trainmode=False
        if self.evalTransform==None: self.trainmode=True

    def __len__(self):
        return len(self.data_chunk)

    def __getitem__(self, idx):
        if self.trainmode:
            randnum = np.random.random()
            transformObj = self.transformNone
            if randnum>0.25:
                transformObj = self.transformFull
            elif randnum>0.1:
                transformObj = self.transformPartial
        else:
            transformObj = self.evalTransform

        sample_arr, existence, padIndex = transformObj.transformSample( self.data_chunk[idx].strip() )
        sample_arr = np.array(sample_arr)
        inputs = sample_arr[:-1]
        outputs = sample_arr[1:]

        if existence in set({0,1}):
            existence = 2
        else:
            existence = 1
        
        begAAindex = np.argwhere(inputs>=self.firstAAidx)[0][0]
        
        inputs = torch.from_numpy(inputs)
        outputs = torch.from_numpy(outputs)
#        print ("inputs_protein_data,outputs_protein_data",inputs,outputs)
        return inputs, outputs, existence, padIndex, begAAindex
