import pickle
import os
import numpy as np
#import matplotlib.pyplot as plt
import time
import random

mapfold = 'mapping_files/'
with open(os.path.join(mapfold,'kw_to_ctrl_idx.p2'),'rb') as handle:
    kw_to_ctrl = pickle.load(handle)

with open(os.path.join(mapfold,'kw_to_name.p2'),'rb') as handle:
    kw_to_name = pickle.load(handle)

with open(os.path.join(mapfold,'kw_to_lineage.p2'),'rb') as handle:
    kw_to_lineage = pickle.load(handle)
key_value_list = list(kw_to_lineage.items())
#print (key_value_list)
antimicrobial_keys = [key for key, value in kw_to_name.items() if value == 'Antimicrobial']
antimicrobial_ctrl_id = kw_to_ctrl[antimicrobial_keys[0]]
print ("antimicrobial_ctrl_id", antimicrobial_ctrl_id)
class transformProtein:
    def __init__(self, mapfold = 'mapping_files/', maxSampleLength = 512, selectSwiss = 1.0,
                 selectTrembl = 1.0, verbose = False, maxTaxaPerSample = 3, 
                 maxKwPerSample = 5, dropRate = 0.0, seqonly = False, noflipseq = False):

        self.maxSampleLength = maxSampleLength
        self.selectSwiss = selectSwiss
        self.selectTrembl = selectTrembl
        self.verbose = verbose
        self.maxTaxaPerSample = maxTaxaPerSample
        self.maxKwPerSample = maxKwPerSample
        self.dropRate = dropRate
        self.seqonly = seqonly
        self.noflipseq = noflipseq

        if self.seqonly:
            with open(os.path.join(mapfold,'aa_to_ctrl_idx_seqonly.p'),'rb') as handle:
                self.aa_to_ctrl = pickle.load(handle)
            print('Sequence only - no CTRL codes')
            self.oneEncoderLength = max(self.aa_to_ctrl.values())+1
        else:
            with open(os.path.join(mapfold,'kw_to_ctrl_idx.p'),'rb') as handle:
                self.kw_to_ctrl = pickle.load(handle)
            with open(os.path.join(mapfold,'taxa_to_ctrl_idx.p'),'rb') as handle:
                self.taxa_to_ctrl = pickle.load(handle)
            with open(os.path.join(mapfold,'aa_to_ctrl_idx.p'),'rb') as handle:
                self.aa_to_ctrl = pickle.load(handle)
            with open(os.path.join(mapfold,'taxa_to_parents.p'),'rb') as handle: # TODO: remove
                self.taxa_to_parents = pickle.load(handle)
            with open(os.path.join(mapfold,'kw_to_parents.p'),'rb') as handle: # TODO: remove
                self.kw_to_parents = pickle.load(handle)
            with open(os.path.join(mapfold,'kw_to_lineage.p'),'rb') as handle:
                self.kw_to_lineage = pickle.load(handle)
            with open(os.path.join(mapfold,'taxa_to_lineage.p'),'rb') as handle:
                self.taxa_to_lineage = pickle.load(handle)
    
            self.oneEncoderLength = max(max(self.kw_to_ctrl.values()),max(self.taxa_to_ctrl.values()),max(self.aa_to_ctrl.values())) + 1
            print('Using one unified encoder to represent protein sample with length', self.oneEncoderLength)
    
    def transformSeq(self, seq):
        """
        Transform the amino acid sequence. Currently only reverses seq--eventually include substitutions/dropout
        """
        if self.noflipseq:
            return seq
        if np.random.random()>=0.5:
            seq = seq[::-1]
        return seq

    def transformSample(self, proteinDict, justidx = True):
        """
        Function to transform/augment a sample.
        If it's not in swiss or trembl, existence set to 3. Or if it's found in swiss/trembl and you sample an other taxa.
        Padding with all zeros
        Returns an encoded sample (taxa's,kw's,sequence) and the existence level
        """
        existence = 3
        seq = proteinDict
#        seq = self.transformSeq(proteinDict['seq'])
            
        encodedSample = []
        if self.oneEncoderLength:
            if justidx:
#                for tax_line in taxa:
#                    encodedSample.extend([self.taxa_to_ctrl[tax] for tax in tax_line])
#                for kw_line in kws:
#                    encodedSample.extend([self.kw_to_ctrl[kw] for kw in kw_line])
                seq_idx = 0
                encodedSample.append(antimicrobial_ctrl_id)
                while (len(encodedSample)<self.maxSampleLength) and (seq_idx<len(seq)):
                    encodedSample.append(self.aa_to_ctrl[seq[seq_idx]])
                    seq_idx += 1
                thePadIndex = len(encodedSample)
                while len(encodedSample)<self.maxSampleLength: # add PAD (index is length of vocab)
                    encodedSample.append(self.oneEncoderLength)
        
        return encodedSample, existence, thePadIndex

if __name__ == "__main__":
    chunknum = 0
    with open('/export/share/amadani/protein-data/train_test/train'+str(chunknum)+'.p','rb') as handle:
        train_chunk = pickle.load(handle)
    uid = 'UPI000000BF1A'
    #uid = random.sample(train_chunk.keys(),1)[0]
    obj = transformProtein(verbose=True, dropRate = 0.0, maxTaxaPerSample = 3, maxKwPerSample = 5, 
                           selectSwiss = 1.0, selectTrembl = 1.0, seqonly = True)
    print(obj.transformSample(train_chunk[uid]))
