from __future__ import print_function
from __future__ import division
import sys
import torch
import os
import tqdm
import pdb
import numpy as np
import platform
import hashlib
import pytorch_transformer_gpu
import re
import argparse
#import tensorflow as tf
#from tensorflow.python import pywrap_tensorflow
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformProtein import transformProtein
from ProteinDataset import ProteinDataset
from torch.utils.data import Dataset, DataLoader
import pickle
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
import sys
from torch import inf
sys.setrecursionlimit(100000)
def save_short_sequences(input_file_path, output_file_path, max_length=50):
    with open(input_file_path, 'r') as file:
        sequences = file.readlines()
    
    short_sequences = [seq.strip() for seq in sequences if len(seq.strip()) < max_length]
    
    with open(output_file_path, 'w') as output_file:
        for seq in short_sequences:
            output_file.write(seq + '\n')

# Example usage:
input_file_path = 'unique_combined_sequences.txt'  # Replace with your input file path
output_file_path = 'short_sequences.txt'  # Replace with your desired output file path

save_short_sequences(input_file_path, output_file_path)
print(f"Sequences less than 50 characters have been saved to {output_file_path}")

def divide_file_into_chunks(filename, chunks):
    # Read the file content
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Calculate the chunk size
    total_lines = len(lines)
    chunk_size = total_lines // chunks
    remainder = total_lines % chunks

    # List to store the chunks
    chunks_list = []

    start = 0
    for i in range(chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk_lines = lines[start:end]
        
        # Append the chunk to the list
        chunks_list.append(chunk_lines)
        
        start = end

    return chunks_list

# Replace 'example.txt' with the path to your file
chunks = divide_file_into_chunks('unique_combined_sequences.txt', 1)

use_py3 = platform.python_version()[0] == '3'

parser = argparse.ArgumentParser(description='TensorFlow code for generating from CTRL')
parser.add_argument('--model_dir', type =str, default='pretrain_progen_full.pth',
                                        help='location of training model checkpoint')
parser.add_argument('--finetuned_model_dir', type =str, default='finetuned_progen.pth',                                                                                            help='location of training model checkpoint')
#parser.add_argument('--model_path', type=str, default='/home/amadani/ctrl/ckpt/seqlen256_36layers_v0.ckpt/model.ckpt-684000', help='location of model *data* checkpoint to load; this is NOT the directory but rather the model checkpoint')
parser.add_argument('--seed', type=int, default=313,
                                        help='random seed for TensorFlow, numpy and PythonHash')
parser.add_argument('--sequence_len', type=int, default=150,
                                        help='sequence len of model being fine-tuned')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--num_layers', type=int, default=36, help='number of transfomer layers. used for loading checkpoint')
parser.add_argument('--batch_size', type=int, default =64, help='batch size for dataloader')
parser.add_argument('--vocab_loc', type=str, default='mapping_files/vocab.txt', help='vocab location')
parser.add_argument('--num_workers', type=int, default=0, help='for dataloader')
parser.add_argument('--warmup_iteration', type=int, default=1000, help='LR warmup cutoff')
parser.add_argument('--save_iter', type=int, default=10, help='save model checkpoint every X iterations')

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '12355'

np.random.seed(args.seed)

# load the vocabulary from file
vocab = open(args.vocab_loc).readlines() if not use_py3 else open(args.vocab_loc, encoding='utf-8').read().split('\n')[:-1]
vocab = list(map(lambda x: x.split(' ')[0], vocab))
# length of the vocabulary
vocab_size = len(vocab)
print('-----vocab size',vocab_size,'------')

# define the numericalization map
# idx2word maps the numericalized ID to the word
# word2idx maps the word to the numericalized ID
word2idx = {u:i for i, u in enumerate(vocab)}
idx2word = np.array(vocab)

# sequence length to use for transfomer
seq_length = args.sequence_len

embedding_dim = 1280

class TiedEmbeddingSoftmax(torch.nn.Module):

  def __init__(self, vocab_size=vocab_size, embedding_size=embedding_dim, **kwargs):
    super(TiedEmbeddingSoftmax, self).__init__()
    self.w = torch.nn.Parameter(torch.normal(0., 1e-2, size=(vocab_size, embedding_size)))
    self.b = torch.nn.Parameter(torch.zeros(vocab_size))

  def forward(self, inputs, embed=True):
    if embed:
      return torch.nn.functional.embedding(inputs, self.w)
    else:
      return torch.tensordot(inputs, self.w.t(), 1) + self.b

class CTRLmodel(torch.nn.Module):
  def __init__(self):
    super(CTRLmodel,self).__init__()
    self.tied_embedding_softmax = TiedEmbeddingSoftmax()
    self.encoder = pytorch_transformer_gpu.Encoder()

  def forward(self, inputs):
    x = self.tied_embedding_softmax(inputs, embed = True)
    print ("shape_of_x",x.shape)
    x = self.encoder(x)
    x = self.tied_embedding_softmax(x, embed = False)
    return x

  def loadCheckpoint(self, model_dir):
#    pytorch_model_hash = hashlib.md5(model_path.encode('utf-8')).hexdigest()

#    if os.path.exists(pytorch_model_hash):
#      print('Found PyTorch checkpoint @', pytorch_model_hash)
      print('Loading instead of converting from TensorFlow')
#      checkpoint = torch.load(model_dir)
      checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
      self.tied_embedding_softmax.load_state_dict(checkpoint['softmax'])
      self.encoder.load_state_dict(checkpoint['encoder'])

#      self.tied_embedding_softmax.to('cuda')
#      self.encoder.to('cuda')

#    else:
#      print('Could not find PyTorch checkpoint')
#      print('Converting weights and will store the PyTorch checkpoint as ', pytorch_model_hash)
#      chkpt_for_reader = model_path # '.'.join(model_path.split('.')[:-1])
#      reader = pywrap_tensorflow.NewCheckpointReader(chkpt_for_reader)
      #self.tied_embedding_softmax.w = torch.nn.Parameter(torch.tensor(reader.get_tensor('w')).to('cuda'))
     #self.tied_embedding_softmax.b = torch.nn.Parameter(torch.tensor(reader.get_tensor('b')).to('cuda'))

#      list_of_variables = list(filter(lambda x: 'Adagrad' not in x, reader.get_variable_to_shape_map().keys()))

#      str2parameter = lambda x: torch.nn.Parameter(torch.tensor(reader.get_tensor(x)).t().to('cuda'))
#      self.encoder.layernorm.weight = str2parameter('encoder/layer_normalization_'+str(int(num_layers*2))+'/gamma')
#      self.encoder.layernorm.bias = str2parameter('encoder/layer_normalization_'+str(int(num_layers*2))+'/beta')
#      for i in tqdm.tqdm(range(num_layers)):
#        if i==0:
#          layer_variables = sorted(filter(lambda x: 'layer/' in x, list_of_variables))
#        else:
#          layer_variables = sorted(filter(lambda x: 'layer_'+str(i)+'/' in x, list_of_variables))

#        current_layer = getattr(self.encoder, 'layer'+str(i))

#        current_layer.layernorm1.bias = str2parameter(layer_variables[0])
#        current_layer.layernorm1.weight = str2parameter(layer_variables[1])

#        current_layer.layernorm2.bias = str2parameter(layer_variables[2])
#        current_layer.layernorm2.weight = str2parameter(layer_variables[3])


#        current_layer.multi_head_attention.Wq.bias = str2parameter(layer_variables[4])
#        current_layer.multi_head_attention.Wq.weight = str2parameter(layer_variables[5])
#        current_layer.multi_head_attention.Wk.bias = str2parameter(layer_variables[6])
#        current_layer.multi_head_attention.Wk.weight = str2parameter(layer_variables[7])
#        current_layer.multi_head_attention.Wv.bias = str2parameter(layer_variables[8])
#        current_layer.multi_head_attention.Wv.weight = str2parameter(layer_variables[9])
#        current_layer.multi_head_attention.dense.bias = str2parameter(layer_variables[10])
#        current_layer.multi_head_attention.dense.weight = str2parameter(layer_variables[11])
#        current_layer.ffn[0].bias = str2parameter(layer_variables[12])
#        current_layer.ffn[0].weight = str2parameter(layer_variables[13])
#        current_layer.ffn[2].bias = str2parameter(layer_variables[14])
#        current_layer.ffn[2].weight = str2parameter(layer_variables[15])
#      torch.save({
#        'softmax': self.tied_embedding_softmax.state_dict(),
#        'encoder': self.encoder.state_dict(),
#      }, pytorch_model_hash)

# initialize ctrl object
# load checkpoint with args.model_path
model = CTRLmodel()
print('model initialized')
checkpoint = torch.load(args.model_dir)
model.load_state_dict(checkpoint)
print('previous checkpoint loaded')
#for module in model.modules():
#    print(module)

#print (model.num_parameters())
#model = model.cuda()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = count_parameters(model)
print(total_params)
lora_config = LoraConfig(
    r=16,  # Low-rank dimension
    lora_alpha=32,  # Scaling parameter
    lora_dropout=0.05,  # Dropout rate
    target_modules=["Wq", "Wk"],
    bias="none"
        # Add more layers as needed
)

model = get_peft_model(model, lora_config)
trainable_params = count_parameters(model)
print (trainable_params/total_params)
#for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
# freeze all weights except embedding
#for p in model.parameters():
#    p.requires_grad=False
#model.tied_embedding_softmax.w.requires_grad=True
#model.tied_embedding_softmax.b.requires_grad=True
#trainable_params = count_parameters(model)
#print (trainable_params/total_params)
accelerator = Accelerator()
device = accelerator.device


class Trainer(object):
    def __init__(self, model, warmup_iteration, seq_length, batch_size, num_workers, vocab_size, finetuned_model_dir, save_iter):
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_size = vocab_size
        self.finetuned_model_dir = finetuned_model_dir
        self.save_iter = save_iter
        self.firstAAidx = self.vocab_size - 26 # Assuming that the pad token is the last token and AAs are at the end
        
        self.optimizer = torch.optim.Adam(self.model.parameters()) #lr, betas
        lambdafn = lambda iteration: min(iteration/(warmup_iteration*1.0),1.0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambdafn)
        
#        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocab_size-1, reduction='none')
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.transformFull = transformProtein(maxSampleLength = seq_length+1, 
                                              selectSwiss = 1.0, selectTrembl = 1.0, 
                                              maxTaxaPerSample = 3, maxKwPerSample = 5, dropRate = 0.0)
        self.transformPartial = transformProtein(maxSampleLength = seq_length+1,   
                                              selectSwiss = 0.9, selectTrembl = 0.9,
                                              maxTaxaPerSample = 3, maxKwPerSample = 5, dropRate = 0.2)
        self.transformNone = transformProtein(maxSampleLength = seq_length+1,   
                                              selectSwiss = 1.0, selectTrembl = 1.0,
                                              maxTaxaPerSample = 0, maxKwPerSample = 0, dropRate = 1.0)
        
        self.writer = SummaryWriter()

    def train(self, num_epochs):
        self.model.train()

        iter_num = 0
#        for epoch in range(num_epochs):
#            loss_e = 0.0
#            num_e = 0

        for chunknum in range(1):
                
            pkl_path=chunks[chunknum] 
            chunk_dataset = ProteinDataset(pkl_path, firstAAidx = self.firstAAidx, transformFull = self.transformFull, transformPartial = self.transformPartial, transformNone = self.transformNone)
            dataloader = DataLoader(chunk_dataset, shuffle = True, batch_size = self.batch_size, num_workers = self.num_workers, pin_memory = False) #TODO pinmem?
            self.model, self.optimizer, dataloader = accelerator.prepare(self.model, self.optimizer, dataloader)
            for epoch in range(num_epochs):
                loss_e = 0.0
                num_e = 0
                for i, (sample, labels, existence, padIndex, begAAindex) in enumerate(dataloader):
#                    print ("sample_shape, labels_shape",sample.shape, labels.shape) 
#                    print ("memory_allocated",torch.cuda.memory_allocated())
                    self.optimizer.zero_grad()
#                    sample, labels, existence, padIndex = sample.cuda(), labels.cuda(), existence.cuda(), padIndex.cuda()
                    output = self.model(sample)
#                    print ("output", output.permute(0,2,1))
#                    print ("labels", labels)
                    loss = self.criterion(output.permute(0,2,1), labels)
                    loss = torch.mean((torch.sum(loss,dim=1)/padIndex)*existence) 
#                    loss.backward()
                    accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                    self.optimizer.step()
                    self.scheduler.step()
                    loss_e += loss.item()
#                    with open('output_gpu.txt', 'a') as file:
#                        file.write(str(loss_e)+'\n')
#                        file.write(str(epoch)+'\n')
                    num_e += sample.shape[0]
                    iter_num += 1
                    self.writer.add_scalar('Loss_iteration',loss.item(),iter_num)

#                    if (i+1)%self.save_iter==0:
#                        torch.save({'epoch': epoch, 'chunknum': chunknum, 'iteration':i,
#                                    'model_state_dict': self.model.state_dict(),
#                                    'optimizer_state_dict': self.optimizer.state_dict(),
#                                    'loss': loss,
#                                   }, self.finetuned_model_dir)
                loss_e/=num_e
                print (loss_e,epoch,chunknum)
#            if epoch%self.save_iter==0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(self.model)
            accelerator.save({
                "model": unwrapped_model.state_dict(),
                "optimizer": self.optimizer.state_dict() # optimizer is an AcceleratedOptimizer object
            },self.finetuned_model_dir)
                    
#                    output_dir = f"epoch_{epoch}"
#                    accelerator.save_state(output_dir)
#                with open('output_gpu.txt', 'a') as file:
#                    file.write(str(chunknum)+'\n')
#                    file.write(str(loss_e)+'\n')
#            with open('output_gpu.txt', 'a') as file:
#                file.write(str(epoch)+'\n')
            self.writer.add_scalar('Loss_epoch',loss_e, epoch)

training = Trainer(model=model, warmup_iteration=args.warmup_iteration, seq_length=seq_length,
                   batch_size=args.batch_size, num_workers=args.num_workers, vocab_size=vocab_size,
                   finetuned_model_dir = args.finetuned_model_dir, save_iter=args.save_iter)
training.train(args.num_epochs)
#import sys
#print(sys.getrecursionlimit())


