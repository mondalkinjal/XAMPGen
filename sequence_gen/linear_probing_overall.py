import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import pickle
import os
import re
from tqdm import tqdm
import argparse
import torch.nn.functional as F
import fast_transformer_linear_probing
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from abc import ABC, abstractmethod
import platform
from peft import LoraConfig, get_peft_model, TaskType
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


class Dataloader(ABC):
    @abstractmethod
    def read_file(self, filepath: Path):
        pass

class CSVLoader(Dataloader):
    def read_file(self, filepath: Path) -> pd.DataFrame:
        return pd.read_csv(filepath)

class DataLoaderContext:
    def __init__(self, data_loader: Dataloader) -> None:
        self._data_loader = data_loader

    def read_file(self, filepath: Path) -> pd.DataFrame:
        return self._data_loader.read_file(filepath)

class ProteinTokenizer:

    def __init__(self, aa_to_ctrl_idx: Dict[str, int], antimicrobial_ctrl_id: int, padding_id:int,  seq_length: int = 150):
        self.aa_to_ctrl_idx = aa_to_ctrl_idx
        self.antimicrobial_ctrl_id = antimicrobial_ctrl_id
        self.max_length = seq_length
        self.padding_id = padding_id
    def tokenize_sequence(self, sequence: str) -> List[int]:
        tokens = [self.antimicrobial_ctrl_id]
        aa_mask = [1]
        seq_idx = 0
        while (len(tokens)<self.max_length) and (seq_idx<len(sequence)):
            tokens.append(self.aa_to_ctrl_idx[sequence[seq_idx]])
            seq_idx += 1
            aa_mask.append(0)
        while len(tokens)<self.max_length:
            tokens.append(self.padding_id)
            aa_mask.append(1)
        return {'tokens': tokens,'aa_mask': aa_mask}
    def batch_tokenize(self, sequences: List[str]) -> np.ndarray:
        all_tokens = []
        all_aa_masks = []
        for seq in sequences:
            result = self.tokenize_sequence(seq)
            all_tokens.append(result['tokens'])
            all_aa_masks.append(result['aa_mask'])

        return {'input_ids': np.array(all_tokens, dtype=np.int64), 'aa_mask': np.array(all_aa_masks, dtype=np.int64)}
    

pattern = re.compile('[^ARNDCQEGHILKMFPSTWYV]')

class DatasetValidator:
    def processing_dataset(self, dataset: pd.DataFrame, output_setting: Dict):
        filtered_df = self.filter_sequences_with_non_natural_amino_acids(dataset, output_setting)
        transform_df = self.transform(filtered_df)
        return transform_df

    @staticmethod
    def not_sequences_with_non_natural_amino_acids(sequence):
        return pattern.search(sequence) is not None

    def filter_sequences_with_non_natural_amino_acids(self, dataset: pd.DataFrame, output_setting: Dict) -> pd.DataFrame:
        csv_file = output_setting['sequences_with_non_natural_amino_acids_file']
        sequence_df = dataset.copy()
        sequences_to_exclude_mask = sequence_df['sequence'].apply(self.not_sequences_with_non_natural_amino_acids)
        sequences_to_exclude = sequence_df[sequences_to_exclude_mask]

        if not sequences_to_exclude.empty:
            sequences_to_exclude.to_csv(csv_file, index=False)
            sequence_df = sequence_df.drop(sequences_to_exclude.index)
            print(f"Sequences with non-natural amino acids excluded: {csv_file}")
        return sequence_df

    @staticmethod
    def transform(dataset) -> pd.DataFrame:
        sequence_df = dataset.copy()
        return sequence_df.assign(length=sequence_df['sequence'].str.len())


class DatasetValidatorContext:
    def __init__(self, dataset_validator: DatasetValidator) -> None:
        self._dataset_validator = dataset_validator

    def processing_dataset(self, dataset: pd.DataFrame, output_setting: Dict):
        return self._dataset_validator.processing_dataset(dataset, output_setting)

def getting_data_by_partition(data: pd.DataFrame) -> List:
    train_data = data[data['partition'] == 1].copy()
    val_data = data[data['partition'] == 2].copy()
    test_data = data[data['partition'] == 3].copy()
    return train_data, val_data, test_data

class ProteinTokenizer:
    def __init__(self, aa_to_ctrl_idx: Dict[str, int], antimicrobial_ctrl_id: int, padding_id:int,  seq_length: int = 150):
        self.aa_to_ctrl_idx = aa_to_ctrl_idx
        self.antimicrobial_ctrl_id = antimicrobial_ctrl_id
        self.max_length = seq_length
        self.padding_id = padding_id
    def tokenize_sequence(self, sequence: str) -> List[int]:
        tokens = [self.antimicrobial_ctrl_id]
        aa_mask = [1]
        seq_idx = 0
        while (len(tokens)<self.max_length) and (seq_idx<len(sequence)):
            tokens.append(self.aa_to_ctrl_idx[sequence[seq_idx]])
            seq_idx += 1
            aa_mask.append(0)
        while len(tokens)<self.max_length:
            tokens.append(self.padding_id)
            aa_mask.append(1)
        return {'tokens': tokens,'aa_mask': aa_mask}
    def batch_tokenize(self, sequences: List[str]) -> np.ndarray:
        all_tokens = []
        all_aa_masks = []
        for seq in sequences:
            result = self.tokenize_sequence(seq)
            all_tokens.append(result['tokens'])
            all_aa_masks.append(result['aa_mask'])
        return {'input_ids': np.array(all_tokens, dtype=np.int64), 'aa_mask': np.array(all_aa_masks, dtype=np.int64)}

class TiedEmbeddingSoftmax(nn.Module):
    def __init__(self, vocab_size, embedding_size, **kwargs):
        super(TiedEmbeddingSoftmax, self).__init__()
        self.w = nn.Parameter(torch.normal(0., 1e-2, size=(vocab_size, embedding_size)))
        self.b = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, inputs, embed=True):
        if embed:
            return nn.functional.embedding(inputs, self.w)
        else:
            return torch.tensordot(inputs, self.w.t(), 1) + self.b

class CTRLmodel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=1280):
        super(CTRLmodel,self).__init__()
        self.tied_embedding_softmax = TiedEmbeddingSoftmax(vocab_size, embedding_dim)
        self.encoder = fast_transformer_linear_probing.Encoder()
        self.num_layers = self.encoder.num_layers

    def forward(self, inputs, past=None):
        x = self.tied_embedding_softmax(inputs, embed=True)
        x, all_attn, past_outputs = self.encoder(x, past)
        x = self.tied_embedding_softmax(x, embed=False)
        return x, all_attn, past_outputs

    def loadCheckpoint(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
        self.tied_embedding_softmax.load_state_dict(checkpoint['softmax'])
        self.encoder.load_state_dict(checkpoint['encoder'])

class EmbeddingExtractor:
    def __init__(self, checkpoint_path: str, vocab_size: int, device: str = 'cuda'):
        base_model = CTRLmodel(vocab_size=vocab_size, embedding_dim=1280)
        lora_config = LoraConfig(r=16, lora_alpha=32,  lora_dropout=0.05, target_modules=["Wq", "Wk"], bias="none")
        self.model = get_peft_model(base_model, lora_config)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.cuda()
        self.model.eval()
        self.num_layers = self.model.num_layers
        self.device = device
    def extract_all_layers_mean_pool(self, sequences: List[str], tokenizer: ProteinTokenizer, batch_size: int = 32):
        all_layers_embeddings = {i: [] for i in range(self.num_layers)}
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting from all layers"):
            batch_seqs = sequences[i:i+batch_size]
            tokenizer_result = tokenizer.batch_tokenize(batch_seqs)
            input_ids = tokenizer_result['input_ids']
            aa_mask = tokenizer_result['aa_mask']
            input_tensor = torch.tensor(input_ids).to(self.device)
            with torch.no_grad():
                _, _, all_layer_outputs = self.model(input_tensor, past=None)
                for layer_idx, layer_embeddings in enumerate(all_layer_outputs):
                    embeddings_np = layer_embeddings.cpu().numpy()
                    batch_pooled = []
                    for j in range(len(batch_seqs)):
                        aa_indices = np.where(aa_mask[j] == 0)[0]
                        aa_embeddings = embeddings_np[j, aa_indices]
                        pooled = np.mean(aa_embeddings, axis=0)
                        batch_pooled.append(pooled)
                    all_layers_embeddings[layer_idx].extend(batch_pooled)
        for layer_idx in all_layers_embeddings:
            all_layers_embeddings[layer_idx] = np.array(all_layers_embeddings[layer_idx])
        return all_layers_embeddings

class ClassificationMetrics:
    """Metrics functions matching your code structure"""

    @staticmethod
    def matthews_correlation_coefficient(y_true, y_pred):
        """Calculate Matthews Correlation Coefficient"""
        return matthews_corrcoef(y_true, y_pred)

    @staticmethod
    def accuracy(y_true, y_pred):
        """Calculate accuracy"""
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def roc_auc(y_true, y_score):
        """Calculate ROC-AUC"""
        return roc_auc_score(y_true, y_score)

    @staticmethod
    def sensitivity(y_true, y_pred):
        """Calculate sensitivity (recall for positive class)"""
        from sklearn.metrics import recall_score
        return recall_score(y_true, y_pred, pos_label=1)

    @staticmethod
    def specificity(y_true, y_pred):
        """Calculate specificity (recall for negative class)"""
        from sklearn.metrics import recall_score
        return recall_score(y_true, y_pred, pos_label=0)

classification_metrics = ClassificationMetrics()

class DNNClassifier(nn.Module):
    """DNN classifier for binary classification with CrossEntropyLoss"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.3):
        super(DNNClassifier, self).__init__()

        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer: 2 classes for CrossEntropyLoss
        layers.append(nn.Linear(prev_dim, 2))

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.layer_norm(x)
        return self.model(x)

def train_and_evaluate_layer_cls(X_train, y_train, X_val, y_val, X_test, y_test, input_dim: int, device: str = 'cuda', epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001, output_dir: str = None, layer_idx: int = 0):
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    train_dataset = TensorDataset(torch.FloatTensor(X_train).to(device), y_train_tensor)
    val_dataset = TensorDataset(torch.FloatTensor(X_val).to(device), y_val_tensor)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = DNNClassifier(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    criterion = nn.CrossEntropyLoss()

    best_mcc = -2
    current_model = {}
    model_with_best_mcc = {}
    bar = tqdm(total=epochs, desc=f"Training Layer {layer_idx}:")
    metrics_data = []
    model_params = {'input_dim': input_dim, 'learning_rate': learning_rate, 'batch_size': batch_size, 'epochs': epochs}
    for epoch in range(1, epochs + 1):
        model.train()
        arr_loss = []
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            output = model(X_batch)  # [batch_size, 2]
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            arr_loss.append(loss.item())
        current_model['epoch'] = epoch
        current_model['model'] = model
        current_model['model_state_dict'] = model.state_dict()
        current_model['optimizer_state_dict'] = optimizer.state_dict()
        current_model['scheduler_state_dict'] = scheduler.state_dict()
        current_model['parameters'] = model_params
        model.eval()
        with torch.no_grad():
            total_num = 0
            total_correct = 0
            y_score = []
            y_true = []
            y_pred = []
            arr_loss = []
            for X_batch, y_batch in val_dataloader:
                output = model(X_batch)
                out = output  # [batch_size, 2]

                loss = criterion(out, y_batch)
                arr_loss.append(loss.item())

                pred = out.argmax(dim=1)
                score = F.softmax(out, dim=1)[:, 1]  # Probability of class 1
                correct = (pred == y_batch).sum().float()
                total_correct += correct
                total_num += len(y_batch)
                y_pred.extend(pred.cpu().detach().data.numpy())
                y_score.extend(score.cpu().detach().data.numpy())
                y_true.extend(y_batch.cpu().detach().data.numpy())

            train_loss = np.mean(arr_loss)
            val_loss = np.mean(arr_loss)
            mcc = classification_metrics.matthews_correlation_coefficient(y_true=y_true, y_pred=y_pred)
            acc = classification_metrics.accuracy(y_true=y_true, y_pred=y_pred)
            auc = classification_metrics.roc_auc(y_true=y_true, y_score=y_score)
            recall_pos = classification_metrics.sensitivity(y_true=y_true, y_pred=y_pred)
            recall_neg = classification_metrics.specificity(y_true=y_true, y_pred=y_pred)
            bar.update(1)
            bar.set_postfix(Epoch=f"{epoch}", Training_Loss=f"{train_loss:.4f}", Validation_Loss=f"{val_loss:.4f}", Validation_MCC=f"{mcc:.4f}", Validation_ACC=f"{acc:.4f}", Validation_AUC=f"{auc:.4f}" if auc else "N/A", Validation_Recall_Pos=f"{recall_pos:.4f}", Validation_Recall_Neg=f"{recall_neg:.4f}")
            metrics_data.append({ 'Epoch': epoch, 'Loss/Train': train_loss, 'Loss/Val': val_loss, 'MCC/Val': mcc, 'ACC/Val': acc, 'AUC/Val': auc, 'Recall_Pos/Val': recall_pos, 'Recall_Neg/Val': recall_neg,})

            if mcc > best_mcc:
                best_mcc = mcc
                acc_of_the_best_mcc = acc
                auc_of_the_best_mcc = auc
                recall_pos_of_the_best_mcc = recall_pos
                recall_neg_of_the_best_mcc = recall_neg
                val_loss_of_the_best_mcc = val_loss
                train_loss_of_the_best_mcc = train_loss
                epoch_of_the_best_mcc = epoch
                model_with_best_mcc = current_model.copy()
        scheduler.step()

    bar.close()

    csv_file = os.path.join(output_dir, f'layer_{layer_idx}_metrics.csv')
    columns = ['Epoch', 'Loss/Train', 'Loss/Val', 'MCC/Val', 'ACC/Val', 'AUC/Val', 'Recall_Pos/Val', 'Recall_Neg/Val']
    metrics_df = pd.DataFrame(metrics_data, columns=columns)
    metrics_df.to_csv(csv_file, index=False)

    final_model_path = os.path.join(output_dir, f'layer_{layer_idx}_final_model.pth')
    torch.save(current_model, final_model_path)

        # Save best model
    best_model_path = os.path.join(output_dir, f'layer_{layer_idx}_best_model.pth')
    torch.save(model_with_best_mcc, best_model_path)

    model.load_state_dict(model_with_best_mcc['model_state_dict'])
    model.eval()

    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        test_output = model(X_test_tensor)
        test_pred = test_output.argmax(dim=1).cpu().numpy()
        test_score = F.softmax(test_output, dim=1)[:, 1].cpu().numpy()
    test_mcc = classification_metrics.matthews_correlation_coefficient(y_test, test_pred)
    test_acc = classification_metrics.accuracy(y_test, test_pred)
    test_auc = classification_metrics.roc_auc(y_test, test_score)

    test_recall_pos = classification_metrics.sensitivity(y_test, test_pred)
    test_recall_neg = classification_metrics.specificity(y_test, test_pred)
    print(f"\n  Layer {layer_idx} Results:")
    print(f"    Best Epoch: {epoch_of_the_best_mcc}")
    print(f"    Best Val MCC: {best_mcc:.4f}")
    print(f"    Test MCC: {test_mcc:.4f}")
    print(f"    Test Acc: {test_acc:.4f}")

    return {
        'best_epoch': epoch_of_the_best_mcc,
        'best_val_mcc': best_mcc,
        'best_val_acc': acc_of_the_best_mcc,
        'test_mcc': test_mcc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_recall_pos': test_recall_pos,
        'test_recall_neg': test_recall_neg,
        'metrics_data': metrics_data
    }

def run_classification(csv_file_path:str, model_checkpoint_path:str, batch_size:int, output_dir:str, epochs:int):

    csv_loader = CSVLoader()
    loader_context = DataLoaderContext(csv_loader)
    raw_data = loader_context.read_file(Path(csv_file_path))
    output_setting = {
        'sequences_with_non_natural_amino_acids_file': Path(output_dir) / 'invalid_sequences.csv',
        'duplicated_sequence_ids_file': Path(output_dir) / 'duplicate_ids.csv',
        'sequences_with_erroneous_activity_file': Path(output_dir) / 'invalid_activities.csv',
        'sequences_with_baseless_partitions_file': Path(output_dir) / 'invalid_partitions.csv'
    }
    validator = DatasetValidator()
    validator_context = DatasetValidatorContext(validator)
    valid_data = validator_context.processing_dataset(raw_data, output_setting)

    train_data, val_data, test_data = getting_data_by_partition(valid_data)
    mapfold = 'mapping_files/'
    with open(os.path.join(mapfold,'kw_to_ctrl_idx.p2'),'rb') as handle:
        kw_to_ctrl = pickle.load(handle)
    with open(os.path.join(mapfold,'taxa_to_ctrl_idx.p'),'rb') as handle:
        taxa_to_ctrl = pickle.load(handle)
    with open(os.path.join(mapfold,'aa_to_ctrl_idx.p'),'rb') as handle:
        aa_to_ctrl = pickle.load(handle)
    with open(os.path.join(mapfold,'kw_to_name.p2'),'rb') as handle:
        kw_to_name = pickle.load(handle)

    with open(os.path.join(mapfold,'kw_to_lineage.p2'),'rb') as handle:
        kw_to_lineage = pickle.load(handle)
    key_value_list = list(kw_to_lineage.items())
    oneEncoderLength = max(max(kw_to_ctrl.values()),max(taxa_to_ctrl.values()),max(aa_to_ctrl.values())) + 1
    antimicrobial_keys = [key for key, value in kw_to_name.items() if value == 'Antimicrobial']
    antimicrobial_ctrl_id = kw_to_ctrl[antimicrobial_keys[0]]
    padding_id = max(max(kw_to_ctrl.values()),max(taxa_to_ctrl.values()),max(aa_to_ctrl.values())) + 1
    tokenizer = ProteinTokenizer(aa_to_ctrl_idx=aa_to_ctrl, antimicrobial_ctrl_id=antimicrobial_ctrl_id, padding_id=padding_id, seq_length=150)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_loc = 'mapping_files/vocab.txt'
    use_py3 = platform.python_version()[0] == '3'
    vocab = open(vocab_loc).readlines() if not use_py3 else open(vocab_loc, encoding='utf-8').read().split('\n')[:-1]
    vocab = list(map(lambda x: x.split(' ')[0], vocab))
    vocab_size = len(vocab)
    print('-----vocab size',vocab_size,'------')
    word2idx = {u:i for i, u in enumerate(vocab)}
    idx2word = np.array(vocab)
    embedding_dim = 1280
    learning_rate = 0.001

    extractor = EmbeddingExtractor(model_checkpoint_path, vocab_size, device)
    train_layers_embeddings = extractor.extract_all_layers_mean_pool(sequences=train_data['sequence'].tolist(), tokenizer=tokenizer,batch_size=batch_size)
    y_train = train_data['activity'].values
    val_layers_embeddings = extractor.extract_all_layers_mean_pool(sequences=val_data['sequence'].tolist(), tokenizer=tokenizer, batch_size=batch_size)
    y_val = val_data['activity'].values
    test_layers_embeddings = extractor.extract_all_layers_mean_pool(sequences=test_data['sequence'].tolist(), tokenizer=tokenizer, batch_size=batch_size)
    y_test = test_data['activity'].values
    num_layers = len(train_layers_embeddings)
    all_results = {}
    for layer_idx in range(num_layers):
        print(f"\n  Layer {layer_idx}:")
        X_train_layer = train_layers_embeddings[layer_idx]
        X_val_layer = val_layers_embeddings[layer_idx]
        X_test_layer = test_layers_embeddings[layer_idx]
        layer_output_dir = os.path.join(output_dir, f'layer_{layer_idx}')
        os.makedirs(layer_output_dir, exist_ok=True)
        results = train_and_evaluate_layer_cls(X_train=X_train_layer,
            y_train=y_train,
            X_val=X_val_layer,
            y_val=y_val,
            X_test=X_test_layer,
            y_test=y_test,
            input_dim=X_train_layer.shape[1],
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=layer_output_dir,
            layer_idx=layer_idx
        )
        all_results[layer_idx] = results

        print(f"  Layer {layer_idx} Summary:")
        print(f"    Best Val MCC: {results['best_val_mcc']:.4f}")
        print(f"    Test MCC: {results['test_mcc']:.4f}")
        print(f"    Test Acc: {results['test_acc']:.4f}")

    print("\n7. Analyzing results...")

    # Find layer with highest test MCC
    best_layer = max(all_results.items(), key=lambda x: x[1]['test_mcc'])
    print(f"\n  [CLS] Best layer: {best_layer[0]} | Test MCC: {best_layer[1]['test_mcc']:.4f} | Test Acc: {best_layer[1]['test_acc']:.4f}")
    
    return all_results

class RegressionMetrics:
    """Metrics functions matching your code structure"""
    
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """Calculate Matthews Correlation Coefficient"""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Calculate accuracy"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def r_squared(y_true, y_pred):
        """Calculate ROC-AUC"""
        return r2_score(y_true, y_pred)

    @staticmethod
    def pcc(y_true, y_pred):
        """Calculate sensitivity (recall for positive class)"""
        pcc, _ = pearsonr(y_true, y_pred)
        return pcc

regression_metrics = RegressionMetrics()

class DNNRegressor(nn.Module):
    """DNN classifier for binary classification with CrossEntropyLoss"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 64],
                 dropout: float = 0.3):
        super(DNNRegressor, self).__init__()

        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer: 2 classes for CrossEntropyLoss
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.layer_norm(x)
        return self.model(x)

def train_and_evaluate_layer_reg(X_train, y_train, X_val, y_val, X_test, y_test, input_dim: int, device: str = 'cuda', epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001, output_dir: str = None, layer_idx: int = 0, mic_scaler: str = None):
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    train_dataset = TensorDataset(torch.FloatTensor(X_train).to(device), y_train_tensor)
    val_dataset = TensorDataset(torch.FloatTensor(X_val).to(device), y_val_tensor)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = DNNRegressor(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=5e-4)
#    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    criterion = torch.nn.MSELoss()

    best_mcc = -2
    current_model = {}
    model_with_best_mcc = {}
    bar = tqdm(total=epochs, desc=f"Training Layer {layer_idx}:")
    metrics_data = []
    model_params = {'input_dim': input_dim, 'learning_rate': learning_rate, 'batch_size': batch_size, 'epochs': epochs}
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        arr_loss = []
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            output = output.squeeze()# [batch_size, 2]
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            arr_loss.append(loss.item())
        train_loss = np.mean(arr_loss)
        current_model['epoch'] = epoch
        current_model['model'] = model
        current_model['model_state_dict'] = model.state_dict()
        current_model['optimizer_state_dict'] = optimizer.state_dict()
        current_model['parameters'] = model_params
        model.eval()
        with torch.no_grad():
            total_num = 0
            total_correct = 0
            y_score = []
            y_val_true = []
            y_val_pred = []
            arr_loss = []
            for X_batch, y_batch in val_dataloader:
                output = model(X_batch)
                output = output.squeeze()  # [batch_size, 2]

                loss = criterion(output, y_batch)
                arr_loss.append(loss.item())


                total_num += len(y_batch)
                y_val_pred.extend(output.cpu().detach().data.numpy())
                y_val_true.extend(y_batch.cpu().detach().data.numpy())


            val_loss = np.mean(arr_loss)

            mse_val = regression_metrics.mean_squared_error(y_true=y_val_true, y_pred=y_val_pred)
            rmse_val = regression_metrics.rmse(y_true=y_val_true, y_pred=y_val_pred)
            r_2_val = regression_metrics.r_squared(y_true=y_val_true, y_pred=y_val_pred)
            pcc_val = regression_metrics.pcc(y_true=y_val_true, y_pred=y_val_pred)

            bar.update(1)
            bar.set_postfix(Epoch=f"{epoch}", Training_Loss=f"{train_loss:.4f}", Validation_Loss=f"{val_loss:.4f}", Validation_mse=f"{mse_val:.4f}", Validation_rmse=f"{rmse_val:.4f}", r_2_val=f"{r_2_val:.4f}", pcc_val=f"{pcc_val:.4f}")
            metrics_data.append({ 'Epoch': epoch, 'Loss/Train': train_loss, 'Loss/Val': val_loss, 'MSE/Val': mse_val, 'RMSE/Val': rmse_val, 'R2/Val': r_2_val, 'PCC/Val': pcc_val})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_with_best_val = current_model.copy()

    bar.close()

    csv_file = os.path.join(output_dir, f'layer_{layer_idx}_metrics.csv')
    columns = ['Epoch', 'Loss/Train', 'Loss/Val', 'MSE/Val', 'RMSE/Val', 'R2/Val', 'PCC/Val']
    metrics_df = pd.DataFrame(metrics_data, columns=columns)
    metrics_df.to_csv(csv_file, index=False)

    final_model_path = os.path.join(output_dir, f'layer_{layer_idx}_final_model.pth')
    torch.save(current_model, final_model_path)

        # Save best model
    best_model_path = os.path.join(output_dir, f'layer_{layer_idx}_best_model.pth')
    torch.save(model_with_best_val, best_model_path)

    model.load_state_dict(model_with_best_val['model_state_dict'])
    model.eval()

    scaler_pkl = mic_scaler
    with open(scaler_pkl, 'rb') as f:
        scaler: pd.DataFrame = pickle.load(f)

    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        test_output = model(X_test_tensor)
        test_output = test_output.squeeze()
        test_pred = test_output.cpu().detach().data.numpy()
        test_true = y_test
    y_pred_scaled = np.array(test_pred).reshape(-1, 1)
    y_true_scaled = np.array(test_true).reshape(-1, 1)

    y_val_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_val_true = scaler.inverse_transform(y_true_scaled).flatten()

    mse_test = regression_metrics.mean_squared_error(y_true=y_val_true, y_pred=y_val_pred)
    rmse_test = regression_metrics.rmse(y_true=y_val_true, y_pred=y_val_pred)
    r_2_test = regression_metrics.r_squared(y_true=y_val_true, y_pred=y_val_pred)
    pcc_test = regression_metrics.pcc(y_true=y_val_true, y_pred=y_val_pred)
    print(f"\n  Layer {layer_idx} Results:")
    print(f"    Best Val loss: {best_val_loss:.4f}")
    print(f"    MSE test: {mse_test:.4f}")
    print(f"    RMSE test: {rmse_test:.4f}")
    print(f"    r_2_test: {r_2_test:.4f}")
    print(f"    pcc_test: {pcc_test:.4f}")

    return {
        'best_val_loss': best_val_loss,
        'MSE_test': mse_test,
        'RMSE_test': rmse_test,
        'R2_test':r_2_test,
        'PCC_test':pcc_test
    }

def run_regression(csv_file_path:str, model_checkpoint_path:str, batch_size:int, output_dir:str, epochs:int, mic_scaler:str):    
    csv_loader = CSVLoader()
    loader_context = DataLoaderContext(csv_loader)
    raw_data = loader_context.read_file(Path(csv_file_path))
    output_setting = {
        'sequences_with_non_natural_amino_acids_file': Path(output_dir) / 'invalid_sequences.csv',
        'duplicated_sequence_ids_file': Path(output_dir) / 'duplicate_ids.csv',
        'sequences_with_erroneous_activity_file': Path(output_dir) / 'invalid_activities.csv',
        'sequences_with_baseless_partitions_file': Path(output_dir) / 'invalid_partitions.csv'
    }
    validator = DatasetValidator()
    validator_context = DatasetValidatorContext(validator)
    valid_data = validator_context.processing_dataset(raw_data, output_setting)

    train_data, val_data, test_data = getting_data_by_partition(valid_data)
    mapfold = 'mapping_files/'
    with open(os.path.join(mapfold,'kw_to_ctrl_idx.p2'),'rb') as handle:
        kw_to_ctrl = pickle.load(handle)
    with open(os.path.join(mapfold,'taxa_to_ctrl_idx.p'),'rb') as handle:
        taxa_to_ctrl = pickle.load(handle)
    with open(os.path.join(mapfold,'aa_to_ctrl_idx.p'),'rb') as handle:
        aa_to_ctrl = pickle.load(handle)
    with open(os.path.join(mapfold,'kw_to_name.p2'),'rb') as handle:
        kw_to_name = pickle.load(handle)

    with open(os.path.join(mapfold,'kw_to_lineage.p2'),'rb') as handle:
        kw_to_lineage = pickle.load(handle)
    key_value_list = list(kw_to_lineage.items())
    oneEncoderLength = max(max(kw_to_ctrl.values()),max(taxa_to_ctrl.values()),max(aa_to_ctrl.values())) + 1
    antimicrobial_keys = [key for key, value in kw_to_name.items() if value == 'Antimicrobial']
    antimicrobial_ctrl_id = kw_to_ctrl[antimicrobial_keys[0]]
    padding_id = max(max(kw_to_ctrl.values()),max(taxa_to_ctrl.values()),max(aa_to_ctrl.values())) + 1
    tokenizer = ProteinTokenizer(aa_to_ctrl_idx=aa_to_ctrl, antimicrobial_ctrl_id=antimicrobial_ctrl_id, padding_id=padding_id, seq_length=150)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_loc = 'mapping_files/vocab.txt'
    use_py3 = platform.python_version()[0] == '3'
    vocab = open(vocab_loc).readlines() if not use_py3 else open(vocab_loc, encoding='utf-8').read().split('\n')[:-1]
    vocab = list(map(lambda x: x.split(' ')[0], vocab))
    vocab_size = len(vocab)
    print('-----vocab size',vocab_size,'------')
    word2idx = {u:i for i, u in enumerate(vocab)}
    idx2word = np.array(vocab)
    embedding_dim = 1280
    learning_rate = 0.001

    extractor = EmbeddingExtractor(model_checkpoint_path, vocab_size, device)
    train_layers_embeddings = extractor.extract_all_layers_mean_pool(sequences=train_data['sequence'].tolist(), tokenizer=tokenizer,batch_size=batch_size)
    y_train = train_data['activity'].values
    print (y_train, flush=True)
    val_layers_embeddings = extractor.extract_all_layers_mean_pool(sequences=val_data['sequence'].tolist(), tokenizer=tokenizer, batch_size=batch_size)
    y_val = val_data['activity'].values
    print (y_val, flush=True)
    test_layers_embeddings = extractor.extract_all_layers_mean_pool(sequences=test_data['sequence'].tolist(), tokenizer=tokenizer, batch_size=batch_size)
    y_test = test_data['activity'].values
    print (y_test, flush=True)
    num_layers = len(train_layers_embeddings)
    all_results = {}
    for layer_idx in range(num_layers):
        print(f"\n  Layer {layer_idx}:")
        X_train_layer = train_layers_embeddings[layer_idx]
        X_val_layer = val_layers_embeddings[layer_idx]
        X_test_layer = test_layers_embeddings[layer_idx]
        layer_output_dir = os.path.join(output_dir, f'layer_{layer_idx}')
        os.makedirs(layer_output_dir, exist_ok=True)
        results = train_and_evaluate_layer_reg(X_train=X_train_layer,
            y_train=y_train,
            X_val=X_val_layer,
            y_val=y_val,
            X_test=X_test_layer,
            y_test=y_test,
            input_dim=X_train_layer.shape[1],
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=layer_output_dir,
            layer_idx=layer_idx,
            mic_scaler=mic_scaler
        )
        all_results[layer_idx] = results


    print("\n7. Analyzing results...")

    # Find layer with highest test MCC
    best_layer = max(all_results.items(), key=lambda x: x[1]['R2_test'])
    
    print(f"\n  [CLS] Best layer: {best_layer[0]} | Test R2: {best_layer[1]['R2_test']:.4f} | Test PCC: {best_layer[1]['PCC_test']:.4f}")

    return all_results

def plot_combined_results(class_results, reg_results, output_dir):
    layers = sorted(class_results.keys())

    mcc_values = [class_results[l]['test_mcc'] for l in layers]
    acc_values = [class_results[l]['test_acc'] for l in layers]
    r2_values = [reg_results[l]['R2_test'] for l in layers]
    pcc_values = [reg_results[l]['PCC_test'] for l in layers]

    plt.rcParams.update({
        'font.size': 26, 'axes.titlesize': 32, 'axes.labelsize': 30,
        'xtick.labelsize': 26, 'ytick.labelsize': 26, 'legend.fontsize': 26,
        'axes.titleweight': 'bold', 'axes.linewidth': 2.5, 'figure.titlesize': 36
    })
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 20), sharex=True, gridspec_kw={'hspace': 0})
    
    mcc_min, mcc_max = min(mcc_values), max(mcc_values)
    acc_min, acc_max = min(acc_values), max(acc_values)
    overall_min = min(mcc_min, acc_min)
    overall_max = max(mcc_max, acc_max)
    y_range = overall_max - overall_min
    
    ax1.set_ylim(overall_min - y_range * 0.1, overall_max + y_range * 0.35)
    
    ax1.plot(layers, mcc_values, 'b-o', linewidth=4, markersize=12, label='MCC', markeredgecolor='darkblue')
    ax1.plot(layers, acc_values, 'g-s', linewidth=4, markersize=12, label='Accuracy', markeredgecolor='darkgreen')
    ax1.set_ylabel('Classif.\nScore', fontweight='bold', labelpad=15)
    ax1.legend(loc='upper left', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.4, linestyle='--', linewidth=2)
    ax1.margins(y=0.1)
    ax1.text(0.5, 0.93, 'A', transform=ax1.transAxes, fontsize=32, fontweight='bold', verticalalignment='bottom', horizontalalignment='center')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
#ax3.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))


    # Panel 2: Regression (R2)
    ax2.plot(layers, r2_values, 'r-^', linewidth=4, markersize=12, label='$R^2$', markeredgecolor='darkred')
    ax2.set_ylabel('$R^2$ Score', fontweight='bold', labelpad=15)
    ax2.grid(True, alpha=0.4, linestyle='--', linewidth=2)
    ax2.margins(y=0.1)
    ax2.text(0.5, 0.98, 'B', transform=ax2.transAxes, fontsize=32, fontweight='bold', verticalalignment='top')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Panel 3: Regression (PCC)
    ax3.plot(layers, pcc_values, 'm-p', linewidth=4, markersize=12, label='PCC', markeredgecolor='purple')
    ax3.set_ylabel('PCC Score', fontweight='bold', labelpad=15)
    ax3.set_xlabel('Layer Number', fontweight='bold', labelpad=15) # Label only on bottom
    ax3.grid(True, alpha=0.4, linestyle='--', linewidth=2)
    ax3.margins(y=0.1)
    ax3.text(0.5, 0.97, 'C', transform=ax3.transAxes, fontsize=32, fontweight='bold', verticalalignment='top')
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Formatting Ticks
    tick_spacing = 5
    selected_ticks = [l for l in layers if l % tick_spacing == 0]
    ax3.set_xticks(selected_ticks)

    # Hide overlapping y-ticks between panels
    plt.setp(ax1.get_yticklabels()[-1], visible=False)  # top label of panel A
    plt.setp(ax2.get_yticklabels()[-1], visible=False)  # top label of panel B
    plt.setp(ax2.get_yticklabels()[0],  visible=False)  # bottom label of panel B
    plt.setp(ax3.get_yticklabels()[0],  visible=False)  # bottom label of panel C

    fig.tight_layout(rect=[0.02, 0, 1, 1], h_pad=0)

    save_path = os.path.join(output_dir, 'combined_probing_3panel.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SUCCESS] Final Combined 3-panel plot saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Master Complete DNN probing (Classif. & Regress.)')
    
    # Accept both CSV files so you can run the whole pipeline at once
    parser.add_argument('--cls_csv_file', type=str, required=True, help='Path to Classification CSV file')
    parser.add_argument('--reg_csv_file', type=str, required=True, help='Path to Regression CSV file')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--output_dir', type=str, default='master_probing_results', help='Master Output directory')
    parser.add_argument('--mic_scaler', type=str, default='mic_scaler.pkl', help='scaling MIC')
    
    args = parser.parse_args()

    # Create directories
    cls_output_dir = os.path.join(args.output_dir, 'classification')
    reg_output_dir = os.path.join(args.output_dir, 'regression')
    os.makedirs(cls_output_dir, exist_ok=True)
    os.makedirs(reg_output_dir, exist_ok=True)

    # 1. Run Classification and grab results dictionary
    cls_results = run_classification(
        csv_file_path=args.cls_csv_file,
        model_checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        output_dir=cls_output_dir,
        epochs=args.epochs
    )

    # 2. Run Regression and grab results dictionary
    reg_results = run_regression(
        csv_file_path=args.reg_csv_file,
        model_checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        output_dir=reg_output_dir,
        epochs=args.epochs,
        mic_scaler=args.mic_scaler
    )

    # 3. Generate combined 3-panel plot
    print("\n--- GENERATING COMBINED 3-PANEL FIGURE ---")
    plot_combined_results(cls_results, reg_results, args.output_dir)

