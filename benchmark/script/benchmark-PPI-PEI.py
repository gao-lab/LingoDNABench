import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import glob
import numpy as np
import copy
import h5py
import sys
import math
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader,TensorDataset,Dataset
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, roc_curve,confusion_matrix,matthews_corrcoef,precision_recall_curve
from sklearn import metrics
from torch.backends import cudnn
import scipy.stats
from prefetch_generator import BackgroundGenerator


model_name=sys.argv[1]
data_dir=sys.argv[2]
layer=int(sys.argv[3])
random_seed=int(sys.argv[4])
output_dim=int(sys.argv[5])



num_targets=output_dim
seed = random_seed
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
'''devices'''
device=torch.device('cuda')


save_dir=f"{data_dir}/{model_name}/checkpoint"
if os.path.exists(save_dir):
    pass
else:
    os.mkdir(save_dir)


import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SingleFileDataset(Dataset):
    def __init__(self, file_path_1, file_path_2, labels):
        """
        Dataset for a single pair of HDF5 files.
        Args:
            file_path_1 (str): Path to the first HDF5 file.
            file_path_2 (str): Path to the second HDF5 file.
            labels (np.ndarray): Label array.
        """
        self.file_path_1 = file_path_1
        self.file_path_2 = file_path_2
        self.labels = labels
        with h5py.File(self.file_path_1, "r") as f1, h5py.File(self.file_path_2, "r") as f2:
            self.num_samples = len(f1["embedding"])
            assert self.num_samples == len(f2["embedding"]), "Mismatched number of samples in input files."
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        with h5py.File(self.file_path_1, "r") as f1, h5py.File(self.file_path_2, "r") as f2:
            embedding1 = f1["embedding"][idx]
            embedding2 = f2["embedding"][idx]
        label = self.labels[idx]
        return (
            torch.tensor(embedding1, dtype=torch.float32),
            torch.tensor(embedding2, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

class InterleavedDataset(Dataset):
    def __init__(self, file_paths_1, file_paths_2, labels, cycle_length):
        """
        Interleaves samples from multiple pairs of HDF5 files.
        Args:
            file_paths_1 (list of str): List of paths to the first set of HDF5 files.
            file_paths_2 (list of str): List of paths to the second set of HDF5 files.
            labels (np.ndarray): Label array.
            cycle_length (int): Number of file pairs to interleave.
        """
        assert len(file_paths_1) == len(file_paths_2), "Mismatched number of file pairs."
        self.datasets = [
            SingleFileDataset(fp1, fp2, lb) for fp1, fp2,lb in zip(file_paths_1, file_paths_2,labels)
        ]
        self.cycle_length = cycle_length
        self.lengths = [len(ds) for ds in self.datasets]
        self.total_length = sum(self.lengths)
    def __len__(self):
        return self.total_length
    def __getitem__(self, idx):
        # Determine the dataset and index within the dataset
        total_length = 0
        for i, ds_length in enumerate(self.lengths):
            if idx < total_length + ds_length:
                dataset_idx = i
                index_within_dataset = idx - total_length
                break
            total_length += ds_length
        return self.datasets[dataset_idx][index_within_dataset]


def numeric_sort_key(s):
    """Convert string to integer for sorting."""
    return int(s.split('-split-')[1].split('-')[0])

data_type = "train"
file_paths_1 = sorted(glob.glob(data_dir + f"/{model_name}/{data_type}_data_0-split-***-embedding-layer_{layer}.h5"),key=numeric_sort_key)
file_paths_2 = sorted(glob.glob(data_dir + f"/{model_name}/{data_type}_data_1-split-***-embedding-layer_{layer}.h5"),key=numeric_sort_key)
labels = np.loadtxt(f"{data_dir}/{data_type}_label.txt", dtype=int).reshape(-1,1)
import numpy as np


group_size = 5000
num_groups = len(labels) // group_size + (1 if len(labels) % group_size != 0 else 0)


split_vector = []

for i in range(num_groups - 1):
    split_vector.append(labels[i * group_size : (i + 1) * group_size])

split_vector.append(labels[(num_groups - 1) * group_size:])

# Create interleaved dataset for dual inputs
cycle_length = len(file_paths_1)  # Number of file pairs to cycle through
train_dataset = InterleavedDataset(file_paths_1, file_paths_2, split_vector, cycle_length)
batch_size=128
# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32, prefetch_factor=2)


# Prefetching DataLoader wrapper
class PrefetchDataLoader(DataLoader):
    def __iter__(self):
        # Use an iterator to preload batches in the background
        return BackgroundGenerator(super().__iter__())


import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EmbeddingDataset(Dataset):
    def __init__(self,data_type):
        """
        Args:
            file_list (list): List of HDF5 file paths.
            data_mode (str): Mode of the dataset ('train', 'valid', 'test').
            embedding_shape (tuple): Shape of the embedding (e.g., [embedding_len, model_dim]).
            label_shape (tuple): Shape of the labels.
        """
        merged_file_pattern_1=fr"{data_type}_data_0-split-***-embedding-layer_{layer}.h5"
        file_to_merge_1=glob.glob(data_dir+f"/{model_name}"+fr"/{merged_file_pattern_1}")
        file_to_merge_1 = sorted(file_to_merge_1, key=numeric_sort_key)
        merged_file_pattern_2=fr"{data_type}_data_1-split-***-embedding-layer_{layer}.h5"
        file_to_merge_2=glob.glob(data_dir+f"/{model_name}"+fr"/{merged_file_pattern_2}")
        file_to_merge_2 = sorted(file_to_merge_2, key=numeric_sort_key)
        self.label=np.loadtxt(f"{data_dir}/{data_type}_label.txt",dtype=int)
        self.file_list1 = file_to_merge_1
        self.file_list2 = file_to_merge_2
        with h5py.File(self.file_list1[0], 'r') as f:
            self.data_shape=f['embedding'][0].shape
        self.file_indices = []  # Map global index to (file_id, index_within_file)
        
        for file_id, file_path in enumerate(self.file_list1 ):
            with h5py.File(file_path, 'r') as f:
                num_samples = len(f['embedding'])
                self.data_shape=f['embedding'].shape
                self.file_indices.extend([(file_id, i) for i in range(num_samples)])
        
    def __len__(self):
        return len(self.file_indices)
    def __getitem__(self, idx):
        
        file_id, index_within_file = self.file_indices[idx]
        file_path_1 = self.file_list1[file_id]
        file_path_2 = self.file_list2[file_id]
        label=self.label[idx]
        with h5py.File(file_path_1, 'r') as f1:
            embedding1 = f1['embedding'][index_within_file]
        with h5py.File(file_path_2, 'r') as f2:
            embedding2 = f2['embedding'][index_within_file]
        embedding1 = torch.tensor(embedding1, dtype=torch.float32)
        embedding2 = torch.tensor(embedding2, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return embedding1,embedding2, label
    

valid_dataset=EmbeddingDataset("dev")
test_dataset=EmbeddingDataset("test")





def testing(test_dataloader, model):
    model.eval().cuda()    
    begin=0
    test_flag=True
    with torch.no_grad():
        for one_batch in test_dataloader:
            data1,data2 ,targets = map(lambda x: x.to(device), one_batch)
            data1=data1.to(torch.float32)
            data2=data2.to(torch.float32)
            target_pred = model(data1,data2)
            if test_flag:
                shape=target_pred.detach().cpu().numpy().shape
                y_pred=np.empty((len(test_dataloader.dataset),*shape[1:]))
                test_flag=False
            num=data1.shape[0]
            targets=targets.to(torch.float32)
            target_pred=target_pred.to(torch.float32)
            y_pred[begin:begin+num,]=target_pred.cpu().detach().numpy()
            begin+=num
    return y_pred


def validating(valid_dataloader, model, loss_fn):
    num_batches = len(valid_dataloader)
    model.eval().cuda()
    test_loss = 0
    step=0
    print("validating")    
    with torch.no_grad():
        for one_batch in valid_dataloader:
            data1,data2 ,targets = map(lambda x: x.to(device), one_batch)
            data1=data1.to(torch.float32)
            data2=data2.to(torch.float32)
            target_pred = model(data1,data2)
            target_pred=target_pred.to(torch.float32)
            targets=targets.to(torch.float32)
            loss_lm = loss_fn(target_pred, targets)
            test_loss += loss_lm.detach().item()
            step+=1
    test_loss /= num_batches
    return test_loss

class DownstreamModel(nn.Module):
    def __init__(self,in_channels):
        super(DownstreamModel, self).__init__()
        self.batchnorm1=nn.BatchNorm1d(in_channels)
        self.conv1=nn.Conv1d(in_channels,256,1)
        self.conv2=nn.Conv1d(256,256,7,padding='same')
        self.batchnorm2=nn.BatchNorm1d(in_channels)
        self.conv3=nn.Conv1d(in_channels,256,1)
        self.conv4=nn.Conv1d(256,256,7,padding='same')
        self.act=nn.GELU()
        self.linear1=nn.Linear(in_features=256*2,out_features=512)
        self.drop=nn.Dropout(0.2)
        self.linear2=nn.Linear(in_features=512,out_features=256)
        self.linear3=nn.Linear(in_features=256,out_features=output_dim)
    def forward(self,x1,x2):
        x1=x1.permute(0,2,1)
        x2=x2.permute(0,2,1)
        #convolutions
        x1=self.batchnorm1(x1)
        x2=self.batchnorm2(x2)
        x1=self.conv1(x1)
        x1=self.conv2(x1)
        x1=self.act(x1)
        x2=self.conv3(x2)
        x2=self.conv4(x2)
        x2=self.act(x2)
        #pooling
        x1=torch.nn.functional.max_pool1d(x1,kernel_size=x1.shape[2])
        x2=torch.nn.functional.max_pool1d(x2,kernel_size=x2.shape[2])
        #flatten
        x1=x1.view(x1.shape[0],-1)
        x2=x2.view(x2.shape[0],-1)
        x=torch.cat([x1,x2],dim=1)
        x=self.linear1(x)
        x=self.act(x)
        x=self.drop(x)        
        x=self.linear2(x)
        x=self.act(x)
        x=self.drop(x)  
        x=self.linear3(x)
        if (output_dim==1) and (not regression):
            x=torch.sigmoid(x)
        return x


valid_dataloader=PrefetchDataLoader(valid_dataset,batch_size,shuffle=True,num_workers=32,pin_memory=True)
test_dataloader=PrefetchDataLoader(test_dataset,batch_size,shuffle=False,num_workers=32,pin_memory=True)


model=DownstreamModel(in_channels=valid_dataset.data_shape[2])
model.to(device)

learning_rate=1e-3
epochs=80
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_fn=torch.nn.BCELoss()


patience=10
count=0
min_valid_loss=1000
final_data={}
import time
tot_time=time.time()
for epoch in range(epochs):
    start=time.time()
    model.train()
    train_epoch_loss = 0
    train_num_batches = len(train_dataloader)
    step=0
    for one_batch in train_dataloader:
        data1,data2 ,targets = map(lambda x: x.to(device), one_batch)
        data1=data1.to(torch.float32)
        data2=data2.to(torch.float32)
        target_pred = model(data1,data2)
        target_pred=target_pred.to(torch.float32)
        targets=targets.to(torch.float32)
        loss_lm = loss_fn(target_pred, targets)
        train_epoch_loss += loss_lm.detach().item()
        print("train loss",end='\t')
        print(loss_lm.detach().item())
        loss_lm.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        print("time used")
        print(time.time()-start)
        start=time.time()
    print(time.time()-tot_time)
    #train epoch loss
    train_epoch_loss /= train_num_batches
    print(f"train_loss\t{train_epoch_loss}")
    #validating
    start=time.time()
    valid_loss = validating(valid_dataloader, model, loss_fn)
    print(time.time()-start)
    print(f"valid_loss\t{valid_loss}")
    current_state={'epoch': epoch,
                    'model_state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                    'step': step,
                    'train_loss': train_epoch_loss,
                    'valid_loss': valid_loss,
                    }
    
    if valid_loss<min_valid_loss:
        print('update')
        min_valid_loss=valid_loss
        final_data=copy.deepcopy(current_state)
        count=0
    
    count+=1
    if count==patience:
        break

print(time.time()-tot_time)


test_model=DownstreamModel(in_channels=valid_dataset.data_shape[2])
test_model.to(device)
test_model.load_state_dict(final_data['model_state_dict'])

y_pred=testing(test_dataloader,test_model)

y_test=test_dataset.label
y_pred=y_pred.reshape(-1)
with open(f"{save_dir}/metrics-{model_name}-{layer}.txt",'a') as f:
    
    auc=roc_auc_score(y_test,y_pred) 
    print(auc)
    y_prediction=y_pred
    y_true=y_test
    precision, recall, _ = precision_recall_curve(y_true, y_prediction)
    pr_auc = metrics.auc(recall, precision)
    fpr, tpr, thresholds = roc_curve(y_true,y_prediction)
    youden = tpr-fpr
    cutoff = thresholds[np.argmax(youden)]
    y_prediction[y_prediction<cutoff]=0
    y_prediction[y_prediction>=cutoff]=1
    recall=recall_score(y_true,y_prediction)
    accuracy=accuracy_score(y_true,y_prediction)
    precision=precision_score(y_true,y_prediction)
    mcc=matthews_corrcoef(y_true,y_prediction)
    f1=f1_score(y_true,y_prediction)
    f.write(f"{model_name}\t{random_seed}\t{cutoff}\t{accuracy}\t{precision}\t{recall}\t{f1}\t{auc}\t{pr_auc}\t{mcc}\n")

np.save(save_dir+f"/test-{model_name}_{layer}_{random_seed}_pred.npy",y_pred)
torch.save(final_data,f"{save_dir}/checkpoint-{model_name}_{layer}_{random_seed}.pt")
