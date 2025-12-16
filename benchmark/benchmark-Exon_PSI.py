
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np
import copy
import sys
import math
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader,TensorDataset
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, roc_curve,confusion_matrix,matthews_corrcoef,precision_recall_curve
from sklearn import metrics
from torch.backends import cudnn
from audtorch.metrics.functional import pearsonr
from scipy.stats import spearmanr


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import splev


task_name=sys.argv[1]
model_name=sys.argv[2]
data_dir=sys.argv[3]
layer=int(sys.argv[4])
random_seed=int(sys.argv[5])
output_dim=int(sys.argv[6])

regression=sys.argv[7]
if regression=="True":
    regression=True
else:
    regression=False
multi_seq=sys.argv[8]
if multi_seq=="True":
    multi_seq=True
else:
    multi_seq=False
save_dir=f"{data_dir}/{model_name}/checkpoint"
if os.path.exists(save_dir):
    pass
else:
    os.mkdir(save_dir)
    
class DiffKL(nn.Module):
    def __init__(self):
        super(DiffKL, self).__init__()
    def forward(self, y_pred, y_true):
        # get mean logit
        mean_logit = y_true[:, :, -1]
        # get true psi value
        psi_true = torch.sigmoid(y_true[:, :, 0] + mean_logit)
        # find no observation mask
        mask = ~torch.isnan(psi_true)
        # use mean logit and prediction to get psi_pred
        pred_logit = y_pred + mean_logit
        psi_pred = torch.sigmoid(pred_logit)
        # clip psi value
        clip_true = torch.clamp(psi_true, min=1e-5, max=1-1e-5)
        clip_pred = torch.clamp(psi_pred, min=1e-5, max=1-1e-5)
        clip_true = torch.where(mask, clip_true, clip_pred)
        # KL divergence
        kl1 = torch.log(clip_true / clip_pred)
        kl1 = clip_true * kl1
        kl2 = torch.log((1 - clip_true) / (1 - clip_pred))
        kl2 = (1 - clip_true) * kl2
        kl = kl1 + kl2
        clean_kl = kl * mask.float()
        return clean_kl.mean()

def evaluate_test(test_true, test_pred):
    corr_list = []
    for x in range(56):
        corr,_ = spearmanr(test_true[:,x], test_pred[:,x], nan_policy='omit')
        corr_list.append(corr)
    return corr_list

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
    model=model.eval().cuda()
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
            loss_lm = loss_fn(target_pred, targets)
            test_loss += loss_lm.detach().item()
            step+=1
    test_loss /= num_batches
    return test_loss


def get_S(n_bases=10, spline_order=3, add_intercept=True):
    S = np.identity(n_bases)
    m2 = spline_order - 1
    for _ in range(m2):
        S = np.diff(S, axis=0)
    S = np.dot(S.T, S)
    S = (S + S.T) / 2
    if add_intercept:
        zeros = np.zeros_like(S[:1, :])
        S = np.vstack([zeros, S])
        zeros = np.zeros_like(S[:, :1])
        S = np.hstack([zeros, S])
    return torch.tensor(S, dtype=torch.float32)


def get_knots(start, end, n_bases=10, spline_order=3):
    # print(end)
    # print(start)
    x_range = end - start
    start -= x_range * 0.001
    end += x_range * 0.001
    m = spline_order - 1
    nk = n_bases - m
    dknots = (end - start) / (nk - 1)
    knots = np.linspace(start - dknots * (m + 1), end + dknots * (m + 1), nk + 2 * m + 2)
    return torch.tensor(knots, dtype=torch.float32)


def get_X_spline(x, knots, n_bases=10, spline_order=3, add_intercept=True):
    if len(x.shape) != 1:
        raise ValueError("x has to be 1-dimensional")
    tck = [knots, np.zeros(n_bases), spline_order]
    X = np.zeros((len(x), n_bases))
    for i in range(n_bases):
        vec = np.zeros(n_bases)
        vec[i] = 1.0
        tck[1] = vec
        X[:, i] = splev(x, tck, der=0)
    if add_intercept:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    return torch.tensor(X, dtype=torch.float32)


class BSpline:
    def __init__(self, start=0, end=1, n_bases=10, spline_order=3):
        self.start = start
        self.end = end
        self.n_bases = n_bases
        self.spline_order = spline_order
        self.knots = get_knots(self.start, self.end, self.n_bases, self.spline_order)
        self.S = get_S(self.n_bases, self.spline_order, add_intercept=False)
    def predict(self, x, add_intercept=False):
        if x.min() < self.start or x.max() > self.end:
            raise ValueError("x out of bounds.")
        return get_X_spline(x, self.knots, self.n_bases, self.spline_order, add_intercept)

class SplineWeight1D(nn.Module):
    """
    A PyTorch implementation of the SplineWeight1D layer.
    Up- or down-weight positions in the activation array of 1D convolutions:
    `x_out[:, :, j, k] = x_in[:, :, j, k] * (1 + f_S^k(j))`,
    where f_S is the spline transformation.
    Args:
        n_bases (int): Number of spline bases used for the positional effect.
        spline_degree (int): Degree of the B-spline.
        share_splines (bool): If True, all channels share the same spline weights.
        l2_smooth (float): L2 regularization strength for smoothness of spline.
        l2 (float): L2 regularization strength for spline coefficients.
        use_bias (bool): Whether to use bias in the spline transformation.
        bias_initializer (str): Method to initialize the bias.
    """
    def __init__(
        self,
        n_bases=10,
        spline_degree=3,
        share_splines=False,
        l2_smooth=0.0,
        l2=0.0,
        use_bias=False,
        bias_initializer="zeros",
    ):
        super(SplineWeight1D, self).__init__()
        self.n_bases = n_bases
        self.spline_degree = spline_degree
        self.share_splines = share_splines
        self.l2_smooth = l2_smooth
        self.l2 = l2
        self.use_bias = use_bias
        # Bias initializer
        if bias_initializer == "zeros":
            self.bias_initializer = nn.init.zeros_
        else:
            raise ValueError(f"Unsupported bias_initializer: {bias_initializer}")
    def build(self, input_shape):
        start = 0
        end = input_shape[1]
        filters = input_shape[2]
        if self.share_splines:
            n_spline_tracks = 1
        else:
            n_spline_tracks = filters
            self.n_spline_tracks = filters
        # Create B-spline basis
        self.positions = np.arange(end)
        self.bs = BSpline(start, end - 1,
                          n_bases=self.n_bases,
                          spline_order=self.spline_degree
                          )
        # print('ok2')
        self.X_spline = Parameter(torch.tensor(self.bs.predict(self.positions), dtype=torch.float32))
        # Spline weights
        self.kernel = Parameter(torch.zeros((self.n_bases, self.n_spline_tracks)))
        # Bias (optional)
        if self.use_bias:
            self.bias = Parameter(torch.zeros(n_spline_tracks))
            self.bias_initializer(self.bias)
        else:
            self.bias = None
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, steps, filters).
        Returns:
            Tensor: Output tensor with positional effect applied.
        """
        
        # Apply positional effects
        spline_track = torch.matmul(self.X_spline, self.kernel)
        if self.use_bias:
            spline_track = spline_track + self.bias
        # Add 1 to spline effect (as per original implementation)
        spline_track = spline_track + 1
        # Apply spline weighting
        output = spline_track * x
        return output



class Double_input_model(nn.Module):
    def __init__(self,in_channels):
        super(Double_input_model, self).__init__()
        #参照原工作
        self.conv1=nn.Conv1d(in_channels,256,1)
        self.conv2=nn.Conv1d(256,256,9,padding='same')
        self.batchnorm1=nn.BatchNorm1d(256)
        self.batchnorm2=nn.BatchNorm1d(256)
        self.sp_model1=SplineWeight1D()
        self.sp_model1.build(torch.randn(4,valid_data_1.shape[1],256).shape)
        self.sp_model2=SplineWeight1D()
        self.sp_model2.build(torch.randn(4,valid_data_1.shape[1],256).shape)
        self.act=nn.ReLU()
        self.linear1=nn.Linear(in_features=256,out_features=64)
        self.batchnorm3=nn.BatchNorm1d(256)
        self.batchnorm4=nn.BatchNorm1d(64)
        self.drop=nn.Dropout(0.2)
        self.linear2=nn.Linear(in_features=64,out_features=output_dim)
        # self.linear3=nn.Linear(in_features=256,out_features=output_dim)
        
    def forward(self,x1,x2):
        x1=x1.permute(0,2,1)
        x2=x2.permute(0,2,1)
        x1=self.conv1(x1)
        x1=self.conv2(x1)
        # print(x1.shape)
        x2=self.conv1(x2)
        x2=self.conv2(x2)
        #convolutions
        x1=self.act(self.batchnorm1(x1))
        x2=self.act(self.batchnorm2(x2))
        # print(x2.shape)
        x1=self.sp_model1(x1.permute(0,2,1))
        
        x2=self.sp_model2(x2.permute(0,2,1))
        # print(x2.shape)
        x=torch.cat([x1.permute(0,2,1),x2.permute(0,2,1)],dim=2)
        # print(x.shape)
        # print(x.shape)
        #pooling
        x=torch.nn.functional.avg_pool1d(x,kernel_size=x.shape[2])
        x=x.view(x.shape[0],-1)
        # print(x.shape)
        x=self.batchnorm3(x)
        
        x=self.linear1(x)
        x=self.act(x)
        x=self.batchnorm4(x)
        x=self.drop(x)        
        x=self.linear2(x)
        if (output_dim==1) and (not regression):
            x=torch.sigmoid(x)
        return x

    
exon=True
if exon:
    train_label=np.load(data_dir+'/train_target.npy')
    valid_label=np.load(data_dir+'/dev_target.npy')
    test_label=np.load(data_dir+'/test_target.npy')
    train_label=torch.as_tensor(train_label)
    valid_label=torch.as_tensor(valid_label)
    test_label=torch.as_tensor(test_label)

    train_data_1=np.load(f"{data_dir}/{model_name}/train_data_0-embedding-layer_{layer}.npy")
    valid_data_1=np.load(f"{data_dir}/{model_name}/dev_data_0-embedding-layer_{layer}.npy")
    test_data_1=np.load(f"{data_dir}/{model_name}/test_data_0-embedding-layer_{layer}.npy")
    train_data_1=torch.as_tensor(train_data_1)
    valid_data_1=torch.as_tensor(valid_data_1)
    test_data_1=torch.as_tensor(test_data_1)

    train_data_2=np.load(f"{data_dir}/{model_name}/train_data_1-embedding-layer_{layer}.npy")
    valid_data_2=np.load(f"{data_dir}/{model_name}/dev_data_1-embedding-layer_{layer}.npy")
    test_data_2=np.load(f"{data_dir}/{model_name}/test_data_1-embedding-layer_{layer}.npy")
    train_data_2=torch.as_tensor(train_data_2)
    valid_data_2=torch.as_tensor(valid_data_2)
    test_data_2=torch.as_tensor(test_data_2)

    train_dataset=TensorDataset(train_data_1,train_data_2,train_label)
    valid_dataset=TensorDataset(valid_data_1,valid_data_2,valid_label)
    test_dataset=TensorDataset(test_data_1,test_data_2,test_label)
    batch_size=256
    train_dataloader = DataLoader(train_dataset,shuffle=True, batch_size=batch_size,)
    valid_dataloader = DataLoader(valid_dataset,shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset,shuffle=False, batch_size=batch_size, num_workers=0)
    model=Double_input_model(in_channels=valid_data_1.shape[2])

    model.to(device)
    learning_rate=1e-3
    epochs=80
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn=DiffKL()
    patience=10
    count=0
    min_valid_loss=1000
    final_data={}
    for epoch in range(epochs):
        model=model.train()
        train_epoch_loss = 0
        train_num_batches = len(train_dataloader)
        step=0
        for one_batch in train_dataloader:
            
            data1,data2 ,targets = map(lambda x: x.to(device), one_batch)
            data1=data1.to(torch.float32)
            data2=data2.to(torch.float32)
            target_pred = model(data1,data2)
            target_pred=target_pred.to(torch.float32)
        
            loss_lm = loss_fn(target_pred, targets)
            train_epoch_loss += loss_lm.detach().item()
            loss_lm.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
        #train epoch loss
        train_epoch_loss /= train_num_batches
        print(f"train_loss\t{train_epoch_loss}")
        #validating
        valid_loss = validating(valid_dataloader, model, loss_fn)
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

    test_model=Double_input_model(in_channels=1*valid_data_1.shape[2])
    test_model.to(device)
    test_model.load_state_dict(final_data['model_state_dict'])
    y_pred=testing(test_dataloader,test_model)

y_test=test_label.cpu().detach().numpy()
result=evaluate_test(y_test,y_pred)
result=[str(elem) for elem in result]

np.save(save_dir+f"/test_{model_name}_{layer}_{random_seed}_pred.npy",y_pred)
torch.save(final_data,f"{save_dir}/checkpoint_{model_name}_{layer}_{random_seed}.pt")
with open(save_dir+f"/metrics.txt",'a') as f:
    f.write(f"{model_name}\t{random_seed}\t"+"\t".join(result)+'\n')
