import numpy as np
import torch
import random
import os
from sklearn.metrics import accuracy_score,roc_auc_score,average_precision_score,f1_score
def totensor(*x):
    x=list(x)
    if isinstance(x[0],torch.Tensor):
        return x
    elif isinstance(x[0],np.ndarray):
        for i in range(len(x)):
            x[i]=torch.from_numpy(x[i])
    elif isinstance(x[0],list):
        for i in range(len(x)):
            x[i]=torch.tensor(x[i])
    else:
        raise NotImplementedError
    return x
def unsqueeze(dim,*x):
    x=list(x)
    for i in range(len(x)):
        x[i]=x[i].unsqueeze(dim)
    return x
def todevice(args,*x):
    x=list(x)
    if isinstance(x[0],torch.Tensor):
        for i in range(len(x)):
            x[i]=x[i].to(args.device)
    elif isinstance(x[0],np.ndarray):
        for i in range(len(x)):
            x[i]=torch.from_numpy(x[i]).to(args.device)
    elif isinstance(x[0],list):
        for i in range(len(x)):
            x[i]=torch.tensor(x[i]).to(args.device)
    else:
        raise NotImplementedError

    return x

def round(d,*x):
    x=list(x)
    for i in range(len(x)):
        x[i]=np.round(x[i],d)
    return x

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def update2(table,a,b,eid):
    if a not in table:
        table[a]={b:[eid]}
    else:
        if b not in table[a]:
            table[a][b]=[eid]
        else:
            table[a][b].append(eid)
    return table

def update1(table,a,eid):
    if a not in table:
        table[a]=[eid]
    else:
        table[a].append(eid)
    return table
def crop(dict,key,max):
    dict[key]=dict[key][-max:]

def get_acc_f1(pred,label,average="macro"):
    pred=(pred>0).astype(np.int64)
    return accuracy_score(label,pred),f1_score(label,pred,average=average)

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def get_auc_ap(pred,label):
    pred=sigmoid(pred)
    return roc_auc_score(label,pred),average_precision_score(label,pred)


def pad_list(need_num,l):
    num=len(l)
    pad_num=need_num-num
    a=[0 for i in range(pad_num)]

    return a+l
def pad_tensor(need_dim,dim,ten):
    size=ten.shape
    pad_num=need_dim-size[dim]
    need_size=list(size)
    need_size[dim]=pad_num
    a=torch.zeros(need_size,device=ten.device)
    pad_ten=torch.cat([a,ten],dim=dim)
    return pad_ten

def list_multiindex(l,ind):
    need=[]
    for i in range(len(ind)):
        need.append(l[ind[i]])
    return need