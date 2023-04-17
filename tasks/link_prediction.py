import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as tnf
from concurrent.futures import ProcessPoolExecutor,as_completed

from sklearn.metrics import f1_score
import os
import time
import math
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from utils.utils import *
from datetime import datetime
from utils.utils import *
from utils.sampler import Sampler
from utils.early_stop import Early_Stopper
from tasks.dataset import *
from models.model import *
class Neg_Sampler():
    def __init__(self,args,data,val_idx, test_idx):
        self.data=data
        self.event_num = len(data)
        self.train_end_idx=int(val_idx*args.training_proportion)
        self.val_idx=val_idx
        self.test_idx=test_idx

        
        self.train_dsts=np.unique(list(self.data[:self.train_end_idx]["i"]))
        self.val_dsts=np.unique(list(self.data[:self.test_idx]["i"]))
        self.test_dsts=np.unique(list(self.data["i"]))

    def sample(self,mode="train"):
        
        
        
        
        
        
        
        if mode == "train":
            r = np.random.randint(1, len(self.train_dsts), 1)
            return self.train_dsts[r[0]]
        elif mode == "val":
            r = np.random.randint(1, len(self.val_dsts), 1)
            return self.val_dsts[r[0]]
        elif mode == "test":
            r = np.random.randint(1, len(self.test_dsts), 1)
            return self.test_dsts[r[0]]



class link_prediction():
    def __init__(self,args,df, n_feat, e_feat,val_idx,test_idx,adj_util):
        
        self.args=args
        self.data=df
        self.sampler=Sampler(args,df,adj_util)
        self.neg_sampler=Neg_Sampler(args,df,val_idx, test_idx)
        class_num=1

        self.model = TGACN(args,n_feat, e_feat,class_num).to(args.device)
        self.trainloader,self.valloader, self.testloader = get_dataloader(args, df, self.sampler, val_idx, test_idx,self.model.event_emb,self.neg_sampler)
        self.early_stopper= Early_Stopper(args,self.model)
    def prepare(self,data):
        
        data,center_skip,pos_skip,neg_skip=data
        
        pos=data[0][:, 0, :], data[1][:, 0, :], data[2][:, 0, :],data[3][:, 0,:]
        neg=data[0][:, 1, :], data[1][:, 1, :], data[2][:, 1, :],data[3][:, 1,:]

        pos=todevice(self.args,*pos)
        neg=todevice(self.args,*neg)

        inputs=[]
        for i in range(4):
            inputs.append(torch.cat([pos[i],neg[i]]))

        
        
        
        
        
        
        
        
        
        
        return inputs


    def eval(self,test=False):
        torch.set_grad_enabled(False)
        self.model.eval()
        args=self.args
        loader=self.testloader if test else self.valloader
        pos=[]
        neg=[]
        for idx, data in enumerate(loader):
            
            bsize = len(data[1][0])
            
            inputs = self.prepare(data)

            
            outputs=self.model(inputs).squeeze(1)

            
            
            

            pos.append(outputs[:bsize])
            neg.append(outputs[bsize:])


        pos=torch.cat(pos).cpu().numpy()
        neg=torch.cat(neg).cpu().numpy()
        outputs=np.hstack([pos,neg])
        labels=np.hstack([np.ones_like(pos),np.zeros_like(neg)])

        acc_f1,auc_ap=get_acc_f1(outputs, labels), get_auc_ap(outputs, labels)
        acc, f1 = acc_f1[0], acc_f1[1]
        auc, ap = auc_ap[0], auc_ap[1]
        acc, f1, auc,ap = round(3, acc, f1,  auc,ap)
        torch.set_grad_enabled(True)
        self.model.train()
        return acc,f1,auc,ap


    def train(self):
        args=self.args
        start_time = time.time()
        args.logger.info("start training on time:{}".format(datetime.now()))
        optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        loss_fn=nn.BCEWithLogitsLoss()

        batches=len(self.trainloader)


        self.model.train()
        eval_every_batch=batches // self.args.eval_every_epoch + 1
        for epoch in tqdm(range(args.epochs)):
            sampling_losses=[]
            losses=[]

            start_time=time.time()
            for idx,data in enumerate(self.trainloader):
                
                
                
                inputs=self.prepare(data)

                bsize=len(inputs[0])//2

                optimizer.zero_grad()

                
                labels = torch.cat([torch.ones(bsize, device=args.device), torch.zeros(bsize, device=args.device)])
                outputs = self.model(inputs).squeeze(1)  
                loss = loss_fn(outputs, labels)
                
                loss.backward()
                losses.append(loss.item())

                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                


                optimizer.step()
                
                end_of_epoch=((idx+1)==batches)
                
                
                
                if (idx+1)%eval_every_batch==0 or end_of_epoch:
                    start=time.time()
                    acc,f1,auc,ap = self.eval()

                    end=time.time()
                    args.logger.info( "\neval result:\nepoch:{},sampling_loss:{},loss:{},time:{}\nacc:{},f1:{}\nauc:{},ap:{}\n".
                                      format(epoch,np.mean(sampling_losses),np.mean(losses),end-start,acc, f1, auc,ap))
                    
                    metric=auc
                    if self.early_stopper.check(metric):
                        args.logger.info(
                            "end training on time:{},training time in total:{}".format(datetime.now(),
                                                                                       time.time() - start_time))
                        return
        args.logger.info(
            "end training on time:{},training time in total:{}".format(datetime.now(),
                                                                       time.time() - start_time))
    def run(self):
        if self.args.testing:
            
            
            self.model.load_state_dict(torch.load(
                os.path.join(self.args.model_path, self.args.load_path + "-" +self.args.task+"-"+ self.args.data + ".pyt")))  
            
            
            
            
            self.model=self.model.to(self.args.device)

        else:
            self.train()

        self.args.logger.info("start testing on time:{}".format(datetime.now()))
        start = time.time()
        acc,f1,auc,ap = self.eval(test=True)
        end = time.time()
        self.args.logger.info("\ntest result:\ntime:{}\nacc:{},f1:{}\nauc:{},ap:{}\n".
                              format(end - start, acc, f1,auc,ap))
        self.args.logger.info("end testing on time:{}".format(datetime.now()))
        
        
        
