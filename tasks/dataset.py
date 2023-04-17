import torch
import torch.utils.data as tud
from utils.utils import *
def padlist(l,num,e):
    pad_num=num-len(l)
    need_l=[e for i in range(pad_num)]
    l=need_l + l
    return l

def normalize(l,window):
    
    eids=[]
    for i in range(len(l)):
        eids.extend(l[i])
    pad_num=window+1-len(eids)
    
    eids=[0 for i in range(pad_num)]+eids
    return eids

















class lp_dataset(tud.Dataset):

    def __init__(self,df,data,args,sampler,neg_sampler,event_emb,mode="train"):
        self.df=df
        self.data =data
        self.args=args
        self.sampler=sampler
        self.neg_sampler=neg_sampler
        self.event_emb=event_emb
        self.mode=mode
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item = self.data[item].item()
        src = self.df.loc[item, "u"]
        dst = self.df.loc[item, "i"]
        dst_neg = self.neg_sampler.sample(self.mode)

        
        window=self.args.window
        recent=self.args.recent
        para=self.args.para

        sample_eids_src = self.sampler.link_prediction(item, src)  
        sample_eids_dst = self.sampler.link_prediction(item, dst)  
        sample_eids_dst_neg = self.sampler.link_prediction(item, dst_neg)  
        
        sample_eids_src=normalize(sample_eids_src,window)
        sample_eids_dst=normalize(sample_eids_dst,window)
        sample_eids_dst_neg=normalize(sample_eids_dst_neg,window)

        
        
        
        

        
        
        if recent==0:
            recent_sample_pos,recent_sample_neg=[],[]
        else:
            recent_sample_pos=sample_eids_src[-recent:]+sample_eids_dst[-recent:]
            recent_sample_neg=sample_eids_src[-recent:]+sample_eids_dst_neg[-recent:]
        
        
        
        

        
        
        if para==0:
            para_sample_pos, para_sample_neg=[],[]
        else:
            para_sample_pos,para_sample_neg=self.para_sample(para,item,dst_neg,sample_eids_src,sample_eids_dst,sample_eids_dst_neg)

        
        
        center=[0],[0],[0],[0]
        pos=[0],[0],[0],[0]
        neg=[0],[0],[0],[0]

        
        sample_pos=sorted(list(set(recent_sample_pos+para_sample_pos)))
        sample_neg=sorted(list(set(recent_sample_neg+para_sample_neg)))
        sample_pos=pad_list((para+recent)*2,sample_pos)
        sample_neg=pad_list((para+recent)*2,sample_neg)
        sample_pos.append(item)
        sample_neg.append(item)
        

        sids_pos, oids_pos, eids_pos, ts_pos = self.sampler.eid2e(sample_pos)
        sids_neg, oids_neg, eids_neg, ts_neg = self.sampler.eid2e(sample_neg)  
        
        
        
        
        
        
        assert oids_neg[-1] == dst
        oids_neg[-1] = dst_neg
        eids_pos[-1] = 0
        eids_neg[-1] = 0  

        sids = [sids_pos, sids_neg]  
        oids = [oids_pos, oids_neg]
        eids = [eids_pos, eids_neg]
        ts = [ts_pos, ts_neg]

        
        
        
        return totensor(sids, oids, eids, ts, [dst_neg]), totensor(*center),totensor(*pos),totensor(*neg)
        

        
        

    def para_sample(self,para,item,dst_neg,sample_eids_src,sample_eids_dst,sample_eids_dst_neg):
        
        c_sids_pos, c_oids_pos, c_eids_pos, c_ts_pos = self.sampler.eid2e([item])
        c_sids_neg, c_oids_neg, c_eids_neg, c_ts_neg = c_sids_pos, [dst_neg], c_eids_pos, c_ts_pos 
        
        
        
        
        
        

        
        
        
        

        pos_center=unsqueeze(0,*todevice(self.args,*totensor(c_sids_pos, c_oids_pos, c_eids_pos, c_ts_pos)))
        neg_center=unsqueeze(0,*todevice(self.args,*totensor(c_sids_neg, c_oids_neg, c_eids_neg, c_ts_neg)))
        sample_src = self.sampler.eid2e(sample_eids_src)
        sample_dst = self.sampler.eid2e(sample_eids_dst)
        sample_dst_neg = self.sampler.eid2e(sample_eids_dst_neg)

        sample_src=unsqueeze(0, *todevice(self.args, *totensor(*sample_src)))
        sample_dst=unsqueeze(0, *todevice(self.args, *totensor(*sample_dst)))
        sample_dst_neg=unsqueeze(0, *todevice(self.args, *totensor(*sample_dst_neg)))

        
        
        with torch.no_grad():
            
            self.event_emb.training=False
            pos_dot_src,pos_dot_dst=self.event_emb(pos_center,sample_src,sample_dst)
            neg_dot_src,neg_dot_dst=self.event_emb(neg_center,sample_src,sample_dst_neg)
            self.event_emb.training=True

        

        pos_idx_src=torch.argsort(pos_dot_src[0])[-para:].cpu().tolist()
        pos_idx_dst=torch.argsort(pos_dot_dst[0])[-para:].cpu().tolist()
        neg_idx_src=torch.argsort(neg_dot_src[0])[-para:].cpu().tolist()
        neg_idx_dst=torch.argsort(neg_dot_dst[0])[-para:].cpu().tolist()

        
        
        para_sample_pos=list_multiindex(sample_eids_src,pos_idx_src)+list_multiindex(sample_eids_dst,pos_idx_dst)
        para_sample_neg=list_multiindex(sample_eids_src,neg_idx_src)+list_multiindex(sample_eids_dst_neg,neg_idx_dst)
        return para_sample_pos,para_sample_neg

    def skip_gram(self,item,sample_eids_src,sample_eids_dst):
    
        
        
        
        pos_sample,neg_sample=self.args.skip_gram
        window=self.args.window
        
        pos=np.random.choice(sample_eids_src[-window:],pos_sample,replace=False).tolist()+np.random.choice(sample_eids_dst[-window:],pos_sample,replace=False).tolist()
        neg=np.random.choice(sample_eids_src[:-window],neg_sample).tolist()+np.random.choice(sample_eids_dst[:-window],neg_sample).tolist()
        
        
        
        
        center=self.sampler.eid2e([item])
        pos=self.sampler.eid2e(pos)
        neg=self.sampler.eid2e(neg)

        
        
        
        
        
        
        

        
        
        return center, pos, neg


def get_dataloader(args,data,sampler,val_idx,test_idx,event_emb,*util):
    event_num=len(data)
    
    train_end_idx=int(val_idx*args.training_proportion)
    traindata=torch.arange(1,train_end_idx)
    valdata=torch.arange(val_idx,test_idx)
    testdata=torch.arange(test_idx,event_num)

    if args.task=="np":
        pass
        
        
        
    elif args.task=="lp":
        neg_sampler=util[0]
        trainset = lp_dataset(data,traindata,args,sampler,neg_sampler,event_emb,mode="train")
        valset = lp_dataset(data,valdata,args,sampler,neg_sampler,event_emb,mode="val")
        testset = lp_dataset(data,testdata,args,sampler,neg_sampler,event_emb,mode="test")
    else:
        raise  NotImplementedError

    trainloader=tud.DataLoader(trainset,batch_size=args.bs,shuffle=True)
    valloader=tud.DataLoader(valset,batch_size=args.bs,shuffle=False)
    testloader=tud.DataLoader(testset,batch_size=args.bs,shuffle=False)

    return trainloader,valloader,testloader


