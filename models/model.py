import torch
from utils.datautil import *
from models.backbone import *
from models.attention_operator import *
import torch.nn.functional as tnf
class MLP(nn.Module):
    def __init__(self,input_dim,output_dim,dropout=0.5):
        super(MLP,self).__init__()
        hidden_dim=input_dim//2
        self.linear1=nn.Linear(input_dim,hidden_dim)
        self.hidden_drop=nn.Dropout(p=dropout)
        self.linear2=nn.Linear(hidden_dim,output_dim)
    def forward(self,x):
        x=self.linear1(x)
        x=tnf.relu(x)
        x=self.hidden_drop(x)
        x=self.linear2(x)
        return x


class TimeEncode(torch.nn.Module):
    def __init__(self, hdim, factor=5):
        super(TimeEncode, self).__init__()
        

        time_dim = hdim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter(torch.empty(hdim).uniform_(-1/np.sqrt(hdim),1/np.sqrt(hdim)))
        self.phase = torch.nn.Parameter(torch.empty(hdim).uniform_(-1/np.sqrt(hdim),1/np.sqrt(hdim)))

        

        

    def forward(self, ts):
        
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1) 
        map_ts = ts * self.basis_freq.view(1, 1, -1)
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)
        
        return harmonic  


class EventEmb(nn.Module):
    def __init__(self,args,nfeat,efeat):
        super(EventEmb,self).__init__()
        
        hdim=args.hdim
        nfeat=pad_tensor(hdim,-1,nfeat)
        efeat=pad_tensor(hdim,-1,efeat)
        

        
        
        
        self.use_time=args.use_timee
        if not self.use_time:
            nemb=torch.empty(nfeat.shape).uniform_(-1/np.sqrt(hdim),1/np.sqrt(hdim))
            nemb[0] = 0  
        else:
            nemb=torch.zeros(nfeat.shape)
        self.nemb=nn.Embedding.from_pretrained(nemb,padding_idx=0,freeze=False)
        self.nfeat=nn.Embedding.from_pretrained(nfeat,padding_idx=0)
        self.efeat=nn.Embedding.from_pretrained(efeat,padding_idx=0)
        self.dropout=nn.Dropout(p=0.2)
        if self.use_time:
            self.temb=TimeEncode(hdim)
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def forward(self,c,pos,neg):
        
        
        c_sids,c_oids,c_eids,c_ts=c
        pos_sids,pos_oids,pos_eids,pos_ts=pos
        neg_sids,neg_oids,neg_eids,neg_ts=neg

        if self.use_time:
            p_ts=torch.cat([pos_ts,c_ts],dim=1)
            
            p_ts=c_ts-p_ts
            p_ts_emb=self.temb(p_ts)

            n_ts=torch.cat([neg_ts,c_ts],dim=1)
            n_ts=c_ts-n_ts
            n_ts_emb=self.temb(n_ts)

            
            c_emb=self.nemb(c_sids)+self.nemb(c_oids)+self.nfeat(c_sids) + self.nfeat(c_oids)+p_ts_emb[:,[-1],:]
            
            pos_emb = self.nemb(pos_sids) + self.nemb(pos_oids) +  +self.nfeat(pos_sids) + self.nfeat(pos_oids) +\
                      self.efeat(pos_eids) + p_ts_emb[:, :-1, :]  
            neg_emb = self.nemb(neg_sids) + self.nemb(neg_oids) + +self.nfeat(neg_sids) + self.nfeat(neg_oids) + \
                      self.efeat(neg_eids) + n_ts_emb[:, :-1, :]  
        else:
            c_emb = self.nemb(c_sids) + self.nemb(c_oids)+self.nfeat(c_sids) + self.nfeat(c_oids)
            pos_emb = self.nemb(pos_sids) + self.nemb(pos_oids) +self.nfeat(pos_sids) + self.nfeat(pos_oids) + self.efeat(pos_eids)
            neg_emb = self.nemb(neg_sids) + self.nemb(neg_oids) +self.nfeat(neg_sids) + self.nfeat(neg_oids) +  self.efeat(neg_eids)
            
            c_emb=self.dropout(c_emb)
            pos_emb=self.dropout(pos_emb)
            neg_emb=self.dropout(neg_emb)

        
        pos_dot=torch.sum(c_emb*pos_emb,dim=-1)
        neg_dot=torch.sum(c_emb*neg_emb,dim=-1)
        
        return pos_dot,neg_dot
    def get_eventemb(self,sids,oids,eids,ts,grad=False):
        
        if grad:
            if self.use_time:
                semb=self.nemb(sids)
                oemb=self.nemb(oids)
                sfeat=self.nfeat(sids)
                ofeat=self.nfeat(oids)
                eemb=self.efeat(eids)
                
                ts=ts[:,[-1]]-ts
                temb=self.temb(ts)
                
                eventemb=semb+oemb+sfeat+ofeat+eemb+temb
            else:
                eventemb=self.nemb(sids)+self.nemb(oids)+self.nfeat(sids)+self.nfeat(oids)+self.efeat(eids)
        else:
            with torch.no_grad():
                if self.use_time:
                    semb = self.nemb(sids)  
                    oemb = self.nemb(oids)
                    sfeat = self.nfeat(sids)
                    ofeat = self.nfeat(oids)
                    eemb = self.efeat(eids)  
                    
                    ts = ts[:, [-1]] - ts  
                    temb = self.temb(ts)  
                    
                    eventemb = semb + oemb + sfeat + ofeat + eemb + temb
                else:
                    eventemb = self.nemb(sids) + self.nemb(oids) + self.nfeat(sids) + self.nfeat(oids) + self.efeat(
                        eids)
        return eventemb


class TGACN(nn.Module):
    def __init__(self, args,n_feat, e_feat,class_num):
        super(TGACN, self).__init__()
        self.args=args

        self.use_time=args.use_timec
        
        if self.use_time:
            self.time_op=TimeOperator(args)
        self.index_op=IndexOperator()
        self.event_emb = EventEmb(args, n_feat, e_feat).to(args.device)
        self.vector_op = VectorOperator(self.event_emb.get_eventemb)
        
        if self.use_time:
            input_channel=self.index_op.channel+self.time_op.channel+self.vector_op.channel
        else:
            input_channel=self.index_op.channel+self.vector_op.channel
        
        
        
        
        
        

        self.dropout=nn.Dropout(p=args.dropout)

        if args.model==0:
            self.cnn = efficientnetv2_s(num_classes=class_num,input_channel=input_channel,stage=args.stage)
        elif args.model==1:
            self.cnn=resnet18(num_classes=class_num,input_channel=input_channel)
        elif args.model==3:
            self.cnn=shufflenet_v2_x0_5(num_classes=class_num,input_channel=input_channel)
        elif args.model==2:
            self.cnn = efficientnetv2_sa(num_classes=class_num,input_channel=input_channel)



    def forward(self, data):
        
        sids, oids, eids, ts=data

        index_image=self.index_op(sids, oids)
        
        if self.use_time:
            time_image=self.time_op(ts)
        
        vector_image=self.vector_op(sids, oids, eids,ts)
        
        
        vector_image=self.dropout(vector_image)

        if self.use_time:
            image=torch.cat([index_image,time_image,vector_image],dim=1)
        else:
            image=torch.cat([index_image,vector_image],dim=1)

        
        
        
        
        
        output = self.cnn(image)

        return output













