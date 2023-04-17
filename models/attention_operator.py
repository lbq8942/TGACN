import torch
import torch.nn as nn















def min_max(ts):
    
    bsize=ts.shape[0]
    ts=ts.type(torch.float32)
    for i in range(bsize):
        t=ts[i]
        max=torch.max(t)
        min=torch.min(t)
        if max==min:
            ts[i]=0
        else:
            ts[i]=(ts[i]-min)/(max-min)
    return ts












































class TimeOperator(nn.Module):
    def __init__(self,args):
        
        super(TimeOperator,self).__init__()
        assert args.alpha>0
        self.args=args
        self.channel=1


    def forward(self,ts):
        

        ts=min_max(ts)
        ts1=ts.unsqueeze(1)
        ts2=ts.unsqueeze(2)
        
        time_image=torch.abs(ts1-ts2)
        
        time_image=torch.exp(-self.args.alpha*time_image)
        time_image = torch.tril(time_image, -1).unsqueeze(1)
        

        return time_image


class IndexOperator(nn.Module):
    def __init__(self):
        super(IndexOperator,self).__init__()
        
        self.channel=3

        def attention(a, b):
            x1 = a.unsqueeze(2)
            x2 = b.unsqueeze(1)
            attn = (x1 == x2).type(torch.float32)
            return attn.unsqueeze(1)
        self.attention=attention

    def forward(self,sids, oids):
        index_image=torch.cat([self.attention(sids,sids),self.attention(oids,oids)],dim=1)
        index_image=torch.cat([index_image,self.attention(sids,oids)],dim=1)

        
        index_image=torch.tril(index_image)

        return index_image


class VectorOperator(nn.Module):
    def __init__(self,get_eventemb):
        super(VectorOperator, self).__init__()

        self.get_eventemb=get_eventemb
        self.channel=1

        def multihead_attention(x, y):
            
            return torch.matmul(x.permute(0,3,1,2),y.permute(0,3,2,1))
        def dot_attention(x,y):
            return torch.matmul(x,y.permute(0,2,1))

        self.attention=dot_attention
        self.bn=nn.BatchNorm2d(self.channel)

    def forward(self,sids,oids,eids,ts):
        

        bsize,seq_len=sids.shape
        
        eventemb=self.get_eventemb(sids,oids,eids,ts,grad=True)
        vector_image=self.attention(eventemb,eventemb).unsqueeze(1)
        vector_image=self.bn(vector_image)
        vector_image=torch.tril(vector_image,0)
        
        return vector_image