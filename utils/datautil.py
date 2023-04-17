
import pandas as pd
from utils.utils import *

bidata=["mooc","wikipedia","lastfm"]
homo_loop_data=["enron"]
homo_no_loop_data=["uci"]


def load_data(args):
    '''
    u,i:1-index
    nfeat:[nodesnum+1,dim1]
    efeat:[edgesnum+1,dim2]
    eid:1-index
    t:in order
    returned dataframe must include columns:u,i,t
    and first line is u:0;i:0,t:0
    '''
    
    data_dir = os.path.join(args.pro_path,"data/{}".format(args.data))
    df=pd.read_csv(os.path.join(data_dir,"ml_{}.csv".format(args.data)),index_col=0)
    df=df[["u","i","ts"]]
    df.columns=["u","i","t"]
    nodes=set(df.u.values).union(set(df.i.values))
    args.nodes_num=len(nodes)
    args.logger.info("nodes number:{}".format(args.nodes_num))

    
    df.index = df.index + 1
    df=df[:0].append({"u":0,"i":0,"t":0}, ignore_index=True).append(df)

    df = df[:int(len(df) * args.data_usage)]  
    event_num = len(df)-1
    val_idx = int(event_num * 0.7)+1
    test_idx = int(event_num * 0.8)+1


    n_feat = np.load(os.path.join(data_dir,'ml_{}_node.npy'.format(args.data)))
    e_feat = np.load(os.path.join(data_dir,'ml_{}.npy'.format(args.data)))
    n_feat=torch.tensor(n_feat,dtype=torch.float32)
    e_feat=torch.tensor(e_feat,dtype=torch.float32)
    
    
    
    
    
    
    


    args.logger.info("nodefeat:{},edgefeat:{}".format(n_feat.shape, e_feat.shape))
    return df,n_feat,e_feat,val_idx,test_idx