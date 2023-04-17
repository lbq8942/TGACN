import numpy as np
from utils.utils import *
def update_time(n_t,node,time):
    if node not in n_t:
        
        n_t[node]=[time]
    else:
        if time!=n_t[node][-1]:
            n_t[node].append(time)
            return True
        
    return False
def update_eid(n_eid,node,eid,new=False):
    if node not in n_eid:
        n_eid[node]=[[eid]]
    else:
        if new:
            n_eid[node].append([eid])
        else:
            n_eid[node][-1].append(eid)
def find_latest(eids,ts,trace_step):
    l_eids=[[] for i in range(trace_step)]
    uniq_ts=sorted(list(set(ts)))
    need_ts=uniq_ts[-trace_step:]
    for i in range(len(ts)):
        for j in range(len(need_ts)):
            if ts[i]== need_ts[j]:
                
                
                l_eids[j].extend(eids[i])
                break
    
    
    
    for i in range(len(l_eids)):
        l_eids[i]=sorted(list(set(l_eids[i])))
    return l_eids


def make_adjlist(data):
    
    
    index_table = {}  
    n_eid = {}
    n_n = {}
    n_t={}
    
    for i in range(1, len(data)):  
        s = data.loc[i, "u"]
        o = data.loc[i, "i"]
        t = data.loc[i, "t"]
        
        new=update_time(n_t,s,t)
        update_eid(n_eid,s,i,new)
        update_eid(n_n,s,o,new)
        if s!=o:
            new = update_time(n_t, o, t)
            update_eid(n_eid, o, i, new)
            update_eid(n_n, o, s, new)  
        

        
        
        index_table[i]=[len(n_eid[s])-1,len(n_eid[o])-1]
        
    return index_table,n_eid,n_n,n_t

class Sampler(object):
    def __init__(self,args,data,adj_util):
        self.args=args
        self.data=data 
        self.index_table, self.n_eid, self.n_n,self.n_t=adj_util
        self.trace_step=args.trace_step

    def get_ngh(self,nodes,eid):
        
        data=self.data
        sampled_eids=[]
        sampled_ns=[]
        sampled_ts=[]

        eu=data.loc[eid,"u"]
        ei=data.loc[eid,"i"]
        et=data.loc[eid,"t"]

        for i in range(len(nodes)):
            node=nodes[i]
            node_eid = self.n_eid[node]
            node_time=self.n_t[node]
            node_n=self.n_n[node]

            if node == eu:
                time_limit=self.index_table[eid][0]
            elif node == ei:
                time_limit=self.index_table[eid][1]
            else:  
                time_limit = np.searchsorted(node_time, et)  
            node_eid = node_eid[:time_limit][-self.trace_step:]
            node_n = node_n[:time_limit][-self.trace_step:]
            node_time=node_time[:time_limit][-self.trace_step:]
            sampled_eids.extend(node_eid)
            sampled_ns.extend(node_n)
            sampled_ts.extend(node_time)
        return sampled_ns,sampled_eids,sampled_ts

    def K_hop_sampling(self,node,eid):
        nodes=[node]
        sampled_eids=[]
        sampled_ts=[]
        visited=[node]
        for i in range(self.args.K):
            if i>0:
                real_nodes = []
                for j in range(len(nodes)):
                    for k in range(len(nodes[j])):
                        if nodes[j][k] not in visited:
                            real_nodes.append(nodes[j][k])
                            visited.append(nodes[j][k])
                nodes=real_nodes
            nodes,eids,ts=self.get_ngh(nodes,eid)
            sampled_eids.extend(eids)
            sampled_ts.extend(ts)
            
            
        if self.args.K>1:
            sampled_eids=find_latest(sampled_eids,sampled_ts,self.trace_step)
        return sampled_eids
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    def link_prediction(self,eid,node):
        cem=self.K_hop_sampling(node,eid)
        
        return cem

    def eid2e(self,eids):
        
        data=self.data
        sids=list(data.loc[eids,"u"])
        oids=list(data.loc[eids,"i"])
        ts=list(data.loc[eids,"t"])
        
        
        return list(sids),list(oids),eids,list(ts)



