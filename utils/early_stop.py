import time
from datetime import datetime
import torch
import os

class Early_Stopper():
    def __init__(self,args,model):
        self.args=args
        self.best=0
        self.model=model
        self.count=0

    def check(self,metric):
        if metric>self.best:
            self.count=0
            self.best=metric
            torch.save(self.model.state_dict(),
                       os.path.join(self.args.model_path, str(self.args.count) + "-" + self.args.task + "-" + self.args.data + ".pyt"))
        else:
            self.count+=1

        if self.count>=self.args.patience:
            self.model.load_state_dict(torch.load(
                       os.path.join(self.args.model_path, str(self.args.count) + "-" + self.args.task + "-" + self.args.data + ".pyt")))
            return True#stop
        else:
            return False

