from utils.datautil import *
from utils.sampler import *
from utils.log import get_logger
from utils.args import *

from tasks.link_prediction import *
from itertools import product

if __name__=="__main__":
    start_time = time.time()
    set_random_seed(42)
    args = load_args()
    args.model_path=os.path.join(args.pro_path,"saved_models")
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    args.logger=get_logger(args)
    df, n_feat, e_feat,val_idx,test_idx=load_data(args)
    file = os.path.join(args.pro_path, "data/{}/time_adj_util".format(args.data))
    if os.path.exists(file):
        adj_util=torch.load(file)
    else:
        adj_util= make_adjlist(df)
        torch.save(adj_util,file)

    args.logger.info(args)
    args.device=torch.device("cuda:{}".format(args.gpu) if args.gpu>=0 and torch.cuda.is_available() else "cpu")
    args.logger.info("#" * 20 + "Start" + "#" * 20)

    task = link_prediction(args, df, n_feat, e_feat, val_idx, test_idx, adj_util)
    task.run()
















    args.logger.info("#"*20+"Done"+"#"*20)
