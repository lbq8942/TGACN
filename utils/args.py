import argparse







def load_args():
    parser = argparse.ArgumentParser('TGACN')

    parser.add_argument('--task', type=str,default="lp",help='classification task')
    parser.add_argument('--data', type=str,default="wikipedia",help='Dataset name')
    parser.add_argument('--bs', type=int, default=128, help='Batch_size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning_rate')
    parser.add_argument('--epochs', type=int, default=50, help='maximum epochs to run')
    parser.add_argument('--patience', type=int, default=5, help='early stop')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use,if cpu,set this to -1')
    parser.add_argument('--data_usage', type=float, default=1, help='use how many data')
    parser.add_argument('--eval_every_epoch', type=int, default=1, help=' evaluate how many times in one epoch')
    parser.add_argument('--metrics', type=str, default="acc", choices=["f1","acc","auc"],help='which metrics to determine model to be saved')
    parser.add_argument('--stage', type=int,nargs='*', default=[0,1],help="want stage 1 and 2 in effnet")
    

    
    
    parser.add_argument('--recent', type=int, default=6, help='nearest sampling')
    parser.add_argument('--para', type=int, default=3, help='parametric sampling')

    
    parser.add_argument('--trace_step', type=int, default=12, help='M in paper')
    parser.add_argument('--window', type=int, default=12,help="how many events within this step are considered as positive")#not used
    parser.add_argument('--skip_gram', type=int,nargs='+', default=[3,3],help="within window,sample how much postive and out of window ,how much negtive events")#not used
    

    
    parser.add_argument('--dropout', type=float,default=0.2,help="dropout for transductive attention")
    parser.add_argument('--use_timee', action="store_true",help="whether time as part of event embedding")
    parser.add_argument('--use_timec', action="store_true",help="whether time as part of channels")
    parser.add_argument('--alpha', type=float, default=5.0, help="exp(-alpha|x-y|) in time encoding")

    parser.add_argument('--neg_weight',type=float, default=1.0 ,help="weight for loss of negative sampling")#not used
    parser.add_argument('--hdim', type=int, default=256,help="dimension of different vector")
    parser.add_argument('--K', type=int, default=1,choices=[1,2],help="k-hop events sampling")#currently not used.
    parser.add_argument('--model', type=int, default=0,help="choose which backbone network")
    parser.add_argument('--training_proportion',type=float, default=1.0 ,help="use only little data but competitve performance")

    parser.add_argument('--pro_path', type=str, default="/home/liubingqing/project/TGACN", help='project path')#you should replace this with your own project path.
    
    parser.add_argument('--testing', action="store_true", help='training or testing')
    parser.add_argument('--load_path', type=str, default="1",help="the path of model  when training is false")
    parser.add_argument('--grid_search', action="store_true",help="search hyperparameter from the hp_range")#currently not used.

    args = parser.parse_args()


    
    assert args.window>=args.recent
    

    return args


