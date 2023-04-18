# TGACN
This repo provides a reference implementation of TGACN as described in paper "Link-aware link prediction over temporal graph by pattern recognition"

Before you run the following command, 
1. download the [datasets](https://drive.google.com/drive/folders/1nqnEXxGe7RBTyFl9CLTfLSzdpI_uFG-X?usp=share_link) and let this folder under the root directory of this project. Name this folder as "data".
2. replace "pro_path" in  utils/args.py as your own project path.
```
python  main.py  --data uci  --gpu 1   --recent 6  --para 0 --patience 3  --model 0   --trace_step 35 --use_timec;  
python  main.py  --data social  --gpu 1   --recent 6   --para 0  --patience 3   --model 0   --trace_step 35 --use_timec; 
python  main.py  --data enron  --gpu 1 --recent 5  --para 4   --patience 3  --model 0   --trace_step 35   --use_timec;
python  main.py  --data wikipedia  --gpu 1   --recent 5  --para 4 --patience 3  --model 0   --trace_step 35 --use_timec; 
python  main.py  --data lastfm  --gpu 1  --recent 6  --para 3  --patience 3  --model 0  --trace_step 35    --use_timec;  
python  main.py  --data mooc  --gpu 1  --recent 9 --para 0 --use_timee --use_timec --dropout 0.07  --patience 3  --model 0  --trace_step 35  
