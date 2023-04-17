# TGACN
code for paper "Link-aware link prediction over temporal graph by pattern recognition"

This repo provides a reference implementation of TGACN as described in paper "Link-aware link prediction over temporal graph by pattern recognition"

before you run the following command, please  replace "pro_path" in  utils/args.py as your own project path.
```
python  main.py  --data uci  --gpu 1   --recent 6  --para 0 --patience 3  --model 0   --trace_step 35 --use_timec;  
python  main.py  --data social  --gpu 1   --recent 6   --para 0  --patience 3   --model 0   --trace_step 35 --use_timec; 
python  main.py  --data enron  --gpu 1 --recent 5  --para 4   --patience 3  --model 0   --trace_step 35   --use_timec;
python  main.py  --data wikipedia  --gpu 1   --recent 5  --para 4 --patience 3  --model 0   --trace_step 35 --use_timec; 
python  main.py  --data lastfm  --gpu 1  --recent 6  --para 3  --patience 3  --model 0  --trace_step 35    --use_timec;  
python  main.py  --data mooc  --gpu 1  --recent 9 --para 0 --use_timee --use_timec --dropout 0.07  --patience 3  --model 0  --trace_step 35  
