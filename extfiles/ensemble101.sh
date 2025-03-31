#!/bin/bash
:'
#onehot 101
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_onehot101.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtOneHot101/fold_${i}/BEST_MuAtOneHot101_fold${i}.pth --valid --full_data
done
'
#onehot 101 pos
for i in {8..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_onehot101pos.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtOneHot101Pos/fold_${i}/BEST_MuAtOneHot101Pos_fold${i}.pth --valid --full_data
done

#onehot 101 pos GES
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_onehot101posGES.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtOneHot101PosGES/fold_${i}/BEST_MuAtOneHot101PosGES_fold${i}.pth --valid --full_data
done