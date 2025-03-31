#!/bin/bash
:'
#MuAt motif esized
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_original_motif.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotif_orig_esize/fold_${i}/BEST_MuAtMotif_orig_esize_fold${i}.pth --valid --full_data
done

#MuAt motif pos esized
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_original_motifpos.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotifPos_orig_esize/fold_${i}/BEST_MuAtMotifPos_orig_esize_fold${i}.pth --valid --full_data
done

#MuAt motif Pos GES esized
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_original_motifposGES.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotifPosGES_orig_esize/fold_${i}/BEST_MuAtMotifPosGES_orig_esize_fold${i}.pth --valid --full_data
done

#MuAt motif epipos
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_orig_epipos.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtEpiPos/fold_${i}/BEST_MuAtEpiPos_fold${i}.pth --valid --full_data
done

#MuAt motif GES epipos
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_orig_epiposGES.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtEpiPosGES/fold_${i}/BEST_MuAtEpiPosGES_fold${i}.pth --valid --full_data
done

#MuAt motif Pos epipos
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_orig_Pos_epipos.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotifPositionEpiPos/fold_${i}/BEST_MuAtMotifPositionEpiPos_fold${i}.pth --valid --full_data
done

#MuAt motif Pos GES epipos 
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_orig_PosGES_epipos.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotifPositionGESEpiPos/fold_${i}/BEST_MuAtMotifPositionGESEpiPos_fold${i}.pth --valid --full_data
done

#MuAt motif
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_original_motif.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotif_orig/fold_${i}/BEST_MuAtMotif_orig_fold${i}.pth --valid --full_data
done

#MuAt motif pos
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_original_motifpos.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotifPos_orig/fold_${i}/BEST_MuAtMotifPos_orig_fold${i}.pth --valid --full_data
done

#MuAt motif Pos GES
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_original_motifposGES.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotifPosGES_orig/fold_${i}/BEST_MuAtMotifPosGES_orig_fold${i}.pth --valid --full_data
done

#continue here
#MuAt motif3

for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_motif3.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotif3/fold_${i}/BEST_MuAtMotif3_fold${i}.pth --valid --full_data
done

#MuAt motif3 Pos

for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_motif3pos.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotif3Pos/fold_${i}/BEST_MuAtMotif3Pos_fold${i}.pth --valid --full_data
done

#MuAt motif3 Pos Ges

for i in {5..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_motif3posGES.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotif3PosGES/fold_${i}/BEST_MuAtMotif3PosGES_fold${i}.pth --valid --full_data
done

#MuAt motif101
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_motif101.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotif101/fold_${i}/BEST_MuAtMotif101_fold${i}.pth --valid --full_data
done

#MuAt motif101 Pos
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_motif101pos.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotif101Pos/fold_${i}/BEST_MuAtMotif101Pos_fold${i}.pth --valid --full_data
done

python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_motif101posGES.ini --fold 1 --load /csc/epitkane/projects/multimodal/models/MuAtMotif101PosGES/fold_1/BEST_MuAtMotif101PosGES_fold1.pth --valid --full_data

#MuAt motif101 Pos Ges
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_motif101posGES.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotif101PosGES/fold_${i}/BEST_MuAtMotif101PosGES_fold${i}.pth --valid --full_data
done

#MuAt onehot motif3
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_onehot3.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtOneHot3/fold_${i}/BEST_MuAtOneHot3_fold${i}.pth --valid --full_data
done

#MuAt onehot motif3 Pos 
for i in {1..10};main.py
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_onehot3pos.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtOneHot3Pos/fold_${i}/BEST_MuAtOneHot3Pos_fold${i}.pth --valid --full_data
done

#MuAt onehot motif3 Pos Ges
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_onehot3posGES.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtOneHot3PosGES/fold_${i}/BEST_MuAtOneHot3PosGES_fold${i}.pth --valid --full_data
done

#MuAt motif3 SNVs only
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_motif3_SNV.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotif3_SNV_only/fold_${i}/BEST_MuAtMotif3_SNV_only_fold${i}.pth --valid --full_data
done

#MuAt motif3 diff
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_motif3_diff.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotif3_diff/fold_${i}/BEST_MuAtMotif3_diff_fold${i}.pth --valid --full_data
done

#MuAt motif3 pos diff
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_motif3Pos_diff.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotif3Pos_diff/fold_${i}/BEST_MuAtMotif3Pos_diff_fold${i}.pth --valid --full_data
done

#MuAt motif3 pos  GES diff
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_motif3PosGES_diff.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtMotif3PosGES_diff/fold_${i}/BEST_MuAtMotif3PosGES_diff_fold${i}.pth --valid --full_data
done
'
#onehot 101
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_onehot101.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtOneHot101/fold_${i}/BEST_MuAtOneHot101_fold${i}.pth --valid --full_data
done

#onehot 101 pos
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_onehot101pos.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtOneHot101Pos/fold_${i}/BEST_MuAtOneHot101Pos_fold${i}.pth --valid --full_data
done

#onehot 101 pos GES
for i in {1..10};
do
    python3 main.py --config_file /csc/epitkane/projects/multimodal/configs/config_onehot101posGES.ini --fold $i --load /csc/epitkane/projects/multimodal/models/MuAtOneHot101PosGES/fold_${i}/BEST_MuAtOneHot101PosGES_fold${i}.pth --valid --full_data
done