#!/bin/bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 

if [ $1 -eq 1 ] # fundus; train on train+valid+test, test on target domain
then
    IFS='_' read -r -a array <<< "$2"
    python da_train.py \
    --dataset fundus \
    --source_root ./data/refuge/train \
    --target_root ./data/refuge/train \
    --split \
    --num-class 2 \
    --resize 512 512 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --epochs 45 \
    --branch_num "${array[@]}" \
    --save-per-epochs 1
elif [ $1 -eq 100 ] # UDA
then
    IFS='_' read -r -a array <<< "$2"
    python mt_da_train.py \
    --dataset rim \
    --source_root ./data/refuge/train \
    --target_root ./data/rim \
    --num-class 2 \
    --resize 512 512 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 100 \
    --batch-size 2 \
    --start-adapt-iters 0 \
    --pair-id 100 \
    --iters-to-stop 10000 \
    --use-logit \
    --mask_fix_on_source_target 0 0 \
    --mask_dy_size_ratio 1 1 \
    --mask_on_source_target 1 1 \
    --Lambda 0.6 \
    --mu 0.3 \
    --beta 1 \
    --rmax 0.75 \
    --mask-ratio 0.35 \
    --mask-ratio-t 0.35 \
    --rmin 0.1 \
    --smin 32 \
    --sel_mask 1 \
    --seed 2 \
    --branch_num "${array[@]}" \
    --pretrained $3
    # --sel_objects $4 \
    # --rand_paired $5 \
# elif [ $1 -eq 1010 ] # few-shot, train: refuge, 1200 * 50%; test: refuge, 1200 * 50%
# then
#     python da_train.py \
#     --dataset fundus \
#     --source_root ./data/refuge/train ./data/refuge/test ./data/refuge/valid \
#     --target_root ./data/refuge/train ./data/refuge/test ./data/refuge/valid \
#     --split \
#     --pair-id 1010 \
#     --num-class 2 \
#     --resize 512 512 \
#     --lr 0.0001 \
#     --lr-update 'poly' \
#     --epochs 45 \
#     --save-per-epochs 1
# elif [ $1 -eq 101 ] # few shot w/ rim + refuge ---> rim
# then
#     python mt_da_train.py \
#     --dataset fewshotrim \
#     --source_root ./data/refuge/train ./data/refuge/test ./data/refuge/valid \
#     --target_root ./data/rim \
#     --few_shot_root "./data/rim/few_shot_5.$2.csv" \
#     --num-class 2 \
#     --resize 512 512 \
#     --lr 0.00006 \
#     --lr-update poly \
#     --iters 20000 \
#     --alpha 0.999 \
#     --save-per-iters 100 \
#     --batch-size 2 \
#     --start-adapt-iters 0 \
#     --pair-id $1 \
#     --iters-to-stop 10000 \
#     --mask_fix_on_source_target 1 0 \
#     --mask_dy_size_ratio 1 1 \
#     --mask_on_source_target 1 1 \
#     --Lambda 1 \
#     --mu 0.3 \
#     --beta 1 \
#     --rmax 0.75 \
#     --rmin 0.1 \
#     --smin 32 \
#     --sel_mask \
#     --use-logit \
#     --pretrained 0.512_pth_Refuge_rim/0.13_0.8097_0.9150.pth
# elif [ $1 -eq 102 ] # few shot w/o rim + refuge ---> rim
# then
#     python mt_da_train.py \
#     --dataset fewshotrim \
#     --source_root ./data/refuge/train ./data/refuge/test ./data/refuge/valid \
#     --target_root ./data/rim \
#     --few_shot_root "./data/rim/few_shot_5.$2.csv" \
#     --few_shot_only \
#     --num-class 2 \
#     --resize 512 512 \
#     --lr 0.00006 \
#     --lr-update poly \
#     --iters 20000 \
#     --alpha 0.999 \
#     --save-per-iters 5 \
#     --batch-size 2 \
#     --start-adapt-iters 0 \
#     --pair-id $1 \
#     --iters-to-stop 10000 \
#     --mask_fix_on_source_target 1 0 \
#     --mask_dy_size_ratio 1 1 \
#     --mask_on_source_target 1 1 \
#     --Lambda 1 \
#     --mu 0.3 \
#     --beta 1 \
#     --rmax 0.75 \
#     --rmin 0.1 \
#     --smin 32 \
#     --sel_mask \
#     --pretrained 0.512_pth_Refuge_50%/1.42_0.8957_0.9589.pth
    # --use-logit \
# elif [ $1 -eq 2 ] # fundus; train on B&M, test on target RIM
# then
#     python da_train.py \
#     --dataset fundus \
#     --source_root ./data/BinRushed ./data/Magrabia \
#     --target_root ./data/rim \
#     --num-class 2 \
#     --resize 512 512 \
#     --lr 0.0005 \
#     --lr-update 'poly' \
#     --epochs 45 \
#     --save-per-epochs 1
# elif [ $1 -eq 200 ] # fundus; UDA mean-teacher version 1
# then
#     python mt_da_train.py \
#     --dataset fundus \
#     --source_root ./data/BinRushed ./data/Magrabia \
#     --target_root ./data/rim \
#     --num-class 2 \
#     --resize 512 512 \
#     --lr 0.00006 \
#     --lr-update poly \
#     --iters 20000 \
#     --alpha 0.999 \
#     --save-per-iters 100 \
#     --batch-size 2 \
#     --start-adapt-iters 0 \
#     --pair-id $1 \
#     --lambda-hyp $2 \
#     --pretrained 0.512_pth_BM_rim/2.51_0.8521_0.9339.pth
# elif [ $1 -eq 3 ] # fundus; train on refuge, test on Drishti-GS
# then
#     python da_train.py \
#     --dataset fundus \
#     --source_root ./data/refuge/train \
#     --target_root ./data/Drishti-GS/train ./data/Drishti-GS/test \
#     --num-class 2 \
#     --resize 512 512 \
#     --lr 0.0005 \
#     --lr-update 'poly' \
#     --epochs 45 \
#     --save-per-epochs 1
elif [ $1 -eq 300 ] # need increment=8
then
    IFS='_' read -r -a array <<< "$2"
    python mt_da_train.py \
    --dataset fundus \
    --source_root ./data/refuge/train \
    --target_root ./data/Drishti-GS/train \
    --test_root ./data/Drishti-GS/test \
    --num-class 2 \
    --resize 512 512 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 100 \
    --batch-size 2 \
    --start-adapt-iters 0 \
    --pair-id $1 \
    --iters-to-stop 10000 \
    --aux-coeff 0.3 \
    --mask_fix_on_source_target 0 0 \
    --mask_dy_size_ratio 1 1 \
    --mask_on_source_target 1 1 \
    --Lambda $4 \
    --mu 0.3 \
    --beta 3 \
    --rmax 0.75 \
    --rmin 0.1 \
    --smin 32 \
    --sel_mask \
    --seed 2 \
    --branch_num "${array[@]}" \
    --pretrained $3
elif [ $1 -eq 4 ] # fundus; train on B&M, test on MESSIDOR_Base1
then
    IFS='_' read -r -a array <<< "$2"
    python da_train.py \
    --dataset basex \
    --num-class 2 \
    --resize 512 512 \
    --lr 0.0005 \
    --lr-update 'poly' \
    --epochs 45 \
    --branch_num "${array[@]}" \
    --save-per-epochs 1 
elif [ $1 -eq 400 ] # fundus; UDA mean-teacher version 1
then
    IFS='_' read -r -a array <<< "$2"
    python mt_da_train.py \
    --dataset basex \
    --target_root MESSIDOR_Base1 \
    --num-class 2 \
    --resize 512 512 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 100 \
    --batch-size 2 \
    --start-adapt-iters 0 \
    --pair-id 400 \
    --iters-to-stop 1000 \
    --mask_fix_on_source_target 1 0 \
    --mask_dy_size_ratio 1 1 \
    --mask_on_source_target 1 1 \
    --Lambda 0.6 \
    --mu 0.3 \
    --beta 0.8 \
    --rmax 0.75 \
    --rmin 0.6 \
    --smin 32 \
    --seed 2 \
    --sel_mask 0 \
    --sel_objects 1 \
    --branch_num "${array[@]}" \
    --pretrained $3
    # --Lambda 0.6 \
elif [ $1 -eq 500 ] # fundus; UDA mean-teacher version 1
then
    IFS='_' read -r -a array <<< "$2"
    python mt_da_train.py \
    --dataset basex \
    --target_root MESSIDOR_Base2 \
    --num-class 2 \
    --resize 512 512 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 100 \
    --batch-size 2 \
    --start-adapt-iters 0 \
    --pair-id 500 \
    --iters-to-stop 1000 \
    --mask_fix_on_source_target 1 0 \
    --mask_dy_size_ratio 1 1 \
    --mask_on_source_target 1 1 \
    --mask-size-t 35 \
    --mask-size 35 \
    --Lambda 0.6 \
    --mu 0.15 \
    --beta 0.8 \
    --rmax 0.75 \
    --rmin 0.6 \
    --smin 8 \
    --seed 2 \
    --sel_mask 0 \
    --sel_objects 1 \
    --branch_num "${array[@]}" \
    --pretrained $3
    # --Lambda 0.6 \
elif [ $1 -eq 600 ] # fundus; UDA mean-teacher version 1
then
    IFS='_' read -r -a array <<< "$2"
    python mt_da_train.py \
    --dataset basex \
    --target_root MESSIDOR_Base3 \
    --num-class 2 \
    --resize 512 512 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 100 \
    --batch-size 2 \
    --start-adapt-iters 0 \
    --pair-id 600 \
    --iters-to-stop 1000 \
    --mask_fix_on_source_target 1 0 \
    --mask_dy_size_ratio 1 1 \
    --mask_on_source_target 1 1 \
    --mask-size-t 55 \
    --mask-size 55 \
    --Lambda 0.6 \
    --mu 0.25 \
    --beta 0.8 \
    --rmax 0.75 \
    --rmin 0.6 \
    --smin 32 \
    --seed 2 \
    --sel_mask 0 \
    --sel_objects 1 \
    --branch_num "${array[@]}" \
    --pretrained $3
    # --Lambda 0.6 \
elif [ $1 -eq 12 ] # polyp; train on CVC-612+kvasir, test on VC-300
then
    IFS='_' read -r -a array <<< "$2"
    python da_train.py \
    --dataset polyp \
    --source_root ./data/polyp/Kvasir \
    --target_root ./data/polyp/Kvasir \
    --split \
    --num-class 1 \
    --resize 352 352 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --epochs 150 \
    --threshold 0.9 \
    --pair-id $1 \
    --branch_num "${array[@]}" \
    --save-per-epochs 1
elif [ $1 -eq 1200 ] # polyp; UDA mean-teacher version 1
then
    IFS='_' read -r -a array <<< "$2"
    python mt_da_train.py \
    --dataset polyp \
    --source_root ./data/polyp/Kvasir \
    --target_root ./data/polyp/ETIS-LaribPolypDB \
    --num-class 1 \
    --resize 352 352 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 100 \
    --batch-size 2 \
    --threshold 0.90 \
    --iters-to-stop 20000 \
    --mask-ratio 0.3 \
    --mask-ratio-t 0.3 \
    --mask-size-t 16 \
    --mask-size 16 \
    --pair-id 1200 \
    --mask_fix_on_source_target 1 0 \
    --mask_dy_size_ratio $6 $7 \
    --mask_on_source_target 1 1 \
    --Lambda 0.6 \
    --mu 0.3 \
    --beta 5.25 \
    --rmax 0.45 \
    --rmin 0.1 \
    --smin 16 \
    --seed 2 \
    --sel_mask $4 \
    --rand_paired $5 \
    --branch_num "${array[@]}" \
    --pretrained $3
# elif [ $1 -eq 13 ] # polyp; train on CVC-612+kvasir, test on VC-300
# then
#     python da_train.py \
#     --dataset polyp \
#     --source_root ./data/polyp/Kvasir \
#     --target_root ./data/polyp/CVC-Endo/Valid ./data/polyp/CVC-Endo/Train ./data/polyp/CVC-Endo/Test \
#     --num-class 1 \
#     --resize 352 352 \
#     --lr 0.0001 \
#     --lr-update 'poly' \
#     --epochs 150 \
#     --threshold 0.9 \
#     --pair-id $1 \
#     --save-per-epochs 1
elif [ $1 -eq 1300 ] # polyp; UDA mean-teacher version 1
then
    IFS='_' read -r -a array <<< "$2"
    python mt_da_train.py \
    --dataset polyp \
    --source_root ./data/polyp/Kvasir \
    --target_root ./data/polyp/CVC-Endo/Valid ./data/polyp/CVC-Endo/Train ./data/polyp/CVC-Endo/Test \
    --num-class 1 \
    --resize 352 352 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 100 \
    --batch-size 2 \
    --threshold 0.90 \
    --iters-to-stop 15000 \
    --mask-ratio 0.3 \
    --mask-ratio-t 0.3 \
    --mask-size-t 16 \
    --mask-size 16 \
    --pair-id $1 \
    --mask_fix_on_source_target 1 0 \
    --mask_dy_size_ratio 1 1 \
    --mask_on_source_target 1 1 \
    --Lambda $4 \
    --mu 0.3 \
    --beta 5 \
    --rmax 0.45 \
    --rmin 0.1 \
    --smin 16 \
    --seed 2 \
    --branch_num "${array[@]}" \
    --pretrained $3
elif [ $1 -eq 14 ] # polyp; train on CVC-612+kvasir, test on VC-300
then
    IFS='_' read -r -a array <<< "$2"
    python da_train.py \
    --dataset polyp \
    --source_root ./data/polyp/ETIS-LaribPolypDB \
    --target_root ./data/polyp/ETIS-LaribPolypDB \
    --split \
    --num-class 1 \
    --resize 352 352 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --epochs 200 \
    --threshold 0.7 \
    --pair-id $1 \
    --branch_num "${array[@]}" \
    --save-per-epochs 1
elif [ $1 -eq 1400 ] # polyp; UDA mean-teacher version 1
then
    IFS='_' read -r -a array <<< "$2"
    python mt_da_train.py \
    --dataset polyp \
    --source_root ./data/polyp/ETIS-LaribPolypDB \
    --target_root ./data/polyp/CVC-Endo/Valid ./data/polyp/CVC-Endo/Train ./data/polyp/CVC-Endo/Test \
    --num-class 1 \
    --resize 352 352 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 100 \
    --batch-size 2 \
    --threshold 0.70 \
    --iters-to-stop 20000 \
    --mask-ratio 0.3 \
    --mask-ratio-t 0.3 \
    --mask-size-t 16 \
    --mask-size 16 \
    --pair-id 1400 \
    --mask_fix_on_source_target 1 0 \
    --mask_dy_size_ratio 1 1 \
    --mask_on_source_target 1 1 \
    --Lambda $4 \
    --mu 0.3 \
    --beta 5 \
    --rmax 0.45 \
    --rmin 0.1 \
    --smin 16 \
    --seed 2 \
    --branch_num "${array[@]}" \
    --pretrained $3
# elif [ $1 -eq 15 ] # polyp; train on CVC-612+kvasir, test on VC-300
# then
#     python da_train.py \
#     --dataset polyp \
#     --source_root ./data/polyp/ETIS-LaribPolypDB \
#     --target_root ./data/polyp/Kvasir \
#     --num-class 1 \
#     --resize 352 352 \
#     --lr 0.0001 \
#     --lr-update 'poly' \
#     --epochs 150 \
#     --threshold 0.9 \
#     --pair-id $1 \
#     --save-per-epochs 1
elif [ $1 -eq 1500 ] # polyp; UDA mean-teacher version 1
then
    IFS='_' read -r -a array <<< "$2"
    python mt_da_train.py \
    --dataset polyp \
    --source_root ./data/polyp/ETIS-LaribPolypDB \
    --target_root ./data/polyp/Kvasir \
    --num-class 1 \
    --resize 352 352 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 100 \
    --batch-size 2 \
    --threshold 0.90 \
    --iters-to-stop 20000 \
    --mask-ratio 0.3 \
    --mask-ratio-t 0.3 \
    --mask-size-t 16 \
    --mask-size 16 \
    --pair-id 1500 \
    --mask_fix_on_source_target 1 0 \
    --mask_dy_size_ratio 1 1 \
    --mask_on_source_target 1 1 \
    --Lambda $4 \
    --mu 0.2 \
    --beta 5 \
    --rmax 0.3 \
    --rmin 0.1 \
    --smin 16 \
    --seed 2 \
    --branch_num "${array[@]}" \
    --pretrained $3
elif [ $1 -eq 16 ] # polyp; train on CVC-612+kvasir, test on VC-300
then
    IFS='_' read -r -a array <<< "$2"
    python da_train.py \
    --dataset polyp_v2 \
    --source_root ./data/polyp/CVC-ColonDB \
    --target_root ./data/polyp/CVC-ColonDB \
    --split \
    --num-class 1 \
    --resize 256 256 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --epochs 150 \
    --threshold 0.9 \
    --pair-id $1 \
    --branch_num "${array[@]}" \
    --save-per-epochs 1
elif [ $1 -eq 1600 ] # polyp; UDA mean-teacher version 1
then
    IFS='_' read -r -a array <<< "$2"
    python mt_da_train.py \
    --dataset polyp_v2 \
    --source_root ./data/polyp/CVC-ColonDB \
    --target_root ./data/polyp/ETIS-LaribPolypDB \
    --test_root ./data/polyp/ETIS-LaribPolypDB \
    --split \
    --num-class 1 \
    --resize 256 256 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 100 \
    --batch-size 2 \
    --threshold 0.90 \
    --iters-to-stop 10000 \
    --mask-ratio 0.3 \
    --mask-ratio-t 0.3 \
    --mask-size-t 16 \
    --mask-size 16 \
    --pair-id $1 \
    --mask_fix_on_source_target 1 0 \
    --mask_dy_size_ratio 1 1 \
    --mask_on_source_target 1 1 \
    --Lambda 0.6 \
    --mu 0.3 \
    --beta 5 \
    --rmax 0.45 \
    --rmin 0.3 \
    --smin 16 \
    --branch_num "${array[@]}" \
    --pretrained $3
elif [ $1 -eq 17 ] # polyp; train on CVC-612+kvasir, test on VC-300
then
    IFS='_' read -r -a array <<< "$2"
    python da_train.py \
    --dataset polyp_v2 \
    --source_root ./data/polyp/ETIS-LaribPolypDB \
    --target_root ./data/polyp/ETIS-LaribPolypDB \
    --split \
    --num-class 1 \
    --resize 256 256 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --epochs 150 \
    --threshold 0.9 \
    --pair-id $1 \
    --branch_num "${array[@]}" \
    --save-per-epochs 1
elif [ $1 -eq 1700 ] # polyp; UDA mean-teacher version 1
then
    IFS='_' read -r -a array <<< "$2"
    python mt_da_train.py \
    --dataset polyp_v2 \
    --source_root ./data/polyp/ETIS-LaribPolypDB \
    --target_root ./data/polyp/CVC-ColonDB \
    --test_root ./data/polyp/CVC-ColonDB \
    --split \
    --num-class 1 \
    --resize 256 256 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 100 \
    --batch-size 2 \
    --threshold 0.90 \
    --iters-to-stop 15000 \
    --mask-ratio 0.3 \
    --mask-ratio-t 0.3 \
    --mask-size-t 16 \
    --mask-size 16 \
    --pair-id 1700 \
    --mask_fix_on_source_target 1 0 \
    --mask_dy_size_ratio 1 1 \
    --mask_on_source_target 1 1 \
    --Lambda 0.6 \
    --mu 0.3 \
    --beta 5 \
    --rmax 0.33 \
    --rmin 0.1 \
    --smin 16 \
    --branch_num "${array[@]}" \
    --pretrained $3
elif [ $1 -eq 181 ] # polyp; train on (CVC-612+kvasir) * 50%, test on CVC-300
then
    python da_train.py \
    --dataset polyp \
    --source_root ./data/polyp/CVC-612 ./data/polyp/Kvasir \
    --target_root ./data/polyp/CVC-300 \
    --num-class 1 \
    --resize 320 320 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --epochs 150 \
    --threshold 0.9 \
    --pair-id $1 \
    --save-per-epochs 1
    # --split \
elif [ $1 -eq 1801 ] # polyp; cvc-612 + Kvasir + few shot with rest
then
    python mt_da_train.py \
    --dataset fewshotcvc300 \
    --source_root ./data/polyp/CVC-612 ./data/polyp/Kvasir \
    --target_root ./data/polyp/CVC-300 \
    --few_shot_root "./data/polyp/CVC-300/few_shot_5.$2.csv" \
    --num-class 1 \
    --resize 320 320 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 100 \
    --batch-size 2 \
    --threshold 0.90 \
    --iters-to-stop 15000 \
    --mask-ratio 0.3 \
    --mask-ratio-t 0.3 \
    --mask-size-t 16 \
    --mask-size 16 \
    --pair-id $1 \
    --mask_fix_on_source_target 1 0 \
    --mask_dy_size_ratio 1 1 \
    --mask_on_source_target 1 1 \
    --Lambda 0.5 \
    --mu 0.3 \
    --beta 5 \
    --rmax 0.45 \
    --rmin 0.1 \
    --smin 16 \
    --pretrained 1.320_pth_cvc612_kvasir/0.33_0.9135.pth
elif [ $1 -eq 1802 ] # polyp; cvc-612 + Kvasir + few shot only
then
    python mt_da_train.py \
    --dataset fewshotcvc300 \
    --source_root ./data/polyp/CVC-612 ./data/polyp/Kvasir \
    --target_root ./data/polyp/CVC-300 \
    --few_shot_root "./data/polyp/CVC-300/few_shot_5.$2.csv" \
    --few_shot_only \
    --num-class 1 \
    --resize 320 320 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 5 \
    --batch-size 2 \
    --threshold 0.90 \
    --iters-to-stop 1000 \
    --mask-ratio 0.3 \
    --mask-ratio-t 0.3 \
    --mask-size-t 16 \
    --mask-size 16 \
    --pair-id $1 \
    --mask_fix_on_source_target 1 0 \
    --mask_dy_size_ratio 1 1 \
    --mask_on_source_target 1 1 \
    --Lambda 0.5 \
    --mu 0.3 \
    --beta 5 \
    --rmax 0.45 \
    --rmin 0.1 \
    --smin 16 \
    --pretrained 1.320_pth_cvc612_kvasir_50%/2.150_0.8822.pth
# elif [ $1 -eq 18 ] # polyp; train on CVC-612+kvasir, test on VC-300
# then
#     python da_train.py \
#     --dataset polyp \
#     --source_root ./data/polyp/CVC-Endo/Valid ./data/polyp/CVC-Endo/Train ./data/polyp/CVC-Endo/Test \
#     --target_root ./data/polyp/ETIS-LaribPolypDB \
#     --num-class 1 \
#     --resize 256 256 \
#     --lr 0.0001 \
#     --lr-update 'poly' \
#     --epochs 150 \
#     --threshold 0.9 \
#     --pair-id $1 \
#     --save-per-epochs 1
# elif [ $1 -eq 1800 ] # polyp; UDA mean-teacher version 1
# then
#     python mt_da_train.py \
#     --dataset polyp \
#     --source_root ./data/polyp/CVC-Endo/Valid ./data/polyp/CVC-Endo/Train ./data/polyp/CVC-Endo/Test \
#     --target_root ./data/polyp/ETIS-LaribPolypDB \
#     --num-class 1 \
#     --resize 256 256 \
#     --lr 0.00006 \
#     --lr-update poly \
#     --iters 20000 \
#     --alpha 0.999 \
#     --save-per-iters 100 \
#     --batch-size 2 \
#     --threshold 0.90 \
#     --pair-id $1 \
#     --mask-size 16 \
#     --mask-size-t 16 \
#     --mask-ratio 0.30 \
#     --mask-ratio-t 0.30 \
#     --iters-to-stop 20000 \
#     --use-hardness \
#     --pretrained 18/1.51_0.8033.pth #85.61
# elif [ $1 -eq 19 ] # polyp; train on CVC-612+kvasir, test on VC-300
# then
#     python da_train.py \
#     --dataset polyp \
#     --source_root ./data/polyp/CVC-612 \
#     --target_root ./data/polyp/ETIS-LaribPolypDB \
#     --num-class 1 \
#     --resize 320 320 \
#     --lr 0.0001 \
#     --lr-update 'poly' \
#     --epochs 150 \
#     --threshold 0.9 \
#     --pair-id $1 \
#     --save-per-epochs 1
# elif [ $1 -eq 1900 ] # polyp; UDA mean-teacher version 1
# then
#     python mt_da_train.py \
#     --dataset polyp \
#     --source_root ./data/polyp/CVC-612 \
#     --target_root ./data/polyp/ETIS-LaribPolypDB \
#     --num-class 1 \
#     --resize 320 320 \
#     --lr 0.00006 \
#     --lr-update poly \
#     --iters 20000 \
#     --alpha 0.999 \
#     --save-per-iters 100 \
#     --batch-size 2 \
#     --threshold 0.90 \
#     --self-kd $2 \
#     --pair-id $1 \
#     --mask-size 15 \
#     --mask-size-t 15 \
#     --mask-ratio 0.30 \
#     --mask-ratio-t 0.30 \
#     --iters-to-stop 20000 \
#     --pretrained 1.320_pth_endo_etis/2.93_0.8372.pth
elif [ $1 -eq 18 ] # mmwhs ct;
then
    python da_train.py \
    --dataset MMWHS \
    --source_root '/home/yongze/Desktop/mmwhs/ct_train_/' \
    --target_root '/home/yongze/Desktop/mmwhs/ct_test_/' \
    --pair-id $1 \
    --num-class 4 \
    --resize 256 256 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --epochs 45 \
    --threshold 0.9 \
    --seed 2 \
    --save-per-epochs 1
    # --test_only \
    # --pretrained 18/1.2_0.9275_0.9377_0.9420_0.9575.pth
elif [ $1 -eq 1800 ] # ct -> mr
then
    python mt_da_train.py \
    --dataset MMWHS \
    --source_root '/home/yongze/Desktop/mmwhs/ct_train_/' \
    --target_root '/home/yongze/Desktop/mmwhs/mr_test_/' \
    --num-class 4 \
    --resize 256 256 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 100 \
    --batch-size 2 \
    --start-adapt-iters 0 \
    --pair-id 1800 \
    --iters-to-stop 20000 \
    --use-logit \
    --mask-ratio 0.3 \
    --mask-ratio-t 0.3 \
    --mask-size-t 64 \
    --mask-size 64 \
    --mask_fix_on_source_target 1 1 \
    --mask_dy_size_ratio 1 1 \
    --mask_on_source_target 1 1 \
    --Lambda 0.5 \
    --mu 0.3 \
    --beta 0.6 \
    --rmax 0.45 \
    --rmin 0.1 \
    --smin 16 \
    --pretrained 3.256_mmwhs_ct_3c/2.42_0.9105_0.9311_0.9368_0.9052.pth
elif [ $1 -eq 19 ] # mmwhs mr;
then
    python da_train.py \
    --dataset MMWHS \
    --source_root '/home/yongze/Desktop/mmwhs/mr_train_/' \
    --target_root '/home/yongze/Desktop/mmwhs/mr_test_/' \
    --pair-id $1 \
    --num-class 4 \
    --resize 256 256 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --epochs 150 \
    --threshold 0.9 \
    --save-per-epochs 1 
elif [ $1 -eq 1900 ] # need increment=8
then
    python mt_da_train.py \
    --dataset MMWHS \
    --source_root '/home/yongze/Desktop/mmwhs/mr_train_/' \
    --target_root '/home/yongze/Desktop/mmwhs/ct_test_/' \
    --num-class 4 \
    --resize 256 256 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 100 \
    --batch-size 2 \
    --start-adapt-iters 0 \
    --pair-id 1900 \
    --iters-to-stop 5000 \
    --use-logit \
    --mask-ratio 0.3 \
    --mask-ratio-t 0.3 \
    --mask-size-t 64 \
    --mask-size 64 \
    --mask_fix_on_source_target 1 1 \
    --mask_dy_size_ratio 1 1 \
    --mask_on_source_target 1 1 \
    --Lambda 0.5 \
    --mu 0.3 \
    --beta 0.6 \
    --rmax 0.45 \
    --rmin 0.1 \
    --smin 16 \
    --pretrained 3.256_mmwhs_mr_3c/1.125_0.9238_0.9511_0.9532_0.9284.pth
elif [ $1 -eq 20 ] # abdominal ct;
then
    python da_train.py \
    --dataset abdominal \
    --source_root '/home/yongze/Desktop/abdominalDATA/train_ct.txt' \
    --target_root '/home/yongze/Desktop/abdominalDATA/val_ct.txt' \
    --pair-id $1 \
    --num-class 4 \
    --resize 256 256 \
    --lr 0.00001 \
    --lr-update 'poly' \
    --epochs 100 \
    --threshold 0.5 \
    --pair-id $1 \
    --save-per-epochs 1 
elif [ $1 -eq 21 ] # abdominal ct;
then
    python da_train.py \
    --dataset abdominal \
    --source_root '/home/yongze/Desktop/abdominalDATA/train_mr.txt' \
    --target_root '/home/yongze/Desktop/abdominalDATA/val_mr.txt' \
    --pair-id $1 \
    --num-class 4 \
    --resize 256 256 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --epochs 45 \
    --threshold 0.9 \
    --pair-id $1 \
    --save-per-epochs 1     
elif [ $1 -eq 22 ] # promise12 50 cases
then
    python da_train.py \
    --dataset prostate \
    --source_root '/home/yongze/Desktop/promise12/images_3c/' \
    --target_root '/home/yongze/Desktop/promise12/images_3c/' \
    --num-class 1 \
    --resize 256 256 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --epochs 100 \
    --threshold 0.9 \
    --pair-id $1 \
    --split \
    --seed $2 \
    --save-per-epochs 1
    # --net 'unet' \
elif [ $1 -eq 23 ] # prostate 30 cases
then
    python da_train.py \
    --dataset prostate \
    --source_root '/home/yongze/Desktop/prostate/images_3c/' \
    --target_root '/home/yongze/Desktop/prostate/images_3c/' \
    --pair-id $1 \
    --num-class 1 \
    --resize 256 256 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --epochs 100 \
    --threshold 0.9 \
    --pair-id $1 \
    --split \
    --seed 2 \
    --test_only \
    --inference_only \
    --save-per-epochs 1 \
    --pretrained 23_2024-05-28_15:44:14/2.25_0.8207.pth
elif [ $1 -eq 2300 ] # need increment=8
then
    python mt_da_train.py \
    --dataset prostate \
    --source_root '/home/yongze/Desktop/prostate/images_3c/' \
    --target_root '/home/yongze/Desktop/promise12/images_3c/' \
    --num-class 1 \
    --resize 256 256 \
    --lr 0.00006 \
    --lr-update poly \
    --iters 20000 \
    --alpha 0.999 \
    --save-per-iters 100 \
    --batch-size 2 \
    --start-adapt-iters 0 \
    --pair-id $1 \
    --iters-to-stop 10000 \
    --use-logit \
    --mask_fix_on_source_target 1 1 \
    --mask_dy_size_ratio 1 1 \
    --mask_on_source_target 1 1 \
    --split \
    --mask-ratio 0.3 \
    --mask-ratio-t 0.3 \
    --mask-size 32 \
    --mask-size-t 32 \
    --Lambda 0.5 \
    --mu 0.3 \
    --beta 0.6 \
    --rmax 0.45 \
    --rmin 0.1 \
    --smin 16 \
    --pretrained 2.256_prosta_3c/2.88_0.8369.pth
elif [ $1 -eq -1 ] # fundus; supervised learning on Drishti-GS
then
    python da_train.py \
    --dataset base_v2 \
    --num-class 1 \
    --resize 352 352 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --epochs 100 \
    --save-per-epochs 1 \
    --pair-id $1 \
    --source_root ./data/polyp/Kvasir \
    --target_root ./data/polyp/Kvasir
elif [ $1 -eq -2 ] # fundus; supervised learning on Drishti-GS
then
    python mt_da_train.py \
    --dataset base_v3 \
    --num-class 1 \
    --resize 352 352 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --pair-id $1 \
    --test_only \
    --source_root ./data/polyp/ETIS-LaribPolypDB/train.csv \
    --target_root ./data/polyp/ETIS-LaribPolypDB/test.csv \
    --pretrained 1200_sup/0.39_0.4538.pth
    # --inference_only \
    # --epochs 100 \
    # --save-per-epochs 1 \
elif [ $1 -eq -3 ] # fundus; supervised learning on Drishti-GS
then
    python da_train.py \
    --dataset base_v2 \
    --num-class 1 \
    --resize 256 256 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --epochs 100 \
    --save-per-epochs 1 \
    --pair-id $1 \
    --source_root ./data/polyp/CVC-ColonDB \
    --target_root ./data/polyp/CVC-ColonDB
elif [ $1 -eq -4 ] # fundus; supervised learning on Drishti-GS
then
    python da_train.py \
    --dataset base_v2 \
    --num-class 1 \
    --resize 256 256 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --epochs 100 \
    --save-per-epochs 1 \
    --pair-id $1 \
    --source_root ./data/polyp/CVC-612 \
    --target_root ./data/polyp/CVC-612
elif [ $1 -eq -5 ] # fundus; supervised learning on Drishti-GS
then
    python da_train.py \
    --dataset base_v3 \
    --num-class 1 \
    --resize 352 352 \
    --lr 0.0001 \
    --lr-update 'poly' \
    --epochs 100 \
    --save-per-epochs 1 \
    --pair-id $1 \
    --test_only \
    --inference_only \
    --source_root ./data/polyp/CVC-Endo/train.csv \
    --target_root ./data/polyp/CVC-Endo/test.csv \
    --pretrained 1400_sup/1.53_0.7418.pth
elif [ $1 -eq 4100 ] # fundus; supervised learning on Drishti-GS
then
    python da_train.py \
    --dataset fundus \
    --num-class 2 \
    --resize 512 512 \
    --lr 0.0005 \
    --lr-update 'poly' \
    --epochs 100 \
    --save-per-epochs 1 \
    --source_root ./data/Drishti-GS/train \
    --target_root ./data/Drishti-GS/test
elif [ $1 -eq 4200 ] # fundus; supervised learning on MESSIDOR_Base1
then
    python da_train.py \
    --dataset base \
    --num-class 2 \
    --resize 512 512 \
    --lr 0.0005 \
    --lr-update 'poly' \
    --epochs 100 \
    --save-per-epochs 1 \
    --source_root ./data/MESSIDOR_Base1 \
    --target_root ./data/MESSIDOR_Base1
elif [ $1 -eq 4300 ] # fundus; supervised learning on Drishti-GS
then
    python da_train.py \
    --dataset base \
    --num-class 2 \
    --resize 512 512 \
    --lr 0.0005 \
    --lr-update 'poly' \
    --epochs 100 \
    --save-per-epochs 1 \
    --source_root ./data/MESSIDOR_Base2 \
    --target_root ./data/MESSIDOR_Base2
elif [ $1 -eq 4400 ] # fundus; supervised learning on Drishti-GS
then
    python da_train.py \
    --dataset base \
    --num-class 2 \
    --resize 512 512 \
    --lr 0.0005 \
    --lr-update 'poly' \
    --epochs 100 \
    --save-per-epochs 1 \
    --source_root ./data/MESSIDOR_Base3 \
    --target_root ./data/MESSIDOR_Base3
elif [ $1 -eq 4500 ] # fundus; supervised learning on Drishti-GS
then
    python da_train.py \
    --dataset base \
    --num-class 2 \
    --resize 512 512 \
    --lr 0.0005 \
    --lr-update 'poly' \
    --epochs 100 \
    --save-per-epochs 1 \
    --source_root ./data/BinRushed ./data/Magrabia \
    --target_root ./data/BinRushed ./data/Magrabia
elif [ $1 -eq 400 ] # fundus; train on rim 1/2, test on rim 2/2
then
    python da_train.py \
    --dataset rim \
    --source_root ./data/rim \
    --target_root ./data/rim \
    --num-class 2 \
    --resize 512 512 \
    --lr 0.001 \
    --lr-update 'poly' \
    --epochs 200 \
    --save-per-epochs 1 \
    # --net 'unet' \
    # --pretrained 0.models_bak/0.8649_0.9529_50%rim.pth
elif [ $1 -eq 4000 ] # fundus; train on train1/2, test on train2/2
then
    python da_train.py \
    --dataset train \
    --source_root ./data/fundus/train \
    --target_root ./data/fundus/train \
    --num-class 2 \
    --resize 512 512 \
    --lr 0.00006 \
    --lr-update 'poly' \
    --epochs 45 \
    --save-per-epochs 1
else
    echo 'unknow args'
fi