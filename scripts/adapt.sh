# example for Refuge pre-train
# step 1: pre-train stage
bash scripts/run.sh 1 4

# step 2: adaptation stage
# example for Refuge --> rim
bash scripts/run.sh 100 4 "add pre-trained checkpoint path generated in step 1"
# example for Refuge --> Drishti-GS
bash scripts/run.sh 300 3_4 "add pre-trained checkpoint path generated in step 1"

