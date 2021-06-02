import os
for i in range(1, 11):
    print(f"-----{i}-----")
    os.system(f"python validate.py --data ./T_data/mosi --path './checkpoints/checkpoint{i}.pt' --task emotion_prediction --valid-subset test --batch-size 4 --num-workers 0 --eval-metric")