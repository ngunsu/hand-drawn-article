# shufflenet
python trainer.py --seed 10 --exp-id best_holes_shufflenet_k4_1_1 --split-type k4_1  --check-val-every-n-epoch 1 --pretrained-model --model-type shufflenet --min-epochs 100 --max-epochs 200 --patience 30 --batch-size 32 --im-size 128 --shuffle --optimizer adamw --scheduler plateau --lr 0.001 --gamma 0.35 --gamma-step 35 --loss 'smooth' --precision '16-mixed'
python trainer.py --seed 30 --exp-id best_holes_shufflenet_k4_1_2 --split-type k4_1  --check-val-every-n-epoch 1 --pretrained-model --model-type shufflenet --min-epochs 100 --max-epochs 200 --patience 30 --batch-size 32 --im-size 128 --shuffle --optimizer adamw --scheduler plateau --lr 0.001 --gamma 0.35 --gamma-step 35 --loss 'smooth' --precision '16-mixed'
python trainer.py --seed 40 --exp-id best_holes_shufflenet_k4_1_3 --split-type k4_1  --check-val-every-n-epoch 1 --pretrained-model --model-type shufflenet --min-epochs 100 --max-epochs 200 --patience 30 --batch-size 32 --im-size 128 --shuffle --optimizer adamw --scheduler plateau --lr 0.001 --gamma 0.35 --gamma-step 35 --loss 'smooth' --precision '16-mixed'
python trainer.py --seed 50 --exp-id best_holes_shufflenet_k4_1_4 --split-type k4_1  --check-val-every-n-epoch 1 --pretrained-model --model-type shufflenet --min-epochs 100 --max-epochs 200 --patience 30 --batch-size 32 --im-size 128 --shuffle --optimizer adamw --scheduler plateau --lr 0.001 --gamma 0.35 --gamma-step 35 --loss 'smooth' --precision '16-mixed'
python trainer.py --seed 60 --exp-id best_holes_shufflenet_k4_1_5 --split-type k4_1  --check-val-every-n-epoch 1 --pretrained-model --model-type shufflenet --min-epochs 100 --max-epochs 200 --patience 30 --batch-size 32 --im-size 128 --shuffle --optimizer adamw --scheduler plateau --lr 0.001 --gamma 0.35 --gamma-step 35 --loss 'smooth' --precision '16-mixed'