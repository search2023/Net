
## Train
'''
python train.py --model="base" --cs_ratio=10 --batch_size=8 --blr=1e-4 --min_lr=1e-6 --epochs=200 --warmup_epochs=20 
'''
Model is saved in the model folder, log is saved in the logs folder.

## Test
'''
python test.py --model="base" --cs_ratio=10 --test_dataset="Set11"
'''
Results is saved in the results folder.
