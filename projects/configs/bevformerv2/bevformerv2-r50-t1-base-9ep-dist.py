_base_ = ['./bevformerv2-r50-t1-base-24ep.py']

# Finetune continuation: run up to epoch 9
total_epochs = 9
runner = dict(type='EpochBasedRunner', max_epochs=9)


