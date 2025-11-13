_base_ = ['./bevformerv2-r50-t1-base-24ep.py']

# Extend fine-tuning to 10 epochs total
total_epochs = 10
runner = dict(type='EpochBasedRunner', max_epochs=10)


