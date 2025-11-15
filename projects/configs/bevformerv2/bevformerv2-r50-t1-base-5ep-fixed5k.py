_base_ = ['./bevformerv2-r50-t1-base-24ep.py']

# Fine-tune with fixed 5000 queries per camera
# Baseline (no sampling): mAP 0.3512
# Fixed 5k (before finetune): mAP 0.3408 (-3.0%)
total_epochs = 5
runner = dict(type='EpochBasedRunner', max_epochs=5)

