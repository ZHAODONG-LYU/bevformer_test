_base_ = ['./bevformerv2-r50-t1-base-24ep.py']

# Override epochs for a short finetune run
total_epochs = 6
runner = dict(type='EpochBasedRunner', max_epochs=6)

# Optional: you can still change LR / warmup via CLI if needed
# Example:
#   --cfg-options optimizer.lr=2e-5 lr_config.warmup_iters=500


