python operating_platform/core/train.py \
  --dataset.repo_id="20251104/experimental/so101-tes-1t" \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=False
