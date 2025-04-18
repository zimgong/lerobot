rm -r checkpoints/pi0ac_onlycritic
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 deepspeed  lerobot/scripts/off_policy_pi0.py\
  --teacher /data/ceph_hdd/main/dev/zim.gong/yidong/lerobot/checkpoints/pi0_new  \
  --policy.type pi0_onestep_ac \
  --output_dir checkpoints/pi0ac_onlycritic \
  --dataset.repo_id robocasa/v0.1   \
  --epochs 10 \
  --batch_size 4 \
  --temperature 1.0 \
  --soft_target_weight 0.5 \
  --hard_target_weight 0.5 \
  --deepspeed_config ds_config.json \
  --wandb.enable true \
  --wandb.project lerobot \
  --only_critic true

# deepspeed --num_gpus=4 lerobot/scripts/distill_pi0_deepspeed.py \
#   --teacher=lerobot/pi0 \
#   --policy.type=pi0_onestep \
#   --output_dir=lerobot/pi0_onestep \
#   --dataset.repo_id=danaaubakirova/koch_test \
#   --epochs=10 \
#   --batch_size=32 \
#   --temperature=1.0 \
#   --soft_target_weight=0.5 \
#   --hard_target_weight=0.5 \
#   --deepspeed \
#   --deepspeed_config=ds_config.json