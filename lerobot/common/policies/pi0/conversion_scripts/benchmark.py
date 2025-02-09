import argparse
import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig

torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_repo_id", type=str, default="danaaubakirova/koch_test")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--episodes", type=str, default="0")
    parser.add_argument("--local_files_only", type=bool, default=False)
    parser.add_argument("--ckpt_torch_dir", type=str, default="lerobot/pi0")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = [int(ep) for ep in args.episodes.split(",")]
    # dataset_repo_id = "danaaubakirova/koch_test"
    # model_name = "pi0_base"
    # ckpt_torch_dir = Path.home() / f".cache/openpi/openpi-assets/checkpoints/{model_name}_pytorch"
    # ckpt_torch_dir = "lerobot/pi0"
    
    if "agibotworld" in args.dataset_repo_id:
        # delta_timestamps = {
        #     "observation.images.top_head": [0]
        # }
        dataset = LeRobotDataset(
            args.dataset_repo_id, 
            root=args.dataset_root, 
            # delta_timestamps=delta_timestamps,
            local_files_only=args.local_files_only
        )
    else:
        dataset = LeRobotDataset(
            args.dataset_repo_id, 
            root=args.dataset_root, 
            episodes=episodes, 
            local_files_only=args.local_files_only
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
    )

    batch = next(iter(dataloader))

    # To device
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device=device, dtype=torch.float32)

    cfg = PreTrainedConfig.from_pretrained(args.ckpt_torch_dir)
    cfg.pretrained_path = args.ckpt_torch_dir
    policy = make_policy(cfg, device, ds_meta=dataset.meta)

    # policy = torch.compile(policy, mode="reduce-overhead")

    warmup_iters = 10
    benchmark_iters = 30

    # Warmup
    for _ in range(warmup_iters):
        torch.cuda.synchronize()
        policy.select_action(batch)
        policy.reset()
        torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(benchmark_iters):
        policy.select_action(batch)
        policy.reset()
    end_event.record()

    # Synchronize and measure time
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)

    avg_time_per_iter = elapsed_time_ms / benchmark_iters
    print(f"Average execution time per iteration: {avg_time_per_iter:.3f} ms")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
