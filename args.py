import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()

    # DDPG
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--start-timesteps", type=int, default=25000)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=5000)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)

    # Environment
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--n-service-providers", type=int, default=10)
    parser.add_argument("--n_tasks", type=int, default=10000)

    # Logger and resume training
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--watch", default=False, action="store_true",
        help="watch the play of pre-trained policy only",
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()
