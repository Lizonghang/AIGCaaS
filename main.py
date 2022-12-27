import argparse
import os
import pprint

from env import make_aigc_env
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from config import *

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import PGPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='pg')
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--episode-per-collect', type=int, default=1)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--training-num', type=int, default=20)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    env, train_envs, test_envs = make_aigc_env(args.training_num, args.test_num)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    # policy network
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        softmax=True
    ).to(args.device)

    # optimizer
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    # policy gradient
    if args.algorithm == 'pg':
        policy = PGPolicy(
            net,
            optim,
            torch.distributions.Categorical,
            args.gamma,
            reward_normalization=args.rew_norm,
            action_space=env.action_space,
            action_scaling=False,
            action_bound_method="",
        )
    else:
        raise NotImplementedError(f"Algorithm {args.algorithm} not supported")

    # orthogonal initialization
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # log
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    log_path = os.path.join(args.logdir, 'aigcaas', args.algorithm, time_now)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        if args.reward_threshold:
            return mean_rewards >= args.reward_threshold
        return False

    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )

    # Watch the performance
    if __name__ == '__main__':
        pprint.pprint(result)
        env, _, _ = make_aigc_env()
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    main(get_args())
