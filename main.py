import argparse
import os
import pprint

from env import make_aigc_env
from config import *

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net


def get_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--algorithm', type=str, default='sac')
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.01)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--deterministic-eval', action='store_true', default=False)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action="store_true", default=False)

    # for pg
    # parser.add_argument('--lr', type=float, default=1e-2)
    # parser.add_argument('--episode-per-collect', type=int, default=1)
    # parser.add_argument('--gamma', type=float, default=0.95)
    # parser.add_argument('--repeat-per-collect', type=int, default=1)

    # for sac
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--auto-alpha', action="store_true", default=False)
    parser.add_argument('--step-per-collect', type=int, default=1000)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--repeat-per-collect', type=int, default=1)

    # for ppo
    # parser.add_argument('--lr', type=float, default=3e-4)
    # parser.add_argument('--gamma', type=float, default=0.99)
    # parser.add_argument('--step-per-collect', type=int, default=1000)
    # parser.add_argument('--repeat-per-collect', type=int, default=10)
    # parser.add_argument('--vf-coef', type=float, default=0.5)
    # parser.add_argument('--ent-coef', type=float, default=0.0)
    # parser.add_argument('--eps-clip', type=float, default=0.2)
    # parser.add_argument('--max-grad-norm', type=float, default=0.5)
    # parser.add_argument('--gae-lambda', type=float, default=0.95)
    # parser.add_argument('--norm-adv', type=int, default=0)
    # parser.add_argument('--recompute-adv', type=int, default=0)
    # parser.add_argument('--dual-clip', type=float, default=None)
    # parser.add_argument('--value-clip', type=int, default=0)

    args = parser.parse_known_args()[0]
    return args


def run_policy_gradient(env, train_envs, test_envs, logger, save_best_fn, stop_fn, args):
    from tianshou.policy import PGPolicy
    from tianshou.trainer import onpolicy_trainer

    # network
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        softmax=True
    ).to(args.device)

    # optimizer
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    # policy
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

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args.resume_path)

    # orthogonal initialization
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # trainer
    result = ""
    if not args.watch:
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

    return policy, result


def run_sac(env, train_envs, test_envs, logger, save_best_fn, stop_fn, args):
    from tianshou.policy import DiscreteSACPolicy
    from tianshou.utils.net.discrete import Actor, Critic
    from tianshou.trainer import offpolicy_trainer

    # model
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(net, args.action_shape, softmax_output=False,
                  device=args.device).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    critic1 = Critic(net_c1, last_size=args.action_shape,
                     device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    net_c2 = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    critic2 = Critic(net_c2, last_size=args.action_shape,
                     device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = 0.98 * np.log(np.prod(args.action_shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    # policy
    policy = DiscreteSACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        args.tau,
        args.gamma,
        args.alpha,
        estimation_step=args.n_step,
        reward_normalization=args.rew_norm
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args.resume_path)

    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # trainer
    result = ""
    if not args.watch:
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False
        )

    return policy, result


def run_ppo(env, train_envs, test_envs, logger, save_best_fn, stop_fn, args):
    from tianshou.policy import PPOPolicy
    from tianshou.trainer import onpolicy_trainer
    from tianshou.utils.net.common import ActorCritic
    from tianshou.utils.net.discrete import Actor, Critic

    # model
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(net, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)

    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    # optimizer
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    # policy
    policy = PPOPolicy(
        actor,
        critic,
        optim,
        torch.distributions.Categorical,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        action_space=env.action_space,
        deterministic_eval=True,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args.resume_path)

    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # trainer
    result = ""
    if not args.watch:
        result = onpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.repeat_per_collect,
            args.test_num,
            args.batch_size,
            step_per_collect=args.step_per_collect,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger
        )

    return policy, result


def main(args=get_args()):
    env, train_envs, test_envs = make_aigc_env(args.training_num, args.test_num)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    # log
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    log_path = os.path.join(args.logdir, args.log_prefix, args.algorithm, time_now)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        if args.reward_threshold:
            return mean_rewards >= args.reward_threshold
        return False

    if args.algorithm == 'pg':
        policy, result = run_policy_gradient(
            env, train_envs, test_envs, logger, save_best_fn, stop_fn, args)
    elif args.algorithm == 'sac':
        policy, result = run_sac(
            env, train_envs, test_envs, logger, save_best_fn, stop_fn, args)
    elif args.algorithm == 'ppo':
        policy, result = run_ppo(
            env, train_envs, test_envs, logger, save_best_fn, stop_fn, args)
    else:
        raise NotImplementedError(f"Algorithm {args.algorithm} not supported")

    if result:
        pprint.pprint(result)

    # Watch the performance
    if __name__ == '__main__':
        env, _, _ = make_aigc_env()
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    main(get_args())
