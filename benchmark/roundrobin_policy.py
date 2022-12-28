from typing import Any, Dict, List, Type, Optional, Union

import argparse
import os
import pprint
import torch
import numpy as np

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import one_hot
from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.data import Batch


class RoundRobinPolicy(BasePolicy):
    """Implementation of round robin policy. This policy assign user tasks to service
    providers in turn.

    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    """

    def __init__(
            self,
            dist_fn: Type[torch.distributions.Distribution],
            action_scaling: bool = True,
            action_bound_method: str = "clip",
            **kwargs: Any
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs)
        self.dist_fn = dist_fn
        self.round_act = 0

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            **kwargs: Any
    ) -> Batch:
        """Compute action at random."""
        act_ = (self.round_act % NUM_SERVICE_PROVIDERS)
        act_ = torch.Tensor([act_] * batch.obs.shape[0]).to(torch.int64)
        logits, hidden = one_hot(act_, num_classes=NUM_SERVICE_PROVIDERS), None
        self.round_act += 1

        # convert to probability distribution
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)

        # use deterministic policy
        if self.action_type == "discrete":
            act = logits.argmax(-1)
        elif self.action_type == "continuous":
            act = logits[0]

        assert act.equal(act_), f"Action mismatch: {act_} != {act}"

        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def learn(
            self,
            batch: Batch,
            batch_size: int,
            repeat: int,
            **kwargs: Any
    ) -> Dict[str, List[float]]:
        return {"loss": [0.]}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--episode-per-collect', type=int, default=1)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--training-num', type=int, default=20)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    env, train_envs, test_envs = make_aigc_env(args.training_num, args.test_num)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    # random policy
    policy = RoundRobinPolicy(
        torch.distributions.Categorical,
        action_space=env.action_space,
        action_scaling=False,
        action_bound_method="",
    )

    # collector
    train_collector = Collector(policy, train_envs)
    test_collector = Collector(policy, test_envs)

    # log
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    root = path.dirname(path.dirname(path.abspath(__file__)))
    log_path = os.path.join(root, args.logdir, 'aigcaas', 'roundrobin', time_now)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # trainer
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
            logger=logger
        )
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
    if __package__ is None:
        import sys
        from os import path

        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from env import make_aigc_env
        from config import *
    else:
        from ..env import make_aigc_env
        from ..config import *

    main(get_args())
