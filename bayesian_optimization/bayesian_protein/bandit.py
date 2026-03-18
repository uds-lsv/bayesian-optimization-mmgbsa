import warnings
from typing import List

import numpy as np


class MultiarmedBanditPlayer:
    def __init__(self, cluster_ids: List[int], confidence: float = 1.0):
        """
        :param cluster_ids: Integer labels representing th clusters
        :param confidence: Confidence of the bandit, a higher value leads to more exploring
        """
        self.confidence = confidence
        self.arms = cluster_ids
        self.n_plays = np.zeros(len(self.arms))
        self.cum_rewards = np.zeros(len(self.arms))
        self.unavailable = set()

    def average_reward(self, arm: int):
        if self.n_plays[arm] == 0:
            return 0
        return self.cum_rewards[arm] / self.n_plays[arm]

    def update(self, arm: int, reward: float):
        # smina affinities are negative -> we expected negative rewards, the lower, the better
        if reward >= 0:
            warnings.warn(
                "Reward is not negative, this may lead to unexpected results."
            )
        self.n_plays[arm] += 1
        self.cum_rewards[arm] += reward

    def remove(self, arm: int):
        self.unavailable.add(arm)

    def ucb(self, arm: int):
        # If a bandit has never been played we
        # really want the player to explore it
        if self.n_plays[arm] == 0:
            return np.inf

        plays = self.n_plays.sum()
        return self.average_reward(arm) + self.confidence * np.sqrt(
            2 * np.log(plays) / self.n_plays[arm]
        )

    def __next__(self) -> int:
        return self.choose()

    def choose(self) -> int:
        """
        Chooses the next cluster based on the highest ucb value
        :return: ID of the cluster
        """
        # We minimize upper confidence bound as the values should be negative
        min_ucb = np.inf
        best_arm = None
        for arm in self.arms:
            if arm in self.unavailable:
                continue
            ucb = self.ucb(arm)
            if ucb < min_ucb:
                min_ucb = ucb
                best_arm = arm

        assert best_arm is not None
        return best_arm
