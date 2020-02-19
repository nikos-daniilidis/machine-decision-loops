import numpy as np


class BaseLFUsers(object):
    """
    Base class for users described by latent factors.
    """
    def __init__(self, n_factors,
                 n_users, n_items,
                 b, eps_features, sigma_features,
                 c,
                 d):
        self.n_users = n_users
        self.n_factors = n_factors
        self.n_items = n_items
        self.b = b
        self.c = c
        self.d = d
        self._user_init_times = {}  # user_id: int -> appearance_time: float
        self._user_exposure = {}  # user_id: int -> (item_id: int -> counts: int)
        self._user_conversions = {}  # user_id: int -> (item_id: int -> counts: int)
        self._user_factors = None
        self._user_features = None

    def initialize_factors(self):
        """
        Initialize the latent factors for all users.
        Returns:
            None
        """
        return

    def update_factors(self):
        """
        Update the values of the latent factors for all active users for one epoch.
        Must make call to _set_user_features
        Returns:
            None
        """
        self._set_user_features()
        return

    def _set_user_features(self):
        """
        Use the latent factors, feature means, feature correlations, and b to determine
        the features for all users.
        Returns:
            None
        """
        return

    def set_features(self):
        """
        Public caller to _set_user_features.
        Returns:
            None
        """
        return

    def get_base_return_probabilities(self):
        """
        Get the base return probabilities for all users.
        Returns:
            None
        """
        return

    def get_base_conversion_probabilities(self):
        """
        Get the base conversion probabilities for all user item pairs.
        Returns:
            None
        """
        return

    def get_return_probabilities(self):
        """
        Get the return probabilities for all users. Adapted for user interest decay.
        Returns:
            numpy.ndarray: The return probabilities. Shape [n_user,]
        """
        return

    def sample_users_in_epoch(self, time_steps):
        # TODO: if I sample users for the entire epoch, I cannot know how much their probabilities have decayed.
        # TODO: sample once per epoch and apply decay per epoch. Or: epoch has batches, sample in batches then combine
        """
        Sample users who appear during an epoch.
        Args:
            time_steps (int): Time steps in an epoch

        Returns:
            (tuple(list[int], list[float])): ids of users who appeared, and their appearance times.
        """
        return

    def measure_users_in_epoch(self, users, items):
        """
        Determine if users presented with items converted.
        Args:
            users (list[int]): The user ids
            items (list[int]): The item ids

        Returns:
            list[int]: The outcomes of the measurements.
        """
        return

    def _activity_decay_curve(self, t):
        """
        Curve which describes reduction of user log odds to return as time passes.
        Requires _user_init_times attribute.
        Args:
            t (float): The current time step.

        Returns:
            numpy.ndarray: The reduction in log odds (additive effect). Shape [n_user,]
        """
        return

    def _interest_decay_curve(self, t):
        """
        Curve which describes reduction in user log odds to convert on any items as time passes.
        Requires _user_init_times attribute.
        Args:
            t (float): The current time step.

        Returns:
            numpy.ndarray: The reduction in log odds (additive effect). Shape [n_user,]
        """
        return

    def update_user_init_times(self, users, times):
        """
        Updates user init times with the time users were first seen.
        Args:
            users (list[int]): The ids of users who visited.
            times (list[float]): The times each user visited.

        Returns:
            None
        """
        return

    def _user_saturation_curve(self):
        """
        Curve which describes how users' interest increases then saturates as they have more conversions.
        Effect is in log odds space.
        Returns:
            numpy.ndarray: The reduction in log odds (additive effect). Shape [n_user,]
        """
        return

    def _user_fatigue_curve(self):
        """
        Curve which describes how users' interest decreases as they are exposed to same content repeatedly.
        Effect is in log odds space.
        Returns:
            numpy.ndarray: The reduction in log odds (additive effect). Shape [n_user,n_item]
        """
        return