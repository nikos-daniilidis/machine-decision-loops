import numpy as np


class BaseLatentFactorUserSimulator(object):
    """
    Base class to create and track the evolution of users described by latent factors.

    This class creates an ensemble of users (the number is specified by n_users).

    The class determines the time evolution of the ensemble. The time evolution occurs in time steps,
    (also called batches). The class determines the probability for each user to appear during a time step,
    the probability for each user to convert on an item if the item is shown, and the evolution of each user's
    latent factors and observable characteristics. The class also provides methods to determine which users
    appear in each time step, and which users

    Each user is described by a number of factors (the number is specified by n_factors).

    The factors of each user determine the probability for the user to return in one batch, the probability of
    the user to convert on an item, and the observable features of the user.

    The appearance probability and conversion probability are also determined byt the user's past experience
    since their first appearance, i.e. amount of time since the user first appeared, the total number of
    conversions of the user, and the number of impressions for the user and each item.

    This class internally tracks the experience (impressions, conversions) of all users.

    This class contains a generator method to iterate through epochs, one time step at a time. From a client
    perspective, we care about being able to keep roughly constant number of users (expected_users_per_step)
    in each time step, and adjust the number of time steps per epoch. For this reason, we compute an internal
    return_probability_adjustment factor to ensure E~p[p] results in one time step.

    From the end user perspective, the API must offer the following methods:

    Pre-action step:
        x: np.ndarray((num_events, num_features)), y: np.ndarray((num_events, num_items)), batch_number: int
            = simulator.get_labeled_batch()

            num_events is a non-deterministic number of events (user appearances) which happen in a time step.
            x are the features for the users who appeared.
            y are the potential binary outcomes should a user see an item.
            batch_number is the number of the current batch, starting at zero.

    The recommender system action happens here (handled by a different object dealiong with the policy at hand).

    Post-action step:
        simulator.update_user_state(batch_number)

    Additional public methods are optional.

    Under the hood, the get_labeled_batch() method call runs the following processes:
    1. Determine which users appear in current batch (n_events is the outcome of n_user independent Bernoulli
    trials).
    2. Determine the user features for the users who do appear.
    3. Determine the outcomes for all users who do appear, if any of the items is shown to them
    (this entails [num_events,n_items] independent bernoulli trials).
    Steps 1 and 3 compute the appearance and conversion probabilities prior to the Bernoulli draws.

    The update_user_state() method updates the state of the users who have appeared so far.
    1. It updates the latent factors for all users who have appeared in any past time steps
    (latent factors have a slow drift in time).
    2. It updates the appearance logs of the users who appeared in the last step.
    3. It updates the impression logs for the user/item pairs which occurred.
    4. It updates the conversion logs for the user/item pairs which occurred.
    """
    def __init__(self, n_factors,
                 n_users, n_items, n_features,
                 b, eps_features, sigma_features,
                 c,
                 d,
                 expected_users_per_step=1000):
        self.n_users = n_users
        self.n_factors = n_factors
        self.n_items = n_items
        self.n_features = n_features
        self.b = b
        self.c = c
        self.d = d
        self._user_init_times = {}  # user_id: int -> appearance_time: float
        self._user_exposure = {}  # user_id: int -> (item_id: int -> counts: int)
        self._user_conversions = {}  # user_id: int -> (item_id: int -> counts: int)
        self._user_factors = np.zeros(shape=(n_users, n_factors), dtype=np.float)
        self._user_features = np.zeros(shape=(n_users, n_features), dtype=np.float)
        self.expected_users_per_step = expected_users_per_step
        self.return_probability_adjustment = None
        self.batch_number = 0

    def get_labeled_batch(self):
        """
        Create labeled events for a single time step.
        Returns:
            (np.array, np.array, np.array, int)
                ids are the ids of users who appeared.
                xs are the features for the users who appeared.
                potential_outcomes are the potential binary outcomes should a user see an item.
                current_batch is the number of the current batch, starting at zero.
        """
        p_rs = self.get_return_probabilities()
        ids = self.get_return_users(p_rs)
        p_cs = self.get_conversion_probabilities(ids)
        potetnial_outcomes = self.get_conversion_outcomes(p_cs)
        xs = self.get_user_features(ids)
        current_batch = self.batch_number
        self.batch_number += 1
        return ids, xs, potetnial_outcomes, current_batch

    def update_user_state(self, ids, impressions, outcomes):
        self.update_factors()
        self.update_user_init_times(ids)
        self.update_user_impressions(ids, impressions)
        

    def initialize_factors(self):
        """
        Initialize the latent factors for all users.
        Returns:
            None
        """
        return

    def update_factors(self):
        """
        Update the values of the latent factors for all active users for one time step.
        Must make call to _set_user_features
        Returns:
            None
        """
        self._set_user_features()
        return

    def get_user_features(self, ids):
        """
        Use the latent factors, feature means, feature correlations, and b to determine
        the features for the active users.
        Args:
            ids (np.array): The ids of active users. Shape is [n_events,].
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

    def get_return_probability_adjustment(self):
        """

        Returns:

        """
        return

    def get_base_conversion_probabilities(self, ids):
        """
        Get the base conversion probabilities for for all items for active users.
        Args:
            ids (np.array): The ids of active users. Shape is [n_events,].
        Returns:
            None
        """
        return

    def get_return_probabilities(self):
        """
        Get the return probabilities for all users. Adapted for user interest decay.
        Returns:
            numpy.array: The return probabilities. Shape [n_user,]
        """
        return

    def get_return_users(self, ps):
        """
        Sample the users who return, given the return probabilities.
        Args:
            ps (numpy.array): Bernoulli trial probabilities for n_users. Shape must be [n_users]

        Returns:
            numpy.array: The indices of users who return. Shape is [n_events,]
        """
        assert (ps.shape == (self.n_users,))
        return np.argwhere(self.independent_bernoulli_trials(ps) == 1)

    def get_conversion_probabilities(self, ids):
        """
        Get the base conversion probabilities for all items for active users. Adapted for user interest
        decay and saturation.
        Args:
            ids (np.array): The ids of active users. Shape is [n_events,].
        Returns:
            numpy.array: The return probabilities. Shape [n_user,n_item]
        """
        return

    def get_conversion_outcomes(self, ps):
        """
        Sample the outcomes of conversions of the users on items
        Args:
            ps (np.array): The conversion probabilities for the active users on all items.

        Returns:
            np.array: The outcomes, shape is [n_events, n_items]
        """
        assert (ps.shape == (self.n_users,self.n_items))
        return self.independent_bernoulli_trials(ps)

    def _activity_decay_curve(self, t):
        """
        Curve which describes reduction of user log odds to return as time passes.
        Requires _user_init_times attribute.
        Args:
            t (float): The current time step.

        Returns:
            numpy.array: The reduction in log odds (additive effect). Shape [n_user,]
        """
        return

    def _interest_decay_curve(self, t):
        """
        Curve which describes reduction in user log odds to convert on any items as time passes.
        Requires _user_init_times attribute.
        Args:
            t (float): The current time step.

        Returns:
            numpy.array: The reduction in log odds (additive effect). Shape [n_user,]
        """
        return

    def update_user_init_times(self, ids):
        """
        Updates user init times with the current time step for users who first appeared now.
        Args:
            ids (numpy.array): The ids of users who visited.

        Returns:
            None
        """
        return

    def update_user_impressions(self, ids, impressions):
        """
        Update the impression logs for users who appeared in the current batch.
        Args:
            ids (np.array): ids of the users who appeared
            impressions (np.array): one hot encoded impressions for the items  the users saw

        Returns:
            None
        """
        return

    def _user_saturation_curve(self):
        """
        Curve which describes how users' interest increases then saturates as they have more conversions.
        Effect is in log odds space.
        Returns:
            numpy.array: The reduction in log odds (additive effect). Shape [n_user,]
        """
        return

    def _user_fatigue_curve(self):
        """
        Curve which describes how users' interest decreases as they are exposed to same content repeatedly.
        Effect is in log odds space.
        Returns:
            numpy.array: The reduction in log odds (additive effect). Shape [n_user,n_item]
        """
        return

    @staticmethod
    def independent_bernoulli_trials(ps):
        """
        Draw independent bernoulli trials with probabilities ps
        Args:
            ps (numpy.array): The probabilities of bernoulli trials.

        Returns:
            numpy.array: The 0,1 outcomes.
        """
        return np.random.binomial(n=1, p=ps)
