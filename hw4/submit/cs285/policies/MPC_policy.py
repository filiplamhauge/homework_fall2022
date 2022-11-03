import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere
        self.data_statistics2 = None
        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' or (self.sample_strategy == 'cem' and obs is None):
            # TODO(Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self.ac_dim) in the range
            # [self.low, self.high]
            random_action_sequences = np.random.uniform(low=self.low, high=self.high, size=(num_sequences, self.horizon, self.ac_dim))

            #print(random_action_sequences.shape)

            return random_action_sequences

        elif self.sample_strategy == 'cem':
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf 
            for i in range(self.cem_iterations):
                # - Sample candidate sequences from a Gaussian with the current 
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)
                # - Get the top `self.cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                # - Update the elite mean and variance
                if i == 0:
                    random_action_seq = np.random.uniform(low=self.low, high=self.high, size=(num_sequences, self.horizon, self.ac_dim))
                    #print(random_action_seq[0])
                    eval_action_seq = self.evaluate_candidate_sequences(random_action_seq, obs)
                    elite_actions = np.argpartition(eval_action_seq, -self.cem_num_elites)[-self.cem_num_elites:]
                    elite_actions = random_action_seq[elite_actions]
                    #print('HERE',elite_actions.shape)
                    self.data_statistics2 = (np.mean(elite_actions, axis=0),np.std(elite_actions, axis=0))
                    #print('TEST', np.mean(elite_actions, axis=0).shape, np.std(elite_actions, axis=0).shape)
                    #self.data_statistics = (np.mean(elite_actions, axis=0),np.cov(elite_actions, bias=True))

                else:
                    #print(self.data_statistics,'\n')
                    #print('YO',self.data_statistics[0],'\n')
                    #print('HEY',self.data_statistics[1],'\n')
                    #print('Before:',random_action_seq.shape)

                    #print(np.random.normal(self.data_statistics[0][0][0], self.data_statistics[1][0][0],  5).shape)
                    random_action_seq = []
                    for j in range(num_sequences):
                        random_action_seq.append(np.array([[np.random.normal(self.data_statistics2[0][k][i], self.data_statistics2[1][k][i],  1) for i in range(len(self.data_statistics2[0][k]))] for k in range(len(self.data_statistics2[0]))]))
                    random_action_seq = np.squeeze(np.array(random_action_seq))
                    #random_action_seq = np.array([np.random.normal(self.data_statistics[0][k], self.data_statistics[1][k],  self.ac_dim) for i in range(num_sequences)])
                    #print('After:',random_action_seq.shape)
                    #print('After2:',np.squeeze(random_action_seq).shape)
                    #print('SEQ',random_action_seq)
                    #random_action_seq = np.random.multivariate_normal(self.data_statistics[0], self.data_statistics[1], self.cem_num_elites)
                    eval_action_seq = self.evaluate_candidate_sequences(random_action_seq, obs)
                    elite_actions = np.argpartition(eval_action_seq, -self.cem_num_elites)[-self.cem_num_elites:]
                    elite_actions = random_action_seq[elite_actions]
                    self.data_statistics2 = (np.mean(elite_actions, axis=0),np.std(elite_actions, axis=0))


            # TODO(Q5): Set `cem_action` to the appropriate action chosen by CEM
            #cem_action = np.mean(elite_actions, axis=0)
            #return cem_action

            #print('WE MADE IT',elite_actions.shape)

            return elite_actions
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # TODO(Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        #
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)
        pred_sums = np.zeros(shape=(candidate_action_sequences.shape[0]))
        for idx,model in enumerate(self.dyn_models):
            pred_sums += self.calculate_sum_of_rewards(obs,candidate_action_sequences,model)

        pred_means = np.divide(pred_sums,1000)
        return pred_means

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon, obs=obs)
        if candidate_action_sequences.shape[0] == 1:
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
            best_action_sequence = candidate_action_sequences[np.argmax(self.evaluate_candidate_sequences(candidate_action_sequences,obs))]  # TODO (Q2)
            #print(best_action_sequence)
            action_to_take = best_action_sequence[0]
            return action_to_take[None] 

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        sum_of_rewards = np.zeros(candidate_action_sequences.shape[0])
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs, action)`
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.

        assert candidate_action_sequences.shape[1] == self.horizon
        obs = np.tile(obs, (candidate_action_sequences.shape[0], 1))
        for t in range(self.horizon):
            actions = candidate_action_sequences[:, t, :]
            rewards, _ = self.env.get_reward(obs, actions)
            sum_of_rewards += rewards
            obs = model.get_prediction(obs, actions, self.data_statistics)

        return sum_of_rewards
