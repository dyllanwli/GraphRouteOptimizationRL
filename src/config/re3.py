config = {}
config["callbacks"] = RE3Callbacks
config["exploration_config"] = {
    "type": "RE3",
     # the dimensionality of the observation embedding vectors in latent space.
     "embeds_dim": 128,
     "rho": 0.1, # Beta decay factor, used for on-policy algorithm.
     "k_nn": 50, # Number of neighbours to set for K-NN entropy estimation.
     # Configuration for the encoder network, producing embedding vectors from observations.
     # This can be used to configure fcnet- or conv_net setups to properly process any
     # observation space. By default uses the Policy model configuration.
     "encoder_net_config": {
         "fcnet_hiddens": [],
         "fcnet_activation": "relu",
     },
     # Hyperparameter to choose between exploration and exploitation. A higher value of beta adds
     # more importance to the intrinsic reward, as per the following equation
     # `reward = r + beta * intrinsic_reward`
     "beta": 0.2,
     # Schedule to use for beta decay, one of constant" or "linear_decay".
     "beta_schedule": 'constant',
     # Specify, which exploration sub-type to use (usually, the algo's "default"
     # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
     "sub_exploration": {
         "type": "StochasticSampling",
     }
}