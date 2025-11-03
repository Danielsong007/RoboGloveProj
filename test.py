from RLmylib.RL_lib import RealEnv, ActorCritic, PPOTrainer

env = RealEnv(buffer_weight_Srope,buffer_weight_Stouch,buffer_rising_CurPos)
policy = ActorCritic(env.state_dim)
trainer = PPOTrainer(policy)

