from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import ProbabilisticActor, TanhDelta, OrnsteinUhlenbeckProcessModule

def agent_policy(actor_network, env):
    #these will be the keys that go intot he tensordict
    low = env.action_spec.space.low
    high = env.action_spec.space.high

    #this key gets the state to input into the network
    state_key = "observation"
    #this key gets the output, the output is an action 
    action_key = "action"
    #this key is for the weights of the paramters of the model
    param_key = "param"

    actor_tensordict = TensorDictModule(
        actor_network,
        in_keys = [state_key],
        out_keys=[param_key]
    )
    

    #this is just the actor network, but wrapped to make sure it follows environment constraints
    #takes care of tanh, scalling, exploration
    policy_distribution = ProbabilisticActor(
        module = actor_tensordict,
        in_keys = [param_key],
        out_keys=[action_key],
        #tanhdelta = scale(tanh(u)) = how actions are produces
        distribution_class = TanhDelta,
        distribution_kwargs = {"low":low, "high":high},
        spec = env.action_spec,
    )

    ou = OrnsteinUhlenbeckProcessModule(
        spec = env.action_spec,
        eps_init = 1.0,
        eps_end = 0.1,
        annealing_num_steps=200_000
    )

    policy_exploration = TensorDictSequential(policy_distribution, ou)
    policy_eval = policy_distribution

    return policy_exploration, policy_eval
