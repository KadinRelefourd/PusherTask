import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#environment
env_name = "Pusher-v4"
max_steps  = 100
#prone to instability
actor_lr =  1e-4
critic_lr = 3e-4
#discount
gamma = 0.99
#target network lr (soft update)
tau = 0.005

#replay 
batch_size = 256
replay_size = 1_000_000
frames_per_batch = 1000
#total training
total_frames = 1_000_000
