# training.py

import torch
from tqdm import tqdm
from torchrl.collectors import SyncDataCollector

from config import *
from ddpg import ddpg
from logger import LiveLogger


@torch.no_grad()
def evaluate(policy_eval, env, max_steps):
    td = env.reset()
    done = torch.zeros(td.batch_size, dtype=torch.bool, device=td.device)
    total_reward = 0.0
    steps = 0

    while (not done.any()) and steps < max_steps:
        td = policy_eval(td)
        td = env.step(td)
        total_reward += td.get(("next", "reward")).sum().item()
        done = td.get(("next", "done")).squeeze(-1)
        td = td.get("next")
        steps += 1

    return total_reward


def train():
    env, policy_exploration, policy_eval, loss, target_updater, actor_optim, critic_optim, replay_buffer = ddpg(env_name, device, max_steps, actor_lr, critic_lr, gamma, tau, replay_size, batch_size)

    #runs the exporitoray model through 
    #has 1,000 timesteps fo data per batch
    #training ends after 1,000,000 timesteps
    #the output of a data colect is given (S,A,R,done) what (S(t+1),A(t+1),done(t+1)) is 
    #this is the informatin pushed to the replay buffer
    collector = SyncDataCollector(
        env,
        policy_exploration,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device="cpu",
    )

    logger = LiveLogger("metrics.csv")

    last_train_reward = 0.0
    last_eval_reward = 0.0
    eval_every = 10

    #progress bar, how long until training is done
    pbar = tqdm(total=total_frames)
    it = 0

    
    for td in collector:
        it += 1
        #each batch gets added to the replay_buffer
        replay_buffer.extend(td.cpu())
        pbar.update(td.numel())
        #we get the rewawrd of the episode by averageing the reward of all rewards of the eppisode
        r = td.get(("next", "episode_reward"), None)
        if r is not None and r.numel() > 0:
            last_train_reward = r.mean().item()

        #we don't start training untile we have enough data in the replay buffer
        if len(replay_buffer) < batch_size:
            logger.log(pbar.n, last_train_reward, last_eval_reward, 0.0, 0.0)
            continue
        
        #use a random batch to learn from
        batch = replay_buffer.sample().to(device)

        # checks if episode ended at this time step: done == terminated
        done = batch.get(("next", "done"))
        batch.set(("next", "terminated"), done.clone())

        #computes all the losses using the current batch experience
        losses = loss(batch)

        #we only want the crtic's loss: How good or bad was the value estimate
        lc = losses["loss_value"]

        #backprop +optim
        critic_optim.zero_grad(set_to_none=True)
        lc.backward()
        critic_optim.step()

        #we don't want the critic to change based off the actor, only off the replay bufffer
        #so freeze the critic ntwowerk
        for p in critic_optim.param_groups[0]["params"]:
            p.requires_grad_(False)

        losses = loss(batch)           
        la = losses["loss_actor"]

        actor_optim.zero_grad(set_to_none=True)
        la.backward()
        actor_optim.step()

        #unfreeze crtc netwok
        for p in critic_optim.param_groups[0]["params"]:
            p.requires_grad_(True)
        
        #update the target network weights
        target_updater.step()

        #we make sure to evalute after every 10 interations, so we run the policy agent with no noise
        if it % eval_every == 0:
            last_eval_reward = evaluate(policy_eval, env, max_steps)

        #send everythgin to lgger
        logger.log(pbar.n, last_train_reward, last_eval_reward, la.item(), lc.item())

        pbar.set_description(
            f"train={last_train_reward:.2f} eval={last_eval_reward:.2f} "
            f"la={la.item():.3f} lc={lc.item():.3f}"
        )
    #stops the collector completely
    collector.shutdown()
    pbar.close()
    print("Training finished. Saved metrics.csv")


if __name__ == "__main__":
    train()
