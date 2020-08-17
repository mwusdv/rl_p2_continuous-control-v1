import torch
from unityagents import UnityEnvironment

import numpy as np
from collections import deque

from ddpg_agent import Agent
import matplotlib.pyplot as plt

def train_ddpg(env, max_episode=1000, max_t=1000, save_every=100, check_history=100,
               sigma_start=0.2, sigma_end=0.01, sigma_decay=0.995):
    # reset
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]  
    env_info = env.reset(train_mode=True)[brain_name]
    
    # action and state size
    action_size = brain.vector_action_space_size 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('State size:', state_size)
    print('Action size: ', action_size)
      
    # initialize agent
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    agent = Agent(state_size, action_size, random_seed=123)
    
    scores_deque = deque(maxlen=check_history)
    scores = []
   
    # learning multiple episodes
    sigma = sigma_start
    for episode in range(max_episode):
        # prepare for training in the current epoc
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]  
        score = 0
        agent.reset(sigma=sigma)
        
        # play and learn in current episode
        for t in range(max_t):
            action = agent.act(state)
            
            env_info = env.step(action)[brain_name]   
            next_state = env_info.vector_observations[0]      # get next state (for each agent)
            reward = env_info.rewards[0]                       # get reward (for each agent)
            done = env_info.local_done[0]                      # see if episode finished

            agent.step(t, state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        
        # update sigma for exlporation
        sigma = max(sigma_end, sigma*sigma_decay)
        
        # record score
        epoc_score = score
        scores_deque.append(epoc_score)
        scores.append(epoc_score)
        
        print('Episode {}\tscore: {:.2f}\tAverage Score: {:.2f}'.format(episode, epoc_score, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 30.0 and episode >= check_history:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - check_history, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.data')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.data')
            break
    
        if episode % save_every == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_ep{}.data'.format(episode))
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_ep{}.data'.format(episode))
    return scores

if __name__ == '__main__':
    env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')
    scores = train_ddpg(env)
    env.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Scores')
    plt.xlabel('Episode')
    plt.show()