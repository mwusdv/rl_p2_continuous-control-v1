# Reinforcement Learning: Continuous Control

## Project Details
For this project, we will train an agent to work with the Reacher environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this project, I work with the **First Version**, which contains a single angent.

The task is episodic. **The envionment is considered as solved if the agent can get an average score of +30 over 100 consecutive episodes.**



## Getting Started
In the project, we need the Unity environment. We can download it that has already been built.

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

Then, place the file in the rl_p2_continuous-control-v1/ folder and unzip (or decompress) the file.

## Instructions
Firt download the necessary zip file according to the above section. Then please modify the 77-th line in the `train.py` and the 40-th line accordingly:

        env = UnityEnvironment(file_name="Reacher_Linux/Reacher.x86_64")
with the correct file name. Currently it is set as `Reacher_Linux/Reacher.x86_64` since I worked on this project in a ubuntu system. 

To launch the training code:

            python train.py
    
At the end of training, two data files `checkpoint_actor.data` and `checkpoint_critic.data` will be saved in the same folder. They are for actor and critic models respectively. In addition to that, intermediate will be save every 100 episodes.

To launch the test code:

            python test.py dqn_fname

