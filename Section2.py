import gym
import numpy as np
import tensorflow as tf
import collections
import matplotlib.pyplot as plt
from Models import PolicyNetwork, BaselineNetwork

env = gym.make('CartPole-v1')

np.random.seed(1)

# Define hyperparameters
state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 1.0  # 0.99
policy_learning_rate = 0.0004
baseline_learning_rate = 0.0001
baseline_layer_sizes = [12]

render = False

# Initialize the policy network
tf.reset_default_graph()
policy_net = PolicyNetwork(state_size, action_size, policy_learning_rate)
baseline_net = BaselineNetwork(state_size, baseline_learning_rate, baseline_layer_sizes)

# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    solved = False
    # Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = np.zeros(max_episodes)
    episode_baseline_losses = np.zeros(max_episodes)
    average_rewards = 0.0

    average_rewards_with_baseline = []

    for episode in range(max_episodes):
        state = env.reset()
        state = state.reshape([1, state_size])
        # episode_transitions = []

        for step in range(max_steps):
            actions_distribution = sess.run(policy_net.actions_distribution, {policy_net.state: state})
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1

            # Compute TD-error and update the network's weights
            baseline_current = sess.run(baseline_net.output, {baseline_net.state: state})
            baseline_next = sess.run(baseline_net.output, {baseline_net.state: next_state})
            baseline_target = reward + discount_factor * baseline_next
            policy_target = baseline_target - baseline_current
            feed_dict = {baseline_net.state: state, baseline_net.value: baseline_target}
            _, baseline_loss = sess.run([baseline_net.optimizer, baseline_net.loss], feed_dict)
            feed_dict = {policy_net.state: state, policy_net.target: policy_target, policy_net.action: action_one_hot}
            _, policy_loss = sess.run([policy_net.optimizer, policy_net.loss], feed_dict)

            episode_rewards[episode] += reward
            episode_baseline_losses[episode] += baseline_loss

            if done:
                average_rewards_with_baseline.append(np.mean(episode_rewards[max(0, episode - 99):episode + 1]))
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    avg_baseline_loss = np.mean(episode_baseline_losses[(episode - 99):episode + 1])
                if average_rewards > 475:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break
            state = next_state

        print("ep: %d \t reward: %.1f \t(100mean: %.1f) \tbaseline_loss: %.2f \t(100mean: %.1f)" % (
            episode, episode_rewards[episode], average_rewards, baseline_loss, avg_baseline_loss))

        if solved:
            break

plt.plot(range(1, len(average_rewards_with_baseline) + 1), average_rewards_with_baseline)
plt.xlabel('episode')
plt.ylabel('last 100 eps. average reward')
plt.show()
