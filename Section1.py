import gym
import numpy as np
import tensorflow as tf
import collections
from Models import PolicyNetwork, BaselineNetwork


def train_agent():
    """
    Train the agent with the REINFORCE with Baseline algorithm.
    :return: list 100 ep. moving average for each episode
    """
    env = gym.make('CartPole-v1')
    np.random.seed(1)

    # Define hyperparameters
    state_size = 4
    action_size = env.action_space.n

    max_episodes = 5000
    max_steps = 501
    discount_factor = 1.0
    policy_learning_rate = 0.0004
    baseline_learning_rate = 0.001
    # policy_layer_sizes = [12]
    # baseline_layer_sizes = [12]

    render = False

    # Initialize the policy network
    tf.reset_default_graph()
    policy_net = PolicyNetwork(state_size, action_size, policy_learning_rate)
    baseline_net = BaselineNetwork(state_size, baseline_learning_rate)

    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0

        average_rewards_total = []
        baseline_losses = []

        for episode in range(max_episodes):
            state = env.reset()
            state = state.reshape([1, state_size])
            episode_transitions = []

            for step in range(max_steps):
                actions_distribution = sess.run(policy_net.actions_distribution, {policy_net.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                if render:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                episode_transitions.append(
                    Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                episode_rewards[episode] += reward

                if done:
                    average_rewards_total.append(np.mean(episode_rewards[max(0, episode - 99):episode + 1]))
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
                state = next_state

            # Compute Rt for each time-step t and update the network's weights
            Gts = []
            advantages = []
            ep_baseline_losses = []
            baselines_before = []
            baselines_after = []
            for t, tr in enumerate(episode_transitions):
                Gt = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))
                Gts.append(Gt)
                baseline = sess.run(baseline_net.output, {baseline_net.state: state})
                baselines_before.append(float(baseline))

                # OLD:
                advantage = Gt - baseline

                # print('%d \tbaseline: %.1f advantage: %.5f loss: %.1f' % (t+1, baseline, advantage, baseline_loss))
                d = {baseline_net.state: tr.state, baseline_net.target: Gt}
                _, baseline, baseline_loss = sess.run([baseline_net.optimizer, baseline_net.output, baseline_net.loss], d)

                # # NEW:
                # advantage = Gt - baseline

                baselines_after.append(float(baseline))
                advantages.append(Gt - float(baseline))
                ep_baseline_losses.append(float(baseline_loss))
                d = {policy_net.state: tr.state, policy_net.target: advantage, policy_net.action: tr.action}
                _, policy_loss = sess.run([policy_net.optimizer, policy_net.loss], d)
            baseline_loss = np.mean(ep_baseline_losses)
            baseline_losses.append(np.mean(ep_baseline_losses))
            avg_baseline_loss = np.mean(baseline_losses[max(0, episode - 99): episode + 1])

            # print()
            # print('Gts: %s' % Gts)
            # print('baselines before: %s' % baselines_before)
            # print('baselines after: %s' % baselines_after)
            # print('advantages: %s' % advantages)
            # print('losses: %s' % ep_baseline_losses)

            print("ep: %d \t reward: %.1f \t(100mean: %.1f) \tbaseline_loss: %.2f \t(100mean: %.1f)" % (
                episode, episode_rewards[episode], average_rewards, baseline_loss, avg_baseline_loss))

            if solved:
                break

    # plt.plot(range(1, len(average_rewards_total) + 1), average_rewards_total)
    # plt.xlabel('episode')
    # plt.ylabel('last 100 eps. average reward')
    # plt.show()

    return average_rewards_total
