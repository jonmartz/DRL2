import policy_gradients as reinforce
import Section1 as with_baseline
import Section2 as actor_critic
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt


n_iterations = 10

existing_results = set()
if not os.path.exists('results.csv'):
    df_existing_results = None
else:
    df_existing_results = pd.read_csv('results.csv')
    for i, row in df_existing_results.iterrows():
        existing_results.add('%d %s' % (row['iteration'], row['algorithm']))

algorithms = {'reinforce': reinforce, 'with_baseline': with_baseline, 'actor_critic': actor_critic}
with open('results.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    if df_existing_results is None:
        header = ['iteration', 'episode', 'algorithm', 'avg 100 rewards', 'avg 100 policy loss', 'avg 100 baseline loss',
                  'time to solve']
        writer.writerow(header)
    for i in range(1, n_iterations + 1):
        print('\niteration = %d/%d' % (i, n_iterations))
        for algorithm_name, algorithm in algorithms.items():
            print('\n\talgorithm = %s' % algorithm_name)
            if '%d %s' % (i, algorithm_name) in existing_results:
                print('\t\tresults already exists!')
                continue
            results = algorithm.train_agent(max_episodes=1000)
            episode = 0
            time_to_solve = results[-1]
            for reward, policy_loss, baseline_loss in zip(*results[:-1]):
                episode += 1
                row = [i, episode, algorithm_name, reward, policy_loss, baseline_loss, time_to_solve]
                writer.writerow(row)
