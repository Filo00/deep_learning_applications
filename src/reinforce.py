import numpy as np
import torch
import torch.nn.functional as F
from src.policy import select_action


def run_episode(env, policy):
    """Gioca un episodio completo con la policy. Restituisce (stati, log-probabilita', reward) di ogni step."""
    state, _ = env.reset()
    states, log_probs, rewards, done = [], [], [], False
    while not done:
        states.append(state)
        action, log_prob = select_action(policy, state)
        state, reward, terminated, truncated, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        done = terminated or truncated
    return torch.as_tensor(np.array(states), dtype=torch.float32), torch.stack(log_probs), rewards


def compute_returns(rewards, gamma):
    """Return scontati G_t = r_t + gamma * G_{t+1}, calcolati a ritroso dalla fine dell'episodio."""
    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    return torch.tensor(returns[::-1], dtype=torch.float32)


@torch.no_grad()
def evaluate_policy(env, policy, M):
    """Esegue M episodi senza aggiornare i pesi. Restituisce (reward totale medio, lunghezza media)."""
    total_rewards, lengths = [], []
    for _ in range(M):
        _, _, rewards = run_episode(env, policy)
        total_rewards.append(sum(rewards))
        lengths.append(len(rewards))
    return sum(total_rewards) / M, sum(lengths) / M


def reinforce(env, policy, num_episodes=1000, gamma=0.99, lr=1e-2, N=25, M=10,
              standardize=True, value_net=None):
    """
    REINFORCE con due baseline opzionali:
      - value_net: rete che stima v(s), allenata per regressione sui return; il peso di ogni step
        diventa G_t - v(s_t) invece di G_t;
      - standardize=True: standardizza i pesi (G_t, oppure G_t - v(s_t)) all'interno dell'episodio.
    Ogni N episodi di training la policy viene valutata su M episodi.
    Restituisce gli episodi di valutazione, il reward totale medio e la lunghezza media.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    if value_net is not None:
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)
    eval_episodes, eval_rewards, eval_lengths = [], [], []

    for episode in range(1, num_episodes + 1):
        states, log_probs, rewards = run_episode(env, policy)
        returns = compute_returns(rewards, gamma)
        weights = returns

        if value_net is not None:
            values = value_net(states)
            weights = returns - values.detach()  # detach: la loss della policy non deve aggiornare la value network
            value_optimizer.zero_grad()
            F.mse_loss(values, returns).backward()
            value_optimizer.step()

        if standardize and len(weights) > 1:  # con un solo step la deviazione standard non e' definita
            weights = (weights - weights.mean()) / (weights.std() + 1e-8)

        loss = -(log_probs * weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % N == 0:
            avg_reward, avg_length = evaluate_policy(env, policy, M)
            eval_episodes.append(episode)
            eval_rewards.append(avg_reward)
            eval_lengths.append(avg_length)
            print(f"Episode {episode} | avg reward: {avg_reward:.1f} | avg length: {avg_length:.1f}")

    return eval_episodes, eval_rewards, eval_lengths
