import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from IPython.display import HTML
from src.policy import select_action


@torch.no_grad()
def animate_episode(env_name, policy=None, interval=20):
    """
    Gioca un episodio e lo restituisce come animazione per Jupyter.
    Con policy=None le azioni sono casuali.
    """
    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset()
    frames, done = [env.render()], False
    while not done:
        action = env.action_space.sample() if policy is None else select_action(policy, state)[0]
        state, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        done = terminated or truncated
    env.close()
    print(f"Episodio durato {len(frames) - 1} step")

    fig, ax = plt.subplots()
    ax.axis("off")
    img = ax.imshow(frames[0])

    def update(i):
        img.set_data(frames[i])
        return [img]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval)
    plt.close(fig)  # evita che sotto l'animazione compaia anche la figura statica
    return HTML(anim.to_jshtml())


def plot_evaluation(episodes, rewards, lengths, threshold=None, title=""):
    """Grafici delle metriche di valutazione: reward totale medio e lunghezza media degli episodi."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(episodes, rewards, marker="o")
    if threshold is not None:
        ax1.axhline(threshold, color="red", linestyle="--", label="soglia 'risolto'")
        ax1.legend()
    ax1.set(title=f"{title} - reward totale medio", xlabel="Episodio di training", ylabel="Reward totale")
    ax2.plot(episodes, lengths, marker="o", color="tab:orange")
    ax2.set(title=f"{title} - lunghezza media", xlabel="Episodio di training", ylabel="Step")
    plt.tight_layout()
    plt.show()