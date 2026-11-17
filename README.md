# Deep Learning Applications: Laboratories

## Lab 1: Convolutional Neural Networks

### Exercise 1.1: A baseline MLP

An MLP classifies MNIST digits, the network is composed of two hidden layer, trained for 20 epochs.

<details>
<summary><b>MLP implementation</b></summary>

```python
class MLP(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_dim=64, depth=1, num_classes=10):
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

</details>

| Training loss | Training accuracy |
|:---:|:---:|
| ![](assets/lab1/dla-lab1_1-train_loss.png) | ![](assets/lab1/dla-lab1_1-train_accuracy.png) |
| **Validation loss** | **Validation accuracy** |
| ![](assets/lab1/dla-lab1_1-val_loss.png) | ![](assets/lab1/dla-lab1_1-val_accuracy.png) |

All networks converge within a few epochs. Depth does not help: the shallow networks reach about
97.5% validation accuracy, while the 16-layer one stays around 96% and trains less stably. Since the
deeper networks also have a higher training loss, deeper plain MLPs are
simply harder to optimize, which motivates residual connections.

### Exercise 1.2: Adding Residual Connections

ResidualMLP groups the hidden layers into residual blocks. For the same depth the plain and residual networks have the same layers, the same number
of parameters and, with the same seed, identical initial weights: the skip connections are the only
difference.

<details>
<summary><b>ResidualMLP implementation</b></summary>

```python
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(dim, dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.f = nn.Sequential(*layers)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.f(x))


class ResidualMLP(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_dim=64, depth=2, num_classes=10, layers_per_block=2):
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        layers += [ResidualMLPBlock(hidden_dim, layers_per_block) for _ in range(depth // layers_per_block)]
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

</details>

| | Training loss | Validation accuracy |
|:---:|:---:|:---:|
| **MLP** | ![](assets/lab1/dla-lab1_2-train_loss.png) | ![](assets/lab1/dla-lab1_2-val_accuracy.png) |
| **ResidualMLP** | ![](assets/lab1/dla-lab1_2-residual-train_loss.png) | ![](assets/lab1/dla-lab1_2-residual-val_accuracy.png) |

The plain MLP degrades with depth, and with 32 hidden layers it does not learn at all: its training
loss stays at $\approx 2.30$ and its accuracy at about 0.1. With residual
connections every depth trains equally well (above 99% training and 97–98% validation accuracy), so
a 32-layer residual MLP is as easy to train as a 2-layer one, confirming the ResNet result. On an easy
dataset like MNIST it also show that deep networks are not so usefull and a simple MLP make a good work too on a dataset like MNIST.

#### Extra: Gradient analysis

To understand why the deep plain MLP fails, we measure the gradient norm of each linear layer on a
single training batch, for every network, before and after training.

<details>
<summary><b>Gradient analysis code</b></summary>

```python
def grad_norms_per_layer(model, x, y):
    model.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    return [m.weight.grad.norm().item() for m in model.modules() if isinstance(m, nn.Linear)]


train_loader = get_mnist(batch_size=config_12["batch_size"], val_set_size=config_12["val_set_size"],
                         seed=config_12["seed"])[0]
x, y = (t.to(device) for t in next(iter(train_loader)))

grad_norms = {}
for trained_model, _, cfg in results_12.values():
    torch.manual_seed(cfg["seed"])
    init_model = MODELS[cfg["architecture"]](
        **{k: cfg[k] for k in ("hidden_dim", "depth", "layers_per_block") if k in cfg}).to(device)
    for stage, model in [("iniziale", init_model), ("post-training", trained_model)]:
        grad_norms[(cfg["architecture"], cfg["depth"], stage)] = grad_norms_per_layer(model, x, y)

ratios = pd.Series({k: norms[1] / norms[-2] for k, norms in grad_norms.items()}).unstack([2, 0])
```

</details>

![Gradient norm per layer](assets/lab1/gradient_analysis.png)

| Depth | MLP (initial) | MLP (trained) | ResidualMLP (initial) | ResidualMLP (trained) |
|:---:|:---:|:---:|:---:|:---:|
| 2  | 1.07 | 1.45 | 1.03 | 2.24 |
| 4  | 0.51 | 2.86 | 0.83 | 2.39 |
| 8  | 1.5 × 10⁻² | 5.02 | 0.67 | 2.35 |
| 16 | 1.6 × 10⁻⁵ | 5.79 | 0.64 | 5.81 |
| 32 | 6.4 × 10⁻¹² | 6.2 × 10⁻¹³ | 0.42 | 13.99 |

*Ratio between the gradient norm of the first and the last hidden layer*

The plain MLP suffers from vanishing gradients: the norm decays from the output to
the input by about 10 orders of magnitude with 32 layers.
In plain MLPs, the gradient is attenuated at every layer as it flows back towards the input: the deeper the network, the weaker it is when it reaches the first layers,
until it prevents them from being trained. In MLPs with skip connections, the gradient has a direct path through which it reaches the first layers without being attenuated, at any depth.

### Exercise 1.3: Rinse and Repeat (but with a CNN)

The same comparison is repeated with CNNs on CIFAR-10. Both networks follow the CIFAR ResNet of the
original paper. The residual CNN uses torchvision's BasicBlock, the plain CNN uses PlainBlock, the same layers without the
skip connection. As before, both start from identical weights.

<details>
<summary><b>CNN implementation</b></summary>

```python
class PlainBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False), nn.BatchNorm2d(planes), nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, 1, 1, bias=False), nn.BatchNorm2d(planes), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.f(x)


class CIFARNet(nn.Module):
    def __init__(self, block, depth=1, base_channels=16, num_classes=10):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(nn.Conv2d(3, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        layers, in_ch = [], c
        for out_ch, stride in [(c, 1), (2 * c, 2), (4 * c, 2)]:
            downsample = None if (stride == 1 and in_ch == out_ch) else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))
            layers.append(block(in_ch, out_ch, stride, downsample))
            layers += [block(out_ch, out_ch) for _ in range(depth - 1)]
            in_ch = out_ch
        self.stages = nn.Sequential(*layers)
        self.fc = nn.Linear(4 * c, num_classes)

    def feature_maps(self, x):
        return self.stages(self.stem(x))

    def forward(self, x):
        return self.fc(self.feature_maps(x).mean(dim=(2, 3)))


MODELS["PlainCNN"] = lambda **kw: CIFARNet(PlainBlock, **kw)
MODELS["ResidualCNN"] = lambda **kw: CIFARNet(BasicBlock, **kw)
```

</details>

| | Training loss | Validation accuracy |
|:---:|:---:|:---:|
| **PlainCNN** | ![](assets/lab1/dla-lab1_3-train_loss.png) | ![](assets/lab1/dla-lab1_3-val_accuracy.png) |
| **ResidualCNN** | ![](assets/lab1/dla-lab1_3-residualCNN_train_loss.png) | ![](assets/lab1/dla-lab1_3-residualCNN_val_accuracy.png) |

`d1` … `d16` correspond to 8, 14, 26, 50 and 98 layers.

![Plain vs residual CNN](assets/lab1/cnn_with_residual.png)

Deeper plain CNNs do not always work better: the best one has 14 layers (84% test accuracy),
then accuracy drops to 81%, 70% and 21% at 98 layers, with the training loss following the same
trend, the degradation problem again.
Even deeper residual CNNs do work better: accuracy grows with depth from 79% to about 86% and
the training loss keeps decreasing. Unlike MNIST, CIFAR-10 is hard enough for extra depth to be
useful, but only when the network can be trained.

### Exercise 2.3: Explaining the predictions of a CNN

Using the best ResidualCNN from Exercise 1.3, the notebook implements CAM to visualize which image regions drive each classification decision, 
and compares it with a ResNet-18 pre-trained on ImageNet, applied to Imagenette.

<details>
<summary><b>CAM implementation</b></summary>

```python
def compute_cam(feature_fn, fc, images):
    with torch.no_grad():
        fmaps = feature_fn(images)
        preds = fc(fmaps.mean(dim=(2, 3))).argmax(dim=1)
        weights = fc.weight[preds]
        cams = torch.einsum("bkhw,bk->bhw", fmaps, weights).clamp(min=0)
        cams = cams - cams.amin(dim=(1, 2), keepdim=True)
        cams = cams / (cams.amax(dim=(1, 2), keepdim=True) + 1e-8)
        cams = nn.functional.interpolate(cams[:, None], size=images.shape[-2:],
                                         mode="bilinear", align_corners=False)
    return cams[:, 0].cpu(), preds.cpu()


cams, preds = compute_cam(cifar_model.feature_maps, cifar_model.fc, images)

resnet_trunk = nn.Sequential(*list(resnet.children())[:-2])
cams, preds = compute_cam(resnet_trunk, resnet.fc, images)
```

</details>

![CAM on CIFAR-10](assets/lab1/CAM1.png)

![CAM on Imagenette](assets/lab1/CAM2.png)

Both models attend to the objects rather than the background, the ResidualCNN produces maps centred on the cat, the ships, the frogs and even the airplane printed on a poster,
while the pre-trained ResNet-18 highlights the discriminative parts of each object

Both networks focus on the objects rather than the background: the cat, the ships, the airplane on
the poster, the body of the tench (not the hand), the face of the dog, the French horn. The maps
also explain the mistakes: the chain saw becomes a lumbermill because the network looks at the
whole woodworking scene, and the gas pump becomes an oxygen mask because it focuses on the hand
in the foreground.




## Lab 2: Deep Reinforcement Learning

### Exercise 1: Improving the REINFORCE implementation

The goal of this exercise is to get REINFORCE working on
[CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) and to replace the
running average of training rewards, a noisy and misleading metric, with a proper evaluation
protocol. The original implementation is refactored into a small `src/` package (`policy.py` for the
networks, `reinforce.py` for the algorithm, `viz.py` for plots and animations).

The policy is an MLP with one hidden layer of 128 units that maps the state to the logits of the
actions; actions are sampled from the resulting categorical distribution. After each episode the
discounted returns $G_t$ are computed and the policy is updated by minimizing
$-\frac{1}{T}\sum_t \log \pi(a_t \mid s_t)\, G_t$. Every `N` training episodes the training is
paused and the current policy is run for `M` episodes without updating the weights, collecting the
**average total reward** and the **average episode length**.

<details>
<summary><b>Evaluation and training loop</b> (click to expand)</summary>

```python
@torch.no_grad()
def evaluate_policy(env, policy, M):
    total_rewards, lengths = [], []
    for _ in range(M):
        _, _, rewards = run_episode(env, policy)
        total_rewards.append(sum(rewards))
        lengths.append(len(rewards))
    return sum(total_rewards) / M, sum(lengths) / M


def reinforce(env, policy, num_episodes=1000, gamma=0.99, lr=1e-2, N=25, M=10,
              standardize=True, value_net=None):
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
            weights = returns - values.detach()
            value_optimizer.zero_grad()
            F.mse_loss(values, returns).backward()
            value_optimizer.step()

        if standardize and len(weights) > 1:
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
```

</details>

The agent is trained for 1000 episodes and evaluated every `N = 25` episodes over `M = 10` episodes.
As in the original implementation, returns are standardized within each episode (the effect of this
choice is studied in Exercise 2).

![REINFORCE on CartPole: evaluation metrics](assets/lab2/lab2_ex1_eval.png)


In CartPole the agent receives +1 for every step the pole stays up, so the average total reward and
the average episode length coincide, both curves rise together, reaching the maximum of 500 around episode 525.
Training is not stable, though: the evaluation suddenly drops to 69 around episode 675 and again to about 200–240 at the end.
The periodic evaluation makes these collapses, caused by the high variance of REINFORCE, clearly visible, while a running average would smooth them out. 
They motivate the baselines of Exercise 2.


### Exercise 2: REINFORCE with a value baseline

Subtracting from the return $G_t$ any baseline that does not depend on the action leaves the expected
gradient unchanged but can greatly reduce its variance. The exercise compares three variants on
CartPole, all starting from the same seed and the same initial weights:

- **no baseline**: the raw returns $G_t$ are used;
- **standardization**: the returns are standardized within each episode, a baseline that is constant
  for the whole episode (the default of the original implementation);
- **value network**: a second network $\tilde{v}(s \mid \mathbf{w})$, with the same architecture as
  the policy but a single output, estimates the value of each state and the policy is updated with
  $G_t - \tilde{v}(s_t)$. The value network is trained separately, by regression on the returns.

<details>
<summary><b>Value network and baseline</b> (click to expand)</summary>

```python
class ValueNet(nn.Module):
    def __init__(self, env, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


if value_net is not None:
    values = value_net(states)
    weights = returns - values.detach()
    value_optimizer.zero_grad()
    F.mse_loss(values, returns).backward()
    value_optimizer.step()


def train(standardize, value_baseline=False):
    torch.manual_seed(SEED)
    env.reset(seed=SEED)
    policy = PolicyNet(env)
    value_net = ValueNet(env) if value_baseline else None
    return reinforce(env, policy, num_episodes=1000, N=25, M=10,
                     standardize=standardize, value_net=value_net)


episodes, rewards_std, _ = train(standardize=True)
_, rewards_raw, _ = train(standardize=False)
_, rewards_value, _ = train(standardize=False, value_baseline=True)
```

</details>

![REINFORCE on CartPole — comparison of baselines](assets/lab2/lab2_ex2_baselines.png)

Without a baseline the policy collapses in the first episodes and never recovers. Standardization reaches 500 from episode
275, but with several sudden collapses. The value network performs best: it reaches 500 already at
episode 175 and its drops are shorter.


### Exercise 3.1: Lunar Lander with REINFORCE

The same implementation is applied to
[Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/), a harder environment:
the state has 8 components, there are 4 actions (do nothing, fire the left, main or right engine) and
an episode lasts at most 1000 steps. The agent is rewarded for landing softly between the two flags
(with a +100 bonus when it comes to rest), penalized for crashing (−100) and for every frame an engine
is firing; the environment is considered solved at an average reward of 200. Unlike CartPole, here
the average reward and the average episode length are different metrics: a long episode can mean
that the lander hovers without landing.

The agent uses the value network baseline of Exercise 2, which was the most stable variant, without
standardization and with a learning rate of 3e-3; it is trained for 4000 episodes and evaluated every
200 episodes over 10 episodes.

<details>
<summary><b>Lunar Lander training</b> (click to expand)</summary>

```python
@torch.no_grad()
def evaluate_policy(env, policy, M):
    total_rewards, lengths = [], []
    for _ in range(M):
        _, _, rewards = run_episode(env, policy)
        total_rewards.append(sum(rewards))
        lengths.append(len(rewards))
    return sum(total_rewards) / M, sum(lengths) / M


def reinforce(env, policy, num_episodes=1000, gamma=0.99, lr=1e-2, N=25, M=10,
              standardize=True, value_net=None):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    if value_net is not None:
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)
    eval_episodes, eval_rewards, eval_lengths = [], [], []

    for episode in range(1, num_episodes + 1):
        states, log_probs, rewards = run_episode(env, policy)
        returns = compute_returns(rewards, gamma)
        weights = returns

        if value_net is not None:  # value baseline, used from Exercise 2
            values = value_net(states)
            weights = returns - values.detach()
            value_optimizer.zero_grad()
            F.mse_loss(values, returns).backward()
            value_optimizer.step()

        if standardize and len(weights) > 1:
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
```


</details>

![REINFORCE on Lunar Lander — evaluation metrics](assets/lab2/lab2_ex3_lunarlander_eval.png)

The average reward grows from about −106, when the lander crashes within 90 steps, to the "solved"
threshold, which it reaches once at episode 1800. After that it settles between 135 and 165,
while the episode length rises to the 1000-step limit from episode 3200 on. The agent has learned
the safe part of the task, slowing down and staying near the landing pad without crashing, but not
to switch the engines off: it hovers until the time runs out, which the video of the trained agent
confirms. This local optimum, where avoiding the −100 crash is easier to learn than earning the +100
landing bonus.
