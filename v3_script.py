# Cell 1
!python -V
!pip -q install --upgrade pip
!pip -q install open-spiel torch matplotlib tqdm
# Cell 2
import random
import re
from collections import deque, namedtuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pyspiel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)
# Cell 4
def legal_actions(state, player_id=0):
    try:
        return list(state.legal_actions(player_id))
    except TypeError:
        return list(state.legal_actions())


def sample_chance_action(state, rng):
    outcomes = state.chance_outcomes()
    actions, probs = zip(*outcomes)
    idx = rng.choice(len(actions), p=np.asarray(probs, dtype=np.float64))
    return actions[idx]


def auto_resolve_chance_nodes(state, rng):
    while state.is_chance_node() and not state.is_terminal():
        a = sample_chance_action(state, rng)
        state.apply_action(a)
    return state


def state_return(state, player_id=0):
    vals = state.returns()
    return float(vals[player_id]) if len(vals) > player_id else 0.0


def state_reward(state, player_id=0):
    vals = state.rewards()
    return float(vals[player_id]) if len(vals) > player_id else 0.0


def parse_board_numbers(state):
    txt = str(state)
    nums = [int(x) for x in re.findall(r"\d+", txt)]
    if len(nums) >= 16:
        nums = nums[-16:]
        return np.array(nums, dtype=np.int64).reshape(4, 4)
    return None


def make_legal_mask(num_actions, legal_actions_list):
    mask = np.zeros(num_actions, dtype=np.float32)
    mask[legal_actions_list] = 1.0
    return mask


def extract_obs_raw(state, player_id=0):
    for fn_name, args in [
        ("observation_tensor", (player_id,)),
        ("observation_tensor", tuple()),
        ("information_state_tensor", (player_id,)),
        ("information_state_tensor", tuple()),
    ]:
        fn = getattr(state, fn_name, None)
        if fn is None:
            continue
        try:
            obs = fn(*args)
            return np.asarray(obs, dtype=np.float32).reshape(-1)
        except TypeError:
            pass
    raise RuntimeError("Could not extract observation tensor.")


# Possible tile values in 2048 (0 means empty cell)
TILE_VALUES = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
TILE_TO_IDX = {v: i for i, v in enumerate(TILE_VALUES)}
OBS_DIM_ONEHOT = 16 * len(TILE_VALUES)  # 16 cells x 16 possible values = 256


def extract_obs(state, player_id=0):
    # One-hot encode each cell independently.
    # Each of the 16 cells gets a 16-dim vector with a 1 at the position of its tile value.
    # This gives the network clearer signal than a single scalar per cell.
    raw = extract_obs_raw(state, player_id)
    onehot = np.zeros((16, len(TILE_VALUES)), dtype=np.float32)
    for i, val in enumerate(raw):
        idx = TILE_TO_IDX.get(int(val), 0)
        onehot[i, idx] = 1.0
    return onehot.reshape(-1)  # shape (256,)


def reward_transform(x):
    # Compress large rewards to avoid gradient spikes when merging high tiles.
    # From EfficientZero: h(x) = sign(x)*(sqrt(|x|+1)-1) + eps*x
    eps = 0.001
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + eps * x


def shaped_reward(board, raw_reward):
    # Add small bonuses to guide the agent toward good 2048 strategies.
    # The bonuses are scaled small enough to not overshadow the actual merge reward.
    if board is None:
        return raw_reward

    max_tile = board.max()

    # Keeping the largest tile in a corner is a well-known 2048 strategy
    corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
    corner_bonus = max_tile * 0.1 if max_tile in corners else 0.0

    # More empty cells = more room to maneuver
    empty_bonus = float(np.sum(board == 0)) * 0.5

    # Monotonic rows/columns (snake pattern) keep large tiles accessible
    mono_score = 0.0
    for row in board:
        fwd = sum(row[i] * 0.01 for i in range(3) if row[i] >= row[i + 1])
        bwd = sum(row[i + 1] * 0.01 for i in range(3) if row[i + 1] >= row[i])
        mono_score += max(fwd, bwd)
    for col in board.T:
        fwd = sum(col[i] * 0.01 for i in range(3) if col[i] >= col[i + 1])
        bwd = sum(col[i + 1] * 0.01 for i in range(3) if col[i + 1] >= col[i])
        mono_score += max(fwd, bwd)

    return raw_reward + corner_bonus + empty_bonus + mono_score


# Sanity check
game = pyspiel.load_game("2048")
test_state = game.new_initial_state()
auto_resolve_chance_nodes(test_state, np.random.default_rng(0))
obs = extract_obs(test_state)
print("one-hot obs shape:", obs.shape, "non-zero:", int(obs.sum()), "(should be 16)")
print("OBS_DIM_ONEHOT =", OBS_DIM_ONEHOT)
# Cell 6
class OpenSpiel2048Env:
    def __init__(self, seed=42):
        self.game = pyspiel.load_game("2048")
        self.player_id = 0
        self.num_actions = self.game.num_distinct_actions()
        self.obs_dim = OBS_DIM_ONEHOT  # 256 with one-hot encoding
        self.rng = np.random.default_rng(seed)
        self.state = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.game.new_initial_state()
        auto_resolve_chance_nodes(self.state, self.rng)
        return extract_obs(self.state, self.player_id)

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Call reset() first.")
        if self.state.is_terminal():
            raise RuntimeError("Episode already ended.")

        legal = legal_actions(self.state, self.player_id)
        if action not in legal:
            raise ValueError(f"Illegal action {action}. Legal: {legal}")

        prev_return = state_return(self.state, self.player_id)
        self.state.apply_action(int(action))
        auto_resolve_chance_nodes(self.state, self.rng)

        if not self.state.is_terminal():
            next_obs = extract_obs(self.state, self.player_id)
        else:
            next_obs = np.zeros(self.obs_dim, dtype=np.float32)

        new_return = state_return(self.state, self.player_id)
        raw_reward = new_return - prev_return
        board = parse_board_numbers(self.state)

        # Apply reward shaping then compress the scale
        reward = reward_transform(shaped_reward(board, raw_reward))
        done = self.state.is_terminal()

        info = {
            "legal_actions": legal_actions(self.state, self.player_id) if not done else [],
            "state_return": new_return,
            "board": board,
            "state_text": str(self.state),
        }
        return next_obs, float(reward), done, info

    def legal_actions(self):
        if self.state is None or self.state.is_terminal():
            return []
        return legal_actions(self.state, self.player_id)

    def render(self):
        if self.state is None:
            print("<env not reset>")
        else:
            print(self.state)
# Cell 8
Transition = namedtuple("Transition", ["obs", "action", "reward", "next_obs", "done", "legal_mask", "next_legal_mask"])


class SumTree:
    """Binary sum-tree for O(log N) priority sampling."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)
        self.data = [None] * self.capacity
        self.ptr = 0
        self.size = 0

    def _propagate(self, idx, delta):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx, s):
        # We MUST use capacity - 1 as the leaf barrier to correctly shape the array space.
        if idx >= self.capacity - 1:
            return idx
            
        left = 2 * idx + 1
        right = left + 1

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def get(self, s):
        s = min(float(s), self.total() - 1e-8)  # prevent float edge case
        idx = self._retrieve(0, s)
        
        # Absolute safety clamp for tree index - ensures numpy bounds are respected unconditionally
        idx = max(0, min(idx, 2 * self.capacity - 2))
        
        data_idx = idx - self.capacity + 1
        data_idx = max(0, min(data_idx, self.capacity - 1))  # safety clamp for data array
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def __len__(self):
        return self.tree.size

    def add(self, *args):
        self.tree.add(self.max_priority ** self.alpha, Transition(*args))

    def sample(self, batch_size, beta=0.4):
        indices, transitions, weights = [], [], []
        total = self.tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, transition = self.tree.get(s)
            prob = priority / total
            weight = (self.tree.size * prob) ** (-beta)
            indices.append(idx)
            transitions.append(transition)
            weights.append(weight)

        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()
        batch = Transition(*zip(*transitions))
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            priority = (abs(err) + 1e-6) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


print("PrioritizedReplayBuffer ready.")
# Cell 10
class NStepBuffer:
    """Accumulates n transitions and computes the n-step discounted return."""

    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = deque(maxlen=n)

    def push(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) < self.n:
            return None
        G = sum(self.gamma ** i * self.buffer[i].reward for i in range(self.n))
        first = self.buffer[0]
        last  = self.buffer[-1]
        return Transition(
            obs=first.obs, action=first.action, reward=G,
            next_obs=last.next_obs, done=last.done,
            legal_mask=first.legal_mask, next_legal_mask=last.next_legal_mask,
        )

    def flush(self):
        results = []
        while self.buffer:
            n_avail = len(self.buffer)
            G = sum(self.gamma ** i * self.buffer[i].reward for i in range(n_avail))
            first = self.buffer[0]
            last  = self.buffer[-1]
            results.append(Transition(
                obs=first.obs, action=first.action, reward=G,
                next_obs=last.next_obs, done=last.done,
                legal_mask=first.legal_mask, next_legal_mask=last.next_legal_mask,
            ))
            self.buffer.popleft()
        return results

    def reset(self):
        self.buffer.clear()


print("NStepBuffer ready.")
# Cell 12
class NoisyLinear(nn.Module):
    """Linear layer with factorized Gaussian noise. Replaces epsilon-greedy exploration."""

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.full((out_features,), sigma_init))

        self.register_buffer("weight_eps", torch.zeros(out_features, in_features))
        self.register_buffer("bias_eps",   torch.zeros(out_features))

        self._reset_parameters()

    def _reset_parameters(self):
        bound = 1.0 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)

    def _factorized_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def sample_noise(self):
        eps_in  = self._factorized_noise(self.in_features)
        eps_out = self._factorized_noise(self.out_features)
        self.weight_eps.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))
        self.bias_eps.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


class RainbowQNetwork(nn.Module):
    """
    Dueling network with NoisyLinear layers.

    V2 uses hidden_dim=512 to handle the larger one-hot input (256 dims).

    Architecture:
        input(256) -> Linear(512) -> ReLU
                   -> value stream:     NoisyLinear(512->512) -> NoisyLinear(512->1)
                   -> advantage stream: NoisyLinear(512->512) -> NoisyLinear(512->4)
        Q = V + A - mean(A)
    """

    def __init__(self, obs_dim, num_actions, hidden_dim=512, sigma_init=0.5):
        super().__init__()
        self.num_actions = num_actions

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_hidden = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.value_out    = NoisyLinear(hidden_dim, 1, sigma_init)

        self.adv_hidden = NoisyLinear(hidden_dim, hidden_dim, sigma_init)
        self.adv_out    = NoisyLinear(hidden_dim, num_actions, sigma_init)

    def forward(self, x):
        feat = self.feature(x)

        v = F.relu(self.value_hidden(feat))
        v = self.value_out(v)                            # (B, 1)

        a = F.relu(self.adv_hidden(feat))
        a = self.adv_out(a)                              # (B, num_actions)

        return v + (a - a.mean(dim=1, keepdim=True))

    def sample_noise(self):
        self.value_hidden.sample_noise()
        self.value_out.sample_noise()
        self.adv_hidden.sample_noise()
        self.adv_out.sample_noise()


@torch.no_grad()
def greedy_action(q_net, obs, legal_actions_list, num_actions, device=DEVICE):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    q = q_net(obs_t).squeeze(0)
    legal_mask = torch.zeros(num_actions, dtype=torch.bool, device=device)
    legal_mask[legal_actions_list] = True
    q_masked = q.masked_fill(~legal_mask, -1e9)
    return int(torch.argmax(q_masked).item())


# Quick check
test_net = RainbowQNetwork(obs_dim=OBS_DIM_ONEHOT, num_actions=4).to(DEVICE)
test_obs = torch.randn(2, OBS_DIM_ONEHOT).to(DEVICE)
print("output shape:", test_net(test_obs).shape)
print(f"total params: {sum(p.numel() for p in test_net.parameters()):,}")
# Cell 14
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_EPISODES          = 10000      # V3: more episodes for better convergence
BUFFER_SIZE           = 100_000
BATCH_SIZE            = 128
GAMMA                 = 0.995
LR                    = 5e-4
LEARN_START           = 2_000
LEARN_EVERY           = 4
MAX_STEPS_PER_EPISODE = 5_000
GRAD_CLIP             = 10.0

TAU                 = 0.005
TARGET_UPDATE_EVERY = 4

N_STEPS = 5

PER_ALPHA      = 0.6
PER_BETA_START = 0.4
PER_BETA_END   = 1.0
PER_BETA_STEPS = NUM_EPISODES * 200


def beta_by_step(step):
    fraction = min(step / PER_BETA_STEPS, 1.0)
    return PER_BETA_START + fraction * (PER_BETA_END - PER_BETA_START)


def soft_update(q_net, target_net, tau):
    for p_t, p_q in zip(target_net.parameters(), q_net.parameters()):
        p_t.data.copy_((1 - tau) * p_t.data + tau * p_q.data)


train_env   = OpenSpiel2048Env(seed=SEED)
obs_dim     = train_env.obs_dim   # 256
num_actions = train_env.num_actions

q_net      = RainbowQNetwork(obs_dim, num_actions, hidden_dim=512).to(DEVICE)
target_net = RainbowQNetwork(obs_dim, num_actions, hidden_dim=512).to(DEVICE)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=LR)

replay    = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
nstep_buf = NStepBuffer(N_STEPS, GAMMA)

print(f"obs_dim={obs_dim}, num_actions={num_actions}")
print(f"total params: {sum(p.numel() for p in q_net.parameters()):,}")
# Cell 16
def rainbow_update(batch, weights, global_step):
    obs_t      = torch.tensor(np.array(batch.obs),      dtype=torch.float32, device=DEVICE)
    next_obs_t = torch.tensor(np.array(batch.next_obs), dtype=torch.float32, device=DEVICE)
    actions_t  = torch.tensor(batch.action,              dtype=torch.long,    device=DEVICE)
    rewards_t  = torch.tensor(batch.reward,              dtype=torch.float32, device=DEVICE)
    dones_t    = torch.tensor(batch.done,                dtype=torch.float32, device=DEVICE)
    next_masks = torch.tensor(np.array(batch.next_legal_mask), dtype=torch.float32, device=DEVICE)
    weights_t  = torch.tensor(weights,                   dtype=torch.float32, device=DEVICE)

    q_net.sample_noise()
    q_vals = q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double DQN: online net selects action, target net evaluates it
        # Note: do NOT call q_net.sample_noise() here — it modifies weight_eps
        # in-place, corrupting the computation graph needed for loss.backward()
        next_q_online = q_net(next_obs_t)
        next_q_online = next_q_online.masked_fill(next_masks < 0.5, -1e9)
        next_actions  = next_q_online.argmax(dim=1, keepdim=True)

        target_net.sample_noise()
        next_q_target = target_net(next_obs_t).gather(1, next_actions).squeeze(1)

        targets = rewards_t + (GAMMA ** N_STEPS) * next_q_target * (1 - dones_t)

    td_errors = (q_vals - targets).detach().cpu().numpy()
    loss = (weights_t * F.huber_loss(q_vals, targets, reduction="none")).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), GRAD_CLIP)
    optimizer.step()

    return loss.item(), td_errors
# Cell 18
episode_returns  = []
episode_lengths  = []
loss_history     = []
eval_returns     = []
max_tile_history = []

global_step = 0

for episode in tqdm(range(1, NUM_EPISODES + 1)):
    obs  = train_env.reset()
    done = False
    ep_return = 0.0
    ep_len    = 0
    nstep_buf.reset()

    q_net.train()

    while not done and ep_len < MAX_STEPS_PER_EPISODE:
        legal      = train_env.legal_actions()
        legal_mask = make_legal_mask(num_actions, legal)

        q_net.sample_noise()
        with torch.no_grad():
            action = greedy_action(q_net, obs, legal, num_actions, DEVICE)

        next_obs, reward, done, info = train_env.step(action)
        next_legal      = info["legal_actions"] if not done else []
        next_legal_mask = make_legal_mask(num_actions, next_legal)

        t = Transition(obs, action, reward, next_obs, done, legal_mask, next_legal_mask)
        n_step_t = nstep_buf.push(t)
        if n_step_t is not None:
            replay.add(*n_step_t)

        obs        = next_obs
        ep_return += reward
        ep_len    += 1
        global_step += 1

        if len(replay) >= LEARN_START and global_step % LEARN_EVERY == 0:
            beta = beta_by_step(global_step)
            batch, indices, weights = replay.sample(BATCH_SIZE, beta=beta)
            loss, td_errors = rainbow_update(batch, weights, global_step)
            replay.update_priorities(indices, td_errors)
            loss_history.append(loss)

            if global_step % TARGET_UPDATE_EVERY == 0:
                soft_update(q_net, target_net, TAU)

    if done:
        for remaining in nstep_buf.flush():
            replay.add(*remaining)

    episode_returns.append(ep_return)
    episode_lengths.append(ep_len)

    final_board = train_env.state and parse_board_numbers(train_env.state)
    max_tile_history.append(int(final_board.max()) if final_board is not None else 0)

    if episode % 20 == 0:
        q_net.eval()
        eval_env  = OpenSpiel2048Env(seed=1000 + episode)
        obs_eval  = eval_env.reset(seed=2000 + episode)
        done_eval, ret_eval, steps_eval = False, 0.0, 0
        while not done_eval and steps_eval < MAX_STEPS_PER_EPISODE:
            legal_eval = eval_env.legal_actions()
            a_eval = greedy_action(q_net, obs_eval, legal_eval, num_actions, DEVICE)
            obs_eval, r_eval, done_eval, _ = eval_env.step(a_eval)
            ret_eval   += r_eval
            steps_eval += 1
        eval_returns.append((episode, ret_eval))
        q_net.train()

        recent_max_tile = max(max_tile_history[-20:]) if max_tile_history else 0
        tqdm.write(
            f"Ep {episode:4d} | avg={np.mean(episode_returns[-20:]):.1f} | "
            f"eval={ret_eval:.1f} | max_tile={recent_max_tile} | "
            f"buf={len(replay):,} | beta={beta_by_step(global_step):.3f}"
        )

print("Training done.")
# Cell 20
def moving_average(x, w=20):
    if len(x) < w:
        return np.asarray(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Rainbow DQN V2 - 2048", fontsize=14)

ax = axes[0, 0]
ax.plot(episode_returns, alpha=0.3, color="steelblue", label="episode")
ax.plot(moving_average(episode_returns, 20), color="steelblue", label="MA-20")
ax.set_title("Training Return")
ax.set_xlabel("Episode")
ax.legend()

ax = axes[0, 1]
ax.plot(episode_lengths, alpha=0.4, color="orange")
ax.plot(moving_average(episode_lengths, 20), color="darkorange")
ax.set_title("Episode Length")
ax.set_xlabel("Episode")

ax = axes[0, 2]
ax.plot(loss_history, alpha=0.6, color="red")
ax.set_title("Loss")
ax.set_xlabel("Update Step")
ax.set_yscale("log")

ax = axes[1, 0]
if eval_returns:
    eps, vals = zip(*eval_returns)
    ax.plot(eps, vals, marker="o", color="green")
ax.set_title("Greedy Eval Return")
ax.set_xlabel("Episode")

ax = axes[1, 1]
if max_tile_history:
    unique, counts = np.unique(max_tile_history, return_counts=True)
    ax.bar([str(u) for u in unique], counts, color="purple", alpha=0.7)
ax.set_title("Max Tile Distribution")
ax.set_xlabel("Tile Value")
ax.set_ylabel("Count")

ax = axes[1, 2]
if max_tile_history:
    ax.plot(max_tile_history, alpha=0.3, color="teal")
    ax.plot(moving_average(max_tile_history, 20), color="teal")
ax.set_title("Max Tile per Episode (MA-20)")
ax.set_xlabel("Episode")

plt.tight_layout()
plt.show()

print("\n=== Summary ===")
if eval_returns:
    print(f"Best eval return: {max(v for _, v in eval_returns):.1f}")
print(f"Avg last-20 return: {np.mean(episode_returns[-20:]):.1f}")
if max_tile_history:
    unique, counts = np.unique(max_tile_history, return_counts=True)
    for tile, cnt in zip(unique, counts):
        print(f"  Tile {tile:>5}: {cnt} times ({100*cnt/len(max_tile_history):.1f}%)")
# Cell 22
q_net.eval()

eval_env = OpenSpiel2048Env(seed=999)
obs      = eval_env.reset(seed=999)
done     = False
greedy_return = 0.0
rollout  = []

while not done and len(rollout) < MAX_STEPS_PER_EPISODE:
    legal  = eval_env.legal_actions()
    action = greedy_action(q_net, obs, legal, num_actions, DEVICE)
    next_obs, reward, done, info = eval_env.step(action)
    rollout.append({"action": action, "reward": reward,
                    "board": info["board"], "state_text": info["state_text"]})
    obs = next_obs
    greedy_return += reward

print(f"Greedy return: {greedy_return:.1f}")
print(f"Steps: {len(rollout)}")
if rollout and rollout[-1]["board"] is not None:
    print(f"Max tile: {rollout[-1]['board'].max()}")
eval_env.render()
# Cell 23
action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
n_show = min(5, len(rollout))
for i, step in enumerate(rollout[-n_show:], start=len(rollout) - n_show + 1):
    print("=" * 50)
    print(f"Step {i} | {action_names[step['action']]} | reward={step['reward']:.2f}")
    if step["board"] is not None:
        print(step["board"])
# Cell 25
checkpoint_path = "rainbow_dqn_2048_v2.pt"
torch.save({
    "model_state_dict":  q_net.state_dict(),
    "target_state_dict": target_net.state_dict(),
    "obs_dim":           obs_dim,
    "num_actions":       num_actions,
    "episode_returns":   episode_returns,
    "episode_lengths":   episode_lengths,
    "loss_history":      loss_history,
    "max_tile_history":  max_tile_history,
    "hyperparams": {
        "N_STEPS":   N_STEPS,
        "PER_ALPHA": PER_ALPHA,
        "TAU":       TAU,
        "GAMMA":     GAMMA,
    },
}, checkpoint_path)
print("Saved:", checkpoint_path)
