import json

with open('Rainbow_DQN_2048_V3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Modify extract_obs
        if "def extract_obs(state, player_id=0):" in source:
            new_source = """
# Possible tile values in 2048 (0 means empty cell)
TILE_VALUES = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
TILE_TO_IDX = {v: i for i, v in enumerate(TILE_VALUES)}
OBS_DIM_ONEHOT = 16 * 16  # 16 channels of 4x4 boards flattened to 256

def extract_obs(state, player_id=0):
    raw = extract_obs_raw(state, player_id)
    # Shape: (channels, H, W) -> (16, 4, 4)
    onehot = np.zeros((len(TILE_VALUES), 4, 4), dtype=np.float32)
    for i, val in enumerate(raw):
        r, c = divmod(i, 4)
        idx = TILE_TO_IDX.get(int(val), 0)
        onehot[idx, r, c] = 1.0
    return onehot.flatten()  # shape (256,)

def reward_transform(x):
    eps = 0.001
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + eps * x

def shaped_reward(board, raw_reward):
    if board is None: return raw_reward
    max_tile = board.max()
    corners = [board[0, 0], board[0, 3], board[3, 0], board[3, 3]]
    corner_bonus = max_tile * 0.1 if max_tile in corners else 0.0
    empty_bonus = float(np.sum(board == 0)) * 0.5
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

game = pyspiel.load_game("2048")
test_state = game.new_initial_state()
auto_resolve_chance_nodes(test_state, np.random.default_rng(0))
obs = extract_obs(test_state)
print("one-hot obs shape:", obs.shape, "non-zero:", int(obs.sum()), "(should be 16)")
"""
            cell['source'] = [line + '\n' for line in new_source.strip().split('\n')]
            
        # Modify RainbowQNetwork
        elif "class RainbowQNetwork(nn.Module):" in source:
            new_source = """
class NoisyLinear(nn.Module):
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

class CNNRainbowQNetwork(nn.Module):
    \"\"\"
    V4: CNN-based Rainbow DQN with One-hot encoding (16 channels).
    Suitable, small, lightweight kernel for 4x4 grid:
        Conv2d(16->64, kernel=2, stride=1) -> 3x3 output
        Conv2d(64->128, kernel=2, stride=1) -> 2x2 output
    Flatten -> 128 * 2 * 2 = 512 dimensions.
    Then split into Dueling streams with NoisyLinear!
    \"\"\"
    def __init__(self, obs_dim, num_actions, hidden_dim=512, sigma_init=0.5):
        super().__init__()
        self.num_actions = num_actions
        self.conv = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU()
        )
        # Output of two 2x2 convolutions on 4x4 is 2x2.
        # Channels: 128. Flat size: 128 * 2 * 2 = 512
        flatten_dim = 128 * 2 * 2

        self.value_hidden = NoisyLinear(flatten_dim, hidden_dim, sigma_init)
        self.value_out    = NoisyLinear(hidden_dim, 1, sigma_init)

        self.adv_hidden = NoisyLinear(flatten_dim, hidden_dim, sigma_init)
        self.adv_out    = NoisyLinear(hidden_dim, num_actions, sigma_init)

    def forward(self, x):
        # x is (B, 256) flat from environment
        batch_size = x.size(0)
        x = x.view(batch_size, 16, 4, 4)
        
        feat = self.conv(x)
        feat = feat.view(batch_size, -1) # flatten to (B, 512)

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
test_net = CNNRainbowQNetwork(obs_dim=OBS_DIM_ONEHOT, num_actions=4).to(DEVICE)
test_obs = torch.randn(2, OBS_DIM_ONEHOT).to(DEVICE)
print("output shape:", test_net(test_obs).shape)
print(f"total params: {sum(p.numel() for p in test_net.parameters()):,}")
"""
            cell['source'] = [line + '\n' for line in new_source.strip().split('\n')]
            
        elif "q_net      = RainbowQNetwork(" in source:
            # Update network instantiation
            new_source = source.replace("RainbowQNetwork", "CNNRainbowQNetwork")
            cell['source'] = [new_source]
            
        elif 'checkpoint_path = "rainbow_dqn_2048_v2.pt"' in source or 'rainbow_dqn_2048_v3.pt' in source:
            new_source = source.replace("v2", "v4").replace("v3", "v4")
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line]

    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "Rainbow DQN V2" in source or "V3" in source:
            cell['source'] = [line.replace("V2", "V4").replace("V3", "V4") + ('\n' if not line.endswith('\n') else '') for line in cell['source']]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'fig.suptitle("Rainbow DQN V2 - 2048", fontsize=14)' in source or 'fig.suptitle("Rainbow DQN V3 - 2048", fontsize=14)' in source:
            new_source = source.replace("Rainbow DQN V2", "Rainbow DQN V4").replace("Rainbow DQN V3", "Rainbow DQN V4")
            cell['source'] = [line + '\n' for line in new_source.split('\n') if line]

with open('Rainbow_DQN_2048_V4.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("V4 successfully created.")
