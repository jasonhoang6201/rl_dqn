import json

with open('Rainbow_DQN_2048_V3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Modify extract_obs
        if "def extract_obs(state, player_id=0):" in source:
            # We want to replace only extract_obs and add constants
            
            new_source = source.replace("""def extract_obs(state, player_id=0):
    # One-hot encode each cell independently.
    # Each of the 16 cells gets a 16-dim vector with a 1 at the position of its tile value.
    # This gives the network clearer signal than a single scalar per cell.
    raw = extract_obs_raw(state, player_id)
    onehot = np.zeros((16, len(TILE_VALUES)), dtype=np.float32)
    for i, val in enumerate(raw):
        idx = TILE_TO_IDX.get(int(val), 0)
        onehot[i, idx] = 1.0
    return onehot.reshape(-1)  # shape (256,)""", """def extract_obs(state, player_id=0):
    raw = extract_obs_raw(state, player_id)
    # Shape: (channels, H, W) -> (16, 4, 4)
    onehot = np.zeros((len(TILE_VALUES), 4, 4), dtype=np.float32)
    for i, val in enumerate(raw):
        r, c = divmod(i, 4)
        idx = TILE_TO_IDX.get(int(val), 0)
        onehot[idx, r, c] = 1.0
    return onehot.flatten()  # shape (256,)""")
            
            new_source = new_source.replace("OBS_DIM_ONEHOT = 16 * len(TILE_VALUES)  # 16 cells x 16 possible values = 256", "OBS_DIM_ONEHOT = 16 * 16  # 16 channels of 4x4 boards flattened to 256")
            
            cell['source'] = [line + '\n' for line in new_source.strip().split('\n')]
            
        # Modify RainbowQNetwork
        elif "class RainbowQNetwork(nn.Module):" in source:
            # We just need to replace RainbowQNetwork with CNNRainbowQNetwork 
            # We will use string split and replace
            import re
            parts = re.split(r'class RainbowQNetwork\(nn\.Module\):', source)
            
            new_network = """class CNNRainbowQNetwork(nn.Module):
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
"""
            # Replace the old RainbowQNetwork class from parts[1] up to the next class or function
            
            # Since greedy_action follows RainbowQNetwork, we can split again
            sub_parts = re.split(r'@torch\.no_grad\(\)', parts[1])
            new_source = parts[0] + new_network + '\n\n@torch.no_grad()' + sub_parts[1]
            new_source = new_source.replace("RainbowQNetwork", "CNNRainbowQNetwork")
            
            cell['source'] = [line + '\n' for line in new_source.strip().split('\n')]
            
        elif "q_net      = RainbowQNetwork(" in source:
            # Update network instantiation
            new_source = source.replace("RainbowQNetwork", "CNNRainbowQNetwork")
            cell['source'] = [line + '\n' for line in new_source.strip().split('\n')]
            
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
