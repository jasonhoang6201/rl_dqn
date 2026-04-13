# 🌈 Rainbow DQN — Phân Tích 9 Cải Tiến So Với DQN Gốc

> So sánh chi tiết 3 files:
> - `DQN_for_2048_game.ipynb` — Vanilla DQN (file gốc của thầy)
> - `Rainbow_DQN_2048.ipynb` — Rainbow V1 (5/6 kỹ thuật Rainbow)
> - `Rainbow_DQN_2048_V2.ipynb` — Rainbow V2 (V1 + One-hot + Reward Shaping)

---

## 🗺️ Bản đồ tổng quan

```
Vanilla DQN (thầy)
    ├─ [1] Thêm Double DQN      → giảm overestimation Q-values
    ├─ [2] Thêm Dueling Network → tách V(s) và A(s,a)
    ├─ [3] Thêm PER             → sample ưu tiên cái quan trọng
    ├─ [4] Thêm N-step Return   → reward lan xa hơn
    ├─ [5] Thêm NoisyNets       → explore thông minh hơn
    └─ [6] Soft Update          → target update mượt, ổn định hơn
            = Rainbow DQN V1

Rainbow DQN V2
    = V1
    ├─ [7] One-hot Encoding     → biểu diễn state rõ ràng hơn (16→256 dims)
    ├─ [8] Reward Shaping       → dạy agent chiến lược 2048
    └─ [9] Nhiều episodes hơn (500 → 5000)
```

---

## 📊 Bảng so sánh tổng hợp

| Thành phần | DQN gốc | Rainbow V1 | Rainbow V2 |
|---|---|---|---|
| **Kiến trúc mạng** | 3 Linear layers | Dueling Network | Dueling (hidden 512) |
| **Exploration** | ε-greedy random | NoisyNets | NoisyNets |
| **Replay Buffer** | Uniform (random đều) | Prioritized (PER) | Prioritized (PER) |
| **Bellman target** | 1-step | N-step (n=3) | N-step (n=3) |
| **Action selection** | Vanilla argmax | Double DQN | Double DQN |
| **Target update** | Hard copy (250 steps) | Soft update (τ=0.005) | Soft update (τ=0.005) |
| **State encoding** | Raw tensor (16 dims) | log2 (16 dims) | One-hot (256 dims) |
| **Reward** | Delta return | reward_transform | Shaped + transform |
| **Hidden dim** | 256 | 256 | 512 |
| **Episodes** | 2000 | 500 | 5000 |
| **Params mạng** | ~200k | ~200k | ~800k |

---

## 🔄 Rainbow là gì?

**Rainbow** là paper của DeepMind (2017) kết hợp **6 cải tiến tốt nhất** của DQN thành một agent duy nhất. Kết quả: vượt trội tất cả các phương pháp riêng lẻ trước đó trên bộ Atari games.

6 kỹ thuật Rainbow:

| # | Kỹ thuật | V1 | V2 | Ghi chú |
|---|---|---|---|---|
| 1 | Double DQN | ✅ | ✅ | |
| 2 | Dueling Network | ✅ | ✅ | |
| 3 | Prioritized Replay (PER) | ✅ | ✅ | |
| 4 | N-step Returns | ✅ | ✅ | |
| 5 | NoisyNets | ✅ | ✅ | |
| 6 | Distributional RL (C51) | ❌ | ❌ | Phức tạp nhất, bỏ qua |

---

## Cải tiến 1 — Double DQN

### Vấn đề: Vanilla DQN overestimate Q-values

```python
# Vanilla DQN: target_net vừa CHỌN action vừa ĐÁNH GIÁ Q-value
next_q = target_net(next_states).max(dim=1).values      # ← cùng 1 mạng
target = reward + gamma * next_q

# Vấn đề: .max() luôn chọn Q cao nhất, nhưng Q cao nhất thường
# là do noise chứ không phải thực sự tốt → overestimate
```

**Tại sao overestimate là xấu?**

Nếu agent nghĩ mọi action đều tốt hơn thực tế → không phân biệt được action nào thực sự tốt hơn → học chậm.

### Giải pháp: Double DQN — Tách chọn và đánh giá

```python
# Double DQN: q_net CHỌN action, target_net ĐÁNH GIÁ
next_actions = q_net(next_states).argmax(dim=1, keepdim=True)   # q_net chọn
next_q = target_net(next_states).gather(1, next_actions).squeeze(1)  # target_net đánh giá
target = reward + gamma * next_q
```

```
Vanilla DQN:
  target_net → [Q_UP=5.0, Q_DOWN=3.2, Q_LEFT=4.8, Q_RIGHT=2.1]
             → max = 5.0  (chọn và đánh giá cùng 1 lúc)

Double DQN:
  q_net      → [Q_UP=5.0, Q_DOWN=3.2, Q_LEFT=4.8, Q_RIGHT=2.1]
             → chọn UP (index=0)
  target_net → Q(s', UP) = 4.3   (đánh giá riêng → ổn định hơn)
```

> **🎯 Ví dụ:**
> - Giám khảo A tự chọn bài và tự chấm → dễ thiên vị, tự khen mình
> - Giám khảo A chọn bài, Giám khảo B chấm điểm → khách quan hơn

---

## Cải tiến 2 — Dueling Network

### Vấn đề với kiến trúc thẳng

```python
# Vanilla QNetwork: học trực tiếp Q(s, a) cho từng action
class QNetwork(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(16, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 4),          # → Q_UP, Q_DOWN, Q_LEFT, Q_RIGHT
        )
```

**Vấn đề:** Trong nhiều trạng thái 2048, tất cả các action đều tốt (hoặc đều xấu) như nhau. Mạng phải học riêng từng Q(s,a) → lãng phí, chậm.

### Giải pháp: Dueling Network — Tách V(s) và A(s,a)

```python
class RainbowQNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_dim=256):
        # Shared: học đặc trưng chung từ state
        self.feature = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())

        # Value stream: "trạng thái này tốt hay xấu?"
        self.value_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.value_out    = NoisyLinear(hidden_dim, 1)           # → 1 số duy nhất V(s)

        # Advantage stream: "action này tốt hơn trung bình bao nhiêu?"
        self.adv_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.adv_out    = NoisyLinear(hidden_dim, num_actions)   # → A(s,a) cho mỗi action

    def forward(self, x):
        feat = self.feature(x)
        v = self.value_out(F.relu(self.value_hidden(feat)))      # (B, 1)
        a = self.adv_out(F.relu(self.adv_hidden(feat)))          # (B, 4)

        # Dueling combine: Q = V + A - mean(A)
        # Trừ mean(A) để A có thể học deviation (không bị offset)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
```

**Ví dụ số cụ thể:**

```
Tình huống: Bàn cờ rất xấu (gần game over)
  V(s) = -8.0   ← "trạng thái xấu"
  A(s) = [UP=0.5, DOWN=-0.2, LEFT=0.3, RIGHT=-0.6]   ← "UP tốt hơn một chút"
  Q(s) = -8.0 + [0.5, -0.2, 0.3, -0.6] - mean([0.5,-0.2,0.3,-0.6])
       = -8.0 + [0.5, -0.2, 0.3, -0.6] - 0.0
       = [-7.5, -8.2, -7.7, -8.6]
  → Chọn UP (ít xấu nhất)

Tình huống: Bàn cờ rất tốt
  V(s) = +5.0   ← "trạng thái tốt"
  A(s) tương tự → mọi Q đều cao
```

> **🎯 Ví dụ:**
> Đánh giá bài thi:
> - Vanilla: chấm điểm từng câu riêng lẻ cho mỗi học sinh
> - Dueling: biết trước "đề thi dễ hay khó" (V) + "học sinh trả lời câu này so với trung bình" (A) → hiệu quả hơn

---

## Cải tiến 3 — Prioritized Experience Replay (PER)

### Vấn đề với Uniform Replay

```python
# Vanilla: mọi transition được chọn với xác suất bằng nhau
class ReplayBuffer:
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)  # hoàn toàn ngẫu nhiên
```

Nếu buffer có 50,000 transitions, mỗi cái có 1/50,000 khả năng được chọn — kể cả những transition agent đã học tốt (không cần ôn lại) hay những transition rất quan trọng (đang sai nhiều).

### Giải pháp: Sample ưu tiên theo TD-error

**TD-error** = độ chênh lệch giữa dự đoán và thực tế:

```
TD-error = |r + γ·V(s') - Q(s,a)|
            ^              ^
         đáp án đúng    dự đoán hiện tại
```

TD-error lớn = đang dự đoán sai nhiều = cần học nhiều hơn → ưu tiên sample hơn.

```python
# PER: xác suất được sample ∝ TD-error^α
P(i) = priority(i)^α / Σ priority(j)^α

# α = 0 → uniform hoàn toàn
# α = 1 → ưu tiên hoàn toàn theo priority
# α = 0.6 → cân bằng (được dùng ở đây)
```

### SumTree — Cấu trúc dữ liệu hiệu quả

```
           [tổng = 42]        ← root
          /            \
      [25]              [17]  ← tổng con
     /    \            /    \
   [13]   [12]       [9]   [8]  ← tổng con
   / \    / \        / \   / \
  [6][7] [5][7]    [4][5] [3][5]  ← priority của từng transition
```

- Tìm transition theo priority: đi từ root xuống → **O(log N)**
- Uniform buffer: phải tính lại toàn bộ → **O(N)**

### IS Weights — Bù lại bias của PER

Vì sample không đều → tính loss không đều → cần bù lại:

```python
# Transition được sample ít → tăng trọng số khi tính loss
weight = (1 / (N × P(i)))^β
loss = weight × (Q(s,a) - target)²

# beta tăng dần từ 0.4 → 1.0 trong quá trình train
# (ban đầu chấp nhận bias, cuối cùng hoàn toàn correct)
```

> **📚 Ví dụ:**
> Học tiếng Anh với flashcard:
> - Uniform: ôn ngẫu nhiên tất cả từ — nhiều thời gian ôn từ đã thuộc
> - PER: Từ nào hay nhớ sai → được ôn nhiều hơn → học nhanh hơn nhiều

---

## Cải tiến 4 — N-step Return

### Vấn đề với 1-step Bellman

```
1-step: Q(s_t, a_t) = r_t + γ · max Q(s_{t+1}, a)
```

Trong 2048, reward **sparse**: hàng chục bước di chuyển mới có 1 lần merge (mới có reward). Những bước chuẩn bị trước merge có reward = 0 → agent không biết chúng có giá trị.

### Giải pháp: N-step Return (n=3)

```
3-step: Q(s_t, a_t) = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·max Q(s_{t+3}, a)
```

```python
class NStepBuffer:
    def __init__(self, n=3, gamma=0.99):
        self.n = n
        self.gamma = gamma
        self.buffer = deque()

    def add(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) == self.n:
            # Tính n-step return
            G = sum(self.gamma**i * self.buffer[i].reward for i in range(self.n))
            # Đẩy vào PER với s_t, a_t, G (n-step return), s_{t+n}
            ...
            self.buffer.popleft()
```

**Ví dụ số:**

```
Không có n-step (mỗi bước học riêng):
  Bước t:   s_t → LEFT  → reward =  0  → học Q(s_t, LEFT)  = 0 + 0.99·Q(s_{t+1})
  Bước t+1: s_{t+1} → DOWN  → reward =  0  → học Q(s_{t+1}, DOWN) = 0 + ...
  Bước t+2: s_{t+2} → RIGHT → reward = 64  → học Q(s_{t+2}, RIGHT) = 64 + ...
  → Bước t và t+1 không nhận được tín hiệu reward 64

Với 3-step:
  Từ bước t, nhìn 3 bước → G = 0 + 0.99×0 + 0.99²×64 = 62.8
  → Bước t học được: Q(s_t, LEFT) ≈ 62.8 → có giá trị!
```

> **🎯 Ví dụ:**
> Chơi cờ vua: thực hiện 3 nước chuẩn bị để rồi ăn quân đối thủ.
> - 1-step: 3 nước chuẩn bị vô nghĩa (reward = 0)
> - 3-step: 3 nước chuẩn bị đều nhận credit từ việc ăn quân → học giá trị thực

---

## Cải tiến 5 — NoisyNets

### Vấn đề với ε-greedy

```python
# ε-greedy: random hoàn toàn với xác suất epsilon
EPS_START = 1.0
EPS_END   = 0.05
# Giảm tuyến tính → lịch trình cứng nhắc

if random.random() < epsilon:
    action = random.choice(legal_actions)  # random không có học
else:
    action = argmax(q_values)              # exploit hoàn toàn
```

**Nhược điểm:**
- Explore hoàn toàn ngẫu nhiên, không có cấu trúc
- Phải tự thiết kế lịch trình decay (cứng nhắc)
- Khi epsilon nhỏ: không còn explore nữa dù vẫn cần

### Giải pháp: NoisyNets — Noise học được

```python
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        # Mỗi weight có 2 thành phần:
        self.weight_mu    = nn.Parameter(...)  # giá trị trung bình (được học)
        self.weight_sigma = nn.Parameter(...)  # độ nhiễu (được học)

        # Buffer noise — sample lại mỗi forward pass
        self.register_buffer("weight_eps", torch.zeros(...))

    def sample_noise(self):
        # Sample noise mới dùng factorized Gaussian
        eps_in  = self._factorized_noise(self.in_features)
        eps_out = self._factorized_noise(self.out_features)
        self.weight_eps.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))

    def forward(self, x):
        if self.training:
            # Weight = mean + sigma × noise
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            # Evaluate: dùng mean weights (deterministic, không noise)
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)
```

**Tại sao sigma là tham số được học?**

```
σ lớn → noise lớn → output Q-values bất định → agent thử nhiều action khác nhau = EXPLORE
σ nhỏ → noise nhỏ → output Q-values ổn định → agent dùng kiến thức đã học = EXPLOIT

Agent tự học: khi nào nên explore (σ↑), khi nào nên exploit (σ↓)
```

| | ε-greedy | NoisyNets |
|---|---|---|
| **Cách explore** | Chọn ngẫu nhiên hoàn toàn | Noise có cấu trúc trong weights |
| **Lịch trình** | Cứng nhắc (linear decay) | Tự động (σ được học) |
| **Theo ngữ cảnh?** | Không — epsilon giống nhau mọi state | Có — σ khác nhau mỗi state |
| **Khi eval** | Tắt exploration (epsilon thấp) | Tắt noise (`.eval()`) |

> **🎯 Ví dụ:**
> - ε-greedy: Học nấu ăn, cứ 5% lần thì nếm ngẫu nhiên mọi thứ (không quan tâm tình huống)
> - NoisyNets: Khi gặp nguyên liệu lạ → tự nhiên thử nghiệm nhiều hơn. Nguyên liệu quen → nấu đúng công thức.

---

## Cải tiến 6 — Soft Target Update

### Hard Update (DQN gốc):

```python
TARGET_SYNC_EVERY = 250

# Mỗi 250 steps: copy toàn bộ
if step % TARGET_SYNC_EVERY == 0:
    target_net.load_state_dict(q_net.state_dict())

# Timeline:
# Steps 1-249:   target = phiên bản 0 (cũ)
# Step 250:      target = phiên bản 250 ← nhảy đột ngột
# Steps 251-499: target = phiên bản 250 (đứng yên)
# Step 500:      target = phiên bản 500 ← nhảy đột ngột lại
```

### Soft Update (Rainbow V1/V2):

```python
TAU = 0.005                # 0.5%
TARGET_UPDATE_EVERY = 4    # update mỗi 4 steps

# Mỗi 4 steps:
for p_target, p_q in zip(target_net.parameters(), q_net.parameters()):
    p_target.data = (1 - TAU) * p_target.data + TAU * p_q.data
    #             = 0.995 × target_cũ + 0.005 × q_net_mới

# Timeline:
# Step 4:   target = 0.995^1 × target₀ + ...q₄
# Step 8:   target = 0.995^2 × target₀ + ...q₈
# ...       (trôi dần, không bao giờ nhảy đột ngột)
```

**Sau bao nhiêu steps target_net "theo kịp" q_net?**

```
τ = 0.005, sau n steps:
  influence của q_net ban đầu giảm còn = (1 - 0.005)^n = 0.995^n

  n = 100 steps → 0.995^100 ≈ 0.61 (vẫn còn 61% ảnh hưởng từ q_net ban đầu)
  n = 500 steps → 0.995^500 ≈ 0.08 (8% ảnh hưởng)
  n = 1000 steps → 0.995^1000 ≈ 0.007 (gần như theo kịp hoàn toàn)
```

---

## Cải tiến 7 — State Encoding: Raw → Log2 → One-hot

### File gốc (thầy): Raw observation

```python
# OpenSpiel trả về raw tensor: giá trị tile thực
obs = [0, 2, 0, 4, 0, 0, 8, 0, 2, 0, 4, 0, 0, 128, 0, 0]   # shape (16,)
```

Phân phối rất skewed: tile nhỏ (2,4,8) chiếm đa số, tile lớn (512, 1024, 2048) hiếm.

### Rainbow V1: Log2 Normalization

```python
def extract_obs(state, player_id=0):
    raw = extract_obs_raw(state)        # [0, 2, 4, 8, ..., 2048]
    log_obs = np.log2(raw + 1.0) / 11.0  # normalize về [0, 1]
    return log_obs

# Tile 0    → log2(1)/11    = 0.00
# Tile 2    → log2(3)/11    ≈ 0.14
# Tile 4    → log2(5)/11    ≈ 0.21
# Tile 8    → log2(9)/11    ≈ 0.29
# Tile 2048 → log2(2049)/11 ≈ 1.00
```

Tốt hơn raw vì normalize về [0,1] → gradient ổn định.

### Rainbow V2: One-hot Encoding

```python
TILE_VALUES    = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, ...]
TILE_TO_IDX    = {0:0, 2:1, 4:2, 8:3, 16:4, ...}
OBS_DIM_ONEHOT = 16 × 16 = 256   # 16 ô × 16 possible tile values

def extract_obs(state, player_id=0):
    raw = extract_obs_raw(state)               # (16,) raw tiles
    onehot = np.zeros((16, 16), dtype=np.float32)
    for i, tile_val in enumerate(raw):
        idx = TILE_TO_IDX[int(tile_val)]       # tile 4 → index 2
        onehot[i, idx] = 1.0                   # ô thứ i đang chứa tile 4
    return onehot.reshape(-1)                  # (256,)

# Ô chứa tile 4:
# Raw:     4
# Log2:    0.21
# One-hot: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#                ↑ index 2 = tile 4
```

**So sánh 3 cách:**

```
Tile 2  vs  Tile 4:
  Raw:     2        vs  4        → Chênh lệch = 2 (tùy scale)
  Log2:    0.14     vs  0.21     → Chênh lệch = 0.07 (khó phân biệt)
  One-hot: [0,1,0...] vs [0,0,1...]  → Hoàn toàn khác nhau, rõ ràng

Tile 2  vs  Tile 2048:
  Raw:     2        vs  2048     → Chênh lệch = 2046 (quá lớn)
  Log2:    0.14     vs  1.00     → Chênh lệch = 0.86 (tốt hơn)
  One-hot: [0,1,...,0] vs [0,...,0,1]  → Cũng rõ ràng
```

**Nhược điểm của One-hot:** Input tăng từ 16 → 256 → mạng cần hidden_dim 512 (thay vì 256) → ~800k parameters (so với ~200k).

---

## Cải tiến 8 — Reward Shaping (chỉ có V2)

### Vấn đề: Sparse reward

```python
# Reward duy nhất từ game = điểm merge (rất thưa)
reward = new_return - prev_return   # hầu hết = 0, thỉnh thoảng = 4/8/64/2048...
```

Agent không biết rằng "đặt tile lớn ở góc" hay "giữ nhiều ô trống" là chiến lược tốt.

### Giải pháp: Reward Shaping

```python
def shaped_reward(board, raw_reward):
    """Thêm bonus để dạy agent chiến lược 2048."""
    max_tile = board.max()

    # 1. Corner bonus: tile lớn nhất ở góc → thưởng nhiều
    corners = [board[0,0], board[0,3], board[3,0], board[3,3]]
    corner_bonus = max_tile * 0.1 if max_tile in corners else 0.0
    # Nếu max tile = 128, ở góc → +12.8

    # 2. Empty bonus: nhiều ô trống = dễ xoay sở
    empty_cells = int(np.sum(board == 0))
    empty_bonus = empty_cells * 2.0
    # 5 ô trống → +10.0

    # 3. Monotonicity: tiles xếp theo thứ tự giảm dần (chiến lược "snake")
    mono_score = 0.0
    for row in board:
        for i in range(3):
            if row[i] >= row[i+1]:
                mono_score += row[i] * 0.01
    # Hàng [128, 64, 32, 16] → mono_score += (128+64+32)*0.01 = 2.24

    return raw_reward + corner_bonus + empty_bonus + mono_score
```

**Ví dụ cụ thể:**

```
Bàn cờ:
  128  64  32  16
    0   8   4   2
    0   0   0   0
    0   0   0   0

raw_reward = 0  (không có merge bước này)

corner_bonus = 128 × 0.1 = 12.8  (tile 128 ở góc trên-trái!)
empty_bonus  = 8 × 2.0   = 16.0  (8 ô trống)
mono_score   ≈ (128+64+32) × 0.01 × ..rows.. ≈ vài đơn vị

total shaped_reward = 0 + 12.8 + 16.0 + ... ≈ 29+

→ Agent học được: bố cục này tốt dù không có merge
```

Sau đó còn áp dụng `reward_transform` (EfficientZero):

```python
def reward_transform(x):
    """Compress reward lớn để tránh gradient exploding."""
    eps = 0.001
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + eps * x

# Ví dụ:
# reward = 4      → transform ≈ 1.24
# reward = 64     → transform ≈ 7.06
# reward = 2048   → transform ≈ 44.3   (so với 2048 raw)
```

> **⚠️ Cảnh báo:** Reward Shaping cần cẩn thận:
> Nếu bonus quá lớn, agent có thể "hack" reward thay vì chơi thật (ví dụ: đứng im để giữ empty_bonus thay vì merge).

---

## ⚙️ Hyperparameters đầy đủ — So sánh 3 file

| Tham số | DQN gốc | Rainbow V1 | Rainbow V2 |
|---|---|---|---|
| `NUM_EPISODES` | 2,000 | 500 | **5,000** |
| `BUFFER_SIZE` | 50,000 | 100,000 | 100,000 |
| `BATCH_SIZE` | 128 | 128 | 128 |
| `GAMMA` | 0.99 | 0.99 | 0.99 |
| `LR` | **1e-3** | **5e-4** | 5e-4 |
| `TARGET_SYNC` | Hard, 250 steps | Soft τ=0.005 | Soft τ=0.005 |
| `LEARN_START` | 1,000 | 2,000 | 2,000 |
| `LEARN_EVERY` | 4 | 4 | 4 |
| `N_STEPS` | 1 | **3** | 3 |
| `PER_ALPHA` | — | 0.6 | 0.6 |
| `PER_BETA_START` | — | 0.4 | 0.4 |
| `Hidden dim` | 256 | 256 | **512** |
| `Obs dim` | 16 | 16 | **256** |

**Tại sao V1 dùng LR = 5e-4 (nhỏ hơn)?**
NoisyNets thêm variance vào weights → gradient có thể dao động nhiều hơn → LR nhỏ hơn để ổn định.

**Tại sao V2 cần nhiều episodes hơn (5000)?**
Input lớn hơn (256 dims), mạng nhiều params hơn (~800k) → cần nhiều dữ liệu hơn để train tốt.

---

## 📐 Kiến trúc mạng — So sánh hình vẽ

### Vanilla DQN (thầy):

```
[16] → Linear(16→256) → ReLU → Linear(256→256) → ReLU → Linear(256→4)
                                                              ↓
                                               [Q_UP, Q_DOWN, Q_LEFT, Q_RIGHT]
```

### Rainbow DQN V1:

```
[16]
  ↓ Linear(16→256) + ReLU
[256] ── shared features ──────────────────────────────────────────
        ↓ Value stream                      ↓ Advantage stream
   NoisyLinear(256→256) + ReLU        NoisyLinear(256→256) + ReLU
   NoisyLinear(256→1)                 NoisyLinear(256→4)
        ↓ V(s)                              ↓ A(s, *)
        └──────────── Q = V + A - mean(A) ──┘
                              ↓
               [Q_UP, Q_DOWN, Q_LEFT, Q_RIGHT]
```

### Rainbow DQN V2:

```
[256]  ← One-hot encoded (thay vì [16])
  ↓ Linear(256→512) + ReLU     ← hidden 512 (thay vì 256)
[512] ── shared features ──────────────────────────────────────────
        ↓ Value stream                      ↓ Advantage stream
   NoisyLinear(512→512) + ReLU        NoisyLinear(512→512) + ReLU
   NoisyLinear(512→1)                 NoisyLinear(512→4)
        ↓ V(s)                              ↓ A(s, *)
        └──────────── Q = V + A - mean(A) ──┘
                              ↓
               [Q_UP, Q_DOWN, Q_LEFT, Q_RIGHT]
```

---

## 🔑 Ghi nhớ nhanh (TL;DR)

| Kỹ thuật | Giải quyết vấn đề gì? | Một câu |
|---|---|---|
| **Double DQN** | Overestimate Q-values | Tách mạng chọn action ≠ mạng đánh giá |
| **Dueling Network** | Học không hiệu quả | Tách V(s) "state tốt" và A(s,a) "action tốt hơn" |
| **PER** | Ôn lại cả cái đã học tốt | Sample nhiều hơn những gì đang sai |
| **N-step** | Sparse reward không lan được | Nhìn N bước trước, reward lan xa hơn |
| **NoisyNets** | Explore cứng nhắc, không theo ngữ cảnh | Noise trong weight, sigma tự học |
| **Soft Update** | Target "nhảy cóc" mỗi N steps | Target trôi dần, không đột ngột |
| **One-hot** | Tile values khó phân biệt qua số | Mỗi tile = vector riêng biệt, rõ ràng |
| **Reward Shaping** | Không biết chiến lược tốt | Thưởng thêm cho góc, ô trống, monotonic |
| **Reward Transform** | Tile lớn → reward lớn → gradient bùng | Compress reward qua sqrt |
