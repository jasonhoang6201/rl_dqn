# Phân tích: DQN gốc vs Rainbow DQN V1 vs V2

> **3 files trong commit "update":**
> - `DQN_for_2048_game.ipynb` — File gốc của thầy (Vanilla DQN)
> - `Rainbow_DQN_2048.ipynb` — V1: Rainbow DQN (5/6 kỹ thuật Rainbow)
> - `Rainbow_DQN_2048_V2.ipynb` — V2: V1 + One-hot encoding + Reward Shaping

---

## Sửa lại những chỗ bạn hiểu chưa chính xác

Trước khi vào chi tiết, mình sẽ làm rõ 2 điểm bạn nói không chắc:

### ✅ Target Network để làm gì?

Bạn diễn đạt đúng cơ bản, nhưng mình giải thích rõ hơn lý do **tại sao cần** nó:

Khi train, mình dùng công thức Bellman để tính "đáp án đúng" (target):

```
target = r + γ · max Q(s', a')
```

Vấn đề: nếu dùng **cùng một mạng** để vừa tính `target` vừa học, thì **đáp án đúng tự thay đổi liên tục** mỗi bước train — giống như học toán mà đề bài thay đổi theo từng lần làm bài. Mạng sẽ không ổn định, dao động hoặc phân kỳ (diverge).

**Giải pháp:** Tạo `target_net` là bản sao "đông lạnh" của `q_net`, chỉ update định kỳ (ít hơn) → đáp án đúng ổn định trong nhiều bước → mạng học được ổn định hơn.

> **Ví dụ dễ hiểu:** Coi như bạn đang học bắn cung. Nếu mục tiêu di chuyển liên tục mỗi mũi tên bạn nhắm, bạn sẽ không bao giờ học được. `target_net` giống như giữ mục tiêu cố định một lúc, rồi mới dịch chuyển.

---

### ✅ Soft Update (99% cũ + 1% mới) để làm gì?

Đây gọi là **Polyak averaging** hay **soft update**:

```
target_net ← (1 - τ) · target_net + τ · q_net
```

Với `τ = 0.01` → target cập nhật 1% mỗi bước.

Tại sao tốt hơn hard copy (copy toàn bộ mỗi N steps)?

- **Hard copy:** Target nhảy đột ngột mỗi 250 steps → training bất ổn quanh các bước đó
- **Soft update:** Target trôi dần theo q_net → mượt mà, ổn định hơn, không có "shock"

> **Ví dụ:** Giống như điều chỉnh nhiệt độ phòng. Hard copy = đột ngột bật điều hòa 100%, soft update = tăng dần nhiệt mỗi phút — thoải mái hơn cho cơ thể.

---

## Tổng quan các thay đổi

| Kỹ thuật | Vanilla DQN (thầy) | Rainbow V1 | Rainbow V2 |
|---|---|---|---|
| **Kiến trúc mạng** | 3 Linear layers đơn giản | Dueling Network | Dueling Network (hidden 512) |
| **Exploration** | ε-greedy (random dần giảm) | NoisyNets | NoisyNets |
| **Replay Buffer** | Uniform (random đều) | Prioritized (PER) | Prioritized (PER) |
| **Bellman target** | 1-step | N-step (n=3) | N-step (n=3) |
| **Chọn action tốt nhất** | Vanilla Q → argmax | Double DQN | Double DQN |
| **Target update** | Hard copy (mỗi 250 steps) | Soft update (τ=0.005) | Soft update (τ=0.005) |
| **State encoding** | Raw / flat tensor | log2 normalization | **One-hot (16→256 dims)** |
| **Reward** | Delta cumulative return | reward_transform (sqrt) | Shaped reward + reward_transform |
| **Số episodes** | 2000 | 500 | 5000 |

---

## Chi tiết từng thay đổi

---

### 1. 🏗️ Kiến trúc mạng: QNetwork → Dueling Network

#### Vanilla DQN (thầy):
```python
class QNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_dim=256):
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),  # 16 → 256
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 256 → 256
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),  # 256 → 4 (Q values)
        )
    def forward(self, x):
        return self.net(x)
```

Mạng học trực tiếp `Q(s, a)` cho cả 4 action.

#### Rainbow V1/V2 — Dueling Network:
```python
class RainbowQNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions):
        # Shared feature extractor
        self.feature = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())

        # Value stream: V(s) — "trạng thái này tốt hay xấu?"
        self.value_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.value_out    = NoisyLinear(hidden_dim, 1)           # → 1 số

        # Advantage stream: A(s,a) — "action này tốt hơn trung bình bao nhiêu?"
        self.adv_hidden = NoisyLinear(hidden_dim, hidden_dim)
        self.adv_out    = NoisyLinear(hidden_dim, num_actions)   # → 4 số

    def forward(self, x):
        feat = self.feature(x)
        v = self.value_out(F.relu(self.value_hidden(feat)))    # (B, 1)
        a = self.adv_out(F.relu(self.adv_hidden(feat)))        # (B, 4)
        # Combine: Q = V + A - mean(A)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
```

**Tại sao Dueling tốt hơn?**

Trong 2048, nhiều trạng thái `s` mà **bất kỳ action nào cũng tốt như nhau** (ví dụ bàn cờ đang rất dễ). Khi đó mạng cần biết "trạng thái này tốt" hơn là "action nào cụ thể tốt hơn".

```
Q(s, a) = V(s) + A(s, a) - mean(A)
           ^         ^
           |         |
        "s tốt"    "a hơn tgt bình quân bao nhiêu"
```

> **Ví dụ cụ thể:**
> - Bàn cờ rất xấu (gần game over) → `V(s) = -10`, mọi action đều tệ
> - Bàn cờ rất tốt → `V(s) = +5`, dù action nào cũng ổn
> - Vanilla DQN phải học lại từ đầu mỗi lần → chậm hơn
> - Dueling tách biệt được 2 khái niệm → học hiệu quả hơn

**V2 khác V1:** hidden_dim tăng từ 256 → 512 vì input tăng từ 16 → 256 chiều (one-hot).

---

### 2. 🎲 Exploration: ε-greedy → NoisyNets

#### Vanilla DQN (thầy):
```python
# Action selection
if random.random() < epsilon:
    action = random.choice(legal_actions)  # random
else:
    action = argmax(q_values)              # exploit

# Epsilon decay
EPS_START = 1.0
EPS_END   = 0.05
# Giảm tuyến tính từ 1.0 → 0.05 trong 20k steps
```

#### Rainbow V1/V2 — NoisyNets:
```python
class NoisyLinear(nn.Module):
    def forward(self, x):
        if self.training:
            # Weight = mean + sigma * noise (noise ~ N(0,1))
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            w = self.weight_mu  # deterministic khi evaluate
            b = self.bias_mu
        return F.linear(x, w, b)
```

**Tại sao NoisyNets tốt hơn ε-greedy?**

ε-greedy explore **hoàn toàn ngẫu nhiên** — không biết gì từ trải nghiệm trước. NoisyNets thêm noise trực tiếp vào **weight của mạng**, nên noise có cấu trúc — agent explore theo cách "có học".

Thêm nữa, `σ` (sigma) là **tham số được học** — agent tự biết khi nào cần explore nhiều (σ lớn) và khi nào cần khai thác (σ nhỏ), thay vì dùng lịch trình cứng nhắc (linear decay).

> **Ví dụ dễ hiểu:**
> - ε-greedy: Bạn học đánh cờ, cứ 5% cơ hội thì đi ngẫu nhiên bất kể tình huống
> - NoisyNets: Khi bạn ở tình huống mới lạ thì đi sáng tạo, tình huống quen thì đi theo kinh nghiệm

---

### 3. 🎯 Replay Buffer: Uniform → Prioritized Experience Replay (PER)

#### Vanilla DQN (thầy):
```python
class ReplayBuffer:
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # random đều
```

Mọi transition đều có xác suất được chọn như nhau.

#### Rainbow V1/V2 — PER với SumTree:
```python
class SumTree:
    """
    Cấu trúc cây nhị phân để sample O(log N).
    Mỗi lá = priority của một transition.
    """

class PrioritizedReplayBuffer:
    def add(self, ..., priority=None):
        # Mặc định priority = TD-error lớn → ưu tiên cao hơn

    def sample(self, batch_size, beta):
        # Sample theo priority thay vì uniform
        # Trả về IS weights để bù lại bias
```

**Tại sao PER tốt hơn?**

Không phải mọi kinh nghiệm đều quan trọng như nhau. Những transition có **TD-error lớn** = agent đang dự đoán sai nhiều nhất = cần học nhiều nhất. PER ưu tiên sample những transition đó.

```
TD-error = |r + γ·V(s') - Q(s,a)|
            ^
          Sai many  → priority cao → được sample thường hơn
```

**SumTree** là cấu trúc dữ liệu giúp tìm kiếm theo priority nhanh: O(log N) thay vì O(N).

**IS weights (Importance Sampling)** được dùng để bù lại bias: vì mình sample không đều, những transition ít được sample cần được "nhân hệ số" hơn khi tính loss.

> **Ví dụ:** Học tiếng Anh, uniform = ôn ngẫu nhiên tất cả từ, PER = tập trung ôn những từ bạn hay viết sai nhất → học nhanh hơn nhiều.

---

### 4. ⏳ N-step Return (N=3)

#### Vanilla DQN (thầy) — 1-step:
```
target = r_t + γ · max Q(s_{t+1}, a)
```

Chỉ nhìn 1 bước tiếp theo.

#### Rainbow V1/V2 — 3-step:
```
target = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³ · max Q(s_{t+3}, a)
```

Nhìn 3 bước tiếp theo, tích lũy reward.

**Tại sao N-step tốt hơn với 2048?**

Trong 2048, reward chỉ xuất hiện khi merge ô — rất **sparse** (thưa thớt). Sau nhiều bước di chuyển "chuẩn bị", mới có 1 bước merge. Với 1-step, agent không biết những bước chuẩn bị đó có giá trị.

Với 3-step, tín hiệu reward từ bước merge "lan ngược" về 3 bước trước đó → agent học được rằng "những nước di chuyển này cũng có giá trị vì dẫn đến merge".

> **Ví dụ:**
> - Bước 1: di chuyển trái (reward = 0)
> - Bước 2: di chuyển xuống (reward = 0)
> - Bước 3: di chuyển phải → merge 2 ô → (reward = 64)
>
> 1-step: bước 1, 2 học được reward = 0 → chúng vô nghĩa
> 3-step: bước 1 nhìn được reward = 0 + 0 + 64 = 64 → học được giá trị thực

---

### 5. 🎰 Double DQN

#### Vanilla DQN (thầy) — overestimation problem:
```python
# Dùng target_net để chọn action VÀ đánh giá Q-value
# → cùng một mạng làm cả 2 việc → overestimate
next_q = target_net(next_obs).max(dim=1).values
target = reward + gamma * next_q * (1 - done)
```

#### Rainbow V1/V2 — Double DQN:
```python
# q_net chọn action tốt nhất ở s'
next_actions = q_net(next_obs_t).argmax(dim=1, keepdim=True)

# target_net ĐÁNH GIÁ Q-value của action đó
next_q = target_net(next_obs_t).gather(1, next_actions).squeeze(1)

target = reward + gamma * next_q * (1 - done)
```

**Tại sao Double DQN tốt hơn?**

Khi dùng cùng 1 mạng để vừa chọn action vừa đánh giá, nó có xu hướng **overestimate** (đánh giá quá cao) Q-values. Vì nó luôn chọn action có Q cao nhất, nhưng Q cao nhất hay có noise cao nhất chứ không phải thực sự tốt nhất.

Tách ra: `q_net` chọn, `target_net` đánh giá → giảm overestimation đáng kể.

> **Ví dụ:**
> - 1 giám khảo vừa chấm điểm vừa tự chọn bài để chấm → thiên vị
> - Giám khảo A chọn bài, giám khảo B chấm → khách quan hơn

---

### 6. 🔄 Target Update: Hard copy → Soft Update

#### Vanilla DQN (thầy) — Hard copy mỗi 250 steps:
```python
TARGET_SYNC_EVERY = 250
# Mỗi 250 steps: target_net = q_net (copy toàn bộ)
if step % TARGET_SYNC_EVERY == 0:
    target_net.load_state_dict(q_net.state_dict())
```

#### Rainbow V1/V2 — Soft update mỗi 4 steps:
```python
TAU = 0.005
# Mỗi 4 steps: trộn dần
for p_target, p_q in zip(target_net.parameters(), q_net.parameters()):
    p_target.data.copy_(
        (1 - TAU) * p_target.data + TAU * p_q.data
    )
# target = 0.995 × target_cũ + 0.005 × q_net_mới
```

**Tại sao Soft update tốt hơn?**

Hard copy tạo ra "shock" đột ngột mỗi 250 steps — target nhảy từ giá trị cũ sang mới hoàn toàn, có thể làm training mất ổn định quanh các bước đó.

Soft update với `τ = 0.005` → target thay đổi rất chậm và mượt mà → learning ổn định hơn.

Đây là kỹ thuật từ **TD3** và **DDPG** trong continuous RL, được áp dụng vào đây.

---

### 7. 🔢 State Encoding: Raw → Log2 (V1) → One-hot (V2)

#### Vanilla DQN (thầy) — Raw tensor:
```python
# OpenSpiel trả về raw observation (có thể là raw tile values: 0, 2, 4, ..., 2048)
obs = state.observation_tensor()
# obs = [0, 2, 0, 4, ..., 2048]  ← shape (16,)
```

#### Rainbow V1 — Log2 normalization:
```python
def extract_obs(state, player_id=0):
    raw = extract_obs_raw(state, player_id)   # [0, 2, 4, ..., 2048]
    log_obs = np.log2(raw + 1.0) / 11.0      # normalize về [0, 1]
    # Tile 2 → log2(3)/11 ≈ 0.14
    # Tile 2048 → log2(2049)/11 ≈ 1.0
    return log_obs  # shape (16,)
```

**Tại sao log2 tốt hơn raw?** Tile values có phân phối rất skewed: 2, 4, 8, ..., 2048 cách nhau theo cấp số nhân. Nếu để raw, gradient sẽ bị dominated bởi tile 2048, còn tile 2 quá nhỏ để học được.

#### Rainbow V2 — One-hot encoding:
```python
TILE_VALUES   = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, ...]
OBS_DIM_ONEHOT = 16 × 16 = 256  # 16 ô, mỗi ô 16 possible values

def extract_obs(state, player_id=0):
    raw = extract_obs_raw(state, player_id)        # (16,) raw tiles
    onehot = np.zeros((16, 16), dtype=np.float32)
    for i, val in enumerate(raw):
        idx = TILE_TO_IDX[int(val)]                # tile 4 → index 2
        onehot[i, idx] = 1.0                       # ô i đang chứa tile 4
    return onehot.reshape(-1)                      # (256,)
```

**So sánh:**

| Encoding | Tile 2 | Tile 4 | Phân biệt? |
|---|---|---|---|
| Raw | 2 | 4 | Được, nhưng scale lớn |
| Log2 | 0.14 | 0.18 | Khó phân biệt, sai số nhỏ |
| One-hot | `[0,1,0,...,0]` | `[0,0,1,...,0]` | **Rõ ràng nhất** |

One-hot giúp mạng phân biệt tile values **tuyệt đối** không phụ thuộc scale. Nhưng đổi lại, input size tăng từ 16 → 256 → mạng cần nhiều parameters hơn và train lâu hơn.

---

### 8. 🎁 Reward Shaping (chỉ có ở V2)

#### V1/thầy — chỉ dùng delta cumulative return:
```python
reward = new_return - prev_return  # điểm merge được bao nhiêu
```

#### V2 — Shaped reward (dạy chiến lược):
```python
def shaped_reward(board, raw_reward):
    max_tile = board.max()

    # 1. Tile lớn nhất ở góc → thưởng thêm
    corners = [board[0,0], board[0,3], board[3,0], board[3,3]]
    corner_bonus = max_tile * 0.1 if max_tile in corners else 0.0

    # 2. Nhiều ô trống → thưởng nhỏ
    empty_bonus = np.sum(board == 0) * 2.0

    # 3. Tiles xếp thứ tự giảm dần (monotonicity) → thưởng
    mono_score = 0.0
    for row in board:
        for i in range(3):
            if row[i] >= row[i+1]:
                mono_score += row[i] * 0.01

    return raw_reward + corner_bonus + empty_bonus + mono_score
```

**Tại sao Reward Shaping có thể tốt trong 2048?**

Các chiến lược tốt trong 2048 (đặt tile lớn ở góc, giữ nhiều ô trống, xếp tiles theo thứ tự) không được thưởng trực tiếp bởi game. Reward Shaping inject domain knowledge vào — nói với agent "làm thế này thì tốt đó".

**⚠️ Nhưng coi chừng:** Reward Shaping cần thiết kế cẩn thận. Nếu bonus không cân bằng, agent có thể học "hack" reward thay vì học chơi game thật sự (ví dụ: đứng im để duy trì empty bonus thay vì merge).

> **Ví dụ thực tế:** Học sinh chỉ được thưởng khi trả lời đúng cuối kỳ (sparse reward = merge reward). Reward shaping = thưởng thêm khi ngồi học ngoan, làm bài tập, giơ tay phát biểu — dù chưa đến cuối kỳ.

---

### 9. 🔃 Reward Transform (V1 và V2 đều có)

```python
def reward_transform(x):
    """EfficientZero-style reward scaling"""
    eps = 0.001
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + eps * x
```

**Tại sao cần scale reward?**

Reward trong 2048 có thể chênh lệch rất lớn: merge 2+2=4 (reward=4) vs merge 1024+1024=2048 (reward=2048). Nếu để nguyên, gradient khi merge tile lớn sẽ rất lớn → **gradient exploding** → mạng mất ổn định.

Hàm `h(x) = sign(x)·(√(|x|+1) - 1) + ε·x` compress reward lớn:
- `h(4) ≈ 1.24`
- `h(2048) ≈ 44` (thay vì 2048)

→ Gradient ổn định hơn nhiều.

---

## Tóm tắt thứ tự cải tiến "từ thấp đến cao"

```
Vanilla DQN (thầy)
    ↓ thêm Double DQN  → giảm overestimation
    ↓ thêm Dueling     → tách V(s) và A(s,a), học hiệu quả hơn
    ↓ thêm PER         → học nhiều hơn từ những gì quan trọng
    ↓ thêm N-step      → reward lan xa hơn, giải quyết sparse reward
    ↓ thêm NoisyNets   → explore thông minh hơn ε-greedy
    ↓ thêm Soft update → ổn định hơn hard copy
    = Rainbow DQN V1

Rainbow DQN V2
    = V1
    + One-hot encoding  → biểu diễn state rõ ràng hơn (16→256 dims, hidden 256→512)
    + Reward Shaping    → inject chiến lược 2048 vào reward
    + Nhiều hơn 10x episodes (500 → 5000)
```

---

## Bảng Rainbow "6 kỹ thuật" — cái nào được implement?

Rainbow (2017, DeepMind) kết hợp 6 cải tiến của DQN. V1/V2 implement **5/6**:

| # | Kỹ thuật | V1 | V2 | Ghi chú |
|---|---|---|---|---|
| 1 | **Double DQN** | ✅ | ✅ | Tách chọn/đánh giá action |
| 2 | **Dueling Network** | ✅ | ✅ | V(s) + A(s,a) |
| 3 | **PER** | ✅ | ✅ | Priority theo TD-error |
| 4 | **N-step Returns** | ✅ (n=3) | ✅ (n=3) | 3-step Bellman |
| 5 | **NoisyNets** | ✅ | ✅ | Thay ε-greedy |
| 6 | **Distributional RL (C51)** | ❌ | ❌ | Học phân phối Q thay vì mean — phức tạp nhất |
