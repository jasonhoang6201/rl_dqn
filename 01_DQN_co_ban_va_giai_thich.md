# 📚 DQN Cơ Bản — Giải Thích & Đính Chính

> Tài liệu này giải thích cách DQN hoạt động trong project 2048,
> đồng thời làm rõ những khái niệm bạn chưa chắc chắn.

---

## ✅ Những gì bạn hiểu đúng

Bạn đã nắm đúng flow cơ bản của DQN:

```
Bàn cờ 2048 (4×4)
    ↓ flatten
Vector 16 số [0, 2, 0, 4, 0, 0, 8, ...]
    ↓ đưa vào Q-Network
Q-values cho 4 action [UP=1.2, DOWN=0.3, LEFT=2.1, RIGHT=0.8]
    ↓ chọn action có Q cao nhất (hoặc random khi explore)
Thực hiện action → nhận reward + trạng thái mới
    ↓ lưu vào Replay Buffer
Sau N steps → sample batch → update Q-Network
```

Mọi bước trên bạn đều hiểu đúng. Còn 2 điểm bạn chưa rõ:

---

## ❓ Điểm 1: Target Network để làm gì?

### Vấn đề nếu không có Target Network

Khi train, mình cần một "đáp án đúng" (gọi là **target**) để so sánh:

```
target = r + γ · max Q(s', a')
loss   = (Q(s, a) - target)²
```

Nếu dùng **cùng một mạng** để tính cả `Q(s,a)` lẫn `target`:

- Mỗi lần update weights → `target` cũng thay đổi theo
- Như học toán mà đề bài tự thay đổi sau mỗi lần làm bài
- Mạng dao động, không hội tụ được

### Giải pháp: Target Network

```
q_net      → đang được train, weights thay đổi liên tục
target_net → bản sao "đông lạnh" của q_net, ít được update hơn
```

```python
# Tính loss dùng target_net để tính phần "đáp án đúng"
next_q_values = target_net(next_states)          # ← dùng target_net, ổn định
target = reward + gamma * next_q_values.max()

# Q hiện tại vẫn dùng q_net
current_q = q_net(states)

# So sánh → tính loss → update q_net (không update target_net)
loss = MSE(current_q, target)
optimizer.step()   # chỉ q_net thay đổi
```

> **🎯 Ví dụ thực tế:**
> Bạn đang học bắn cung.
> - **Không có target_net:** Mục tiêu di chuyển mỗi mũi tên bạn nhắm → không bao giờ học được
> - **Có target_net:** Mục tiêu đứng yên một lúc (250 steps), bạn tập bắn → giỏi dần → mục tiêu mới được đặt ra

---

## ❓ Điểm 2: Soft Update (99% cũ + 1% mới) để làm gì?

Đây là 2 cách update target_net:

### Cách 1: Hard Copy (file gốc của thầy)

```python
TARGET_SYNC_EVERY = 250  # mỗi 250 steps

if step % TARGET_SYNC_EVERY == 0:
    target_net.load_state_dict(q_net.state_dict())  # copy toàn bộ
```

```
Step 249: target_net = phiên bản cũ
Step 250: target_net = phiên bản mới hoàn toàn  ← "shock" đột ngột
Step 251: target_net = phiên bản mới (đứng yên)
...
Step 499: target_net = phiên bản cũ (từ step 250)
Step 500: target_net = phiên bản mới hoàn toàn  ← "shock" lại
```

→ Target bị "nhảy cóc" mỗi 250 steps → training có thể mất ổn định quanh các bước đó.

### Cách 2: Soft Update — Polyak Averaging (Rainbow V1/V2)

```python
TAU = 0.005  # 0.5%

for p_target, p_q in zip(target_net.parameters(), q_net.parameters()):
    p_target.data = (1 - TAU) * p_target.data + TAU * p_q.data
    #              = 0.995 × target_cũ  +  0.005 × q_net_mới
```

```
Step 1:   target = 0.995 × target₀ + 0.005 × q₁
Step 2:   target = 0.995 × target₁ + 0.005 × q₂
Step 3:   target = 0.995 × target₂ + 0.005 × q₃
...       (target trôi dần theo q_net, không bao giờ nhảy đột ngột)
```

> **🌡️ Ví dụ:**
> Điều chỉnh nhiệt độ phòng từ 30°C → 20°C:
> - **Hard copy:** Đột ngột bật điều hòa hết công suất → sốc nhiệt
> - **Soft update:** Giảm 0.5°C mỗi phút → dễ chịu, ổn định

---

## 🔁 Flow đầy đủ của DQN (file gốc thầy)

```
┌─────────────────────────────────────────────────────┐
│                   TRAINING LOOP                      │
│                                                      │
│  Bàn cờ s → q_net → Q(s,·) → chọn action a         │
│                ↓                                     │
│       epsilon-greedy (1.0→0.05)                      │
│                ↓                                     │
│  Thực hiện a → nhận r, s'                            │
│                ↓                                     │
│    Lưu (s, a, r, s') vào Replay Buffer               │
│                ↓ (mỗi 4 steps)                       │
│    Sample batch ngẫu nhiên từ Replay Buffer          │
│                ↓                                     │
│  target = r + γ · target_net(s').max()               │
│  loss   = MSE(q_net(s, a), target)                   │
│                ↓                                     │
│    Backprop → update q_net                           │
│                ↓ (mỗi 250 steps)                     │
│    target_net ← copy q_net  ← Hard Update           │
└─────────────────────────────────────────────────────┘
```

---

## 📋 Hyperparameters của file gốc và ý nghĩa

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `NUM_EPISODES` | 2000 | Số ván game để train |
| `BUFFER_SIZE` | 50,000 | Replay buffer chứa tối đa 50k transitions |
| `BATCH_SIZE` | 128 | Mỗi lần update, lấy 128 transitions |
| `GAMMA` | 0.99 | Hệ số discount — reward tương lai giá trị ~99% reward hiện tại |
| `LR` | 1e-3 | Learning rate của optimizer Adam |
| `TARGET_SYNC_EVERY` | 250 | Cứ 250 steps thì copy q_net → target_net |
| `LEARN_START` | 1,000 | Chờ 1000 steps đủ dữ liệu mới bắt đầu train |
| `LEARN_EVERY` | 4 | Update mạng mỗi 4 steps (không phải mỗi step) |
| `EPS_START` | 1.0 | Epsilon ban đầu = explore 100% random |
| `EPS_END` | 0.05 | Epsilon cuối = 5% random, 95% dùng mạng |
| `EPS_DECAY_STEPS` | 20,000 | Giảm epsilon từ 1.0 → 0.05 trong 20k steps |
| `GRAD_CLIP` | 10.0 | Giới hạn gradient tối đa — tránh gradient exploding |

### GAMMA = 0.99 có nghĩa gì?

```
reward bây giờ     = 1.0
reward sau 1 bước  = 0.99
reward sau 2 bước  = 0.99² = 0.98
reward sau 10 bước = 0.99¹⁰ = 0.904
reward sau 100 bước = 0.99¹⁰⁰ = 0.366
```

Agent coi trọng reward gần hơn, nhưng vẫn quan tâm reward xa (không bỏ qua hoàn toàn).

### Epsilon-greedy là gì?

```python
if random.random() < epsilon:
    action = random.choice(legal_actions)   # EXPLORE: thử ngẫu nhiên
else:
    action = argmax(q_values)               # EXPLOIT: dùng kiến thức đã học
```

Đây là sự cân bằng **explore vs exploit**:
- Lúc đầu (epsilon=1.0): thử hết mọi thứ để khám phá
- Về sau (epsilon=0.05): phần lớn dùng kiến thức đã học, thỉnh thoảng vẫn thử mới

---

## 🏗️ Kiến trúc Q-Network (file gốc)

```
Input: [16 số] — bàn cờ được flatten
    ↓ Linear(16 → 256) + ReLU
    ↓ Linear(256 → 256) + ReLU
    ↓ Linear(256 → 4)
Output: [Q_UP, Q_DOWN, Q_LEFT, Q_RIGHT]
```

```python
class QNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_dim=256):
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),    # 16 → 256
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # 256 → 256
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions), # 256 → 4
        )
    def forward(self, x):
        return self.net(x)  # ra Q-values cho 4 action
```

Mạng đơn giản, học trực tiếp `Q(s, a)` — không tách biệt "trạng thái tốt" vs "action tốt".

---

## 📖 Glossary — Từ Điển Thuật Ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| **Q-value / Q(s,a)** | Điểm dự đoán: "nếu ở trạng thái s, chọn action a, thì tổng reward tương lai là bao nhiêu?" |
| **Bellman equation** | Công thức cập nhật: Q(s,a) = r + γ·max Q(s',a') |
| **Replay Buffer** | Bộ nhớ lưu các trải nghiệm (s, a, r, s') để train lại sau |
| **Batch sampling** | Lấy ngẫu nhiên một nhóm transitions từ buffer để train |
| **Gradient** | Hướng và độ lớn cần thay đổi weights để giảm loss |
| **Backprop** | Lan truyền ngược — tính gradient từ loss về từng layer |
| **Epsilon-greedy** | Chiến lược explore với xác suất ε, exploit với xác suất (1-ε) |
| **Target network** | Bản sao ổn định của q_net, dùng để tính "đáp án đúng" |
| **Hard update** | Copy toàn bộ target_net ← q_net sau N steps |
| **Soft update** | Trộn dần: target ← (1-τ)·target + τ·q_net mỗi step |
| **Legal action masking** | Che các action không hợp lệ (Q = -∞) để không bao giờ chọn chúng |
| **Sparse reward** | Reward hiếm xuất hiện — như 2048, chỉ có reward khi merge ô |
