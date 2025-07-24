import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# --- Config ---
torch.manual_seed(42)
dtype = torch.float64
lr = 0.01
PRINT_PRECISION = 6
torch.set_printoptions(precision=PRINT_PRECISION, sci_mode=False)

# --- Load and prepare real dataset (1 feature regression) ---
data = load_diabetes()
X = data.data[:, [2]]  # Use only 1 feature (e.g., BMI)
y = data.target.reshape(-1, 1)

# Normalize
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)

X = torch.tensor(X, dtype=dtype)
y = torch.tensor(y, dtype=dtype)

X_train = X[:44]
Y_train = y[:44]

x45 = X[44:45]
y45 = y[44:45]

# --- Define Model ---
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(3, 1, dtype=dtype))
        self.W2 = nn.Parameter(torch.randn(3, 3, dtype=dtype))
        self.W3 = nn.Parameter(torch.randn(1, 3, dtype=dtype))

    def forward(self, x):
        x = x.view(1, 1)
        h1 = F.softplus(self.W1 @ x)
        h2 = F.softplus(self.W2 @ h1)
        out = self.W3 @ h2
        return out.view(-1)

def loss_fn(y_pred, y_true):
    return 0.5 * (y_pred - y_true)**2

# --- Step 1: Train on first 44 examples ---
net = SimpleNet()
opt = torch.optim.SGD(net.parameters(), lr=lr)

for i in range(44):
    opt.zero_grad()
    y_pred = net(X_train[i])
    loss = loss_fn(y_pred, Y_train[i])
    loss.backward()
    opt.step()

# Save weights after 44th example
W1_44 = net.W1.clone().detach()
W2_44 = net.W2.clone().detach()
W3_44 = net.W3.clone().detach()

print("\n✅ Weights after 44 examples:")
print("W1_44:\n", W1_44)
print("W2_44:\n", W2_44)
print("W3_44:\n", W3_44)

# --- Step 2: Train on 45th example ---
opt.zero_grad()
y_pred_45 = net(x45)
loss_45 = loss_fn(y_pred_45, y45)
loss_45.backward()
opt.step()

W1_45 = net.W1.clone().detach()
W2_45 = net.W2.clone().detach()
W3_45 = net.W3.clone().detach()

print("\n📈 Weights after training on 45th example:")
print("W1_45:\n", W1_45)
print("W2_45:\n", W2_45)
print("W3_45:\n", W3_45)

# --- Step 3: Construct anti-example (x46, y46) ---
x46 = torch.tensor([[0.0]], requires_grad=True, dtype=dtype)
y46 = torch.tensor([0.0], requires_grad=True, dtype=dtype)
unroll_net = SimpleNet()

# Copy W45 weights
with torch.no_grad():
    unroll_net.W1.copy_(W1_45)
    unroll_net.W2.copy_(W2_45)
    unroll_net.W3.copy_(W3_45)

meta_opt = torch.optim.Adam([x46, y46], lr=0.01)

for step in range(500):
    meta_opt.zero_grad()

    # Clone weights for simulation
    W1_sim = unroll_net.W1.clone()
    W2_sim = unroll_net.W2.clone()
    W3_sim = unroll_net.W3.clone()

    # Forward pass and loss
    y_pred = unroll_net(x46)
    loss = loss_fn(y_pred, y46)

    # Get gradients
    grads = torch.autograd.grad(loss, [unroll_net.W1, unroll_net.W2, unroll_net.W3],
                                create_graph=True)

    # Simulate one SGD step
    W1_new = unroll_net.W1 - lr * grads[0]
    W2_new = unroll_net.W2 - lr * grads[1]
    W3_new = unroll_net.W3 - lr * grads[2]

    # Meta-loss: distance to W1_44, W2_44, W3_44
    meta_loss = ((W1_new - W1_44)**2).sum() + \
                ((W2_new - W2_44)**2).sum() + \
                ((W3_new - W3_44)**2).sum()

    meta_loss.backward()
    meta_opt.step()

    if step % 50 == 0:
        print(f"Step {step} | Meta Loss: {meta_loss.item():.8f} | x46: {x46.item():.6f}, y46: {y46.item():.6f}")

print("\n🎯 Constructed anti-example:")
print(f"x46 = {x46.item():.6f}, y46 = {y46.item():.6f}")

# --- Step 4: Train on anti-example (46th) ---
opt.zero_grad()
y_pred_46 = net(x46)
loss_46 = loss_fn(y_pred_46, y46)
loss_46.backward()
opt.step()

print("\n🔁 Weights after training on anti-example (46th):")
print("W1_final:\n", net.W1.detach())
print("W2_final:\n", net.W2.detach())
print("W3_final:\n", net.W3.detach())

# Compare final weights to W1_44
print("\n📏 Weight recovery error after untraining:")
print("W1 diff:", ((net.W1.detach() - W1_44)**2).sum().item())
print("W2 diff:", ((net.W2.detach() - W2_44)**2).sum().item())
print("W3 diff:", ((net.W3.detach() - W3_44)**2).sum().item())
