import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix
import os

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('figures', exist_ok=True)

# Personalized parameters for Student ID 502808
r = 808
a, b, c = 8, 0, 8
alpha_q1 = 0.05 + r / 10000.0  # 0.1308
lambda_q6 = 2.0 + r / 1000.0   # 2.808
noise_q7 = 0.10 + (r % 20) / 100.0  # 0.18
h_q7 = 3 + (r % 4)                  # 3
q_q8 = r % 3                        # 1

print(f"Personalized parameters: r={r}, a={a}, b={b}, c={c}")
print(f"Q1 alpha = {alpha_q1:.4f}")
print(f"Q6 lambda = {lambda_q6:.4f}")
print(f"Q7 noise = {noise_q7:.4f}, h = {h_q7}")
print(f"Q8 q = {q_q8}")

# ==============================================================================
# FIGURE 1: Q1 Activation Function and Derivative Forensics
# ==============================================================================
z = np.linspace(-5, 5, 1000)

def g_func(z_val):
    return np.where(z_val < 0, alpha_q1 * z_val, np.tanh(z_val) + 0.2 * z_val)

def g_prime_func(z_val):
    return np.where(z_val < 0, alpha_q1, (1.0 - np.tanh(z_val)**2) + 0.2)

gz = g_func(z)
gp = g_prime_func(z)

sigmoid = 1.0 / (1.0 + np.exp(-z))
sigmoid_p = sigmoid * (1.0 - sigmoid)
tanh = np.tanh(z)
tanh_p = 1.0 - tanh**2
relu = np.maximum(0, z)
relu_p = np.where(z > 0, 1.0, 0.0)
leaky_relu = np.where(z > 0, z, 0.01 * z)
leaky_relu_p = np.where(z > 0, 1.0, 0.01)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Plot g(z)
axes[0].plot(z, gz, label=f"g(z) ($\\alpha = {alpha_q1:.4f}$)", color='#1f77b4', lw=2.5)
axes[0].plot(z, tanh, label="tanh(z)", color='#2ca02c', lw=1.5, ls='--')
axes[0].plot(z, relu, label="ReLU(z)", color='#d62728', lw=1.5, ls=':')
axes[0].plot(z, leaky_relu, label="Leaky ReLU(z) ($\\alpha=0.01$)", color='#9467bd', lw=1.5, ls='-.')
axes[0].axhline(0, color='black', lw=0.8, alpha=0.5)
axes[0].axvline(0, color='black', lw=0.8, alpha=0.5)
axes[0].scatter([0], [0], color='#1f77b4', s=60, zorder=5)
axes[0].set_title(f"Personalized Activation $g(z)$ vs Common Functions ($r={r}$)", fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel("$z$", fontsize=12)
axes[0].set_ylabel("$g(z)$", fontsize=12)
axes[0].set_xlim(-5, 5)
axes[0].legend(loc="upper left", frameon=True)
axes[0].grid(True, alpha=0.3)

# Plot g'(z)
axes[1].plot(z[z < 0], gp[z < 0], color='#1f77b4', lw=2.5, label=f"$g'(z)$ ($r={r}$)")
axes[1].plot(z[z >= 0], gp[z >= 0], color='#1f77b4', lw=2.5)
axes[1].scatter([0], [alpha_q1], facecolors='white', edgecolors='#1f77b4', s=60, zorder=5, lw=2, label="Left deriv: 0.1308")
axes[1].scatter([0], [1.2], color='#1f77b4', s=60, zorder=5, label="Right deriv: 1.2000")
axes[1].plot(z, sigmoid_p, label="$\sigma'(z)$", color='#ff7f0e', lw=1.5, ls='--')
axes[1].plot(z, tanh_p, label="tanh'(z)", color='#2ca02c', lw=1.5, ls='--')
axes[1].plot(z, relu_p, label="ReLU'(z)", color='#d62728', lw=1.5, ls=':')
axes[1].plot(z, leaky_relu_p, label="Leaky ReLU'(z)", color='#9467bd', lw=1.5, ls='-.')
axes[1].axhline(0, color='black', lw=0.8, alpha=0.5)
axes[1].axvline(0, color='black', lw=0.8, alpha=0.5)
axes[1].set_title(f"Derivative $g'(z)$ Gradient Behavior ($r={r}$)", fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel("$z$", fontsize=12)
axes[1].set_ylabel("$g'(z)$", fontsize=12)
axes[1].set_xlim(-5, 5)
axes[1].set_ylim(-0.05, 1.35)
axes[1].legend(loc="upper right", frameon=True, fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/q1_activation_plot.png', dpi=300)
plt.close()
print("Saved figures/q1_activation_plot.png")


# ==============================================================================
# FIGURE 2: Q6 Gradient Descent Behavior and Trajectories
# ==============================================================================
def run_gd_path(eta, iters=4):
    w1, w2 = 2.0, -1.0
    path = [(w1, w2)]
    for _ in range(iters):
        d1 = 2.0 * w1
        d2 = 2.0 * lambda_q6 * w2
        w1 -= eta * d1
        w2 -= eta * d2
        path.append((w1, w2))
    return np.array(path)

path_small = run_gd_path(0.05, 4)
path_large = run_gd_path(0.5, 4)

w1_grid = np.linspace(-3, 3, 400)
w2_grid = np.linspace(-12, 12, 400)
W1, W2 = np.meshgrid(w1_grid, w2_grid)
C_grid = W1**2 + lambda_q6 * W2**2

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Path for eta = 0.05
axes[0].contour(W1, W2, C_grid, levels=25, cmap='viridis', alpha=0.7)
axes[0].plot(path_small[:, 0], path_small[:, 1], 'o-', color='#1f77b4', lw=2.2, markersize=7, label=r"GD steps ($\eta=0.05$)")
for idx, (p1, p2) in enumerate(path_small):
    axes[0].annotate(f"t={idx}", (p1, p2), textcoords="offset points", xytext=(8, -5), fontsize=10, fontweight='bold', color='#0b406b')
axes[0].scatter([0], [0], color='red', marker='*', s=150, zorder=5, label="Global Min (0,0)")
axes[0].set_title(f"Gradient Descent with $\\eta = 0.05$ (Stable Convergence)\n$\\lambda = {lambda_q6:.3f}$", fontsize=12, fontweight='bold')
axes[0].set_xlabel("$w_1$", fontsize=12)
axes[0].set_ylabel("$w_2$", fontsize=12)
axes[0].set_xlim(-0.5, 2.5)
axes[0].set_ylim(-1.5, 0.5)
axes[0].legend(loc="upper left")
axes[0].grid(True, alpha=0.3)

# Path for eta = 0.5
axes[1].contour(W1, W2, C_grid, levels=25, cmap='viridis', alpha=0.7)
axes[1].plot(path_large[:, 0], path_large[:, 1], 's--', color='#d62728', lw=2.2, markersize=7, label=r"GD steps ($\eta=0.50$)")
for idx, (p1, p2) in enumerate(path_large):
    offset = (8, 5) if p2 > 0 else (8, -12)
    c_val = p1**2 + lambda_q6 * p2**2
    axes[1].annotate(f"t={idx}\n(C={c_val:.1f})", (p1, p2), textcoords="offset points", xytext=offset, fontsize=9, fontweight='bold', color='#7a1112')
axes[1].scatter([0], [0], color='red', marker='*', s=150, zorder=5, label="Global Min (0,0)")
axes[1].set_title(f"Gradient Descent with $\\eta = 0.50$ (Severe Divergence)\n$\\lambda = {lambda_q6:.3f} > 1/\\eta$", fontsize=12, fontweight='bold')
axes[1].set_xlabel("$w_1$", fontsize=12)
axes[1].set_ylabel("$w_2$", fontsize=12)
axes[1].set_xlim(-1, 3)
axes[1].set_ylim(-13, 13)
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/q6_gd_trajectories.png', dpi=300)
plt.close()
print("Saved figures/q6_gd_trajectories.png")


# ==============================================================================
# FIGURE 3 & 4: Q7 Programming - ANN from Scratch Training & Decision Boundary
# ==============================================================================
X_data, y_data = make_moons(n_samples=300, noise=noise_q7, random_state=r)
X = X_data.T # (2, 300)
Y = y_data.reshape(1, -1) # (1, 300)

def sigmoid_np(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def init_q7(n_x=2, n_h=h_q7, n_y=1, seed=r):
    np.random.seed(seed)
    return {
        'W1': np.random.randn(n_h, n_x) * 0.1,
        'b1': np.zeros((n_h, 1)),
        'W2': np.random.randn(n_y, n_h) * 0.1,
        'b2': np.zeros((n_y, 1))
    }

def forward_q7(X_in, params):
    Z1 = np.dot(params['W1'], X_in) + params['b1']
    A1 = np.tanh(Z1)
    Z2 = np.dot(params['W2'], A1) + params['b2']
    A2 = sigmoid_np(Z2)
    return A2, {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

def compute_loss_q7(A2, Y_true):
    m = Y_true.shape[1]
    eps = 1e-15
    clipped = np.clip(A2, eps, 1.0 - eps)
    return float(- (1.0 / m) * np.sum(Y_true * np.log(clipped) + (1.0 - Y_true) * np.log(1.0 - clipped)))

def backward_q7(params, cache, X_in, Y_true):
    m = Y_true.shape[1]
    dZ2 = cache['A2'] - Y_true
    dW2 = (1.0 / m) * np.dot(dZ2, cache['A1'].T)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(params['W2'].T, dZ2)
    dZ1 = dA1 * (1.0 - np.power(cache['A1'], 2))
    dW1 = (1.0 / m) * np.dot(dZ1, X_in.T)
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

epochs = 3000
lr = 0.5
params = init_q7()

loss_list = []
acc_list = []
norm_dW1_list = []
norm_dW2_list = []

for ep in range(epochs):
    A2, cache = forward_q7(X, params)
    cost = compute_loss_q7(A2, Y)
    grads = backward_q7(params, cache, X, Y)
    
    preds = (A2 > 0.5).astype(int)
    acc = np.mean(preds == Y)
    
    loss_list.append(cost)
    acc_list.append(acc)
    norm_dW1_list.append(np.linalg.norm(grads['dW1']))
    norm_dW2_list.append(np.linalg.norm(grads['dW2']))
    
    params['W1'] -= lr * grads['dW1']
    params['b1'] -= lr * grads['db1']
    params['W2'] -= lr * grads['dW2']
    params['b2'] -= lr * grads['db2']

max_ep_dW1 = int(np.argmax(norm_dW1_list))
max_ep_dW2 = int(np.argmax(norm_dW2_list))

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

axes[0, 0].plot(loss_list, color='#1f77b4', lw=2)
axes[0, 0].set_title("Training BCE Loss vs Epoch", fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel("Epoch", fontsize=11)
axes[0, 0].set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(acc_list, color='#2ca02c', lw=2)
axes[0, 1].set_title(f"Classification Accuracy vs Epoch (Final: {acc_list[-1]*100:.1f}%)", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel("Epoch", fontsize=11)
axes[0, 1].set_ylabel("Accuracy", fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(norm_dW1_list, color='#ff7f0e', lw=2)
axes[1, 0].scatter([max_ep_dW1], [norm_dW1_list[max_ep_dW1]], color='red', s=80, zorder=5)
axes[1, 0].annotate(f"Max ||dW1|| = {norm_dW1_list[max_ep_dW1]:.5f}\nat epoch {max_ep_dW1}", 
                    (max_ep_dW1, norm_dW1_list[max_ep_dW1]), 
                    textcoords="offset points", xytext=(20, -10),
                    arrowprops=dict(arrowstyle="->", color='red', lw=1.2), fontweight='bold')
axes[1, 0].set_title(r"Hidden Layer Gradient Norm $||dW(1)||$ vs Epoch", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel("Epoch", fontsize=11)
axes[1, 0].set_ylabel(r"$||dW(1)||_2$", fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(norm_dW2_list, color='#9467bd', lw=2)
axes[1, 1].scatter([max_ep_dW2], [norm_dW2_list[max_ep_dW2]], color='red', s=80, zorder=5)
axes[1, 1].annotate(f"Max ||dW2|| = {norm_dW2_list[max_ep_dW2]:.5f}\nat epoch {max_ep_dW2}", 
                    (max_ep_dW2, norm_dW2_list[max_ep_dW2]), 
                    textcoords="offset points", xytext=(20, -10),
                    arrowprops=dict(arrowstyle="->", color='red', lw=1.2), fontweight='bold')
axes[1, 1].set_title(r"Output Layer Gradient Norm $||dW(2)||$ vs Epoch", fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel("Epoch", fontsize=11)
axes[1, 1].set_ylabel(r"$||dW(2)||_2$", fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/q7_training_curves.png', dpi=300)
plt.close()
print("Saved figures/q7_training_curves.png")

# Decision boundary plot
x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()].T
Z_mesh, _ = forward_q7(grid_points, params)
Z_mesh = Z_mesh.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
contour = ax.contourf(xx, yy, Z_mesh, levels=50, cmap="RdBu", alpha=0.7)
ax.contour(xx, yy, Z_mesh, levels=[0.5], colors='black', linewidths=2.0, linestyles='--')
scatter = ax.scatter(X[0, :], X[1, :], c=y_data, cmap="coolwarm", edgecolors='k', s=45, lw=0.6)
ax.set_title(f"Learned Non-Linear Decision Boundary (2 -> 3 -> 1 Architecture)\nAccuracy = {acc_list[-1]*100:.1f}%, Noise = {noise_q7:.2f}, Seed = {r}", fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel("$x_1$", fontsize=12)
ax.set_ylabel("$x_2$", fontsize=12)
plt.colorbar(contour, ax=ax, label="P(y = 1 | x)")

final_preds = (forward_q7(X, params)[0] > 0.5).astype(int).reshape(-1)
cm = confusion_matrix(y_data, final_preds)

cm_text = (f"Confusion Matrix:\n"
           f"TN: {cm[0,0]}  |  FP: {cm[0,1]}\n"
           f"FN: {cm[1,0]}  |  TP: {cm[1,1]}\n"
           f"Accuracy: {acc_list[-1]*100:.1f}%")
ax.text(0.03, 0.05, cm_text, transform=ax.transAxes, fontsize=11, family='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#333333', alpha=0.9))

plt.tight_layout()
plt.savefig('figures/q7_decision_boundary.png', dpi=300)
plt.close()
print("Saved figures/q7_decision_boundary.png")


# ==============================================================================
# FIGURE 5: Q8 Failure Diagnosis (Dead ReLU) vs Single Design Repair
# ==============================================================================
def relu(z_val):
    return np.maximum(0, z_val)

def relu_p(z_val):
    return (z_val > 0).astype(float)

# Run failure (b1 = -3.0)
params_fail = {
    'W1': np.random.RandomState(r).randn(h_q7, 2) * 0.1,
    'b1': np.full((h_q7, 1), -3.0),
    'W2': np.random.RandomState(r).randn(1, h_q7) * 0.1,
    'b2': np.zeros((1, 1))
}

# Run repair (only change: b1 = 0.0)
params_rep = {
    'W1': np.random.RandomState(r).randn(h_q7, 2) * 0.1,
    'b1': np.zeros((h_q7, 1)),
    'W2': np.random.RandomState(r).randn(1, h_q7) * 0.1,
    'b2': np.zeros((1, 1))
}

def train_q8_experiment(p_init, epochs=3000, lr=0.5):
    p = {k: v.copy() for k, v in p_init.items()}
    losses, accs, dW1_norms, dead_pcts = [], [], [], []
    m = Y.shape[1]
    
    for ep in range(epochs):
        Z1 = np.dot(p['W1'], X) + p['b1']
        A1 = relu(Z1)
        Z2 = np.dot(p['W2'], A1) + p['b2']
        A2 = sigmoid_np(Z2)
        
        cost = compute_loss_q7(A2, Y)
        dead = np.mean(np.all(A1 == 0, axis=1)) * 100.0  # % units dead across all samples
        
        dZ2 = A2 - Y
        dW2 = (1.0 / m) * np.dot(dZ2, A1.T)
        db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dA1 = np.dot(p['W2'].T, dZ2)
        dZ1 = dA1 * relu_p(Z1)
        dW1 = (1.0 / m) * np.dot(dZ1, X.T)
        db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)
        
        p['W1'] -= lr * dW1
        p['b1'] -= lr * db1
        p['W2'] -= lr * dW2
        p['b2'] -= lr * db2
        
        losses.append(cost)
        accs.append(np.mean((A2 > 0.5).astype(int) == Y))
        dW1_norms.append(np.linalg.norm(dW1))
        dead_pcts.append(dead)
        
    return losses, accs, dW1_norms, dead_pcts, p

fail_l, fail_acc, fail_grad, fail_dead, p_f = train_q8_experiment(params_fail)
rep_l, rep_acc, rep_grad, rep_dead, p_r = train_q8_experiment(params_rep)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

# Loss Comparison
axes[0].plot(fail_l, color='#d62728', lw=2.2, label=r"Failure ($b_1=-3.0$)")
axes[0].plot(rep_l, color='#2ca02c', lw=2.2, label=r"Repaired ($b_1=0.0$)")
axes[0].set_title("Training Loss: Dead ReLU vs Repaired", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Epoch", fontsize=11)
axes[0].set_ylabel("BCE Loss", fontsize=11)
axes[0].legend(frameon=True)
axes[0].grid(True, alpha=0.3)

# Accuracy Comparison
axes[1].plot(fail_acc, color='#d62728', lw=2.2, label="Failure (Stuck at 50% / Chance)")
axes[1].plot(rep_acc, color='#2ca02c', lw=2.2, label=f"Repaired (Final: {rep_acc[-1]*100:.1f}%)")
axes[1].set_title("Accuracy: Dead ReLU vs Repaired", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylabel("Accuracy", fontsize=11)
axes[1].legend(frameon=True)
axes[1].grid(True, alpha=0.3)

# Gradient Norm Comparison
axes[2].plot(fail_grad, color='#d62728', lw=2.2, label=r"Failure ($||dW_1|| = 0.0$)")
axes[2].plot(rep_grad, color='#2ca02c', lw=2.2, label="Repaired (Active Gradient Flow)")
axes[2].set_title(r"Hidden Layer Gradient Norm $||dW_1||$", fontsize=12, fontweight='bold')
axes[2].set_xlabel("Epoch", fontsize=11)
axes[2].set_ylabel(r"$||dW_1||_2$", fontsize=11)
axes[2].legend(frameon=True)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/q8_failure_vs_repair.png', dpi=300)
plt.close()
print("Saved figures/q8_failure_vs_repair.png")
