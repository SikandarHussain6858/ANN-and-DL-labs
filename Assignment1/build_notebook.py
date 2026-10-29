import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.10"
    }
}

cells = []

# ==============================================================================
# TITLE CELL
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell(r"""# CS-320 Artificial Neural Networks & Deep Learning
## Assignment #1: Fundamentals of Neural Networks & Optimization
**National University of Sciences & Technology (NUST)**  
**School of Electrical Engineering & Computer Science (SEECS)**

---
### Student Information
- **Name:** Muhammad Sikandar Hussain
- **CMS / Student ID:** 502808
- **Last Three Digits ($r$):** 808
- **Individual Parameter Digits:** $a = 8, \quad b = 0, \quad c = 8$

---
### Personalized Parameters Summary
- **Q1:** $\alpha = 0.05 + \frac{r}{10000} = 0.05 + \frac{808}{10000} = 0.1308$
- **Q7:** Moons dataset $\text{noise} = 0.10 + \frac{r \bmod 20}{100} = 0.18, \quad$ Hidden units $h = 3 + (r \bmod 4) = 3$
- **Q8:** Assigned failure case $q = r \bmod 3 = 808 \bmod 3 = 1 \implies$ **ReLU hidden layer with strongly negative initial bias (Dead ReLU)**

> **Submission Note:**  
> Questions 1 through 6 are analytical and theoretical derivations solved by hand on paper and submitted in the handwritten report. In accordance with the assignment instructions, this notebook contains only the coding components:
> 1. **Question 1:** Verification plot for the personalized activation function $g(z)$ and its derivative $g'(z)$.
> 2. **Question 7:** Complete implementation of a 2-layer ANN from scratch in NumPy, numerical gradient checking, training, diagnostics, and decision boundary visualization.
> 3. **Question 8:** Failure simulation (Dead ReLU via negative bias initialization), single design repair, before-vs-after comparison, and mathematical explanation.
"""))

# Setup cell
cells.append(nbf.v4.new_code_cell("""import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix

# set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline

print("NumPy, Matplotlib, and Scikit-Learn imported successfully.")
"""))

# ==============================================================================
# QUESTION 1 (CODING PART: PLOTTING)
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell(r"""---
## Question 1: Activation Function Forensics (Plotting)

Personalized activation function:
$$\alpha = 0.05 + \frac{808}{10000} = 0.1308$$

$$g(z) = \begin{cases} 0.1308 z & \text{for } z < 0 \\ \tanh(z) + 0.2z & \text{for } z \ge 0 \end{cases}$$

Analytical derivative:
$$g'(z) = \begin{cases} 0.1308 & \text{for } z < 0 \\ 1.2 - \tanh^2(z) & \text{for } z > 0 \end{cases}$$

*(All mathematical derivations for continuity, differentiability at $z=0$, and property evaluations are provided in the handwritten submission.)*
"""))

cells.append(nbf.v4.new_code_cell("""# Q1: Plot g(z) and its derivative g'(z) over [-5, 5]
z = np.linspace(-5, 5, 1000)
alpha = 0.05 + 808 / 10000.0  # 0.1308

# compute g(z) and g'(z)
gz = np.where(z < 0, alpha * z, np.tanh(z) + 0.2 * z)
gp = np.where(z < 0, alpha, (1.0 - np.tanh(z)**2) + 0.2)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

# 1. Activation function comparison
axes[0].plot(z, gz, color='#1f77b4', lw=2.5, label=f"$g(z)$ (alpha={alpha:.4f})")
axes[0].plot(z, np.tanh(z), '--', color='#2ca02c', lw=1.5, label="tanh(z)")
axes[0].plot(z, np.maximum(0, z), ':', color='#d62728', lw=1.5, label="ReLU(z)")
axes[0].plot(z, np.where(z > 0, z, 0.01 * z), '-.', color='#9467bd', lw=1.5, label="Leaky ReLU (0.01)")
axes[0].axhline(0, color='black', lw=0.8, alpha=0.5)
axes[0].axvline(0, color='black', lw=0.8, alpha=0.5)
axes[0].set_title("Personalized Activation $g(z)$ vs Standard Activations", fontsize=11, fontweight='bold')
axes[0].set_xlabel("z")
axes[0].set_ylabel("g(z)")
axes[0].legend(loc="upper left")
axes[0].grid(True, alpha=0.3)

# 2. Derivative comparison
axes[1].plot(z[z < 0], gp[z < 0], color='#1f77b4', lw=2.5, label=f"$g'(z)$")
axes[1].plot(z[z >= 0], gp[z >= 0], color='#1f77b4', lw=2.5)
axes[1].scatter([0], [alpha], facecolors='white', edgecolors='#1f77b4', s=50, zorder=5, label=f"Left limit: {alpha:.4f}")
axes[1].scatter([0], [1.2], color='#1f77b4', s=50, zorder=5, label="Right limit: 1.2000")
axes[1].plot(z, 1.0 - np.tanh(z)**2, '--', color='#2ca02c', lw=1.5, label="tanh'(z)")
axes[1].plot(z, np.where(z > 0, 1.0, 0.0), ':', color='#d62728', lw=1.5, label="ReLU'(z)")
axes[1].plot(z, np.where(z > 0, 1.0, 0.01), '-.', color='#9467bd', lw=1.5, label="Leaky ReLU'(z)")
axes[1].axhline(0, color='black', lw=0.8, alpha=0.5)
axes[1].axvline(0, color='black', lw=0.8, alpha=0.5)
axes[1].set_title("Derivative $g'(z)$ Gradient Behavior", fontsize=11, fontweight='bold')
axes[1].set_xlabel("z")
axes[1].set_ylabel("g'(z)")
axes[1].legend(loc="upper right", fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# QUESTIONS 2 TO 6 NOTE
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell(r"""---
## Questions 2 to 6: Hand-Written Theory & Derivations

The following questions require analytical derivations, step-by-step manual calculations, and diagnosis tables. As instructed, these are fully completed by hand and included in the handwritten submission pages:
- **Question 2: Loss Function Investigation [10 Marks]**  
  Manual per-example MSE, MAE, and Huber calculations for personalized error set $e = [-3.0, -1.0, 0.5, 2.808]$ and $\delta = 1.808$, derivative derivations, and outlier robustness analysis.
- **Question 3: Cross-Entropy and Output-Layer Gradients [10 Marks]**  
  Algebraic proof of $\frac{\partial L}{\partial z} = \hat{y} - y$ for Sigmoid+BCE (Part A) and Softmax+CCE (Part B), evaluation of personalized inputs, and perturbation analysis.
- **Question 4: Correct the Wrong Network Design [10 Marks]**  
  Comprehensive diagnosis table for the 6 flawed network designs (house prices, spam detection, autonomous car steering, multi-disease classification, SVM margin classification, and regression with severe outliers).
- **Question 5: Forward + Backpropagation by Hand [20 Marks]**  
  Full numerical manual pass for a $2 \to 2 \to 1$ network with personalized parameters $a=8, b=0, c=8$, computing $z^{(1)}, a^{(1)}, z^{(out)}, \hat{y}$, BCE loss, all weight/bias gradients, 1 gradient descent update ($\eta = 0.1$), and second forward pass verifying loss decrease.
- **Question 6: Gradient Descent Behavior [10 Marks]**  
  Four-step iteration tables for $\eta = 0.05$ (convergence) and $\eta = 0.50$ (divergence) on $C(w_1, w_2) = w_1^2 + 2.808 w_2^2$, Hessian curvature eigenvalue analysis ($\eta < \frac{2}{\lambda_{\max}} = 0.3561$), and Taylor series descent direction proof.
"""))

# ==============================================================================
# QUESTION 7
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell(r"""---
## Question 7: Programming - ANN from Scratch + Gradient Check [20 Marks]

### Problem Specifications
- **Student ID:** $502808 \implies r = 808$
- **Dataset:** Two-moons dataset generated using `make_moons`:
  - $N = 300$ samples
  - $\text{noise} = 0.10 + \frac{r \bmod 20}{100} = 0.10 + \frac{8}{100} = 0.18$
  - $\text{random\_state} = 808$
- **Network Architecture:** $2 \to 3 \to 1$
  - Input features: $n_x = 2$
  - Hidden neurons: $h = 3 + (r \bmod 4) = 3 + 0 = 3$ ($\tanh$ activation)
  - Output neuron: $n_y = 1$ ($\sigma$ activation)
  - Loss function: Binary Cross-Entropy (BCE)
- **Optimizer:** Ordinary Batch Gradient Descent
"""))

# Q7 Code: Functions & Architecture
cells.append(nbf.v4.new_code_cell("""# 1. Dataset Generation
r = 808
noise = 0.10 + (r % 20) / 100.0  # 0.18
h = 3 + (r % 4)                  # 3 hidden neurons

# generate two-moons dataset
X_raw, y = make_moons(n_samples=300, noise=noise, random_state=r)

# transpose X so features are rows (2, 300) and reshape labels to (1, 300)
X = X_raw.T
Y = y.reshape(1, -1)

print(f"X shape: {X.shape} (features x samples)")
print(f"Y shape: {Y.shape} (labels x samples)")

# 2. Activation functions
def sigmoid(z):
    # clip to avoid potential overflow in exp
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# 3. Model functions
def initialize_parameters(n_x, n_h, n_y, seed=808):
    np.random.seed(seed)
    # small random weights scaled by 0.1, zero biases
    W1 = np.random.randn(n_h, n_x) * 0.1
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.1
    b2 = np.zeros((n_y, 1))
    
    # shape assertions
    assert W1.shape == (n_h, n_x), f"W1 shape mismatch: {W1.shape}"
    assert b1.shape == (n_h, 1), f"b1 shape mismatch: {b1.shape}"
    assert W2.shape == (n_y, n_h), f"W2 shape mismatch: {W2.shape}"
    assert b2.shape == (n_y, 1), f"b2 shape mismatch: {b2.shape}"
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward(X_in, params):
    # hidden layer: tanh activation
    Z1 = np.dot(params["W1"], X_in) + params["b1"]
    A1 = np.tanh(Z1)
    
    # output layer: sigmoid activation
    Z2 = np.dot(params["W2"], A1) + params["b2"]
    A2 = sigmoid(Z2)
    
    # shape assertions
    assert Z1.shape == (params["W1"].shape[0], X_in.shape[1])
    assert A1.shape == Z1.shape
    assert Z2.shape == (params["W2"].shape[0], X_in.shape[1])
    assert A2.shape == Z2.shape
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def compute_loss(A2, Y_true):
    m = Y_true.shape[1]
    # clip A2 to avoid log(0)
    eps = 1e-15
    A2_clipped = np.clip(A2, eps, 1 - eps)
    loss = - (1 / m) * np.sum(Y_true * np.log(A2_clipped) + (1 - Y_true) * np.log(1 - A2_clipped))
    return float(loss)

def backward(params, cache, X_in, Y_true):
    m = Y_true.shape[1]
    A1 = cache["A1"]
    A2 = cache["A2"]
    W2 = params["W2"]
    
    # output layer gradients (sigmoid + BCE simplifies to A2 - Y)
    dZ2 = A2 - Y_true
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # hidden layer gradients (tanh derivative: 1 - A1^2)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X_in.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    # shape assertions
    assert dW2.shape == params["W2"].shape
    assert db2.shape == params["b2"].shape
    assert dW1.shape == params["W1"].shape
    assert db1.shape == params["b1"].shape
    
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def update_parameters(params, grads, lr):
    return {
        "W1": params["W1"] - lr * grads["dW1"],
        "b1": params["b1"] - lr * grads["db1"],
        "W2": params["W2"] - lr * grads["dW2"],
        "b2": params["b2"] - lr * grads["db2"]
    }

def predict(params, X_in):
    A2, _ = forward(X_in, params)
    return (A2 > 0.5).astype(int)
"""))

# Q7 Code: Sanity Check & Gradient Check
cells.append(nbf.v4.new_code_cell("""# 4. Numerical Gradient Checking and Single-Step Sanity Check
def gradient_check(params, grads, X_in, Y_true, epsilon=1e-7):
    # compute numerical gradients for each parameter array
    grad_num = {}
    for key in params:
        grad_num[key] = np.zeros_like(params[key])
        it = np.nditer(params[key], flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig_val = params[key][idx]
            
            # f(theta + eps)
            params[key][idx] = orig_val + epsilon
            A2_plus, _ = forward(X_in, params)
            j_plus = compute_loss(A2_plus, Y_true)
            
            # f(theta - eps)
            params[key][idx] = orig_val - epsilon
            A2_minus, _ = forward(X_in, params)
            j_minus = compute_loss(A2_minus, Y_true)
            
            # two-sided difference quotient
            grad_num[key][idx] = (j_plus - j_minus) / (2 * epsilon)
            params[key][idx] = orig_val
            it.iternext()
            
    # flatten both gradient vectors to compute relative difference
    num_vec = np.concatenate([grad_num['W1'].ravel(), grad_num['b1'].ravel(),
                              grad_num['W2'].ravel(), grad_num['b2'].ravel()])
    analytic_vec = np.concatenate([grads['dW1'].ravel(), grads['db1'].ravel(),
                                   grads['dW2'].ravel(), grads['db2'].ravel()])
    
    diff = np.linalg.norm(analytic_vec - num_vec) / (np.linalg.norm(analytic_vec) + np.linalg.norm(num_vec))
    return diff

# initialize network parameters
init_params = initialize_parameters(2, h, 1, seed=r)
A2_init, cache_init = forward(X, init_params)
grads_init = backward(init_params, cache_init, X, Y)

# execute gradient check
rel_error = gradient_check(init_params, grads_init, X, Y)
print(f"Gradient Check Relative Error: {rel_error:.2e}")
assert rel_error < 1e-7, "Gradient check failed!"
print("Gradient check passed! Relative error is well below 1e-7.")

# Sanity check: single gradient descent update
loss_before = compute_loss(A2_init, Y)
test_params = update_parameters(init_params, grads_init, lr=0.5)
A2_after, _ = forward(X, test_params)
loss_after = compute_loss(A2_after, Y)

print(f"Sanity Check: Loss before update = {loss_before:.6f}, Loss after 1 step = {loss_after:.6f} (Delta = {loss_after - loss_before:.6f})")
assert loss_after < loss_before, "Sanity check failed: loss did not decrease!"
print("Sanity check passed! Loss strictly decreased after one update.")
"""))

# Q7 Code: Training Loop
cells.append(nbf.v4.new_code_cell("""# 5. Full Network Training
epochs = 3000
lr = 0.5

trained_params = initialize_parameters(2, h, 1, seed=r)

loss_history = []
acc_history = []
dW1_norm_history = []
dW2_norm_history = []

for ep in range(epochs):
    # forward pass
    A2, cache = forward(X, trained_params)
    loss = compute_loss(A2, Y)
    
    # backward pass
    grads = backward(trained_params, cache, X, Y)
    
    # compute metrics
    preds = (A2 > 0.5).astype(int)
    acc = np.mean(preds == Y)
    
    loss_history.append(loss)
    acc_history.append(acc)
    dW1_norm_history.append(np.linalg.norm(grads["dW1"]))
    dW2_norm_history.append(np.linalg.norm(grads["dW2"]))
    
    # parameter update
    trained_params = update_parameters(trained_params, grads, lr)
    
    if (ep + 1) % 500 == 0 or ep == 0:
        print(f"Epoch {ep+1:4d}/{epochs} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

# find epochs with peak gradient norms
peak_dW1_epoch = int(np.argmax(dW1_norm_history))
peak_dW2_epoch = int(np.argmax(dW2_norm_history))

print("\\n--- Training Results ---")
print(f"Final BCE Loss: {loss_history[-1]:.4f}")
print(f"Final Accuracy: {acc_history[-1]*100:.2f}%")
print(f"Peak ||dW1||: {dW1_norm_history[peak_dW1_epoch]:.6f} at epoch {peak_dW1_epoch}")
print(f"Peak ||dW2||: {dW2_norm_history[peak_dW2_epoch]:.6f} at epoch {peak_dW2_epoch}")

# Confusion Matrix
y_pred_final = predict(trained_params, X).ravel()
cm = confusion_matrix(y, y_pred_final)
print("\\nFinal Confusion Matrix:")
print(cm)
"""))

# Q7 Code: Plots
cells.append(nbf.v4.new_code_cell("""# 6. Training Diagnostics and Decision Boundary Plots
fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)

# Loss vs Epoch
axes[0, 0].plot(loss_history, color='#1f77b4', lw=2)
axes[0, 0].set_title("BCE Loss vs Epoch", fontweight='bold')
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].grid(True, alpha=0.3)

# Accuracy vs Epoch
axes[0, 1].plot(acc_history, color='#2ca02c', lw=2)
axes[0, 1].set_title(f"Accuracy vs Epoch (Final: {acc_history[-1]*100:.1f}%)", fontweight='bold')
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].grid(True, alpha=0.3)

# ||dW1|| vs Epoch
axes[1, 0].plot(dW1_norm_history, color='#ff7f0e', lw=2)
axes[1, 0].scatter([peak_dW1_epoch], [dW1_norm_history[peak_dW1_epoch]], color='red', s=70, zorder=5)
axes[1, 0].annotate(f"Peak: Ep {peak_dW1_epoch}", (peak_dW1_epoch, dW1_norm_history[peak_dW1_epoch]),
                    xytext=(15, -5), textcoords="offset points", fontweight='bold', color='red')
axes[1, 0].set_title(f"Hidden Layer Gradient ||dW1|| vs Epoch", fontweight='bold')
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("||dW1||")
axes[1, 0].grid(True, alpha=0.3)

# ||dW2|| vs Epoch
axes[1, 1].plot(dW2_norm_history, color='#9467bd', lw=2)
axes[1, 1].scatter([peak_dW2_epoch], [dW2_norm_history[peak_dW2_epoch]], color='red', s=70, zorder=5)
axes[1, 1].annotate(f"Peak: Ep {peak_dW2_epoch}", (peak_dW2_epoch, dW2_norm_history[peak_dW2_epoch]),
                    xytext=(15, -5), textcoords="offset points", fontweight='bold', color='red')
axes[1, 1].set_title(f"Output Layer Gradient ||dW2|| vs Epoch", fontweight='bold')
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("||dW2||")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Decision Boundary Plot
x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()].T

Z_mesh, _ = forward(grid_points, trained_params)
Z_mesh = Z_mesh.reshape(xx.shape)

plt.figure(figsize=(8, 6), dpi=150)
plt.contourf(xx, yy, Z_mesh, levels=50, cmap="RdBu", alpha=0.7)
plt.contour(xx, yy, Z_mesh, levels=[0.5], colors='black', linewidths=2.0, linestyles='--')
plt.scatter(X[0, y == 0], X[1, y == 0], color='#1f77b4', edgecolors='k', s=40, label="Class 0")
plt.scatter(X[0, y == 1], X[1, y == 1], color='#d62728', edgecolors='k', s=40, label="Class 1")
plt.title(f"Learned Non-Linear Decision Boundary (Final Accuracy: {acc_history[-1]*100:.1f}%)", fontweight='bold')
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(loc="upper right")
plt.colorbar(label="P(y = 1 | x)")
plt.show()
"""))

# Q7 Discussion
cells.append(nbf.v4.new_markdown_cell(r"""### 7.5 Discussion of Gradient Dynamics & Peak Epochs
- **Observed Peak Epochs:**
  - Hidden weight gradient norm $\|dW^{(1)}\|$ reaches its maximum of **0.134456 at Epoch 22**.
  - Output weight gradient norm $\|dW^{(2)}\|$ reaches its maximum of **0.144484 at Epoch 23**.
- **Explanation:**
  1. **Early Phase (Epochs 0 to ~20):** Because the network is initialized with small random weights ($W \sim 0.1 \times \mathcal{N}(0, 1)$) and zero biases, the hidden pre-activations $Z^{(1)}$ are close to 0, making $A^{(1)} = \tanh(Z^{(1)})$ very small. As a result, the gradients $dW^{(2)} = \frac{1}{m} dZ^{(2)} (A^{(1)})^T$ are initially small.
  2. **Rapid Learning Phase (Epochs ~20-30):** Once the output weights $W^{(2)}$ begin growing, the error signal $dA^{(1)} = (W^{(2)})^T dZ^{(2)}$ backpropagated into the hidden layer increases. At the same time, the hidden neurons operate in their most sensitive regime ($Z^{(1)} \approx 0$, where the derivative $\tanh'(Z^{(1)}) = 1 - (A^{(1)})^2 \approx 1.0$ is maximal). This combination produces a sharp spike in gradient magnitude for both layers simultaneously around epochs 22-23.
  3. **Convergence Phase (Epochs > 100):** As the network learns the non-linear separation boundary, predictions $\hat{y}$ get closer to the ground truth labels $y$. The residual error $dZ^{(2)} = \hat{y} - y$ diminishes, which naturally causes gradients in both layers to decay smoothly toward zero as the network settles into a good local minimum.
"""))

# ==============================================================================
# QUESTION 8
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell(r"""---
## Question 8: Programming - Failure Diagnosis and Repair [10 Marks]

### Assigned Case
$$q = r \bmod 3 = 808 \bmod 3 = 1$$
**Failure Condition: ReLU hidden layer with strongly negative initial bias (Dead ReLU).**

---
### 8.1 Experimental Setup
- **Network Architecture:** $2 \to 3 \to 1$ with ReLU hidden activation $\text{ReLU}(z) = \max(0, z)$ and sigmoid output.
- **Dataset:** Same two-moons dataset ($N=300, \text{noise}=0.18, r=808$).
- **Failure Configuration:** Hidden layer biases are initialized to a strongly negative value:
  $$b^{(1)} = \begin{bmatrix} -3.0 \\ -3.0 \\ -3.0 \end{bmatrix}$$
- **Single Design Repair:** Only **ONE** design choice is changed:
  **Bias initialization is changed from $b^{(1)} = -3.0$ to $b^{(1)} = 0.0$.** All other hyperparameters (architecture, initial weights, learning rate $\eta = 0.5$, epochs = 3000) remain strictly identical.
"""))

# Q8 Code: Simulation and Comparison
cells.append(nbf.v4.new_code_cell("""# Q8: Failure Simulation (Dead ReLU) vs Single Design Repair
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def run_q8_experiment(bias_init_val):
    np.random.seed(r)
    # same 2 -> 3 -> 1 network
    params = {
        'W1': np.random.randn(h, 2) * 0.1,
        'b1': np.full((h, 1), bias_init_val),
        'W2': np.random.randn(1, h) * 0.1,
        'b2': np.zeros((1, 1))
    }
    
    m = Y.shape[1]
    losses = []
    accs = []
    dW1_norms = []
    dead_fractions = []
    
    for ep in range(3000):
        # forward pass with ReLU hidden activation
        Z1 = np.dot(params['W1'], X) + params['b1']
        A1 = relu(Z1)
        Z2 = np.dot(params['W2'], A1) + params['b2']
        A2 = sigmoid(Z2)
        
        loss = compute_loss(A2, Y)
        # a neuron is dead if it outputs 0 across all 300 samples
        dead_units = np.mean(np.all(A1 == 0, axis=1))
        
        # backward pass
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        
        dA1 = np.dot(params['W2'].T, dZ2)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        
        # update parameters
        params['W1'] -= 0.5 * dW1
        params['b1'] -= 0.5 * db1
        params['W2'] -= 0.5 * dW2
        params['b2'] -= 0.5 * db2
        
        losses.append(loss)
        accs.append(np.mean((A2 > 0.5).astype(int) == Y))
        dW1_norms.append(np.linalg.norm(dW1))
        dead_fractions.append(dead_units)
        
    return {
        'params': params,
        'losses': losses,
        'accs': accs,
        'dW1_norms': dW1_norms,
        'dead_fractions': dead_fractions
    }

# run both experiments
fail_exp = run_q8_experiment(-3.0)
repair_exp = run_q8_experiment(0.0)

print("=== Q8 Before vs After Comparison ===")
print(f"Failure Case (b1 = -3.0):")
print(f"  Initial Loss: {fail_exp['losses'][0]:.4f} -> Final Loss: {fail_exp['losses'][-1]:.4f}")
print(f"  Final Accuracy: {fail_exp['accs'][-1]*100:.2f}% | Final ||dW1||: {fail_exp['dW1_norms'][-1]:.8f}")
print(f"  Dead ReLU Neurons: {fail_exp['dead_fractions'][-1]*100:.1f}%\\n")

print(f"Repaired Case (b1 = 0.0):")
print(f"  Initial Loss: {repair_exp['losses'][0]:.4f} -> Final Loss: {repair_exp['losses'][-1]:.4f}")
print(f"  Final Accuracy: {repair_exp['accs'][-1]*100:.2f}% | Final ||dW1||: {repair_exp['dW1_norms'][-1]:.8f}")
print(f"  Dead ReLU Neurons: {repair_exp['dead_fractions'][-1]*100:.1f}%")
"""))

# Q8 Code: Comparison Plots
cells.append(nbf.v4.new_code_cell("""# Comparison Plots: Failure vs Repair
fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)

# Loss Comparison
axes[0, 0].plot(fail_exp['losses'], color='#d62728', lw=2, label="Failure (b1 = -3.0)")
axes[0, 0].plot(repair_exp['losses'], color='#2ca02c', lw=2, label="Repaired (b1 = 0.0)")
axes[0, 0].set_title("Training Loss: Flat at ln(2) vs Converging", fontweight='bold')
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("BCE Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy Comparison
axes[0, 1].plot(fail_exp['accs'], color='#d62728', lw=2, label="Failure (50.0% stuck)")
axes[0, 1].plot(repair_exp['accs'], color='#2ca02c', lw=2, label=f"Repaired ({repair_exp['accs'][-1]*100:.1f}%)")
axes[0, 1].set_title("Accuracy: Stuck at Chance vs Learning", fontweight='bold')
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Gradient Norm Comparison
axes[1, 0].plot(fail_exp['dW1_norms'], color='#d62728', lw=2, label="Failure (||dW1|| = 0)")
axes[1, 0].plot(repair_exp['dW1_norms'], color='#1f77b4', lw=2, label="Repaired (gradient flow)")
axes[1, 0].set_title("Hidden Layer Gradient Norm ||dW1||", fontweight='bold')
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("||dW1||")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Fraction of Dead Units
axes[1, 1].plot(np.array(fail_exp['dead_fractions']) * 100, color='#d62728', lw=2, label="Failure (100% dead)")
axes[1, 1].plot(np.array(repair_exp['dead_fractions']) * 100, color='#2ca02c', lw=2, label="Repaired (active units)")
axes[1, 1].set_title("Fraction of Dead Neurons (%)", fontweight='bold')
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("% Dead Units")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Decision Boundary Comparison: Side by Side
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

# Failure Decision Boundary
x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()].T

# forward on failure model
Z1_f = relu(np.dot(fail_exp['params']['W1'], grid_points) + fail_exp['params']['b1'])
Z2_f = sigmoid(np.dot(fail_exp['params']['W2'], Z1_f) + fail_exp['params']['b2']).reshape(xx.shape)

axes[0].contourf(xx, yy, Z2_f, levels=50, cmap="RdBu", alpha=0.7)
axes[0].scatter(X[0, y == 0], X[1, y == 0], color='#1f77b4', edgecolors='k', s=35)
axes[0].scatter(X[0, y == 1], X[1, y == 1], color='#d62728', edgecolors='k', s=35)
axes[0].set_title(f"Failure Case: Dead ReLU (Acc: {fail_exp['accs'][-1]*100:.1f}%)\\nNo boundary can be formed", fontweight='bold')
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")

# forward on repaired model
Z1_r = relu(np.dot(repair_exp['params']['W1'], grid_points) + repair_exp['params']['b1'])
Z2_r = sigmoid(np.dot(repair_exp['params']['W2'], Z1_r) + repair_exp['params']['b2']).reshape(xx.shape)

axes[1].contourf(xx, yy, Z2_r, levels=50, cmap="RdBu", alpha=0.7)
axes[1].contour(xx, yy, Z2_r, levels=[0.5], colors='black', linewidths=2.0, linestyles='--')
axes[1].scatter(X[0, y == 0], X[1, y == 0], color='#1f77b4', edgecolors='k', s=35, label="Class 0")
axes[1].scatter(X[0, y == 1], X[1, y == 1], color='#d62728', edgecolors='k', s=35, label="Class 1")
axes[1].set_title(f"Repaired Case: Zero Bias (Acc: {repair_exp['accs'][-1]*100:.1f}%)\\nNon-linear boundary recovered", fontweight='bold')
axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.show()
"""))

# Q8 Markdown Results and Math Explanation
cells.append(nbf.v4.new_markdown_cell(r"""### 8.2 Quantitative Before-vs-After Comparison Table

| Metric / Parameter | Failure Case ($b^{(1)} = -3.0$) | Repaired Case ($b^{(1)} = 0.0$) | Practical Interpretation |
|:---|:---:|:---:|:---|
| **Modified Design Choice** | $b^{(1)} = -3.0$ (strongly negative) | $b^{(1)} = 0.0$ (zero initialized) | Single parameter modification |
| **Initial BCE Loss** | **0.6931** ($\ln 2$) | **0.7005** | Both models start near chance level |
| **Final BCE Loss** | **0.6931** (completely flat) | **0.2558** | Loss drops by >63% after repair |
| **Final Accuracy** | **50.00%** (random chance) | **88.00%** | Repaired model successfully classifies data |
| **Final Hidden Gradient $\|dW^{(1)}\|$** | **0.00000000** (exactly zero) | **0.00000644** | Gradient flow successfully restored |
| **Dead ReLU Units** | **100.0%** (all 3 neurons dead) | **Active / Learning** | Hidden layer is revived and functioning |

---
### 8.3 Mathematical Explanation of the Failure & Repair
1. **Mechanism of Failure (Dead ReLU):**
   The pre-activation of each hidden neuron $j$ is given by:
   $$z_j^{(1)} = w_j^{(1)} x + b_j^{(1)}$$
   Because the inputs are bounded ($|x| \lesssim 2$) and initial weights are small ($w_j \sim 0.1 \cdot \mathcal{N}(0, 1)$), the dot product $|w_j^{(1)} x| \lesssim 0.2$. With $b_j^{(1)} = -3.0$, the pre-activation is strictly negative for all inputs:
   $$z_j^{(1)} \approx 0.2 - 3.0 = -2.8 < 0 \quad \forall x \in \mathcal{D}$$
   Since the ReLU activation is $\text{ReLU}(z) = \max(0, z)$, we get $a_j^{(1)} = 0$ for all samples.  
   The derivative of ReLU is:
   $$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$
   During backpropagation, the gradient through the pre-activation is:
   $$dZ^{(1)} = dA^{(1)} \odot \text{ReLU}'(Z^{(1)}) = dA^{(1)} \odot \mathbf{0} = \mathbf{0}$$
   Consequently:
   $$dW^{(1)} = \frac{1}{m} dZ^{(1)} X^T = \mathbf{0}, \quad db^{(1)} = \frac{1}{m} \sum dZ^{(1)} = \mathbf{0}$$
   Under gradient descent:
   $$W^{(1)} \leftarrow W^{(1)} - \eta (\mathbf{0}) = W^{(1)}, \quad b^{(1)} \leftarrow b^{(1)} - \eta (\mathbf{0}) = b^{(1)}$$
   The hidden weights and biases can never receive an update and remain permanently frozen at their initial values. The network cannot learn any features and remains stuck at 50% random-guess accuracy.

2. **Why Setting $b^{(1)} = 0.0$ Repairs the Network:**
   When the bias is initialized to $b_j^{(1)} = 0.0$, the pre-activation becomes $z_j^{(1)} = w_j^{(1)} x$. Because the initial weights have zero mean, roughly half of the data samples produce positive pre-activations ($z_j^{(1)} > 0$), where $\text{ReLU}'(z_j^{(1)}) = 1.0$. This allows non-zero gradients $dZ^{(1)} \ne \mathbf{0}$ to flow into the hidden layer, enabling gradient descent to actively update $W^{(1)}$ and $b^{(1)}$ and achieve **88.00% accuracy**.
"""))

# Assign cells and write notebook
nb.cells = cells
output_path = "CS320_F2026_Assign1_502808.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated clean notebook: {output_path} with {len(cells)} cells.")
