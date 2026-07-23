# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: .venv (3.14.4)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Discrete-time signals and systems

# %% [markdown]
# #### Example 1
#
# Calculate this signal:
#
# $x[n] = u[n-2] - u[n-5]$

# %%
import numpy as np
import matplotlib.pyplot as plt

# Define the index range
n = np.arange(-2, 9)

# Define step function u[n]
def u(n):
    return np.where(n >= 0, 1, 0)

# Signals
u_n2 = u(n - 2)
u_n5 = u(n - 5)
x_n = u_n2 - u_n5

# Plot step-by-step
fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

# Plot 1: u[n-2]
axes[0].stem(n, u_n2, basefmt="C0-")
axes[0].set_ylabel('$u[n-2]$', fontsize=12)
axes[0].set_title('Step 1: First Shifted Unit Step $u[n-2]$', fontsize=12)
axes[0].set_yticks([0, 1])
axes[0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: u[n-5]
axes[1].stem(n, u_n5, basefmt="C0-")
axes[1].set_ylabel('$u[n-5]$', fontsize=12)
axes[1].set_title('Step 2: Second Shifted Unit Step $u[n-5]$', fontsize=12)
axes[1].set_yticks([0, 1])
axes[1].grid(True, linestyle='--', alpha=0.6)

# Plot 3: x[n] = u[n-2] - u[n-5]
axes[2].stem(n, x_n, basefmt="C0-")
axes[2].set_ylabel('$x[n]$', fontsize=12)
axes[2].set_xlabel('Sample Index $n$', fontsize=12)
axes[2].set_title('Step 3: Resulting Signal $x[n] = u[n-2] - u[n-5]$', fontsize=12)
axes[2].set_yticks([0, 1])
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("signal_visualization.png", dpi=300)
plt.show()

print("Signal values x[n]:")
for ni, val in zip(n, x_n):
    print(f"n={ni:2d} : {val}")
