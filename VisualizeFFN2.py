import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# FFN architecture parameters
d_model = 4
d_ff = 16

# The Feed Forward Network (FFN)
ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),  # Linear Layer 1 (FC1)
    nn.ReLU(),                 # ReLU Activation
    nn.Linear(d_ff, d_model),  # Linear Layer 2 (FC2)
)

# Random input tensor with a batch size of 3
# Note: Use torch.randn for a wider, more interesting distribution for visualization
x = torch.randn(3, d_model)

# We'll visualize a single sample (the first one)
x_sample = x[0].unsqueeze(0)

# Ensure the weights are initialized (although nn.Linear does this, calling it ensures fresh calculation)
with torch.no_grad():
    # Step 1: Pass through the first linear layer
    output_fc1 = ffn[0](x_sample)

    # Step 2: Pass through the ReLU activation function
    output_relu = ffn[1](output_fc1)

    # Step 3: Pass through the second linear layer
    output_fc2 = ffn[2](output_relu)

# --- Visualization Setup ---

# Collect all tensor data to determine global min/max for consistent Y-axis scaling
all_data = [
    x_sample.squeeze().detach().numpy(),
    output_fc1.squeeze().detach().numpy(),
    output_relu.squeeze().detach().numpy(),
    output_fc2.squeeze().detach().numpy()
]

# Calculate global minimum and maximum for consistent Y-axis
min_val = min(d.min() for d in all_data)
max_val = max(d.max() for d in all_data)
y_limit_buffer = max(abs(min_val), abs(max_val)) * 0.1 # Add 10% buffer

plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Feed Forward Network (FFN) Step-by-Step Transformation (Sample Values)', fontsize=16)

# --- Plot 0: Input (d_model=4) ---
# FIX: Use d_model for plotting index
axs[0].bar(range(d_model), x_sample.squeeze().detach().numpy(), color='skyblue')
axs[0].set_title(f'Step 0: Input (Size={d_model})', fontsize=12)
axs[0].set_xlabel('Input Dimension Index')
axs[0].set_ylabel('Value')
axs[0].axhline(y=0, color='r', linestyle='--', linewidth=0.8)

# --- Plot 1: Output of Linear Layer 1 (d_ff=16) ---
axs[1].bar(range(d_ff), output_fc1.squeeze().detach().numpy(), color='lightcoral')
axs[1].set_title(f'Step 1: Linear Layer 1 Output (Size={d_ff})', fontsize=12)
axs[1].set_xlabel('Hidden Dimension Index')
axs[1].set_ylabel('Value')
axs[1].axhline(y=0, color='r', linestyle='--', linewidth=0.8)

# --- Plot 2: Output of ReLU Activation (d_ff=16) ---
axs[2].bar(range(d_ff), output_relu.squeeze().detach().numpy(), color='orange')
axs[2].set_title('Step 2: ReLU Activation Output (Negative $\\rightarrow$ 0)', fontsize=12)
axs[2].set_xlabel('Hidden Dimension Index')
axs[2].set_ylabel('Value')
axs[2].axhline(y=0, color='r', linestyle='--', linewidth=0.8)

# --- Plot 3: Output of Linear Layer 2 (d_model=4) ---
# FIX: Use d_model for plotting index
axs[3].bar(range(d_model), output_fc2.squeeze().detach().numpy(), color='mediumseagreen')
axs[3].set_title(f'Step 3: Linear Layer 2 Output (Size={d_model})', fontsize=12)
axs[3].set_xlabel('Output Dimension Index')
axs[3].set_ylabel('Value')
axs[3].axhline(y=0, color='r', linestyle='--', linewidth=0.8)

# Apply consistent Y-axis limits for visual comparison
for ax in axs:
    ax.set_ylim(min_val - y_limit_buffer, max_val + y_limit_buffer)
    ax.tick_params(axis='x', rotation=45) # Rotate X-axis labels for d_ff=16

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
