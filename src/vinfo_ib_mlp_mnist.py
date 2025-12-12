import torch
import torch.nn as nn

# --- 1a. The Encoder (The LeNet-300-100 body) ---
# Maps X (image) to Z (representation)
class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        # LeNet-300-100 structure (Fully Connected Layers)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 300), # 784 = 28*28 for MNIST
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, z_dim) # Output layer: Z
        )

    def forward(self, x):
        # We can optionally add Gaussian noise here for standard VIB
        return self.encoder(x)

# --- 1b. The Observer (The V_Y Predictive Family) ---
# Maps Z (representation) to Y' (prediction)
class Observer_Vy(nn.Module):
    def __init__(self, z_dim, num_classes=10):
        super().__init__()
        # THE CORE CONSTRAINT: V_Y is just a simple linear layer
        self.predictor = nn.Linear(z_dim, num_classes)

    def forward(self, z):
        return self.predictor(z) # Logits/Scores

# Initialize the components
# z_dim is the size of the bottleneck layer Z
z_dim = 20
encoder = Encoder(z_dim)
observer = Observer_Vy(z_dim)

# 2. V-IB LOSS DEFINITION

def vib_loss_compression(z_mean, z_log_var):
    # This is the standard KL Divergence term from Variational Information Bottleneck (VIB)
    # It acts as a bound on I(X; Z) and encourages compression towards a standard Gaussian prior.
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return kl_loss

def v_ib_loss(x, y, encoder, observer, beta):
    # Step 1: Forward pass through the Encoder
    # Assume the encoder outputs the mean and log-variance for a VAE-style VIB
    # (If using Deterministic IB, the encoder just outputs Z directly)
    z_mean = encoder(x)
    z_log_var = torch.zeros_like(z_mean) # Deterministic example

    # Step 2: Sample Z (or use Z_mean for deterministic case)
    z = z_mean

    # Step 3: Compute Relevance Term (V-Information)
    # Maximizing I_V(Z -> Y) is Minimizing the V-Entropy H_V(Y|Z)
    logits = observer(z)
    relevance_loss = nn.functional.cross_entropy(logits, y) # Standard prediction loss

    # Step 4: Compute Compression Term
    compression_loss = vib_loss_compression(z_mean, z_log_var)

    # Step 5: Total V-IB Loss
    total_loss = relevance_loss + beta * compression_loss

    return total_loss

# 3. TRAINING LOOP (Conceptual)

# Hyperparameter for the Information Bottleneck trade-off
beta = 0.001
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(observer.parameters()))

for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        # Zero gradients
        optimizer.zero_grad()

        # Compute the V-IB Loss
        loss = v_ib_loss(x_batch, y_batch, encoder, observer, beta)

        # Backpropagate and update
        loss.backward()
        optimizer.step()

        # ... (Log loss, calculate accuracy using the observer(encoder(X)) prediction)
