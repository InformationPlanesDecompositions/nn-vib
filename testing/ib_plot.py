# ib_plot_simple.py
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Fake (but realistic-looking) IB curves
# -------------------------------------------------
comp = np.linspace(0.1, 4.0, 150)                 # I(X; Xhat)

# Empirical curve (black) – what we see on the training data
rel_emp = 1.5 * (1 - np.exp(-1.8 * comp))
rel_emp += 0.03 * np.random.randn(len(comp))      # tiny jitter

# Worst-case upper bound (red) – empirical + error that grows with K
error = 0.9 * np.exp(0.7 * comp) / np.sqrt(10_000) * 120
rel_bound = rel_emp + error

# Maximum possible relevance
H_Y = 2.0                                         # e.g. log2(4) bits

# Optimal safe point = lowest point on the red curve
i_opt = np.argmin(rel_bound)
R_star = comp[i_opt]
I_star = rel_bound[i_opt]

# -------------------------------------------------
# Plot
# -------------------------------------------------
plt.figure(figsize=(9, 6))

plt.plot(comp, rel_emp,   'k-',  linewidth=3, label='Empirical IB Curve')
plt.plot(comp, rel_bound, 'r--', linewidth=3, label='Worst-case Bound (Red)')

plt.plot(R_star, I_star, 'y*', markersize=18,
         label=f'Optimal Safe Point ({R_star:.2f}, {I_star:.2f})')

# Labels (raw strings → no escape warnings)
plt.xlabel(r'Compression: $I(X;\hat{X})$ [bits]', fontsize=13)
plt.ylabel(r'Relevance: $I(\hat{X};Y)$ [bits]',   fontsize=13)
plt.title('Information Bottleneck Curve\n(with Generalization Bounds)', fontsize=15)

# Limits
plt.axhline(H_Y, color='gray', linestyle=':', linewidth=2,
            label=f'Max Relevance = $H(Y)$ = {H_Y}')
plt.axvline(0, color='gray', linestyle='-', alpha=0.5)

plt.xlim(0, 4.1)
plt.ylim(0, 2.6)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# -------------------------------------------------
# Save + (optionally) show
# -------------------------------------------------
plt.tight_layout()
plt.savefig('ib_curve.png', dpi=300)   # high-res PNG
print("Figure saved as 'ib_curve.png'")

# Uncomment the next line if you are in an interactive environment:
# plt.show()
