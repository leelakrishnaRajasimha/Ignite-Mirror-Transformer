import matplotlib.pyplot as plt

# Your actual training losses
epochs = list(range(1, 16))
losses = [2.5694, 0.8064, 0.2478, 0.0775, 0.0253, 0.0093, 
          0.0049, 0.0026, 0.0017, 0.0023, 0.0052, 0.0025, 
          0.0014, 0.0013, 0.0021]

plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o', linestyle='-', color='#1f77b4', linewidth=2)

# Adding labels to make it look research-standard
plt.title("Mirror Transformer: Training Loss Convergence", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Cross-Entropy Loss", fontsize=12)
plt.yscale('log') # Log scale shows the small improvements at the end better
plt.grid(True, which="both", ls="-", alpha=0.5)

# Save the file
plt.savefig("loss_curve.png", dpi=300)
print("Graph saved successfully as 'loss_curve.png'")