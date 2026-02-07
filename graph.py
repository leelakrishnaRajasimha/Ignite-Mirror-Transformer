import matplotlib.pyplot as plt

# Your actual training losses
epochs = list(range(1, 31))
losses = [2.5325, 0.7898, 0.2420, 0.0755, 0.0242, 0.0124, 0.0058, 0.0031, 0.0018, 0.0018, 0.0018, 0.0040, 0.0039, 0.0022, 0.0013, 0.0010, 0.0011, 0.0014, 0.0016, 0.0017, 0.0028, 0.0018, 0.0018, 0.0022, 0.0016, 0.0013, 0.0016, 0.0015, 0.0008, 0.0014]

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