from adaptesting import datasets

# CIFAR-10 adversarial images
# adv = datasets.CIFAR10Adversarial(N=100, M=100, attack_method='PGD')#, t1_check=True)
cifar10_1 = datasets.CIFAR10_1(N=500, M=500)
X, Y = cifar10_1()

import matplotlib.pyplot as plt

# Create a beautiful visualization with 25 original and 25 adversarial images
fig, axes = plt.subplots(5, 10, figsize=(20, 10))
# fig.suptitle('CIFAR-10: Original Images (Left) vs Adversarial Images (Right)', fontsize=16, fontweight='bold')

# Remove axes and adjust spacing
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.92)

# Plot original images (left 5x5 grid)
for i in range(5):
    for j in range(5):
        idx = i * 5 + j
        img = X[idx].permute(1, 2, 0).cpu().numpy()  # Convert from (C,H,W) to (H,W,C)
        axes[i, j].imshow(img)
        # axes[i, j].set_title(f'Original {idx+1}', fontsize=8, pad=2)

# Plot adversarial images (right 5x5 grid)
for i in range(5):
    for j in range(5):
        idx = i * 5 + j
        img = Y[idx].permute(1, 2, 0).cpu().numpy()  # Convert from (C,H,W) to (H,W,C)
        axes[i, j+5].imshow(img)
        # axes[i, j+5].set_title(f'Adversarial {idx+1}', fontsize=8, pad=2)

# Add section labels
fig.text(0.25, 1.0, 'Original Images', ha='center', fontsize=14, fontweight='bold', color='blue')
# fig.text(0.75, 1.0, 'Adversarial Images', ha='center', fontsize=14, fontweight='bold', color='red')
fig.text(0.75, 1.0, 'CIFAR-10.1 Images', ha='center', fontsize=14, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('cifar10_1_plot.png', dpi=300, bbox_inches='tight')
plt.show()

