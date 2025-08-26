from adaptesting import datasets

# CIFAR-10 adversarial images
adv = datasets.CIFAR10Adversarial(N=100, M=100, attack_method='PGD')#, t1_check=True)
X, Y = adv()

import matplotlib.pyplot as plt

# Plot one original and one adversarial image
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(X[0].permute(1, 2, 0))  # Change from CxHxW to HxWxC for plotting
plt.title('Original Image')
plt.axis('off')

# Adversarial image  
plt.subplot(1, 2, 2)
plt.imshow(Y[0].permute(1, 2, 0))  # Change from CxHxW to HxWxC for plotting
plt.title('Adversarial Image')
plt.axis('off')

plt.tight_layout()
plt.savefig('cifar_adv_plot.png', bbox_inches='tight')

plt.show()
