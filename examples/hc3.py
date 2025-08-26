from adaptesting import datasets

# CIFAR-10 adversarial images
hc3 = datasets.HC3(N=100, M=100)#, t1_check=True)
X, Y = hc3()

# Create a clean text comparison visualization
import matplotlib.pyplot as plt
import textwrap


# Example texts
human_text = X[0]  # First human-written text
machine_text = Y[0]  # First machine-generated text

print(human_text)
print(machine_text)
