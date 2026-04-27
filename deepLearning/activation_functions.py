# activation_functions.py

import numpy as np
import matplotlib.pyplot as plt

def identity_function(x):
    return x

def step_function(x):
    y = x>0
    return y.astype(int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)

plt.figure(figsize = (12,4))

plt.subplot(1,3,1)
plt.plot(x, step_function(x))
plt.title("Step Function")
plt.xlabel("x_input")
plt.ylabel("y_output")
plt.ylim(-0.1, 1.1)

plt.subplot(1,3,2)
plt.plot(x, sigmoid(x))
plt.title("Sigmoid Function")
plt.xlabel("x_input")
plt.ylabel("y_output")
plt.ylim(-0.1, 1.1)

plt.subplot(1,3,3)
plt.plot(x, relu(x))
plt.title("Relu Function")
plt.xlabel("x_input")
plt.ylabel("y_output")
plt.ylim(-1.0, 5.5)

plt.tight_layout()
plt.show()

