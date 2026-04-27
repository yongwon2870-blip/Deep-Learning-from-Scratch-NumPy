# neuralnet_forward.py

import numpy as np
from activation_functions import sigmoid, identity_function

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    print("데이터 형태(Shape) 변화 추적")
    print(f"입력 데이터 x : {x.shape}")

    a1 = x@W1 + b1
    z1 = sigmoid(a1)
    print(f"1층 통과 후 z1 : {z1.shape}")
    a2 = z1@W2 + b2
    z2 = sigmoid(a2)
    print(f"2층 통과 후 z2 : {z2.shape}")
    a3 = z2@W3 + b3

    y = identity_function(a3)
    print(f"최종 출력 y   : {y.shape}")
    print("===============================================\n")

    return y

if __name__ =='__main__':
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)

    print(f"최종 계산 결과값 : {y}")

