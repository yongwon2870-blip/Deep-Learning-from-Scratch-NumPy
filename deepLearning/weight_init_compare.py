# weight_init_compare.py

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# 1. 입력 데이터 생성(Input Signal)
x_input = np.random.randn(1000, 100)

# 2. 시스템 설정
node_num = 100
hidden_layer_size = 5

# 3. 4가지 실험 조건 설정(활성화 함수, 초기값, 함수, 표준편차 비율)
experiments = [
    ('Sigmoid', 'std = 1', sigmoid, lambda n: 1.0),             # 기울기 소실 유발
    ('Sigmoid', 'std = 0.01', sigmoid, lambda n: 0.01),         # 표현력 제한 유발
    ('Sigmoid', 'Xavier', sigmoid, lambda n: np.sqrt(1.0 / n)), # Sigmoid 최적
    ('ReLU', 'He', relu, lambda n: np.sqrt(2.0 / n)),           # ReLU 최적
]

plt.figure(figsize = (18,22))

# 4. 실험을 순차 진행 및 그래프 시각화
for exp_idx, (act_name, init_name, act_func, weight_scale) in enumerate(experiments):

    x = x_input.copy()
    activations = {}

    # 각 실험마다 5개(은닉층 개수)의 은닉층을 통과시킴
    # 첫 번째 층(i가 0일 때)이 아니라면, 이전 층(i-1)에서 계산되어 activations에 저장해 둔 결과값을 꺼내서 현재 층의 입력 데이터(x)로 덮어씀.
    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i-1]

        # 조건에 맞는 가중치 행렬 생성 및 계층 통과
        w = np.random.randn(node_num, node_num) * weight_scale(node_num)
        a = x@w
        z = act_func(a)

        # 현재 층의 번호(i)를 이름표로 삼아, 방금 계산한 출력값(z)을 딕셔너리에 저장.(다음 반복 때 위의 if문에서 다음 층의 입력 데이터로 쓰임)
        activations[i] = z

#-----------------------------------------------------------------------------------------------------------------------
        # 1. 그래프 위치 계산 및 설정
        plot_num = exp_idx * hidden_layer_size + (i+1)
        plt.subplot(4, hidden_layer_size, plot_num)

        if i == 0:
            plt.title(f"[{act_name} + {init_name}]\nLayer {i + 1}", fontsize=13, fontweight='bold')
        else:
            plt.title(f"Layer {i + 1}", fontsize=11)

        # 2. 히스토그램 그리기
        if act_name == 'ReLU':
            plt.hist(activations[i].flatten(), 30, range = (0, 5))
        else:
            plt.hist(activations[i].flatten(), 30, range = (0, 1))

        plt.xticks(fontsize = 8)
        plt.yticks(fontsize = 8)

plt.subplots_adjust(hspace = 0.6, wspace = 0.3)
plt.show()
