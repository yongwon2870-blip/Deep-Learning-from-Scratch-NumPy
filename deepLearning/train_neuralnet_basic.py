# train_neuralnet_basic.py

# 미니 배치(데이터 100장씩 뽑기) -> 수치 미분으로 기울기 계산 -> 가중치 숫자 업데이트 -> 오차 기록. 이 과정을 수만 번 반복하는 전체 흐름을 작성.

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 1. 데이터 준비
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 2. 하이퍼파라미터 설정
iters_num = 10000 # 총 반복 횟수
train_size = x_train.shape[0]
batch_size = 100  # 미니배치 크기
lr = 0.1          # 학습률(learning rate)

# 1epoch당 반복 횟수 계산
iters_per_epoch = max(train_size / batch_size, 1)

# 3. 신경망 객체 생성
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 4. 훈련 루프(수만 번 반복)
for i in range(iters_num):
    # 미니 배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.gradient(x_batch, t_batch)

    # 매개변수 갱신
    for key in network.params.keys():
        network.params[key] -= lr*grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1 epoch 반복할 때마다 오차와 정확도를 채점하고 출력
    if i%iters_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"반복: {i} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# 5. 그래프 시각화
plt.figure(figsize = (12,5))

# [첫 번째 칸] 오차 하강 그래프
plt.subplot(1,2,1)
plt.plot(train_loss_list)
plt.title("Loss Drop")
plt.xlabel("Iteration")
plt.ylabel("Loss")

# [두 번째 칸] 정확도 상승 그래프
plt.subplot(1,2,2)
x_acc = np.arange(len(train_acc_list))*iters_per_epoch
plt.plot(x_acc, train_acc_list, label = 'Train Acc', linestyle = '--')
plt.plot(x_acc, test_acc_list, label = 'Test acc')

plt.title("Accuracy Rise")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

