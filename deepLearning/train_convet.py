# train_convet.py(학습 실행 파일)
# - SimpleConvNet(설계도)에 실제 데이터(MNIST)를 주입하여 학습을 진행

import numpy as np
import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

# 1. 데이터 읽기 및 준비
# CNN은 픽셀의 공간 정보(가로, 세로)가 필요하므로 flatten = False로 설정하여 (데이터 개수, 1채널, 28세로, 28가로)의 4차원 형태를 그대로 유지
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20 # 전체 데이터를 반복해서 볼 횟수

# 2. 신경망 객체 생성
network = SimpleConvNet(input_dim=(1,28,28), conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

# 3. 훈련 전담 클래스(Trainer) 설정 및 학습 시작
# Trainer는 내부적으로 미니배치를 추출하고, network.gradient()를 호출하여 기울기를 구한 뒤, Adam 최적화 기법을 사용하여 가중치 숫자를 갱신하는 반복문을 자동으로 수행.
trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001}, evaluate_sample_num_per_epoch=1000)

print("학습을 시작.")
trainer.train()

# 4. 학습된 가중치 영구 저장
network.save_params("params.pkl")
print("최적화된 가중치가 'params.pkl' 파일로 저장.")

# 5. 에폭(Epoch)별 정확도 변화 시각화 및 이미지 저장
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')

plt.savefig('cnn_accuracy_graph.png')
print("정확도 그래프가 'cnn_accuracy_graph.png'로 저장.")

plt.show()