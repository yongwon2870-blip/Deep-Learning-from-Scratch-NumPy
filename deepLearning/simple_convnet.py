# simple_convet.py

import numpy as np
import sys, os
import pickle
from collections import OrderedDict

sys.path.append(os.pardir)

# 1. 계층 불러오기
from common.layers import Relu, Affine, SoftmaxWithLoss, Convolution, Pooling

class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):

        # 가중치 생성을 위한 크기 계산
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]

        # 출력 데이터 크기 계산
        conv_output_size = (input_size - filter_size + 2 * filter_pad) // filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # 가중치와 편향 초기화 및 저장
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)

        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)

        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 계층 조립
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    # 2. 순전파
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # 3. 오차 계산
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    # 4. 정확도 계산
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0

        # 데이터를 한 번에 넣으면 메모리가 터질 수 있으므로 batch_size만큼 잘라서 채점
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    # 5. 역전파(가중치 수정값 계산)
    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads

    # 6. 가중치 저장 및 불러오기
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        # 불러온 가중치를 순전파 연산 계층들 내부에 다시 덮어쓰기
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]

# 직전 단원의 신경망(다층 퍼셉트론)과 전체적인 조립 뼈대(predict, loss, gradient의 연쇄 흐름)가 같음.
# 두 신경망은 '오차를 계산하고 역전파하여 가중치를 수정한다'는 딥러닝의 기본 수학적 엔진을 완벽하게 공유.
# but 내부에서 배열을 다루는 방식과 계층의 구성에서 결정적인 3가지 차이가 존재.

#   1. 데이터의 형태(Shape) 유지 (공간 정보의 활용)
#      이전 신경망 : 이미지를 입력받을 때 flatten=True를 사용하여 (28 x 28) 이미지를 784개의 1차원 가로줄로 쫙 펴서 입력. 이 과정에서 픽셀들의 상하좌우 위치 관계(공간 정보)가 완전히 파괴.
#      CNN : 이미지를 1차원으로 펴지 않고 (1, 28, 28) 형태의 4차원 다차원 배열 구조를 끝까지 유지. 필터가 이미지 위를 2차원적으로 이동(im2col)하며 연산하기 때문에, 픽셀의 상하좌우 관계를 그대로 보존하면서 패턴을 인식할 수 있음.

#   2. 연산 계층(Layer)의 역할 분업
#      이전 신경망 : 처음부터 끝까지 Affine (완전연결) 계층만 사용. 입력된 모든 픽셀 숫자들을 다음 노드들과 한 번에 전부 곱해서 점수를 계산하는 단일 구조
#      CNN : 철저한 분업 구조
#       - 특징 추출 파트 (Conv + Pool) : 먼저 이미지에서 윤곽선이나 특징적인 패턴을 뽑아내고, 풀링으로 데이터 크기를 압축
#       - 최종 분류 파트 (Affine) : 앞에서 잘 정제되고 압축된 핵심 특징 데이터만 넘겨받아, 최종적으로 0~9 중 어떤 숫자인지 확률 점수를 매기는 역할만 수행

#   3. 가중치(W)의 개수와 재사용 방식
#      이전 신경망 : 784개의 픽셀이 100개의 노드와 모두 개별적으로 연결되려면 784x100 = 78,400개의 거대한 가중치 행렬이 필요. 연산량이 엄청나고 메모리도 많이 차지.
#      CNN : 5x5 크기의 아주 작은 필터(가중치)를 만들어, 이를 이미지 전체에 훑고 지나가며 반복해서 재사용(Parameter Sharing). 덕분에 학습해야 할 가중치 숫자가 획기적으로 줄어들어 연산이 훨씬 빠르고 과적합(Overfitting)도 덜 발생