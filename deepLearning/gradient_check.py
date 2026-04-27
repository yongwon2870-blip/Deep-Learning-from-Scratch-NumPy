# gradient_check.py

import numpy as np
import sys, os

# 부모 폴더의 파일(dataset, common 등)을 가져오기 위한 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset.mnist import load_mnist
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 1. 가중치 초기화
        self.params = {}

        # 가중치(W): 대칭성 파괴(Symmetry Breaking)를 위해 정규분포 난수로 초기화
        # weight_init_std: 값이 과도하게 커져 활성화 함수에서 0이나 1로 치우치는 현상을 방지
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)

        # 편향(b): 데이터의 치우침이 없도록 0으로 초기화
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)

        # 2. 계층(layer) 생성
        self.layers = OrderedDict()
        # OrderedDict 사용 목적: 순전파/역전파 시 계층이 추가된 순서(또는 역순)를 정확히 보장하기 위함

        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        # 각 행(데이터)마다 확률이 가장 높은 클래스의 인덱스 추출
        y = np.argmax(y, axis=1)

        # 정답 레이블이 원-핫 인코딩 형태일 경우, 비교를 위해 인덱스 형태로 변환
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        # 예측과 정답이 일치하는 비율 계산
        accuracy = np.sum(y == t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        # 외부 수치 미분 함수와의 인터페이스를 맞추기 위한 람다(lambda) 함수 래핑
        loss_W = lambda W : self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # 1. 순전파 - 역전파 계산을 위해 중간 데이터(각 계층의 x 등)를 메모리에 기록
        self.loss(x, t)

        # 2. 역전파 시작: 최종 출력의 오차(미분값)는 1에서 출발
        dout = 1
        dout = self.lastLayer.backward(dout)

        # 3. 계층 역순 정렬 및 피드백 전달
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)
            # 과정 중 Affine 계층은 내부적으로 dW, db를 계산하여 저장

        # 결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db


        return grads

# numerical_grads & backprop 비교
if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    # 연산 시간 단축을 위해 3개의 데이터만 추출하여 테스트
    x_batch, t_batch = x_train[:3], t_train[:3]

    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    print('수치 미분vs 역전파 기울기 차이')
    for key in grad_numerical.keys():
        # 각 가중치 배열 간의 절대 오차 평균 계산
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(f"{key}의 오차 : {diff}")

#-----------------------------------------------------------------------------------------------------------------------

# 수치 미분(Numerical Gradient) vs 오차역전파법(Backpropagation)
#  - 두 방식은 기울기를 구한다는 목적은 같지만, 실무에서의 역할은 완전히 다름

# 1. 연산 속도와 효율성
#    - 수치 미분: 매개변수가 100만 개라면, 손실 함수를 200만 번(위로 한 번, 아래로 한 번)이나 실행해야 기울기를 구할 수 있다.
#                -> 사실상 딥러닝 학습에는 사용이 불가능할 정도로 느림.

#    - 오차역전파법: 순전파와 역전파를 딱 1번씩만 통과시키면 연쇄 법칙(Chain Rule)에 의해 100만 개의 미분값이 한 번에 출력.
#                   -> 연산 속도가 기하급수적으로 빠름.

# 2. 정확도 (해석적 해 vs 근사치)
#    - 수치 미분: h(1e-4)라는 아주 작은 값을 이용한 '근사치'이므로 수학적 오차가 필연적으로 발생.
#    - 오차역전파법: 수학의 미분 공식을 코드로 그대로 옮긴 것이므로, 오차가 없는 100% 정확한 '해석적 해(정답)'를 구함.

# 3. 수치 미분의 진짜 존재 이유(Gradient Check)
#    - 오차역전파법은 빠르고 정확하지만, 수식이 조금만 틀려도 버그를 찾기 매우 어렵다는 치명적인 단점이 존재.
#    - but 수치 미분은 느리지만 코드가 아주 단순해서 버그가 섞일 위험이 거의 없다.
#    -  -> 복잡하게 짠 '역전파 코드'가 수학적으로 완벽한지 검증하기 위해, 믿을 수 있는 '수치 미분 결과'와 비교해보는 용도로 사용.(Gradient Check)


