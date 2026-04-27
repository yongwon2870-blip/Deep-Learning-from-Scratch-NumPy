# layers.py
import numpy as np
from common.functions import softmax, cross_entropy_error

class ReLu:
    def __init__(self):
        self.mask = None
        # self.mask : 순전파 때의 스위치 상태를 기억할 변수. True/False로 구성된 ndarray
        # True면 '차단(0 이하)', False면 '통과(0 초과)'를 의미

    def forward(self, x):    # [순전파] 입력 신호 x 중에서 0보다 작은 값을 cut, 0보다 큰 값은 그대로
        self.mask = (x <= 0) # 1단계: 0 이하인 숫자의 위치를 찾아 기록함
        out = x.copy()       # 2단계: 원본 데이터를 복사함
        out[self.mask] = 0   # 3단계: 기록해둔 위치의 숫자를 0으로 바꿈
        return out           # 4단계: 다음 층으로 보냄

    def backward(self, dout): # [역전파] 미분값을 순전파 때 신호를 보냈던 곳은 보내고, 신호를 끊었던 곳은 0으로 만듦
        dx = dout.copy()      # 1단계: 넘어온 피드백을 안전하게 복사함
        dx[self.mask] = 0     # 2단계: 순전파 때 차단했던 위치를 복사본에서도 0으로 만듦
        return dx             # 3단계: 이전 층으로 보냄

#----------------------------------------------------------------------------------------------------------------------------------------------

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        # Sigmoid 역전파 공식 유도 결과 : y(1-y)
        dx = dout * (1.0 - self.out) * self.out
        return dx

#----------------------------------------------------------------------------------------------------------------------------------------------

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.x_init_shape = None # 원래 데이터의 모양을 기억할 변수 추가

        self.dw = None
        self.db = None

    def forward(self, x):
        # 1. 원래 데이터의 모양을 기록
        self.x_init_shape = x.shape

        # 2. 데이터 개수(x.shape[0])는 유지하고, 나머지는 하나로 쭉 펴서 2차원으로 만듦
        x = x.reshape(x.shape[0], -1)
        self.x = x

        # 3. 행렬 곱셈
        out = x@(self.W) + self.b
        return out

    def backward(self, dout):
        dx = dout@(self.W.T)
        self.dW = (self.x.T)@dout
        self.db = np.sum(dout, axis = 0)

        # 4. shape 복구
        # 계산된 dx는 현재 2차원이므로, 이전 계층으로 돌려보내기 전에, 원래 모양으로 다시 복구하여 반환
        dx = dx.reshape(*self.x_init_shape)

        # 5. 원래 shape를 되찾은 dx(피드백)을 이전 계층으로 넘겨줌.
        return dx

#----------------------------------------------------------------------------------------------------------------------------------------------

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실함수
        self.y = None     # softmax의 출력
        self.t = None     # 정답 레이블(원-핫 인코딩 형태)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:                           # 정수 레이블 형태일 때
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
