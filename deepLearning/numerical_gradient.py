# numerical_gradient.py

import numpy as np
import sys, os
from common.functions import cross_entropy_error, softmax

def cross_entropy_error(y, t):
    # y : 예측 확률 배열(softmax 출력), t : 정답 라벨(one-hot encoding)

    # 1. 차원 통일 : 1차원 데이터가 입력되어도 2차원 배치(배치 크기 1)처럼 처리
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 2. 배치 크기 확인 : 데이터 1장당 평균 오차를 구하기 위한 전체 개수
    batch_size = y.shape[0]

    # 3. 오차 계산(one-hot encoding 방식) : 무한대(inf) 에러 방지
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # x와 같은 모양인 0으로 채워진 배열 생성

    # 1. 탐색 도구 생성: 배열의 모든 원소를 순차적으로 방문
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    # flags=['multi_index']: 탐색 중인 현재 위치의 다차원 좌표(행, 열)를 기억
    # op_flags=['readwrite']: 미분 계산을 위해 원본 데이터를 임시로 수정할 권한 부여

    # 2. 미분(중앙차분) 계산: 모든 원소를 돌며 f(x+h)와 f(x-h)를 구해 기울기 산출
    while not it.finished:
        idx = it.multi_index  # 현재 위치 (예: (0, 1))
        tmp_val = x[idx]  # 원본 값 임시 보관

        x[idx] = float(tmp_val) + h # f(x+h)
        fxh1 = f(x)

        x[idx] = float(tmp_val) - h # f(x-h)
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val # 계산 완료 후 원본 값으로 복구
        it.iternext() # 다음 좌표로 이동

    return grad

class SimpleNet:
    def __init__(self):
    # 2x3 형태의 가중치 행렬을 정규분포 난수로 초기화
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return x@self.W

    def loss(self,x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

# 임시 신경망(SimpleNet)의 가중치 미분값 계산 테스트
if __name__ == '__main__':
    net = SimpleNet()
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])

    # net.W 값을 변경할 때마다 loss 함수를 호출하도록 lambda 함수로 래핑
    f = lambda W: net.loss(x, t)
    dW = numerical_gradient(f, net.W)

    print(f"\n가중치 W에 대한 수치 미분 결과 : {dW}")
