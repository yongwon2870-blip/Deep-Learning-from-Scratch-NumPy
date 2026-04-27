# optimizer_analysis.py

import numpy as np

# 확률적 경사 하강법(SGD)
# 현재 위치에서 가장 가파른 방향으로만 이동하여 지그재그 움직임이 발생.

class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    # 매개변수 갱신
    def update(self, params, grads):
        # params: 신경망의 모든 가중치와 편향이 담긴 딕셔너리 (W1, b1, W2, b2...)
        # grads: 각 가중치별 기울기가 담긴 딕셔너리 (dW1, db1, dW2, db2...)

        for key in params.keys():
            params[key] -= self.lr*grads[key]

#-----------------------------------------------------------------------------------------------------------------------

# Momentum
# SGD의 문제점
# : 기울기가 조금만 바뀌어도 방향을 급하게 튼다. -> 최적값에 도달하는데 시간이 오래 걸림.

# Momentum의 해결책(관성)
# : (공이 굴러오던)속도를 기억, 작은 기울기 변화를 만나도 관성으로 직진 -> 급하게 틀던 움직임은 줄어들고, 내려가는 속도는 증가

class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None # 아직 가중치 모양을 모르니 '미정' 상태로 둠

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                #  key 변수에는 'W1' / val 변수에는 W1의 가중치 행렬이 들어감
                self.v[key] = np.zeros_like(val)
                # 가중치(W)와 똑같은 모양의 '속도 기록'을 남김 - 처음에는 정지 상태이므로 0으로 채움.

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key] # 이전 속도에 관성을 주고, 기울기만큼 힘을 가함
            params[key] += self.v[key] # 위치를 속도만큼 이동

#-----------------------------------------------------------------------------------------------------------------------

# AdaGrad

# SGD의 문제점 : 매개변수마다 학습률을 다르게 적용함.
# AdaGrad의 해결책(적응형 학습): 크게 움직인(학습이 많이 된) 변수는 보폭을 줄이고, 적게 움직인 변수는 보폭을 키움.

class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            # 기울기의 제곱을 계속 누적
            self.h[key] += grads[key]**2
            # 누작된 값이 클수록 분모가 커져 실제 학습률(보촉)이 작아짐
            params[key] -= (self.lr*grads[key]) / (np.sqrt(self.h[key]) + 1e-7)

#-----------------------------------------------------------------------------------------------------------------------

# Adam

# Momentum: 핸들링을 부드럽게 해서 관성을 유지함 (방향 제어).
# AdaGrad: 길 상태(기울기)를 보고 엑셀을 밟을지 말지 결정함 (속도 제어).
# ---> Adam: 이 둘을 동시에

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1  # Momentum용 계수(방향 기억력, 0.9)
        self.beta2 = beta2  # AdaGrad용 계수(보폭 기억력, 0.999)
        self.m = None  # 1차 모멘텀(Momentum의 속도 역할)
        self.v = None  # 2차 모멘텀(AdaGrad의 '기울기 제곱' 역할)

        self.iter = 0  # 몇 번째 반복인지 세는 카운터

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1

        for key in params.keys():
            # 1. 방향 유지 - m 보정
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]

            # 2. 보폭 제어 - v 보정
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # 3. 초기 편향 보정
            m_init = self.m[key] / (1 - self.beta1 ** self.iter)
            v_init = self.v[key] / (1 - self.beta2 ** self.iter)

            # 4. 매개변수 최종 갱신
            params[key] = params[key] - self.lr * m_init / (np.sqrt(v_init) + 1e-7)

#-----------------------------------------------------------------------------------------------------------------------

# 최적화 기법(Optimizer)의 발전 계보
# 손실 함수(Loss Function)의 오차를 최소화하기 위해 매개변수 갱신 방식을 개선해 온 과정.

# 1. SGD (기본형)
#    - 특징: 현재 위치의 기울기만 보고 다음 갱신 방향을 결정.
#    - 한계: 매개변수마다 기울기의 가파른 정도가 다를 경우, 목표 지점을 향해 곧장 가지 못하고 상하좌우로 심하게 진동하며 비효율적으로 이동.

# 2. Momentum (방향의 개선)
#    - 특징: 과거의 매개변수 갱신 방향(속도)을 누적하여 다음 업데이트에 반영.
#    - 효과: 지그재그로 흔들리는 불필요한 진동은 더해지면서 상쇄되고, 일관된 진행 방향으로는 가속도가 붙어 SGD의 비효율성을 크게 줄임.

# 3. AdaGrad (보폭의 개선)
#    - 특징: 각 매개변수별로 갱신되는 크기(학습률)를 다르게 조절.
#    - 효과: 지금까지 값이 크게 변동했던 매개변수는 갱신 폭을 줄여 세밀하게 조정하고, 변동이 적었던 매개변수는 갱신 폭을 키워 학습이 빠르게 진행되도록 만듦.

# 4. Adam (완성형)
#    - 특징: Momentum의 '방향 누적'과 AdaGrad의 '개별 학습률 조절'을 융합한 기법.
#    - 결과: 진행 방향이 부드럽고 각 매개변수에 맞는 갱신 폭을 자동으로 찾아가므로, 현재 실무에서 가장 안정적이고 널리 쓰이는 표준 최적화 기법.