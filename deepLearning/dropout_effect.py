# dropout_effect.py
# - 오버피팅을 막기 위해 학습할 때마다 노드(뉴런)를 무작위로 몇 개씩 끄면서 학습시키는 기술

import numpy as np

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None  # 어떤 노드를 on/off하는지 기록해둘 빈 공간

    def forward(self, x, train_flg=True):
        # 1) 학습 중일 때 (train_flg = True)
        if train_flg:
            # 입력 데이터(x)와 똑같은 모양으로 0.0~1.0 사이의 난수(무작위 숫자)를 생성
            # 그 숫자가 dropout_ratio(예: 0.5)보다 큰지 작은지 판단해서 True(살림) 또는 False(끔)로 mask에 기록

            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            # 입력 데이터 x에 mask를 곱한다. True(1)인 자리는 데이터가 그대로 통과, False(0)인 자리는 0이 되어 데이터가 지워짐
            return x * self.mask

        # 2) 실제 평가/테스트 중일 때 (train_flg = False)
        else:
            # 평가 시: 모든 노드를 켜되, 학습 시의 출력 크기와 맞추기 위해 비율을 곱함.
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        # 앞에서 데이터가 통과할 때 살려두었던(True) 노드에만 피드백(dout)을 그대로 전달하고, 꺼버렸던(False) 노드에는 피드백도 0으로 만들어 전달을 차단.
        return dout * self.mask
