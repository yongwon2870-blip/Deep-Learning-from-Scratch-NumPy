# visualize_filter.py

import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet

def filter_show(filters, nx=8):
    # 4차원 배열인 합성곱 필터를 2차원 이미지 그리드 형태로 변환하여 화면에 출력하는 함수
    # filters: (필터 개수, 채널 수, 세로 크기, 가로 크기) 형태의 4차원 배열, nx(number of axis): 가로로 나열할 필터의 개수

    FN, C, FH, FW = filters.shape # (30, 1, 5, 5)
    ny = int(np.ceil(FN / nx))
    # 필터를 가로로 8개씩 그릴 때, FN(30)개의 필터를 다 그리려면 세로로 몇 줄(ny)이 필요한지 계산. (30/8 = 3.75)
    # np.ceil(반올림 함수) 3.75 -> 4.0

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i + 1, xticks=[], yticks=[])
        # 전체를 세로 4줄(ny), 가로 8칸(nx)으로 나눈 뒤, 그 중 몇 번째 칸에 그림을 그릴지 구역(ax)을 할당.
        # 동시에 픽셀 그림 주변에 나타나는 x츅, y축 눈금 숫자를 지워버림.

        ax.imshow(filters[i, 0], cmap= plt.cm.gray_r, interpolation='nearest')
        # filter의 i번째 덩어리에서 0번째 채널을 골라내면 뒤의 세로/가로 정보만 남아 2차원 표 1개가 추출
        # plt.cm.gray_r : matplotlb에 내장된 'Color Map'이라는 다른 객체를 연결
        # 선택된 칸(ax)에 실제 5x5 배열 데이터를 이미지로 그림(imshow). 배열의 숫자가 크면 하얀색, 크면 검은색으로 변환(gray_r)
        # 픽셀 사이의 경계를 흐리게 뭉개지 않고 뚜렷한 모자이크 블록처럼 표현(nearest)

    plt.show()

if __name__ == '__main__':
    # 1. 학습 전 (무작위 초기화 상태) 필터 시각화
    print("학습 전의 무작위 가중치(노이즈)를 출력. 창을 닫으면 다음 결과가 나옴.")
    network = SimpleConvNet()
    filter_show(network.params['W1'])  # 네트워크를 생성한 직후의 가중치(W1)는 np.random.randn에 의해 무작위 소수로 채워져 있음.

    # 2. 학습 후 (특징 추출 상태) 필터 시각화
    print("학습된 가중치(특징 패턴)를 출력.")
    network.load_params("params.pkl")  # 앞서 train_convnet.py 실행으로 만들어진 params.pkl 파일을 불러와 가중치를 덮어씌움.

    # 오차 역전파를 통해 갱신된 최적의 가중치(W1)를 확인.
    filter_show(network.params['W1'])

    # 동일한 신경망 객체를 두고, '무작위 상태의 숫자 배열'을 한 번 출력하고, 하드디스크에서 '최적화된 숫자 배열'을 주입한 뒤 다시 한번 출력하여,
    # 미분과 행렬 곱셈이 만들어낸 수학적 최적화의 결과물을 인간의 눈으로 비교 및 검증하는 완벽한 실험 구조를 구성
