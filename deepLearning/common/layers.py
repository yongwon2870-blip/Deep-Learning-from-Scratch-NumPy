import numpy as np
from common.functions import softmax, cross_entropy_error

# 1. 다차원 배열 전개 함수 (CNN Helpers)
def im2col(x, FH, FW, stride=1, pad=0):
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - FH) // stride + 1
    out_w = (W + 2 * pad - FW) // stride + 1

    # 1. 패딩 추가: 입력 이미지(x)의 상하좌우 가장자리에 0을 덧댐
    img = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

    # 2. 임시 공간 생성: 필터가 추출할 픽셀들을 담아둘 6차원 빈 배열
    col = np.zeros((N, C, FH, FW, out_h, out_w))

    # 3. 픽셀 추출: 필터를 이동시키며 맞닿는 픽셀들을 잘라내어 저장
    for y in range(FH):
        y_max = y + stride * out_h
        for x in range(FW):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 4. 2차원 표 압축: 축 순서를 바꾸고, 행렬 연산을 위해 2차원으로 평탄화(flatten)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col

def col2im(col, x_shape, FH, FW, stride=1, pad=0):
    N, C, H, W = x_shape
    out_h = (H + 2 * pad - FH) // stride + 1
    out_w = (W + 2 * pad - FW) // stride + 1

    # 1. 6차원 복원: 2차원 오차 배열을 6차원 임시 공간 형태로 쪼개고 축 순서 복구
    col = col.reshape(N, out_h, out_w, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)

    # 2. 빈 이미지 생성: 오차를 누적할 원본 픽셀 크기(패딩 포함)의 빈 배열 준비
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    # 3. 오차 누적: 필터가 겹치며 연산된 픽셀들의 오차를 모두 기존 값에 합산(+=)
    for y in range(FH):
        y_max = y + stride * out_h
        for x in range(FW):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    # 4. 패딩 제거: 처음에 덧댔던 여백(pad) 부분을 잘라내고 원래의 입력 이미지 크기만 반환
    return img[:, :, pad:H + pad, pad:W + pad]

# 2. 활성화 함수 및 출력 계층 (Activation & Output Layers)

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:  # 정수 레이블 형태일 때
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx

# 3. 완전연결 및 합성곱 계층 (Affine & CNN Layers)
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.x_init_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x_init_shape = x.shape
        x = x.reshape(x.shape[0], -1)  # 텐서를 2차원으로 변환
        self.x = x
        out = x @ self.W + self.b
        return out

    def backward(self, dout):
        dx = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.x_init_shape)  # 원래의 다차원 형태로 복구
        return dx

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # 1. 데이터 및 가중치를 2차원 행렬로 변환
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        # 2. 행렬 곱셈 연산 후 4차원으로 재조립
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 1. 풀링 영역만큼 데이터를 잘라 2차원 배열로 전개
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 2. 최댓값 추출 및 위치 기억
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))

        # 순전파 때 선택되었던 최댓값 위치에만 오차를 전달
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx