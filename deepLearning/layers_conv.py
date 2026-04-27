# layers_conv.py
import numpy as np

# im2col : 다차원 이미지를 컴퓨터가 계산하기 가장 빠른 2차원 표(행렬)로 전개하는 함수
def im2col(x, FH, FW, stride=1, pad=0):
    N, C, H, W = x.shape
    out_h = (H + 2*pad - FH)//stride + 1
    out_w = (W + 2*pad - FW)//stride + 1

    # 1. 패딩 추가: 입력 이미지(x)의 상하좌우 가장자리에 0을 덧댐
    img = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

    # 2. 임시 공간 생성: (데이터 수, 채널, 필터 세로, 필터 가로, 출력 세로, 출력 가로)
    # 필터가 이동하며 추출할 픽셀들을 독립적으로 담아둘 6차원 빈 배열
    col = np.zeros((N, C, FH, FW, out_h, out_w))

    # 3. 픽셀 추출: 필터 내의 픽셀 위치(y, x)를 기준으로, 이미지 전체에서 맞닿는 픽셀들을 한 번에 잘라내어 저장
    for y in range(FH):
        y_max = y + stride*out_h
        for x in range(FW):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 4. 2차원 표 압축: 축 순서를 (N, out_h, out_w, C, FH, FW)로 바꾼 뒤, 행렬 연산을 위해 2차원으로 flatten
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

#-----------------------------------------------------------------------------------------------------------------------

# col2im : 역전파된 2차원 오차 배열을 다시 원래의 다차원 이미지 형태로 복원하는 함수
def col2im(col, x, FH, FW, stride=1, pad=0):
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - FH) // stride + 1
    out_w = (W + 2 * pad - FW) // stride + 1

    # 1. 6차원 복원: 2차원 오차 배열을 원래의 6차원 임시 공간 형태로 쪼개고 축 순서 복구
    col = col.reshape(N, out_h, out_w, C, FH, FW).transpose(0, 3, 4, 5, 1, 2) # 기존의 col : (N*out_h*out_w, C*FH*FW)

    # 2. 빈 이미지 생성: 오차를 누적할 원본 픽셀 크기(패딩 포함)의 빈 배열 준비
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    # 3. 오차 누적: 필터가 겹치며 연산된 픽셀들의 오차를 모두 기존 값에 더함(+=)
    for y in range(FH):
        y_max = y + stride * out_h
        for x in range(FW):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        # 4. 패딩 제거: 처음에 덧댔던 여백(pad) 부분을 잘라내고 원래의 입력 이미지 크기(H, W)만 반환
        return img[:, :, pad:H + pad, pad:W + pad]

#-----------------------------------------------------------------------------------------------------------------------

class Convolution:
    def __init__(self, W, b,stride = 1, pad = 0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 역전파(backward) 연산 시 사용할 중간 데이터 저장용 변수
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        # 1. 데이터 밒 가중치를 2차원 행렬로 변환(im2col 활용)
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        # 2. 행렬 곱셈 연산
        out = col@col_W + self.b

        # 3. 출력 데이터를 딥러닝 표준 규격인 4차원(N,C,H,W)으로 재조립
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape

        # 역전파 신호(dout)을 2차원으로 flatten
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # 편향(b)과 가중치(W)의 기울기 계산
        self.db = np.sum(dout, axis = 0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # 이전 계층으로 넘겨줄 입력 데이터의 오차(dcol) 계산 후 원래 4차원(col2im)으로 복구
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x, FH, FW, self.stride, self.pad)

        return dx

#-----------------------------------------------------------------------------------------------------------------------

class Pooling:
    def __init__(self, pool_h, pool_w, stride = 2, pad = 0):
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

        # 2. 역전파를 위해 최댓값 위치를 기억하고, 실제 최댓값 추출 (가로줄 단위)
        arg_max = np.argmax(col, axis = 1)
        out = np.max(col, axis = 1)

        # 3. 4차원 형태로 복구
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        # 1차원 배열로 flatten
        dout = dout.transpose(0,2,3,1)

        # 풀링 크기만큼의 빈 2차원 배열(dmax) 생성
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))

        # 순전파 때 가장 컸던 위치(arg_max)에만 오차를 그대로 전달하고, 나머지는 0으로 둠
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        # 2차원 배열을 다시 6차원으로 쪼개고 순서 재배치
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)

        # 원래의 4차원 형태로 복구하여 이전 계층으로 전달
        dx = col2im(dcol, self.x, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


