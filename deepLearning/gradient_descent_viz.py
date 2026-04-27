# gradient_descent_viz.py

import numpy as np
import matplotlib.pyplot as plt
from common.gradient import numerical_gradient

def gradient_descent(f, x_init, lr=0.01, step_num=100):
    x = x_init
    x_history = [] # 점이 이동한 궤적을 기록할 빈 보관함

    for i in range(step_num):
        x_history.append(x.copy())      # 현재 위치를 복사해서 보관함에 기록
        grad = numerical_gradient(f, x) # 기울기 계산
        x -= lr * grad                  # 위치 업데이트

    return x, np.array(x_history)

def test_function(x):
    return x[0]**2 + x[1]**2

if __name__ == '__main__':
    # 시작 위치 설정 (-3.0, 4.0)
    init_x = np.array([-3.0, 4.0])

    # 1. 정상적인 학습 (lr = 0.1)
    x_final_normal, x_hist_normal = gradient_descent(test_function, x_init=init_x.copy(), lr=0.1, step_num=100)

    # 2. 학습률이 너무 클 때 (발산)
    x_final_large, x_hist_large = gradient_descent(test_function, x_init=init_x.copy(), lr=10.0, step_num=100)

    # 3. 학습률이 너무 작을 때 (정체)
    x_final_small, x_hist_small = gradient_descent(test_function, x_init=init_x.copy(), lr=1e-10, step_num=100)

    # 그래프 그리기
    plt.figure(figsize=(15, 5))

    # --- [첫 번째 칸] 정상 학습 (파란색) ---
    plt.subplot(1, 3, 1) # (1줄, 3칸 중, 1번째 칸)
    plt.plot([-5, 5], [0,0], '--b'); plt.plot([0,0], [-5, 5], '--b')
    plt.plot(x_hist_normal[:,0], x_hist_normal[:,1], 'o-') # 'o-' : 점과 점 사이를 선으로 잇기
    plt.xlim(-3.5, 3.5); plt.ylim(-4.5, 4.5)
    plt.title("1. Normal (lr=0.1)")
    plt.xlabel("x0"); plt.ylabel("x1")

    # --- [두 번째 칸] 학습률이 너무 클 때 (빨간색) ---
    plt.subplot(1, 3, 2) # (1줄, 3칸 중, 2번째 칸)
    plt.plot([-5, 5], [0,0], '--b'); plt.plot([0,0], [-5, 5], '--b')
    plt.plot(x_hist_large[:,0], x_hist_large[:,1], 'ro-')
    # 숫자가 너무 커져서 화면 밖으로 튕겨 나가므로 xlim, ylim으로 화면을 고정하지 않음
    plt.title("2. Too Large (lr=10.0)")
    plt.xlabel("x0"); plt.ylabel("x1")

    # --- [세 번째 칸] 학습률이 너무 작을 때 (초록색) ---
    plt.subplot(1, 3, 3) # (1줄, 3칸 중, 3번째 칸)
    plt.plot([-5, 5], [0,0], '--b'); plt.plot([0,0], [-5, 5], '--b')
    plt.plot(x_hist_small[:,0], x_hist_small[:,1], 'go-')
    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.title("3. Too Small (lr=1e-10)")
    plt.xlabel("x0"); plt.ylabel("x1")

    plt.tight_layout()
    plt.show()