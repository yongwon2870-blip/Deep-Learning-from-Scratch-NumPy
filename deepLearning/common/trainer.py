import numpy as np

# [핵심 수정 포인트] 우리가 만든 optimizer 파일에서 클래스들을 직접 불러옵니다.
from common.optimizer import *

class Trainer:
# 신경망 훈련 전담 클래스
# - 역할: 데이터(미니배치) 추출, 역전파(기울기 계산), 가중치 갱신, 정확도 채점의 반복 주기를 자동 수행

    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr': 0.01},
                 evaluate_sample_num_per_epoch=None):

        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # 최적화 기법(Optimizer) 매핑 딕셔너리 구성
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'adagrad': AdaGrad, 'adam': Adam}

        # 사용자가 입력한 문자열('Adam' 등)을 소문자로 변환하여 딕셔너리에서 찾아 객체 생성
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        # 훈련 데이터 총 개수 및 1 에폭(epoch)당 반복 횟수 계산
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        # 훈련 결과 기록용 리스트
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        # 1. 미니배치 데이터 무작위 추출
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 2. 기울기 계산 (오차역전파법 수행)
        grads = self.network.gradient(x_batch, t_batch)

        # 3. 매개변수 갱신 (선택된 Optimizer 작동)
        self.optimizer.update(self.network.params, grads)

        # 4. 오차 기록
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)

        # 5. 1 에폭이 지날 때마다 평가(채점) 진행
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test

            # (속도 최적화) 데이터가 너무 많을 경우, 전체가 아닌 일부 샘플만 뽑아서 채점
            if self.evaluate_sample_num_per_epoch is not None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)

            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            print(f"=== epoch:{self.current_epoch:02d}, train acc:{train_acc:.4f}, test acc:{test_acc:.4f} ===")

        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        # 학습 완료 후 최종 정확도 출력
        test_acc = self.network.accuracy(self.x_test, self.t_test)
        print("=============== 학습 종료 ===============")
        print(f"최종 테스트 정확도 (Test Accuracy): {test_acc:.4f}")