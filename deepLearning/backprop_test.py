# backprop_test.py
# - 사과와 오렌지 쇼핑 예제를 통해 계산 그래프 및 연쇄 법칙(Chain Rule) 검증

class MulLayer:
    def __init__(self):
        # 역전파 시 사용할 입력값 보관용 변수
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x*y

        return out

    def backward(self,dout):
        # 곱셈 계층: 상류의 미분값(dout)에 순전파 시의 입력을 서로 바꿔 곱하여 하류로 전달
        dx = dout*self.y
        dy = dout*self.x

        return dx, dy

# ---------------------------------------------------------------------------------------------------------------------

class AddLayer:
    def __init__(self):
        pass           # 덧셈 계층(AddLayer)은 역전파를 계산할 때 처음에 입력되었던 숫자(x, y)를 기억할 필요가 없음

    def forward(self, x, y):
        out = x + y    # 들어온 숫자를 더하기만 하고 넘김

        return out

    def backward(self, dout):
        # 덧셈 계층: 상류의 미분값(dout)을 변형 없이 그대로 하류로 전달 (1 곱하기)
        dx = dout * 1
        dy = dout * 1

        return dx, dy

# ---------------------------------------------------------------------------------------------------------------------

# 1. 입력 데이터 준비
apple, apple_num = 100, 2
orange, orange_num = 150, 3
tax = 1.1

# 2. 계층(layer) 객체 생성
mul_apple_layer = MulLayer()        # 사과 총액
mul_orange_layer = MulLayer()       # 오렌지 총액
add_apple_orange_layer = AddLayer() # 두 과일값 합산
mul_tax_layer = MulLayer()          # 소비세 적용 최종액


# 순전파(Forward) : 가격 계산
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

print(f"최종 지불 금액 : {int(price)}원")

# 역전파(Backward) : 각 변수가 최종 가격에 미치는 영향(미분값) 계산
dprice = 1 # 최종 출력의 미분값은 항상 1로 시작. (Trigger)

# 1. 소비세 계층 (MulLayer): 입력값을 서로 바꿔 곱함
dall_price, dtax = mul_tax_layer.backward(dprice)

# 2. 합산 계층 (AddLayer): 상류의 미분값(dall_price)을 그대로 전달
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)

# 3. 오렌지 계층 (MulLayer): 입력값을 서로 바꿔 곱함
dorange, dorange_num = mul_orange_layer.backward(dorange_price)

# 4. 사과 계층 (MulLayer): 입력값을 서로 바꿔 곱함
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

# 결과 출력
print("--- 역전파(미분) 결과 ---")
print(f"사과 가격 미분: {dapple:.2f}, 사과 개수 미분: {dapple_num:.2f}")
print(f"오렌지 가격 미분: {dorange:.2f}, 오렌지 개수 미분: {dorange_num:.2f}")
print(f"소비세 미분: {dtax:.2f}")