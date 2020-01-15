# ch03 p099
#### 1. 임의로 생성한 텐서들을 
#### 2. 근사하고자 하는 정답 함수에 넣어 정답(y)를 구하고
#### 3. 그 정답과 신경망을 통과한 y_hat과의 차이(loss)를 평균제곱오차(MSE)를 통해 구하여
#### 4. SGD를 통해 최적화
#### -------------------------------------------------------------------------

##### Modele class를 상속받아 완전 연결 계층을 만드는 class 만들기
```python
import torch
import torch.nn as nn


class MyModel(nn.Module):
    """ 완전 연결 계층 신경망 """
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()

        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        y = self.linear(x)
        return y
```

##### y_true를 계산할 함수 만들기
```python
def f(x):
    return 3 * x[:, 0] + x[:, 1] -2 * x[:, 2]
```

##### 손실(loss)계산, gradient계산, 오차역전파 --> 훈련
```python
def train(network, x, y, optimizer):
    """ 손실계산, gradient 계산, 오차역전파 -> 훈련 """
    optimizer.zero_grad()

    y_hat = network(x)                          # y 예측값
    loss = ((y - y_hat)**2).sum() / x.size(0)   # 손실 계산
    loss.backward()                             # 오차 역전파
    optimizer.step()                            # optimizer(ex, sgd, adam)를 통한 gradient 갱신
    return loss.data
```

##### 함수 test
```python
if __name__ == '__main__':
    batch_size = 1
    epoch = 1000
    iterator = 10_000

    network = MyModel(3, 1)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.0001, momentum=0.1)

    for epoch in range(epoch):
        avg_loss = 0
        for i in range(iterator):
            x = torch.rand(batch_size, 3)   # random_number.shape = (1, 3)
            y = f(x.data)
            loss = train(network, x, y, optimizer)
            avg_loss += loss
        avg_loss = avg_loss / iterator

        x_true = torch.FloatTensor([[.3, .2, .1]])
        y_true = f(x_true.data)                     # tensor의 값만 줄 때 tensor.data

        network.eval()                              # 모델 테스트
        y_hat = network(x_true)
        network.train()                             # 모델 훈련

        print(avg_loss, y_true.data[0], y_hat.data[0])
        if avg_loss < 0.001:
            break
```

#### -------------------------------------------------------------------------
#### 파이토치에서 딥러닝 수행 과정
##### 1. nn.Module 클래스를 상속받아(forward 함수를 통해) 모델 아키텍처 class 선언
##### 2. 해당 class 객체 생성
##### 3. optimizer를 생성(ex, SGD, Adam), 생성한 모델의 파라미터 갱신
##### 4. 데이터로 미니배치를 구성하여 피드포워드 연산 그래프 생성
##### 5. 손실 함수를 통해 최종 결과값과 손실 계산
##### 6. 손실에 대해서 backward() 호출 -> 연산 그래프 상의 텐서들의 gradient가 채워짐
##### 7. 3번의 optimizer에서 step()을 호출하여 경사하강법 수행
##### 8. 4번으로 돌아가 수렴조건(ex, loss < .001)이 만족할 때까지 반복 수행
