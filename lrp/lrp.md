# [LRP](Layer-wise Relevance Propagation)

**L** ayer-wise    **(레이어 단위)** *로* <br>
**R** elevance    *결과에 영향을 주는 **(관련성)** 을 구하는* <br>
**P** ropagation  *역 **(전파)** 기술* <br>

즉, 입력 데이터 관점에서 **분류 결과** 뿐만 아니라 **결정에 영향** 을 미치는 구조를 설명
## 1. Introduction 
<center> <img src="./img/intro.png" width="500" height="350"/> </center>

- 뉴럴네트워크의 동작을 이해하기 위한 연구들은 크게 두 종류로 나눌 수 있다. 
- 첫 번째는 **모델 자체를 해석하는 방법** 이고, 두 번째는 **‘왜 그런 결정을 내렸는지’ 파악하는 방법** 이다.
- Layer-wise Relevance Propagation (이하 LRP)는 두 번째 종류인 ‘왜 그런 결정을 내렸는지’ 파악하는 방법에 속하며, 그 중에서도 decomposition을 이용한 방법이다.

## 2. How?
![lrp_example](./img/example_1_rooster.png)

- 잘 훈련된 네트워크에 input(x):수탉 사진/ouput(f(x)):'수탉'이 경우, 이 '수탉'이라는 출력 f(x)를 얻기 위해 입력 샘플의 각 pixel들이 기여하는 바를 계산하는 방법
- 아래의 그림1에서 보이는 것처럼 heatmap이라고 적힌 그림에 pixel들의 기여도(relevance score)가 색깔로 표시되며, 수탉의 부리나 머리 등을 보고 해당 입력의 클래스가 '수탉'임을 출력했다는 것을 알 수 있다.

### LRP의 기본적인 가정 및 작동 방식
<center> <img src="./img/relevance_propagation.png" width="500" height="350"/> </center>

LRP(Layer-wise Relevance Propagation)의 이름에서 볼 수 있듯이 이 method는 relevance score를 출력단에서 입력단 방향으로 top-down 방식으로 기여도를 재분배 하는 방법이다.
<br>
~~~
## 각 뉴런은 어느 정도의 기여도(certain relevance)를 갖고 있다.
## 기여도는 top-down 방식으로 각 뉴런의 출력단에서 입력단 방향으로 재분배 된다.
## (재)분배시 기여도는 보존된다.
  - 예를 들어 그림 1에서와 같이 특정 사진 입력에 대해 ‘수탉’이라는 분류를 했고 그 출력값 f(x)가 0.9였다고 하자.
  그러면 각 layer의 뉴런들은 0.9라는 출력에 대한 기여도를 모두 조금씩은 갖고 있으며,
  relevance score를 분배한 후 각 layer에서의 relevance score의 합은 0.9가 되어야 한다는 뜻이다.
~~~

### Backpropagation?
- gradient를 이용한다는 점과 출력단에서 입력단으로 거꾸로 계산해간다는 점은 비슷하지만, LRP의 목적은 전혀 다르다.
- 가장 큰 차이점은 back propagation은 뉴럴넷 학습을 위해 사용되고, LRP는 학습이 다 된 뉴럴넷에다가 적용한다는 점이다.
- LRP는 back propagation 처럼 relevance score를 최적화하기 위한 값이 아니다.

### Calculate relevance score with Taylor series
- Relevance score : x가 출력에 얼마나 영향을 주는가? -> x의 변화가 y의 변화에 얼마나 큰 변화를 주는것인가? -> y에 대한 x의 기여도(relevance score)
![taylor](./img/taylor.jpg)

## 3. Conclusion
![propagation2](./img/relevance_propagation2.png)


### 장점
- 비교적 직관적
- CNN/RNN 등 다양한 네트워크에 사용가능 

### 단점
- 기여도의 해석일 뿐 설명이 되려면 추가적인 맥락이 요구됨. 일일이 히트맵으로 기여도를 보고 객체를 인식해야 한다는 번거로움
출력에 가까운 은닉층일수록 히트맵으로 나타난 추상적 개념은 해석이 어려움

### sample
- https://lrpserver.hhi.fraunhofer.de/handwriting-classification

