# [LRP](Layer-wise Relevance Propagation)

**L** ayer-wise    **(레이어 단위)** *로* <br>
**R** elevance    *결과에 영향을 주는 **(관련성)** 을 구하는* <br>
**P** ropagation  *역 **(전파)** 기술* <br>

즉, 입력 데이터 관점에서 **분류 결과** 뿐만 아니라 **결정에 영향** 을 미치는 구조를 설명


## 1. How?
![lrp_example](/img/example_1_rooster.jpg)

- 잘 훈련된 네트워크에 input(x):수탉 사진/ouput(f(x)):'수탉'이 경우, 이 '수탉'이라는 출력 f(x)를 얻기 위해 입력 샘플의 각 pixel들이 기여하는 바를 계산하는 방법
- 아래의 그림1에서 보이는 것처럼 heatmap이라고 적힌 그림에 pixel들의 기여도(relevance score)가 색깔로 표시되며, 수탉의 부리나 머리 등을 보고 해당 입력의 클래스가 '수탉'임을 출력했다는 것을 알 수 있음

LRP(Layer-wise Relevance Propagation)의 이름에서 볼 수 있듯이 이 method는 relevance score를 출력단에서 입력단 방향으로 top-down 방식으로 기여도를 재분배 하는 방법이다.

LRP의 기본적인 가정 및 작동 방식은 다음과 같다.

- 각 뉴런은 어느 정도의 기여도(certain relevance)를 갖고 있다.
- 기여도는 top-down 방식으로 각 뉴런의 출력단에서 입력단 방향으로 재분배 된다.
- (재)분배시 기여도는 보존된다.
- 
- **결과를 역추적해서 입력 이미지에 히트맵을 출력**
- DNN 출력 값을 Decomposition하여 각각의 피처에 대한 기여도(relevance score)를 계산
-  Relevance score를 Output layer에서 Input layer 방향으로 계산해나가며 그 비중을 재분배하는 방법


### 장점
- 비교적 직관적
- CNN/RNN 등 다양한 네트워크에 사용가능 

### 단점
- 기여도의 해석일 뿐 설명이 되려면 추가적인 맥락이 요구됨. 일일이 히트맵으로 기여도를 보고 객체를 인식해야 한다는 번거로움
출력에 가까운 은닉층일수록 히트맵으로 나타난 추상적 개념은 해석이 어려움

### sample
- https://lrpserver.hhi.fraunhofer.de/handwriting-classification


## 추가본


L
layer-wise (레이어 단위)로
R
Relevance 결과에 영향을 주는 (관련성)을 구하는
P
Propagation 역(전파) 기술

- 입력 데이터 관점에서 **분류 결과** 뿐만 아니라 **결정에 영향**을 미치는 구조를 설명

