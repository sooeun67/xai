## [LRP](Layer-wise Relevance Propagation)
- **결과를 역추적해서 입력 이미지에 히트맵을 출력**
- DNN 출력 값을 Decomposition하여 각각의 피처에 대한 기여도(relevance score)를 계산
-  Relevance score를 Output layer에서 Input layer 방향으로 계산해나가며 그 비중을 재분배하는 방법
- 모든 모델에 적용 가능

- 잘 훈련된 네트워크에 input(x):수탉 사진/ouput(f(x)):'수탉'이 경우, 이 '수탉'이라는 출력 f(x)를 얻기 위해 입력 샘플의 각 pixel들이 기여하는 바를 계산하는 방법
- 아래의 그림1에서 보이는 것처럼 heatmap이라고 적힌 그림에 pixel들의 기여도(relevance score)가 색깔로 표시되며, 수탉의 부리나 머리 등을 보고 해당 입력의 클래스가 '수탉'임을 출력했다는 것을 알 수 있음
![lrp_example](https://user-images.githubusercontent.com/12220234/142083511-32da108a-b6d5-4827-879a-00e89f55238a.png)

### 장점
- 비교적 직관적
- CNN/RNN 등 다양한 네트워크에 사용가능 

### 단점
- 기여도의 해석일 뿐 설명이 되려면 추가적인 맥락이 요구됨. 일일이 히트맵으로 기여도를 보고 객체를 인식해야 한다는 번거로움
출력에 가까운 은닉층일수록 히트맵으로 나타난 추상적 개념은 해석이 어려움

### sample
- https://lrpserver.hhi.fraunhofer.de/handwriting-classification
