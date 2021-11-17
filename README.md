

# XAI
Explainable Artificial Intelligence(XAI) algorithms / research papers

## 대리분석 (Surrogate Analysis)
본래 기능을 흉내내는 간단한 대체재를 만들어 prototype이 동작하는지 판단하는 분석기법
![모델 f를 흉내내는 g1과 g2](https://github.com/sooeun67/xai/blob/main/images/surrogate_analysis.png)

다시 말해, 블랙 박스 모델 f 가 존재하고, f를 흉내 내는 해석 가능한 ML 모델 g 를 만드는 것이 대리 분석의 목표. 모델 f 가 SVM을 사용해 학습한 모델이라면 모델 g 는 트리나 linear regression 일 수도 있다. 모델 g의 결정 조건은 (1) f 보다 학습하기 쉽고 (2) 설명 가능하며 (3) 모델 f 를 유사하게 흉내낼 수 있으면 된다

- **장점**:
	- model-agnostic: 모델에 대한 지식 없이 학습 가능
	- 적은 학습 데이터로도 ok! (학습 데이터, 예측 모델만 있으면 됨)
	- 모델이 바뀌더라도 feature 만 같다면 대리 분석 수행 가능

### 로컬 대리분석 (Local Surrogate)
데이터 하나에 대해 원래 모델인 블랙박스 모델이 분류한 결과를 해부하고 해석하는 과정을 분석하는 기법으로 대표적으로 **LIME** 과 **SHAP** 이 있다

## [LIME](https://github.com/marcotcr/lime) (Local Interpretable Model-agnostic Explanation)

- LIME 은 input data 에 대해 부분적으로 변화(permutation)를 준다
- 40명의 얼굴을 학습한 후, 어떤 인풋 이미지가 들어왔을 때 40명 중 한 명을 구분하는 모델이 있다고 가정하자. 이때 어떤 이미지 x 가 모델 입력값으로 들어온다면 LIME은 입력 이미지에 대해 아래와 같이 해석 가능하도록 인식 단위를 쪼개고 이미지를 해석한다
![lime_1](https://github.com/sooeun67/xai/blob/main/images/lime_1.png)

그리고 이렇게 나뉜 영역을 조합해서 원본 모델이 대상을 가장 잘 분류할 수 있는 대표 이미지를 구성

어떤 이미지가 입력 값으로 주어졌을 때, 이미지 내 특정 관심 영역을 x라 하고, 초점 주변으로 관심 영역을 키워갈 때 기준 x 로부터 동일한 정보를 가지고 있다고 간주할 수 있을 때, 이 영역을 πx​이라 하고 이를 슈퍼 픽셀(super pixel)이라 한다


### 장점
- model-agnostic: 모델에 대한 지식 없이 학습 가능
- Deep Learning 이나 GPU 사용하지 않고 적용 가능한, 가벼운 XAI 기법
- matrix 로 표현 가능한 데이터(text/image)에 작동하는 기법

### 단점 및 고려사항
- **불확실성**: 슈퍼 픽셀 알고리즘에 따라 마스킹 데이터가 달라지며, 모델 g는 sampling 위치에 따라 random한 결과를 보일 수 있다 --> ***non-deterministic*** (even for the same input, algoirthm can exhibit different behaviors/output)
- 데이터 하나에 대한 설명이기 때문에 모델 전체에 대한 일관성을 보전하지 못한다
> **Note:** [논문](https://arxiv.org/pdf/1602.04938.pdf)에서 SP-LIME(서브모듈러 최적화) 알고리즘을 통해 데이터 셋 전체를 대표하는  예시들을 뽑아 신뢰가 갈만한 모델을 만드는 기법 소개


### 샘플 (To Do: use case  주피터 노트북 추가 )
- Data: 뉴스 기사와 해당 기사의 20가지의 카테고리
- 아래는 LIME 출력 결과물
--  88-89와 SE 모델이 자동차(auto) 카테고리를 결정하는 서브모듈러(highlighted)
![lime_result](https://github.com/sooeun67/xai/blob/main/images/lime_result.png)

## [SHAP](https://github.com/slundberg/shap)
- Shapely Value: 전체 성과를 창출하는 데 각 참여자가 얼마나 공헌했는지를 수치로 표현. 각 사람의 기여도는 그 사람의 기여도를 제외했을 때의 전체 성과 변화 정도로 나타낼 수 있다
- 원리: 모델이 표현할 수 있는 모든 조합과 feature 




#### Limitations & 차이점
- Local Explanation 을 기반으로 하여, 데이터의 **전체적인 영역에 대한 해석(Global Surrogate)** 이 가능하다는 게 LIME과의 차이
- negative(-) 기여도 계산 가능 

## [LRP](Layer-wise Relevance Propagation)
- **결과를 역추적해서 입력 이미지에 히트맵을 출력**
- DNN 출력 값을 Decomposition하여 각각의 피처에 대한 기여도(relevance score)를 계산
-  Relevance score를 Output layer에서 Input layer 방향으로 계산해나가며 그 비중을 재분배하는 방법
- 모든 모델에 적용 가능

- 잘 훈련된 네트워크에 input(x):수탉 사진/ouput(f(x)):'수탉'이 경우, 이 '수탉'이라는 출력 f(x)를 얻기 위해 입력 샘플의 각 pixel들이 기여하는 바를 계산하는 방법
- 아래의 그림1에서 보이는 것처럼 heatmap이라고 적힌 그림에 pixel들의 기여도(relevance score)가 색깔로 표시되며, 수탉의 부리나 머리 등을 보고 해당 입력의 클래스가 '수탉'임을 출력했다는 것을 알 수 있음
- ![lrp_example](https://user-images.githubusercontent.com/12220234/142083511-32da108a-b6d5-4827-879a-00e89f55238a.png)

### 장점
- 비교적 직관적
- CNN/RNN 등 다양한 네트워크에 사용가능 

### 단점
- 기여도의 해석일 뿐 설명이 되려면 추가적인 맥락이 요구됨. 일일이 히트맵으로 기여도를 보고 객체를 인식해야 한다는 번거로움
출력에 가까운 은닉층일수록 히트맵으로 나타난 추상적 개념은 해석이 어려움

### sample
- https://lrpserver.hhi.fraunhofer.de/handwriting-classification

## [FV] (Filter Visualization)

### Filter
![image](https://user-images.githubusercontent.com/12220234/142094772-7d22112a-fe88-4fa3-ae42-bf14a74e7d75.png)
- 필터는 원본 이미지에서 특정 요소를 추출하기 위해 사용하는 것으로, 주파수 필터을 함.
- 다섯번 째 그림은 저주파만 통과하기 때문에 블러리한 이미지를 결과로 얻으며, 세번째 그림인 Laplace Filter는 Edge를 찾는 역할. 네번째 그림인 high-pass filter는 이미지가 선명해지는 결과를 얻음.
- **즉, 학습된 CNN 필터들은 이런식으로 경계선을 찾거나 블러리한 면을 찾는 등 다양한 주파수 필터의 기능을 한다.**
- **피처맵 시각화 방식으로, 모델이 입력 이미지에 어떻게 반응하는지 조사하는 방법**

### Occlusion Experiment(Zeiler & Fergus 2013)
![image](https://user-images.githubusercontent.com/12220234/142085160-81dd04e3-489a-4dc5-9aa0-af6f067cc23d.png)
- 위 그림은 image의 *어떤 부분이 이미지 분류에 큰 영향*을 미치는지 알아본 결과
- 방법: (a)와 같은 input image가 있을 때, 작은 회색 상자를 그리고,모델에 통과시켜서 나온 결과를 기록 -> 이 회색상자를 조금씩 이동시키면서 위 과정을 반복 
- 결과: 결과를 heatmap으로 시각화한 것이 (d), (e)로, (d)는 회색상자로 일부가 지워진 그림이 포메라니안일 확률이 높으면 빨간색이고, 낮으면 파란색
- 즉, 파란색으로 부분이 지워지면 포메라니안으로 분류될 확률이 낮으므로 이 부분이 분류 결과를 결정하는 중요한 부분임을 암시함. Input image에서 파란 부분은 강아지의 얼굴
- 결론: 본 실험은 CNN이 사람이 물체를 인식하는 과정과 유사하다는 것을 검증

### CAM visualization(2016)
![2019 Seminar-18](https://user-images.githubusercontent.com/12220234/142095727-483622ea-9fcb-433e-af6e-436492593769.jpg)
![image](https://user-images.githubusercontent.com/12220234/142095885-27a6a13b-9f76-43d1-a8b5-3eb83ec561b6.png)

-------
#### Reference (참고문헌)
- LIME paper: https://arxiv.org/pdf/1602.04938.pdf
- XAI 설명가능한 인공지능 도서
- https://velog.io/@tobigs_xai/1%EC%A3%BC%EC%B0%A8-LIME-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Why-Should-I-Trust-You-Explaining-the-Predictions-of-Any-Classifier
- https://yjjo.tistory.com/3#:~:text=SP%2DLIME%3A%20%EB%AA%A8%ED%98%95%20%EC%A0%84%EC%B2%B4%EC%9D%98,%EB%A5%BC%20%EC%84%A0%ED%83%9D%ED%95%B4%EC%A3%BC%EB%8A%94%20%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EC%9E%85%EB%8B%88%EB%8B%A4.
- 
- lrp paper: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140
- https://angeloyeo.github.io/2019/08/17/Layerwise_Relevance_Propagation.html

- CAM :https://tyami.github.io/deep%20learning/CNN-visualization-Grad-CAM/
