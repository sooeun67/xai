## [SHAP] SHAP (SHapley Additive exPlanations)

### Keyword
 - "변수 X1이 이 모델로부터 제거될 때 얼마나 예측 P에 변화를 줄까?"

### 개요
- Shapley Value를 기반으로 하는 XAI
	- Lundberg와 Lee가 제안한 SHAP (SHapley Additive exPlanations)은 각 예측치를 설명할 수 있는 방법이다. SHAP은 게임 이론을 따르는 최적의 Shapley Value를 기반으로한다.
	- Shapley Value : 하나의 특성 대한 중요도를 알기위해 여러 특성들의 조합을 구성하고 해당 특성의 유무에 따른 평균적인 변화를 통해 얻어낸 값

### 정의
- SHAP의 목적은 예측에 대한 각 특성의 기여도를 계산하여 관측치 x의 예측값을 설명하는 것이다. SHAP 설명 방법은 연합 게임 이론(coalitional game theory)을 사용하여 Shaply value를 계산하고 관측치(data instance)의 특성값은 연합에서 플레이어로서 역할을 한다. Shaply Value는 특성들 사이에 “지불(payout) (=예측(prediction))”을 공정하게 분배하는 방법을 알려준다. 예를 들어 tabular data에서 플레이어는 각각의 특성값이 될 수 있으며 특성값의 그룹이 될 수도 있다. 예를 들어 이미지를 설명하기 위해 픽셀을 수퍼 픽셀(픽셀들의 그룹)로 그룹화하고 수퍼 픽셀 간의 예측값의 분포를 확인할 수 있다. 

### 특징
- SHAP는 이와 다르게 피처 간 의존성까지 고려해서 모델 영향력을 계산한다(SHAP가 계산한 모든 피처 영향력의 합은 1)
- 학습된 모델에 대해서만 설명할 수 있으므로, 피처의 추가와 삭제가 빠른 모델을 설명하기에는 적합하지 않다.
- negative(-) 기여도 계산 가능

### 종류
SHAP는 Shapley value (데이터 한 개에 대한 설명, Local)을 기반으로, 데이터 셋의 ‘전체적인 영역’에 대한 해석이 가능하다(Global)
모델의 특징에 따라, 계산법을 달리하여 빠르게 처리한다.
- Kernel SHAP : Linear LIME + Shapley Value
- Tree SHAP : Tree Based Model
- Deep SHAP : Deeplearning based model


Ref.

https://velog.io/@sjinu/개념정리SHAPShapley-Additive-exPlanations
https://datanetworkanalysis.github.io/2019/12/23/shap1
https://datanetworkanalysis.github.io/2019/12/24/shap2
https://moondol-ai.tistory.com/378?category=947304
https://tootouch.github.io/IML/start/




## [SHAP](https://github.com/slundberg/shap)
- Shapely Value: 전체 성과를 창출하는 데 각 참여자가 얼마나 공헌했는지를 수치로 표현. 각 사람의 기여도는 그 사람의 기여도를 제외했을 때의 전체 성과 변화 정도로 나타낼 수 있다
- 원리: 모델이 표현할 수 있는 모든 조합과 feature 
- 
![shap](https://github.com/sooeun67/xai/blob/main/images/shap.png)
- 예시) 집값을 결정짓는 요인으로 [숲세권, 면적/층, 고양이 양육가능여부] Feature 존재
-- '고양이 양육가능여부' 의 집값에 대한 기여도를 평가해보자
-- 나머지 feature 들이 동일하다는 전제 하에, 310,000 (`cat-banned`) - 320,000 (`cat-allowed`) = -10,000
-- 다시 말해, `cat-banned` 의 기여도는 -10,000 유로
-- 이 계산 과정을 모든 가능한 combination 에 대해 반복 

### 장점
- model-agnostic: 다양한 모델에 적용 가능
- consistent: 계산할 때마다 같은 결과 출력
- negative(-) 기여도 계산 가능 
- Local Explanation 을 기반으로 하여, 데이터의 **전체적인 영역에 대한 해석(Global Surrogate)** 이 가능하다는 게 LIME과의 차이



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
