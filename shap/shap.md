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

이렇게 나뉜 영역(슈퍼픽셀)을 조합해서 원본 블랙박스 모델 f의 예측 결과와 가장 유사한 대표 이미지를 구성
![lime_4](https://github.com/sooeun67/xai/blob/main/images/lime_4.png)

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


## [SHAP] 추가 작업 예정 (Test)
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
