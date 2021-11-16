

# XAI
Explainable Artificial Intelligence(XAI) algorithms / research papers

## 대리분석 (Surrogate Analysis)
본래 기능을 흉내내는 간단한 대체재를 만들어 prototype이 동작하는지 판단하는 분석기법
![모델 f를 흉내내는 g1과 g2](https://github.com/sooeun67/xai/blob/main/images/surrogate_analysis.png)

다시 말해, 블랙 박스 모델 f 가 존재하고, f를 흉내 내는 해석 가능한 ML 모델 g 를 만드는 것이 대리 분석의 목표. 모델 f 가 SVM을 사용해 학습한 모델이라면 모델 g 는 트리나 linear regression 일 수도 있다. 모델 g의 결정 조건은 (1) f 보다 학습하기 쉽고 (2) 설명 가능하며 (3) 모델 f 를 유사하게 흉내낼 수 있으면 된다

- 장점:
	- model-agnostic: 모델에 대한 지식 없이 학습 가능
	- 적은 학습 데이터로도 ok! (학습 데이터, 예측 모델만 있으면 됨)
	- 모델이 바뀌더라도 feature 만 같다면 대리 분석 수행 가능

### 로컬 대리분석 (Local Surrogate)
데이터 하나에 대해 블랙박스가 해석하는 과정을 분석하는 기법

## LIME (Local Interpretable Model-agnostic Explanation)
- [official doc](https://github.com/marcotcr/lime)

LIME 은 input data 에 대해 부분적으로 변화(permutation)를 준다. 


## SHAP
- [official doc](https://github.com/slundberg/shap)

## Filter Visualization

## LRP(Layer-wise Relevance Propagation)

