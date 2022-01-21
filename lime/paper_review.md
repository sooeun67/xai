
LIME Paper Review

# I. INTRODUCTION

-  [LIME Intro Video](https://youtu.be/hUnRCxnydCc)

### Trust 에 대한 두 가지 정의를 구분해야할 필요가 있다고 한다

- **(i) trusting a prediction: 사용자가 개별 예측을 충분히 믿어서 “trust”를 기반으로 다음 행동을 취하는 것**

개별 예측에 대해 믿을 것인지에 대한 결정은, 모델이 의사결정 과정에 사용될 때에 특히 중요한 역할을 한다. 의학 진단이나 테러 탐지같은 문제들에 대해 머신 러닝을 적용하는데, 예측들에 대한 이해가 없는 상태로 진행된다면 그 결과가 매우 끔찍할 수 있다.

  - **(ii) trusting a model: 사용자가 모델이 배포되어 잘 행동할지 모델 자체에 대해 믿는 것**

위 포인트와는 조금 다르게, 모델을 쌩야생에 배포한다고 생각보자. 배포하기 전 모델이 잘 작동할 건지, 믿을 만한 놈(?) 인지에 대한 평가가 필요할 것이다.

현재 일반적으로 validation datasets에 accuracy metrics 를 사용해 모델을 평가한다. 하지만 real-world data 가 너무 다르거나 평가 지표가 우리가 보고자 하는 것들을 제대로 알려주지 못할 때도 있다. 

따라서 이 논문에서는 두 가지 방법을 소개하는데, 

- **(i) LIME: 어떤 classifier/regressor 예측이든 설명할 수 있는 알고리즘 — a solution to a “trusting a prediction” problem**

- **(ii) SP-LIME: 대표적인 instances 들을 선택하는 알고리즘 — a solution to a “trusting a model” problem**


# II. Explainer

설명에서 필요한 특성 (Desired Characters)을 다음과 같이 정리

**1) interpretable;**  입력과 그 결과를 “사용자가 이해할 수 있는” 설명을 제공해야 한다.

**2) local fidelity;**  유사한 데이터에 대해서는 유사한 설명이 이뤄져야 한다.

**3) model-agnostic;**  어떤 형태의 모델에도 적용할 수 있어야 한다. 

**4) global perspective;**  모델에 대한 전반적인 설명이 필요하다.

(1),(2)를 설명에서 필수적인 특성으로 꼽았고, 이를 기반으로 방법론을 구성하였습니다. 
LIME은 두 가지 방식으로 설명하는데, 첫 번째는 입력 attribute의 중요도(importance)를 찾는 방식. 두 번째는 중요한 attribute set을 선택하는 방식. 각각 LIME, SP-LIME(Submodular Pick)으로 표현







# III. Thoughts & To-Do

- **LIME 한국어 적용?** 예시는 못 찾았고 관련 [논문 링크](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002787808)
- 
