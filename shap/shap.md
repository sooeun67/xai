## [SHAP] SHAP (SHapley Additive exPlanations)

### Keyword
 - "변수 X1이 모델의 결과값(예측) P에 얼마나 기여를 할까?"

### 개요
- Shapley Value를 기반으로 하는 XAI
	- Lundberg와 Lee가 제안한 SHAP (SHapley Additive exPlanations)은 Shapley Vlues를 기반으로 예측 값에 대한 각 feature의 기여도를 계산하여 Black box 모델 해석 제공 [(논문링크1)](https://arxiv.org/pdf/1705.07874.pdf) [(논문링크2)](https://arxiv.org/pdf/1802.03888.pdf)
	- Shapley Value : 하나의 특성에 대한 중요도를 알기위해 특성들의 모드 조합을 구성하고 해당 특성의 유무에 따른 평균적인 변화를 통해 얻어낸 값

### 설명
- SHAP의 목적은 예측에 대한 각 특성의 기여도를 계산하여 관측치 x의 예측값을 설명하는 것이다. 
- SHAP 설명 방법은 연합 게임 이론(coalitional game theory)을 사용하여 Shaply value를 계산하고 관측치(data instance)의 특성값은 연합에서 플레이어로서 역할을 한다. 
- Shaply Value는 특성들 사이에 “지불(payout) (=예측(prediction))”을 공정하게 분배하는 방법을 알려준다. 예를 들어 tabular data에서 플레이어는 각각의 특성값이 될 수 있으며 특성값의 그룹이 될 수도 있다. 
- 이미지를 설명하기 위해 픽셀을 수퍼 픽셀(픽셀들의 그룹)로 그룹화하고 수퍼 픽셀 간의 예측값의 분포를 확인할 수 있다.

<details>
<summary>추가 설명 PPT</summary>
<div markdown="1">
	<img src="https://github.com/sooeun67/xai/blob/main/images/SHAP_PPT01.jpg"/>
	<img src="https://github.com/sooeun67/xai/blob/main/images/SHAP_PPT02.jpg"/>
	<img src="https://github.com/sooeun67/xai/blob/main/images/SHAP_PPT03.jpg"/>
	(출처 : https://www.youtube.com/watch?v=BQSkV95Dy4s)
	<img src="https://github.com/sooeun67/xai/blob/main/images/SHAP_PPT04.jpg"/>
	(출처 : https://www.youtube.com/watch?v=uh7j_cj9Yf8)
</div>
</details>

### 특징
- LIME과 달리 모든 학습 데이터에 대하 예측값을 기준으로 시작한다.
- 전체 데이터 중 관심 관측치의 위치를 파악하고 해석하는 것이 가능하다.
- 정형데이터의 경우 관측치별 해석과 전반적인 변수 중요도가 산출 가능하다.
- SHAP은 계산해야 하는 Feature 조합의 수가 많아지는 경우 연산 시간이 길어지는 단점이 존재한다.
- SHAP는 Shapley value (데이터 한 개에 대한 설명, Local)을 기반으로, 데이터 셋의 ‘전체적인 영역’에 대한 해석이 가능하다(Global)
- SHAP는 피처 간 의존성까지 고려해서 모델 영향력을 계산한다 (SHAP가 계산한 모든 피처 영향력의 합은 1)
- 학습된 모델에 대해서만 설명할 수 있으므로, Feature의 추가와 삭제가 빠른 모델을 설명하기에는 적합하지 않다.
- negative(-) 기여도가 계산 가능하다.
- Shapley value는 모델 학습 이후 산출하는 것이므로, 원인 결과으 관계로 해석하면 안된다.

### Python 구현시 특이사항
- Tensorflow와 관련해서 몇몇 호환성 issue가 있음 (2021.11.22 현재)
	- https://github.com/slundberg/shap/issues/930
	- https://github.com/slundberg/shap/issues/850

### 종류
- 모델의 특징에 따라 계산법을 달리하여 빠르게 처리한다.
	- Kernel SHAP : Linear LIME + Shapley Value
	- Tree SHAP : Tree Based Model
	- Deep SHAP : Deeplearning based model
- 공식 문서에 대양한 예제가 있음 (https://shap.readthedocs.io/)

### 시각화

![SHAP 시각화1](https://christophm.github.io/interpretable-ml-book/images/shap-importance.png)
- SHAP Feature Importance: 피쳐 중요도는 평균 절대 샤플리 값으로 측정된다. 위 결과에 따르면 호르몬 피임약을 사용한 연수가 가장 중요한 특징으로, 예측된 암 발생 확률을 평균 2.4%포인트 변경했다.(x축 : 0.024)

![SHAP 시각화2](https://christophm.github.io/interpretable-ml-book/images/shap-importance-extended.png) 
- SHAP Summary Plot: 호르몬 피임약을 복용하는 기간이 적을수록 암의 위험이 감소하고, 많은 해가 되면 그 위험이 증가한다. 단, 위 결과는 모델의 결과를 보여주며 현실 세계에서 반드시 인과관계가 있는 것은 아니다.

![SHAP 시각화3](https://christophm.github.io/interpretable-ml-book/images/shap-dependence.png) 
- SHAP Dependence Plot: 호르몬 피임약들에 대한 SHAP 의존도. 0년에 비해 몇 년은 예측 확률을 낮추고 높은 햇수는 예측된 암 확률을 높인다.

![SHAP 시각화4](https://christophm.github.io/interpretable-ml-book/images/shap-dependence-interaction.png) 
- SHAP Interaction Values: SHAP 피쳐 의존도와 상호작용 시각화. 호르몬 피임약의 해는 성병과 상호작용을 한다. 0년에 가까운 경우, 성병의 발생은 예측된 암 위험을 증가시킨다. 피임약에서 더 많은 해 동안, 성병의 발생은 예측된 위험을 감소시킨다. 다시 말하지만, 이것은 인과 모델이 아니다. 효과는 교란 요인에 기인할 수 있다(예: 성병과 낮은 암 위험은 더 많은 의사 방문과 상관관계가 있을 수 있다).

![SHAP 시각화5](https://christophm.github.io/interpretable-ml-book/images/shap-clustering.png)
- Clustering SHAP values : 설명 유사성으로 클러스터링된 스택형 SHAP. x축의 각 위치는 데이터의 인스턴스(instance)이다. 빨간색 SHAP 값은 예측을 증가시키고, 파란색 값은 예측을 감소시킨다. 오른쪽에는 암 발병 리스크가 높게 예측되는 그룹이 있다.


Ref.

https://velog.io/@sjinu/개념정리SHAPShapley-Additive-exPlanations

https://datanetworkanalysis.github.io/2019/12/23/shap1

https://datanetworkanalysis.github.io/2019/12/24/shap2

https://moondol-ai.tistory.com/378?category=947304

https://tootouch.github.io/IML/shap/

https://christophm.github.io/interpretable-ml-book/shap.html

https://www.youtube.com/watch?v=BQSkV95Dy4s

https://www.youtube.com/watch?v=uh7j_cj9Yf8
