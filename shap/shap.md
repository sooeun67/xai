---
## [SHAP](https://github.com/slundberg/shap) SHAP (SHapley Additive exPlanations)

### Keyword
 - "(관심 데이터 xi에 대해) 변수 X1이 모델의 결과값(예측) P에 얼마나 기여를 할까?"

### 개요
- Shapley Value를 기반으로 하는 XAI
	- Lundberg와 Lee가 제안한 SHAP (SHapley Additive exPlanations)은 Shapley Vlues를 기반으로 예측 값에 대한 각 feature의 기여도를 계산하여 Black box 모델 해석 제공 [(논문링크1)](https://arxiv.org/pdf/1705.07874.pdf) [(논문링크2)](https://arxiv.org/pdf/1802.03888.pdf)
	- Shapley Value : 하나의 특성에 대한 중요도를 알기위해 특성값들의 모든 조합을 구성하고 해당 특성의 유무에 따른 평균적인 Y 변화 값
	- 전체 성과를 창출하는 데 각 참여자가 얼마나 공헌했는지의 기여도는 그 사람을 제외했을 때의 전체 성과 변화 정도로 나타낼 수 있다


### 설명
- SHAP의 목적은 예측에 대한 각 특성의 기여도를 계산하여 관측치 x의 예측값을 설명하는 것이다. 
- SHAP 설명 방법은 연합 게임 이론(coalitional game theory)을 사용하여 Shaply value를 계산하는 방식이며, 이때 관측치(data instance)의 특성값은 연합 게임 이론의 플레이어 역할을 한다.
- 다시 말해 Shaply Value는 특성들 사이에 “지불(payout) (=예측(prediction))”을 공정하게 분배하는 방법을 알려준다. 예를 들어 tabular data에서 플레이어는 각각의 특성값이 될 수 있으며 특성값의 그룹이 될 수도 있다. 
- 이미지를 설명하기 위해 픽셀을 수퍼 픽셀(픽셀들의 그룹)로 그룹화하고 수퍼 픽셀 간의 예측값의 분포를 확인할 수 있다.

<details>
<summary>계산방법의 직관적 이해 - 추가 설명 PPT</summary>
<div markdown="1">
	<img src="https://github.com/sooeun67/xai/blob/main/images/SHAP_PPT01.jpg"/>
	<img src="https://github.com/sooeun67/xai/blob/main/images/SHAP_PPT02.jpg"/>
	<img src="https://github.com/sooeun67/xai/blob/main/images/SHAP_PPT02_2.png"/>
	<img src="https://github.com/sooeun67/xai/blob/main/images/SHAP_PPT03.jpg"/>
	(출처 : https://www.youtube.com/watch?v=BQSkV95Dy4s)
	<img src="https://github.com/sooeun67/xai/blob/main/images/SHAP_PPT04.jpg"/>
	(출처 : https://www.youtube.com/watch?v=uh7j_cj9Yf8)
</div>
</details>

### Calculation

---
### Advantages and Dis-Advantages
- Advantages
	- LIME과 달리 SHAP는 Shapley value (데이터 한 개에 대한 설명, Local)을 기반으로, 데이터 셋의 **‘전체적인 영역’**에 대한 해석이 가능하다(Global)
	- Determistic : 계산할 때마다 같은 결과 출력
	- 정형데이터의 경우 관측치별 해석과 전반적인 변수 중요도가 산출 가능하다.
	- Text, Image 에 대한 해석도 가능
	- 이론을 토대로 "Symmetry","Addtivie", "Dummy" 가 만족되는 현재 유일한 방법
- Dis-Advantages
	- SHAP은 계산해야 하는 Feature 조합의 수가 많아지는 경우 연산 시간이 길어지는 단점이 있다. 
		- 이를 커버하기 위해 모델알고리즘에 따른 다른 계산법을 지원함 (Kernal, Tree, Deep, Gradient 등)
	- Shapley value는 모델 학습 이후 산출하는 것이므로, 원인 결과 관계로 해석의하면 안된다. (모델의 결과에 대한 설명 O, 인과관계 X)


### 종류
- 모델의 특징에 따라 계산법을 달리하여 빠르게 처리한다.
	- Kernel SHAP : Linear LIME + Shapley Value
	- Tree SHAP : Tree Based Model
	- Deep SHAP : Deeplearning based model
- 공식 문서에 대양한 예제가 있음 (https://shap.readthedocs.io/)

### Python 구현시 특이사항
- Tensorflow와 관련해서 몇몇 호환성 issue가 있음 (2021.11.22 현재)
	- https://github.com/slundberg/shap/issues/930
	- https://github.com/slundberg/shap/issues/850
---
### 시각화

#### 시각화 1) Titanic

![SHAP 시각화1](https://github.com/sooeun67/xai/blob/main/shap/SHAP_summary_01.jpg)
>- SHAP summary plot (SHAP Feature Importance)
>	- |SHAP value| : 모델의 예측값에 기여하는 Feature의 기여도를 동일한 Unit의 수치로 표현, 해당 Plot에서는 절대값을 취한다.
>	- Sex : 각 인스턴스가 가진 SHAP Value의 평균값이 1.2 수준을 가진다. (절대값이므로 '-', '+'와 무관하게 기여도의 크기만을 표현, 남1,여0) 
>	- Pclass : 각 인스턴스가 가진 SHAP Value의 평균값이 0.8 수준을 가진다. (절대값이므로 '-', '+'와 무관하게 기여도의 크기만을 표현) 
>	- Age : 각 인스턴스가 가진 SHAP Value의 평균값이 0.5 수준을 가진다. (절대값이므로 '-', '+'와 무관하게 기여도의 크기만을 표현)

---

![SHAP 시각화2](https://github.com/sooeun67/xai/blob/main/shap/SHAP_summary_02.jpg)
>- SHAP summary plot
>	- Feature Value : 붉은색일 수록 높은 값을 가지고 있고, 파란색일 수록 낮은 값을 가지고 있다.
>	- SHAP value : 모델의 예측값에 기여하는 Feature의 기여도를 동일한 Unit의 수치로 표현
>	- Sex : Feature의 값이 높을 수록(=남성) SHAP Value 는 Negative값을 가진다 (생존률의 예측 결과값에 '-' 기여함)
>	- Pclass : Feature의 값이 높을 수록(=등급이 낮을 수록) SHAP Value 는 Negative값을 가진다 (생존률의 예측 결과값에 '-' 기여함)
>	- Age : Feature의 값이 높을 수록(=나이가 많을 수록) SHAP Value 는 Negative값을 가진다 (생존률의 예측 결과값에 '-' 기여함)

---

![SHAP 시각화3](https://github.com/sooeun67/xai/blob/main/shap/SHAP_dependency_01.png) 
>- SHAP Dependence Plot (with Interaction Values)
>	- SHAP value for Fare : y축의 각 Point는 각 인스턴스의 Fare에 대한 SHAP Value를 표현한다. 
>		- Fare : x축의 각 Point는 Feature(=Fare)의 실제 값(Value)을 표현한다.
>		- Sex : 각 (x, y)의 Point의 색으로 Sex의 Value를 표현한다. (붉은색 1, 파란색 0)
>		- SHAP value : 모델의 예측값에 기여하는 Feature의 기여도를 동일한 Unit의 수치로 표현

---

![SHAP 시각화4](https://github.com/sooeun67/xai/blob/main/shap/SHAP_force_01.png) 
>- SHAP Force plot (개별 인스턴스에 대해 예측 결과값에 대한 해석을 제공)
>	- base value : 해당 모델의 예측 평균값은 0.3361이다.
>	- f(x) = 0.14 : 해당 인스턴스(=관측치)의 예측값은 0.14로 평균보다 낮다 (생존 확률이 낮음)
>	- 각 bar(또는 화살표 표현)으로 각 예측값에 공헌한 Feature를 시각화 하였다. (red는 + 기여, blue는 - 기여)
>	- Sex = 1(남성) 이며 Pclass = 3(3등석)이 다른 여타 + 공헌도를 가지는 Feature보다 예측값(=생존율)의 하락을 가져오는데 기여하였다.

<br>
<br>

![SHAP 시각화5](https://github.com/sooeun67/xai/blob/main/shap/SHAP_force_02.png) 
>- SHAP Force plot (개별 인스턴스에 대해 예측 결과값에 대한 해석을 제공)
>	- base value : 해당 모델의 예측 평균값은 0.3361이다.
>	- f(x) = 0.14 : 해당 인스턴스(=관측치)의 예측값은 0.90으로 평균보다 매우 높다 (생존 확률이 높음)
>	- 각 bar(또는 화살표 표현)으로 각 예측값에 공헌한 Feature를 시각화 하였다.(red는 + 기여, blue는 - 기여)
>	- Sex = 0(여성) 이며 Pclass = 1(1등석)이 다른 여타 + 공헌도를 가지는 Feature보다 예측값(=생존율)의 증가를 가져오는데 기여하였다.

---

![SHAP 시각화6](https://github.com/sooeun67/xai/blob/main/shap/SHAP_clustering.png)
>- Clustering SHAP values : 설명 유사성으로 클러스터링된 스택형 SHAP.
>	- x축의 각 위치는 데이터의 인스턴스(instance)이다. 빨간색 SHAP 값은 예측을 증가시키고, 파란색 값은 예측을 감소시킨다. 
>	- 왼쪽에 생존율이 높은 그룹이 모여있으며, 중간으로 갈수록 생존율이 낮아지다가, 오른쪽으로 가면 생존율이 약간 상승한다.
>	- 마우스 이동을 통해 각 instance의 값을 확인할 수 있다.

---
#### 시각화 2) MNIST

![SHAP 시각화6](https://github.com/sooeun67/xai/blob/main/shap/SHAP_deep_explainer.png)
>- SHAP image plot : Plots SHAP values for image inputs.
>	- 좌측에는 input image가 표현되며,
>	- 우측에는 각 클래스별 예측 결과를 SHAP Value를 통해 표현한다.

---

Ref.

https://velog.io/@sjinu/개념정리SHAPShapley-Additive-exPlanations

https://datanetworkanalysis.github.io/2019/12/23/shap1

https://datanetworkanalysis.github.io/2019/12/24/shap2

https://moondol-ai.tistory.com/378?category=947304

https://tootouch.github.io/IML/shap/

https://christophm.github.io/interpretable-ml-book/shap.html

https://www.youtube.com/watch?v=BQSkV95Dy4s

https://www.youtube.com/watch?v=uh7j_cj9Yf8
