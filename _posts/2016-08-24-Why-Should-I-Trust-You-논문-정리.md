---
layout: post
title: "Why Should I Trust You?" Explaining the Predictions of Any Classifier 논문 정리
categories : [Deep Learning]
tags : [Deep Learning, Machine Learning]
use_math : true
shortinfo : LIME (Local Interpretable Model-agnostic Explanations) 알고리즘에 대해 설명하는 "Why Should I Trust You?" Explaining the Predictions of Any Classifier 논문을 간략하게 정리한 개인적 정리 노트입니다.
---

## Paper
["Why Should I Trust You?" Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)

## Introduction
- 용어 정의
	- Trust Prediction : Prediction 자체를 믿고, 이에 따라 행동을 개시
	- Trust Model : 모델 자체를 믿는것
- Model Trust의 경우, 단순 validation data의 accuracy 만으로는 부족할 수 있다

## Goals (for explainer)
1. Interpretable
	- Input Variable (혹은 그에 준하는 '원인'?) 들과 예측 결과 사이의 관계를 설명할 수 있어야함
	- 이해하기 쉬워야하고, 누가 봐도 이해할 수 있어야함
2. Local Fieldity
	- 모델 전체가 왜 이렇게 움직였는가에 대해 얘기하는 것 보다는, 하나의 instance에 대해 왜 그렇게 움직이는지 설명할 수 있어야함
	- global fieldity와는 약간 독자적으로 생각될 수 있어야함
3. Model Agnostic
	- Any model을 설명할 수 있어야함 ; original model을 black-box로 가정
4. Global Perspective
	- 각각의 테스트 결과 말고도, 모델 전체가 왜 이런 경향성으로 움직이는지에 대한 global perspective도 가져야 모델에 대한 trust를 가질 수 있음

## LIME (Local Interpretable Model-agnostic Explanations)
- 주의 : 실제 모델에서 사용하는 feature (ex : word embedding) 과 interpretable data representation (ex : one-hot word vector) 는 다름
- 정의
	- $ f(x) $ : 평가하려고 하는 모델 f가 x를 넣었을 때 원하는 label에 대한 확률로 내놓는 값
	- $ g \in G $ :
		- G : class of (potentially) interpretable models
		- g : d' dimension의 {0, 1} vector. Interpretable component가 있는지, 없는지?
    - $ \Omega(g) $ : explanable model의 complexity. ex : decision tree의 depth, linear model의 # of non-zero weights..
    - $ \pi_x(z) $ : x와 z의 가까운 정도의 measure
    - $ L(f, g, \pi_x(z)) $ : 정의된 pi와 주어진 f에 대해, g가 주어진 f를 approximate 하는 데에 대한 uncertainity (주어진 instance z 근처에서)
- LIME에 의해 나오는 explanation : explanational model 중 하나로, uncertainity를 줄임과 동시에 human-interpretable 하도록 complexity를 줄인다
$$ \xi(x) = {argmin}_{g \in G} L(f, g, \pi) + \Omega(g)$$
- 결국 이 LIME은 주어진 instance에 대강 local한 위치에서 model의 decision function을 예측할 수 있는 interpretable model을 찾는 것이 목적이라고 볼 수 있겠다

#### Algorithm
- G = Linear Model을 가지고 예시를 들자
- Given instnace x와 x의 interpretable version x' 가 있다고 하자
	1. x'를 가지고, uniform random하게 element 값들을 바꿔서 주위의 interpretable한 샘플 z'를 N개 뽑는다
	2. z'들에 대해 original representation으로 바꾸어서 z를 만들고, 이걸 가지고 원래 모델에 넣은 값 f(z)와 similarity $\pi_x(z) $를 계산
	3. Loss $ L = \sum_z \pi_x(z) (f(z) - g(z'))^2 $ 정도로 locally weighted square loss로 삼고, 여기에 complexity function을 붙여서 최적화
		- complexity function의 예시 : word-of-bag의 경우, linear model의 non-zero element가 K개 넘으면 complexity function 값을 무한으로 줘버린다는 식으로
		- 다른 방식은 K-LASSO : LASSO를 해서 K개의 feature를 선택한 후, 이 K개의 feature만으로 least-square로 weight 구한다

## SP(Submodular Pick) - LIME
- LIME은 한 instance의 explanation을 할 수 있지만, Model의 Trust는 말할 수 없음
- B개의 budget limit을 골라놓고, 중요한 feature들이 골라지도록 B 가지 미만의 test instance를 고르는 문제임.
	- 이 때 feature의 weight는 단순 LIME을 가지고 계산한 interpretable model의 weight의 sqrt 값의 합을 이용한다
- weighted pick cover (NP-HARD)라서, greedy 하게 제일 목표함수값이 높아지는 instance를 선택해 나감
- 사용자는 B개의 instance 설명 결과를 보고 이 model의 행동 방식을 유추할 수 있을 것

## 정리
- LIME은 결국 모델의 복잡한 decision bound를 설명하기 보다는 현재 보는 instance의 local한 부분의 decision bound를 설명하는(근사하는) interpretable model을 찾음으로서 머신러닝 사용자에게 모델/해석에 대한 믿음을 주려는 연구.
- Model-agnostic을 지향하는 연구인 이상 이렇게 결과값을 가지고 거꾸로 유추하는 방식밖에 사용할 수 없다고는 생각되지만, 아무래도 이런 방식을 택하다보니 느릴 수밖에 없는 것 같다
	- Inception Net의 한 instance를 설명하는 데에 10분이 걸린다고 함
- 또 한 가지 단점은 원래 모델에 사용하는 input <-> interpretable feature의 가역적인 변환이 가능한 경우에만 사용할 수 있다는 것
- 단순 학술적인 관점에서는 글쎄? 라고 말할 수 있을 논문이지만, 실무적인 관점에서는 굉장히 필요한 논문이라고 생각한다. 실무에서는 black-box 모델이 아무리 accuracy가 높게 작동한다고 해도 내부 작동방식을 모르기 때문에 잘 대입해서 사용하지 못한다는 말을 들었기 때문. 실제로 주식 같은데에서 trend 예측에도 이런 이유때문에 선형회귀 모델만을 사용하고 의사결정 도구로만 사용한다고 하는데, 이렇게 black-box model을 설명할 수 있는 연구가 있다면 실무에 머신러닝을 도입하는 데에 큰 도움이 될듯
- Neural Net Specific하게 바꿔서 빠르게 동작할 수 있는 연구도 필요할듯 싶다
