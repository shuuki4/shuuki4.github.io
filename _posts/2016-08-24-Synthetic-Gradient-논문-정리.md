---
layout: post
title: Decoupled Neural Interfaces using Synthetic Gradients 논문 정리
categories : [Deep Learning]
tags : [Deep Learning, Neural Network, Synthetic Gradient, DeepMind]
use_math : true
shortinfo : DeepMind에서 발표한 Decoupled Neural Interfaces using Synthetic Gradients 논문을 간략하게 정리한 개인적 정리 노트입니다.
---

## Paper
[Decoupled Neural Interfaces using Synthetic Gradients](http://arxiv.org/abs/1502.04623)

## Intro
- 지금의 Neural Network은 Sequential 하게 진행되어야 하므로, 현재 layer의 값을 계산하려면 앞 레벨에서 값이 다 계산되서 와야하고, backprop 할 때도 모든 dependent 한 노드들의 계산값이 다 계산되서 와야하고, ... 와 같은 단점이 있음
	- 특히 update locking : 어떤 module도 모든 dependent module들이 forward pass를 거치지 않았을 경우 update될 수 없다
- multiple asynchronus agent가 있거나, 분산처리 할 경우 slowest client에 종속될수밖에 없기 때문에 computation time이 생각보다 길어질 수 있다
- 이 논문에선 update locking을 없앤다
	- 현재 layer의 weight의 gradient를 계산하기 위해 backprop 대신, 현재 hidden layer의 output만으로 gradient를 유추할 수 있는 방식을 사용
	- Decoupled Neural Interface 를 만들었다

## Decoupled Neural Interface
- 원래 방식이라면
	- layer a에서 $h_a$를 다음 layer b로 보냄
	- b에서 또 밑으로 쭉 간후 마지막에 loss를 계산한 후 다시 위로 gradient $ \delta_a$ 를 보내는 방식
- 여기서는 Asynchronous learning을 위해, $\delta_a$를 approx. 하는 새로운 모듈을 만든다
	- $h_a, s_B$ (state of layer B), $c $ (additional information, ex : label) 을 input으로 받아서 $\hat{\delta_a}$를 layer a로 되돌려주는 모듈을 추가할 것이다. 이렇게 함으로서 layer a와 layer b는 *update decoupled* 된다.
	- $ \hat{\delta_a} $를 **Synthetic Gradient**라고 한다.

### Synthetic Gradient for Feed Forward Nets
- N layer짜리 DNN을 생각하자. layer는 각각 input $x_i$를 받고 output $h_i = f_i(x_i) $를 내보낸다. $(x_i = h_{i-1})$
- $F_1^N$을 layer 1부터 N까지의 네트워크로 표기하자
- 원래 backprop에 따른 update rule

{:.center}
$$ \theta_i := \theta_i - \alpha \frac{\partial L}{\partial \theta_i} = \theta_i - \alpha \delta_i \frac{\partial h_i}{\partial \theta_i}; \; \delta_i = \frac{\partial L}{\partial h_i}$$

- 여기서는 1~i와 i+1~ 을 decouple 시키기 위해 $h_i$를 module로 보내서 error gradient의 approximation $\hat \delta_i$를 받아와서,

{:.center}
$$ \theta_n := \theta_n - \alpha \hat\delta_i \frac{\partial h_i}{\partial \theta_i}; \; n = \{ 1, ..., i \}$$

- 모듈의 경우 실제 error $\delta_i$가 계산될 때까지 기다린 후, loss를 대충 실제 error와 approx. error의 L2 distance 정도로 잡아서 update. synthetic gradient 계산하는 방식은 단순 MLP (0~3 layers)
- 같은 방식으로 모든 layer를 decouple할 수도 있음

### Synthetic Gradient for Recurrent Networks
- Recurrent Net의 경우, unroll할 timestep의 크기 T를 상정한 후
	- $L_t$에서 $L_{t+T}$까지 계산한 뒤 여기서 synthetic gradient를 사용해서 update
	- 그 뒤 f를 update하고, 다시 $L_{t+T}$에서 $L_{t+2T}$계산한 뒤 synthetic gradient를 사용... 하는 느낌

## Results
- MNIST와 CIFAR-10으로 실험을 해보았을 때, small accuracy loss가 있었지만 어쨌든 gradient approx.로도 학습이 가능했음
- gradient decouple의 장점? 을 살린 실험을 위해, 항상 backprop이 일어나는 게 아니라 every layer에 대해 각각 p의 확률로만 backward pass가 일어나는 상황을 가정. 이 경우에도 대충 학습이 되었음
	- update만 decouple 된 상황이 아니라, forward pass도 decouple된 상황을 만들 수 있음. 이에 대한 실험 결과도 대충 나옴
	- forward pass도 update pass에서 module 만들었던 것 같이 각 layer 앞에 input x만으로 그 layer로의 input을 approx. 하는 모듈을 하나 놓을 수 있을것
- RNN의 경우, 신기하게도 synthetic gradient를 사용한 모델이 원래 RNN보다 잘함
- Multi-Network의 경우. Feedback이 같은 time range에 일어나지 않는 경우, feedback을 많이 받을 수 있는 Network가 feedback을 받지 못해서 훨씬 느리게 학습할 수 있다. DNI를 사용하면 이를 해결해서, 많은 update 가능.
	- Multi-Network의 효율적인 연결에 관한 가능성 제시
	
## 정리
- 결국 backprop에 필요한 error gradient를 (한 layer의 값을 이용한) approximation 만으로 예측해서 업데이트 할 수 있다는 것이다. 막연하게 해 본 생각인 것 같기도 한데.. 실제로 된다는 게 신기하다. RNN Structure의 경우 기존 모델보다 나은 결과를 낸 실험결과도 있었고
- 이 DNI라는 concept 자체는 일단 지금으로서는 무지막지한 병렬화가 가능한, 즉 자본과 머신들이 있는 상황이 아니면 큰 의미가 없을 것 같다는 생각이 든다.
- 하지만 Multi-Network Model을 보면 어느 정도 활용 가능성이 생길 것 같기도. frequency가 다른 모델들의 효과적인 결합/학습 등을 노릴 수 있다.