---
date: 2025-05-18
layout: post
title: "푸리에 변환과 Positional Encoding의 철학"
subtitle: Sinusoidal APE부터 RoPE, YaRN까지.
description:
image:
optimized_image:
category:
tags:
- RoPE
- Positional encoding
- Relative positional encoding
- RoPE scaling
- YaRN
- Fourier transform
author: minseopjung
paginate: true
math: true
---

# Positional Encoding의 등장
Transformer 아키텍쳐에 대해 조금이라도 알고 있는 사람이라면 positional encoding에 대해 들어봤을 것이다. 순서가 내장된 RNN 아키텍쳐와 달리, 순수하게 Attention 메커니즘으로 구성된 Transformer 아키텍쳐는 모든 입력을 동시에 처리하기에 positional encoding과 같은 방법으로 위치 정보를 별도로 입력해주어야 한다.

예를 들어, "친구가 밥을 사줘서 같이 밥을 먹었다."라는 문장에서 첫번째 "밥"과 두번째 "밥"은 동일한 임베딩 벡터 값을 가지기 때문에, 위치 정보가 포함되지 않는다면 위 문장은 "밥을 밥을 친구가 사줘서 같이 먹었다."와 같은 이상한 문장과 같은 문장으로 받아들여 질 것이다. 그렇다면 우리가 학습시키고자 하는 모델이 어떻게 정상적인 문장을 이해하거나 생성할 수 있겠는가?

"Attention Is All You Need"에서는 "Positional Encoding"이라는 **위치 정보를 나타내는, 그리고 학습되지 않는** 별도의 벡터들을 구성하고, 이를 입력 임베딩 레이어에 더해주었다(Summed). 이후에 다양한 positional encoding의 변종들이 나왔기 때문에, 이 전통적 방법은 흔히 "Sinusoidal Absolute Positional Encoding"이라는 이름으로 종종 분류된다.

## "Sinusoidal" (Absolute) Positional Encoding
이름도 복잡한 이 인코딩을 본 게시글에서는 SinAPE라고 줄여서 지칭하겠다. 먼저 수식을 살펴보자.
$$
PE(pos,2i)=sin(pos/\omega)\newline
PE(pos,2i+1)=cos(pos/\omega)\newline
\text{where}\space\omega=10000^{2i/d}
$$
이름에서 알 수 있듯, SinAPE는 sin/cos 함수를 이요해 위치 정보를 표현한다. 구체적으로, 특정 위치 ($pos$)에서의 임베딩 공간에 대해 짝수번 인덱스($i=0, 2, 4, ...$)는 sin함수, 홀수번 인덱스($i=1, 3, 5, ...$)는 cos함수로 나타낸다.

이렇게 sin과 cos를 함께(번갈아) 사용하는 또 다른 사례는 푸리에 변환(Fourier Transform)에서 찾아볼 수 있다. 푸리에 변환은 시간에 대한 함수(시간->진폭)를 주파수에 대한 함수(주파수->진폭)으로 변환해주는데, 이 때 신호를 다양한 주파수 (고주파/중주파/저주파) 성분으로 분해한다. 구체적으로 말하면, **서로 다른 주파수를 가지는 사인파와 코사인파의 선형 조합**으로 분해한다. 다시 말해 시간 도메인의 정보를 주파수 도메인으로 변환함으로써 신호가 어떤 주기성을 가지고 있는 지를 정밀하게 분석할 수 있게 해준다.

sin함수와 cos함수는 서로 직교하기 때문에, 어떤 신호든 이 둘의 선형 조합으로 완벽하게 표현할 수 있으며, sin과 cos는 모든 신호 표현의 보편적인 기저(basis) 역할을 한다. SinAPE는 토큰 시퀀스 내의 각각의 임베딩 벡터들을 서로 다른 신호들로 간주한다. 임베딩 벡터의 각 차원($i$)은 특정 주파수($\omega$)를 담당하며, 토큰의 위치($pos$)에 따라 해당 주파수의 위상이 변화하는 구조를 갖는다.

 $PE(pos,2i)$와 $PE(pos,2i+1)$는 동일한 위치($pos$)에 대해 동일한 주파수를 공유하지만, 서로 위상이 90도만큼 차이 나는 사인파와 코사인파로 구성되어 있다. 낮은 차원(고주파)은 짧은 거리의 미세한 변화를 인식하고, 높은 차원(저주파)는 긴 거리의 전체적, 구조적 차이를 인식한다.

 SinAPE는 학습 파라미터가 아닌 학습되지 않는 결정론적 파라미터들을 사용하지 때문에, 추가적인 학습의 소요나 오버피팅의 위험이 없으며, 학습 중 본적 없는 긴 문장에 대해서도 사용할 수 있다. Attention Is All You Need의 저자들은 학습 가능한 파라미터 매트릭스를 이용해서도 Absolute Positional Encoding (Learnable APE)을 구현해보았는데, 더 떨어지는 성능을 보였다고 한다. (Learnable APE는 학습 중 본 적 없는 문장 길이에 대해서는 일반화되지 못하낟.)

SinAPE에도 한계는 존재한다. 먼저, SinAPE는 절대적(absolute) 위치만 인코딩한다. 만약 "나는 밥을 먹었다"와 "밥을 나는 먹었다"라는 두 문장이 입력된다면, 둘 사이의 의미 있는 차이를 학습하는 데 어려움을 겪을 수 있다.

## Relative Positional Encoding (RPE)
RPE는 이와 같은 APE의 한계를 해결하고자 등장하였다. RPE는 두 토큰 사이의 상대적 거리($pos_i - pos_j$)를 인코딩한다. 추가적으로, 입력 임베딩에 더해주던 APE와 달리 RPE는 어텐션 스코어를 계산해주는 부분(qk 내적)에 위치 정보를 삽입해준다. 그 구현 방식은 여러가지가 있으며, Transformer-XL, T5, DeBERTa와 같은 논문들에서 사용하였다. Transformer-XL은 Sinusoidal function을, T5는 learnable scalar bias를 어텐션 스코어에 더해주는 방식을 사용하였다.

RPE는 (구현에 따라 다르긴 하지만) 일반적으로 표현 가능한 거리의 범위가 제한되는 점, 학습 가능한 파라미터가 도입되는 점, 위치 정보가 어텐션 스코어에 제한적인 영향을 미치는 점이 한계로 지적되고 있다.

---page-break---

# Rotary Positional Encoding (RoPE)
최근 LLaMA를 비롯한 대부분의 최신 LLM에서는 RoPE를 positional encoding으로 사용한다. 마치 RoPE는 위치 인코딩 분야의 끝판왕인것처럼 보인다. 하지만 정말 그럴까? RoPE의 개념과 함께 왜 많은 최신 LLM들이 RoPE를 차용하고 있으며, RoPE의 한계점과 이를 극복하기 위한 여러 방안들을 알아보자.

RoPE는 이름에서 알 수 있듯, 벡터의 회전(Rotation)을 이용하여 임베딩에 위치 정보를 삽입하는 방식으로 작동한다. 잠시 학부 컴퓨터 그래픽스 시간의 기억을 더듬어 회전 행렬에 대해 상기해보자. 벡터 $\left[x, y\right]^T$를 각도 $\theta$만큼 회전하려면, 다음과 같이 회전 행렬을 행렬곱해주면 된다.
$$
\begin{bmatrix}
x' \\
y' 
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\ 
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
x \\
y 
\end{bmatrix}
$$

위와 같은 회전 연산을 복소수 공간에서 오일러 공식을 이용해 구현하기도 한다. 가령 위와 같은 벡터 $\left[x, y\right]^T$를 복소수로 나타내면 $z=x+iy$와 같은 꼴이 되는데, 여기에 오일러 공식 $e^{i\theta}=\cos\theta+i\sin\theta$을 곱해주면 결과적으로 $z'=(x\cos\theta+y\sin\theta)+i(-x\sin\theta+y\cos\theta)$로 위의 행렬곱 결과를 복소수로 표현한 것과 같은 결과가 나온다. 복소수를 이용한 회전 연산과 RoPE의 원리는 [이 Medium 블로그](https://medium.com/@hugmanskj/mastering-llama-rotary-positional-embedding-rope-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-9b1963a22852)에 정말 잘 정리되어 있고 필자도 도움을 얻었다. 궁금하신 분들은 참고하시도록.

아무튼 $d$ 차원의 임베딩 벡터에 대해 회전 연산을 수행하기 위해, $d$ 차원을 두 개 씩 짝지었다. 이를 통해 총 $d/2$개의 Pair가 만들어지고, 여기에 서로 다른 각도만큼 회전 연산을 적용하여 위치 정보를 삽입할 수 있다. 이 때 적용할 회전의 각도는 다음 수식과 같이 계산된다.
$$
PE(pos,2i)=\cos(\theta_i(pos))\\
PE(pos,2i+1)=\sin(\theta_i(pos))\\
\text{where}\space\theta_i(pos)=\frac{pos}{10000^{2i/d}}
$$

이 때 $i$는 $pos$ 위치의 임베딩 벡터에서의 차원 인덱스이다. SinAPE 수식에서 많이 본 꼴이다. 실제로 SinAPE와 RoPE의 위치 임베딩 같은 푸리에 변환적 motivation을 공유하며, 동일한 주파수 구조를 가진다. 다만 가장 핵심적인 차이가 있다면, SinAPE는 입력 임베딩 벡터에 더해(Summation)주는 반면, **RoPE는 어텐션 계산 시 query와 key 벡터에 직접적으로 곱해진다**. 이러한 덧셈과 곱셈 연산의 차이로 인해 단순히 사인/코사인 주기들을 입력에 더해주는 SinAPE와 달리, **RoPE는 벡터를 실제로 회전시켜주는 효과**를 보여준다. 

또한 query와 key에 위치 정보를 임베딩해주기 때문에, qk의 dot product시 위치 정보를 구조적으로 유지시켜줄 수 있고, 각도간의 차이를 통해 토큰 간의 상대적 거리 정보도 임베딩해줄 수 있다.

## RoPE Scaling: RoPE의 컨텍스트 한계를 넘어서
이러한 RoPE에도 실질적인 한계가 존재하는데, 바로 RoPE가 훈련 당시 설정된 최대 sequence length 이상으로는 일반화가 잘 되지 않는다는 점이다. RoPE의 각도 계산 수식을 따르면, 토큰의 위치 인덱스($pos$)가 클수록 각도 $\theta_i$가 빠르게 증가한다(즉, 과도하게 회전한다). 특히 $\omega=10000^{2i/d}$가 큰 고주파 성분에서는 위치 $pos$가 조금만 증가해도 각도는 훨씬 빠르게 회전한다. 그러다가 각도가 $2\pi$에 도달하면 sin/cos값이 주기적으로 되풀이되고, 이 지점에서부터는 위치 정보의 구분이 어려워 진다. 

RoPE Scaling은 이러한 한계를 방지하고자 회전 각도의 증가를 늦추는 기법이다. 가장 간단한 Rope Scaling 방법은 Position Interpolation(PI)로, scaling factor $s$를 곱해주어 위치 인덱스를 스케일링하고, 더욱 촘촘하게 만들어 긴 시퀀스 길이에서 일반화 가능하도록 만들어준다.
$$
\theta_i(pos)=\frac{pos\cdot s}{\omega}
$$
이 때 scaling factor $s$는 일반적으로 $\frac{L}{L'}$로 설정한다. $L$은 원래 길이, $L'$는 확장하고자 하는 길이이다.

PI는 고주파든 저주파든 똑같이 줄이기 때문에, 서로 가까운 위치에 있는 토큰들 간의 위치 정보가 희미해지는 한계가 있었다. 즉, 긴 거리 학습은 가능해지지만, 짧은 거리 정밀도는 희생하는 것이다. **YaRN(Yet another RoPE extrapolatioN)**에서는 PI에서 더 나아가 **저주파수(고차원)는 적게 줄이고, 고주파(저차원)은 더 많이 줄이자**는 철학으로 디자인된 RoPE Scaling을 제안한다. 저주파수는 원래 회전을 유지하여 긴 길이에서의 표현력을 유지하는 한편, 고주파는 강하게 스케일링하여 위치 정보가 중첩되는 것을 억제하고자 했다.

이번 포스팅에서는 글이 길어져 YaRN의 기본적인 도입 철학에 대해 짧게 다루었다. 기회가 된다면 NTK-aware scaling부터 시작하는 YaRN의 원리에 대해 자세히 다뤄보겠다.