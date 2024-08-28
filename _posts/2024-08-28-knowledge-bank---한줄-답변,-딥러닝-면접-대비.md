---

date: 2024-08-28 06:56:38
layout: post
title: "Knowledge Bank - 한줄 답변, 딥러닝 면접 대비"
subtitle: "개인적으로 딥러닝/머신러닝 엔지니어 면접을 준비하며 정리한 지식들입니다. 가능한 한 짧고 간결하지만 핵심을 담은 답변을 준비하려 했습니다. 다만 경우에 따라 일부 내용은 너무 minor한 부분이 포함되어 있을 수 있습니다."
description:
image: "https://st2.depositphotos.com/2121483/7207/v/450/depositphotos_72072995-stock-illustration-people-profile-heads-in-dialogs.jpg"
optimized_image:
category: "Career"
tags:
- "Job Interview"
- "Career"
- "Deep Learning"
- "Machine Learning"
author: minseopjung
paginate: true
math: true
---

# 1. Data Structure / Algorithm
# 2. Artificial Intelligence

## Deep Learning

**CNN (Convolutional Neural Network)**란 무엇인가? 이미지 데이터를 학습하는 데 주로 사용되는 특징 추출자(Feature extractor)다. Convolution 연산을 주요 구성 요소로 가지며, pooling이나 배치 정규화 레이어 등을 포함하기도 한다. 2차원의 경우, $k \times k$의 convolution kernel(혹은 filter)를 데이터에 대해 sliding하며, kernel의 가중치들과 데이터의 값들 (이미지 데이터의 경우 pixel value)과 element-wise product를 수행한 뒤, aggregate한다. 이러한 계산 과정을 반복하면 데이터의 지역적 특징들이 aggreagte된 feature map을 얻을 수 있다.

**Convolution 연산의 출력 차원은 어떻게 계산되는가?** $\frac{W - k + 2*P}{S} + 1$ 이 때, W는 데이터의 width, k는 kernel size, P는 padding size, S는 stride이다.

**왜 CNN은 이미지 데이터에 효과적인가?** 이미지 데이터는 지역적 정보가 중요한 데이터이다. 예를 들어, 사람 얼굴 이미지라고 하면 눈, 코, 입, 귀 등등의 semantic한 정보는 인접한 픽셀들이 담고 있다. 따라서 이러한 지역적 정보를 convolution을 통해 추출하는 CNN이 이미지 데이터의 특징 추출에 효과적이다.

**Fully Convolutional Neural Network**는 convolution & deconvolution 연산으로만 구성된 CNN이다. 기존 CNN과 달리 fully-connected layer가 들어가지 않기 때문에, 입력 사이즈를 보존해야 하는 segmentation 태스크 등에서 유리하며, 더 파라미터 개수 측면에서 효율적이다.

**Deconvolution** 연산은 무엇인가? Convolution 연산을 역으로 수행하는 연산으로, convolution 이전의 사이즈로 데이터를 upsampling하는 연산이다. 주로 transposed convolution이나 upsampling 기법을 사용한다.

**Transposed convolution**은 무엇인가? 학습가능한 커널을 두고 입력 원소를 커널과 곱해 출력의 대응되는 자리에 배치시킨다.

그렇다면 transposed convolution이 아닌 **Upsampling** 기법은 무엇이 있는가? 최근접 픽셀값으로 채운다. 0으로 채운다. 보간값을 채운다. (pooling의 경우) pooling 전의 값을 기억해두었다가 채운다. 등등 여러 기법이 있다.

**CapsuleNet**은 CNN의 max pooling 연산이 가지는 translation invariance를 지적하며 제안되었다. Max pooling 과정에서 위치 정보가 손실될 수 있고, convolution 연산은 태생적으로 이미지 내 객체의 위치에 무관하게 학습되기 때문에 전역적 정보가 충분히 보존되지 않을 수 있다. Convolution의 수용 영역 한계는 **ViT**의 attention mechanism으로도 극복할 수 있다.

**Pointwise Convolution**은 무엇인가? 1x1 컨볼루션을 적용하여 이미지 크기는 유지하되 차원(채널)을 조절하기 위해 사

**Depthwise Convolution**은 무엇인가? 한 개의 커널이 한 개의 채널에 대해 컨볼루션을 수행한다. 따라서 이미지 크기는 변하되 채널 수는 불변한다. 

**Depthwise Separable Convolution**은 무엇인가? Depthwise convolution으로 이미지 크기를 조절하고, Pointwise convolution으로 채널 수를 조절하는 방식이다. 두 과정을 분리해서 적용하기 때문에 기존 convolution에 비해 더 적은 연산량을 요구한다.

**Vision Transformer**란 무엇인가? 이미지를 픽셀의 집합인 패치 단위로 나누어 이를 토큰으로써 트랜스포머 모델에 입력해 학습시키는 방법. Attention mechanism을 통해 각 이미지 패치간의 global dependency를 학습할 수 있다.

**RNN (Recurrent Neural Network)**이란 무엇인가? RNN은 여러 time-step에 걸쳐서, 현재 time-step에서 신경망의 출력이 다음 time-step의 신경망의 입력으로 들어가는 구조를 띄고 있다. 이러한 구조는 태생적으로 순서가 중요한 자연어 데이터, 시계열 데이터 등의 시퀀스 데이터의 특징을 추출하는데 유리하다.

**LSTM (Long-Short Term Memory)**이란 무엇인가? LSTM은 RNN의 한 종류인데, vanilla RNN에 비해 더 복잡한 구조를 지닌다. RNN과 달리 장기기억 메모리와 단기기억 메모리 두 개의 path로 hidden state가 time-step을 걸쳐 전달된다.

**RNN의 한계는 무엇인가?** Long-term dependency와 gradient vanishing 문제가 있다. Long-term dependency는 시퀀스의 길이가 길어질 수록, RNN이 시퀀스 앞쪽의 내용을 까먹는 문제이다. Gradient vanishing도 비슷하게 시퀀스 길이가 길어질수록, 역전파 때 gradient가 앞쪽까지 전달되지 못하는 문제이다. Gradient가 1보다 작은 값으로 계산될 때, chain rule에 의해 gradient가 곱해지면서 출력 레이어에서 멀어질수록 0에 가까운 값이 계산되기 때문이다. 이를 완화하기 위해 LSTM이 고안되었으나 순환 구조의 구조적 문제 때문에 완전히 해결하지는 못한다.

**Attention mechanism이란 무엇인가?** Query, key, value 세 가지 input에 대해 query와 key의 유사도(i.e. attention score)를 계산하고, 이를 다시 value값과 내적하는 방식으로 계산되는 특징 추출 방식이다. 위와 같은 RNN의 문제를 완화하고자 제안되었는데, 매 RNN time-step마다 현재 step의 RNN hidden state를 query, 이전 step들의 RNN hidden state들을 key, value로 사용하여 attend하면 time-step이 길어지더라도 이전 값들을 참조하기 때문에 long-term dependency와 gradient vanishing 문제를 해결할 수 있다.

**Transformer란 무엇인가?** Attention mechanism으로만 구성된 신경망 구조이다. 기본적으로 기계번역을 위한 encoder-decoder 구조로 제안되었으나, encoder나 decoder만 사용하여 다양한 태스크에 적용된다. Self-attention mechanism을 새롭게 제안하여 입력 시퀀스 내에서의 장거리 종속성 학습을 유리하게 만들었으며, 효율적인 병렬처리가 가능하게 했다. 

**BERT**란 무엇인가?

Diffusion Models, DDPM, DDIM

--page-break--


## Information Theory & Statistics

**정보량**이란 무엇인가? 어떤 정보를 표현하는데 필요한 자원(비트)의 수를 의미한다. $-\log{P(x)}$와 같이 표현된다. 예를 들어, 동전을 던져서 앞면이 나오는 사건의 정보량은 $-\log{\frac{1}{2}}=1$로, 1비트로 해당 사건을 표현할 수 있다. 확률이 낮은 사건일수록 (예측하기 힘들고 깜짝 놀랄만한 일이기에) 정보량이 높다.

**엔트로피**란 무엇인가? 사건의 확률 분포에 대한 정보량이다. 즉, 정보량의 기댓값이다. $-\sum_{x\in X}P(x)\log{P(x)}$ 균등분포에서 최대값$\log{n}$을 가진다.

**교차 엔트로피**란 무엇인가? 두 확률 분포 사이의 차이를 측정하는 지표이다. $Q(x)$를 예측 확률 분포, $P(x)$를 실제 확률 분포라고 하면, 교차 엔트로피는 $-\sum_{x\in X}P(x)\log{Q(x)}$와 같이 계산된다. $P(x)$와 $Q(x)$가 같은 값을 가질 때 0의 값을 가지고, 완전히 다른 값(0/1)의 값을 가질 때 최소값$-\inf$을 가진다.

**이진 교차 엔트로피**란 무엇인가? 클래스의 개수가 두개일때의 교차 엔트로피이다. $-\sum{Q(x)\log{P(x)}+(1-Q(x))\log{(1-P(x))}}$

**KL-Divergence**란 무엇인가? 두 확률 분포의 차이를 나타낸다. 비대칭적이며, 항상 0보다 큰 값을 가진다. $\sum{P(x)\log{\frac{P(x)}{Q(x)}}}$.

**KLD와 교차 엔트로피의 관계는 무엇인가?**교차 엔트로피는 실제 확률 $P(x)$에 대한 엔트로피 $H(P)$와 $D_{KL}(P||Q)$의 합으로 나타낸다. KLD는 교차 엔트로피에서 엔트로피를 뺀 값이다. 즉, 교차 엔트로피는 두 분포의 절대적인 차이를 나타내는 반면, KLD는 상대적인 차이를 나타낸다 (KLD는 상대 엔트로피라고도 불린다). 다중클래스 분류 문제에서, 실제 확률 P(x)는 정답 레이블에서 1, 나머지는 0이기 때문에, $H(P)=0$이 되고, 따라서 KLD=교차엔트로피가 된다.

**교차 엔트로피는 왜 딥러닝 손실함수에 사용되는가?** 먼저, 두 분포의 차이를 나타내는 지표이다. 교차 엔트로피의 최소화는 최대우도추정과 같다. 교차 엔트로피는 로그우도의 기댓값의 음수와 같기 때문이다. 또한 교차 엔트로피는 예측와 정답이 일치할 떄의 확률의 하한과 같다.

**우도**란 무엇인가? 주어진 데이터가 특정 모델에서 나올 확률을 의미한다. 즉, 주어진 데이터가 특정 모델에서 나올 확률이 높을수록, 그 모델이 데이터를 잘 설명한다고 볼 수 있다. 우도는 $P(D|M)$로 표현되며, 데이터 D가 주어졌을 때 모델 M이 맞는지를 나타낸다. 이를 최대화하는 것이 최대우도추정이다. 최대우도추정은 주어진 데이터에 대해 가장 가능성이 높은 모델 파라미터를 찾는 방법이다. 이는 교차 엔트로피와 같다.

**우도와 확률의 차이**는 무엇인가? 확률은 어떤 확률 분포가 주어졌을 때, 관측값이 해당 분포에서 얼마의 확률로 존재하는가를 나타낸다. 우도는 관측값이 해당 분포에서 나왔을 확률을 나타낸다. 확률에서는 분포가 고정되고, 우도에서는 관측값이 고정된다. 확률밀도함수에서 관측구간의 면적이 확률이고, 특정 관측값의 y값이 해당 확률분포에서의 우도이다. 참고로 연속확률분포에서 특정 관측값의 확률은 0이다 (적분을 생각해

--page-break--


## Machine Learning

**Precision**이란 양성으로 예측한 것 중에 실제 양성인 것의 비율 $\frac{TP}{TP+FP}$, **Recall**은 실제 양성인 것 중에 양성으로 예측한 것의 비율 $\frac{TP}{TP+FN}$이다. Precision은 오탐(FP)을 줄이고자 할 때, Recall은 미탐(FN)을 줄이고자 할 때 유용하다. 예를 들어, 스팸 메일을 분류할 때, 스팸으로 분류된 정상 메일을 줄이고 싶다면 Precision을 높이는 것이, 병 진단에서 병에 걸린 환자를 놓치지 않으려는 경우 Recall을 높이는 것이 유용하다.

**F1-score**는 Precision과 Recall의 조화평균으로 계산된다 $\frac{2PR}{P+R}$. 조화평균의 특성상 한쪽만 높을 때 전체 수치가 올라가는 것을 방지하기에, F1-score는 Precision과 Recall 모두 균형있게 평가하고자 할 때 유용하다. 다중분류 태스크에서 F1-score 등의 지표를 계산할 때, **Macro-Average**와 **Micro Average**가 사용된다. 전자는 각 클래스별로 평균 낸 것이고 $\frac{1}{C}\sum_{i=1}^C F1_i$, 후자는 각 샘플별로 평균 낸 것이다 $\frac{1}{N}\sum_{i=1}^N F1_i$.

    

**Sensitivity**는 Recall과 같은 의미로, TPR이다. **Specificity**는 실제 음성인 것 중에 음성으로 예측한 것의 비율로, TNR이다. Sensitivity(TPR)와 1 - Specificity(FPR)을 이용해 만든 그래프가 ROC Curve인데, x축에 FPR이, y축에 TPR이 들어간다. 그래프가 좌상단에 위치할 수록 성능이 좋다는 것을 의미하고, 이에 대한 Area Under Curve (AUC)가 그것을 수치로 나타낸다. ROC Curve는 분류 임계값의 변화에 따라 변화하는 성능을 확인할 수 있어 유용하다.

    

    

**Soft Vector Machine (SVM)**이란 무엇인가? 데이터 공간을 둘로 나누는 hyperplane을 찾는 알고리즘이다. Hyperplane 가까이에 위치한 데이터포인트(i.e. Support vectors)와 초평면 사이의 거리(Margin)을 최대화하는 hyperplane을 찾으면 된다.

    

    

**Random Forest**란 무엇인가? 여러 개의 Decision Tree를 독립적으로 학습시켜 최종적으로 개별 예측을 평균내거나 투표를 통해 결정하는 앙상블 기법이다. 이 때 **Decision Tree**란, 데이터의 특징을 노드로 갖는 트리 구조를 만들어 여러 가지 특징에 따라 분류/회귀를 수행하는 알고리즘이다.

    

    

**앙상블 학습**은 무엇인가? 여러 개의 모델을 결합하여 하나의 정확하고 안정적인 예측 모델을 만드는 기법이다. **Bagging (Boostrap Aggregating)**은 여러 개의 모델을 학습시키고 그 결과를 평균내거나 투표로 결합한다. **Boosting**은 여러 개의 모델을 순차적으로 학습시키며 이전 모델의 오차를 다음 모델이 줄이는 식으로 학습을 진행한다. RF는 Bagging에, GBM은 Boosting에 속한다.

--page-break--


## Reinforcement Learning

**Monte Carlo** 방법은 한 에피소드가 끝날 때까지 에이전트가 환경과 상호작용하며 보상을 수집하고, 에피소드 종료 후 계산한 보상으로 가치 함수를 업데이트 하는 방법이다. 반면 **Temporal Difference**는 특정 time-step마다 현재 가치 함수의 차이를 이용해 업데이트하는 방법이다. MC는 에피소드가 끝나야 학습이 가능하기 때문에 긴 에피소드를 갖는 문제에서 비효율적이며, 총 보상을 한번에 업데이트 하기에 분산이 크다. 반면 TD는 에피소드가 끝나기 전에 총 보상의 근사치로 학습하기 때문에 편향이 발생할 수 있다.

TD에는 여러 가지 변형이 있는데, 먼저 **TD(0)**는 가장 기본적인 형태로 매 time-step마다 가치 함수를 업데이트 하는 방법이다. **n-step TD**는 n번의 time-step동안의 보상을 고려하여 학습하기 때문에 TD(0)에 비해 편향이 감소한다. **TD($\lambda$)**는 $\lambda$라는 파라미터를 두어 TD(0)와 MC 사이의 균형을 조절하는 방법이다. $\lambda$가 1이면 MC와 동일하다. 

**On-policy** 학습과 **Off-policy** 학습은 무엇인가? 전자는 에이전트가 행동을 선택하는 정책과 학습되는 정책이 동일한 반면, 후자는 다르다. 예를 들어, 에이전트가 행동을 취할때는 $\epsilon$-greedy하게, 학습할 때는 greedy하게 행동을 선택한다면 off-policy이다. 전자는 매 step마다 데이터 샘플을 한번 보고 버리기 때문에 sample efficiency가 비교적 낮은 반면 후자는 과거 정보를 꺼내 쓸 수 있어 환경과의 상호작용을 적게 해도 된다. 하지만 과거와 현재의 policy가 많이 달라진 경우 학습이 어려울 수 있다.

**Online** 학습은 에이전트가 직접 환경과 상호작용하며 데이터를 수집하고 학습하는 방법이고, **Offline** 학습은 실시간 상호작용이 불가능하며 이전에 기록된 플레이 기록을 가지고 학습한다. Offline 학습은 필연적으로 off-policy이지만, Online 학습은 on-policy와 off-policy가 둘 다 가능하다.

**Policy Gradient**란 정책 함수를 파라미터화하여 gradient를 계산하고, gradient ascent를 통해 보상을 최대화 하도록 하는 방법이다. 매우 큰 상태 공간에 대해 효율적으로 처리할 수 있다. **REINFORCE**는 Monte Carlo 방식으로 여러 에피소드로부터 얻은 return을 기반으로 policy gradient를 수행하는데, 간단하지만 큰 분산을 갖는다. **Actor-Critic**은 정책을 업데이트하는 Actor와 가치 함수를 추정하는 Critic을 각각 학습시켜 Critic의 추정을 바탕으로 Actor를 학습시키는 방식이다.

**Policy gradient**

--page-break--

# 3. Data Science / Mining / Analysis

**독립변수**란 무엇인가? 다른 변수에 영향을 주는 "원인"이 되는 변수로, $y=ax+b$에서 $x$에 해당한다. **종속변수**란 무엇인가? 다른 변수에 영향을 받는 "결과"가 되는 변수로, $y=ax+b$에서 $y$에 해당한다.

**가설 검성**이란 무엇인가? 통계적 기법으로 가설의 타당성을 검증하는 방법이다. 어떤 결과가 통계적으로 유의미한 결과인지, 아니면 단순 샘플링 에러에 불과한지를 판단한다. 이는 **귀무 가설**과 **대립 가설**을 세우고, 표본 데이터를 통해 귀무 가설을 채택할지 기각할 지를 통해 결정한다. 이 때 귀무 가설은 "A와 B가 차이가 없다"는 가설이고, 대립 가설은 "차이가 있다"는 가설이다. 귀무 가설을 거짓으로 판단하고 기각하면, A와 B는 통계적으로 유의미한 차이가 있다고 말할 수 있다. 이 때 t-test나 ANOVA-test와 같은 방법을 사용하여 표본 데이터를 통해 기각과 채택을 판단한다.

**Type I Error**와 **Type II Error**는 무엇인가? 위의 귀무 가설이 참인데 (즉 차이가 없는데) 차이가 있다고 판단하여 기각하는 것이 Type I Error고, 반대로 귀무 가설이 거짓인데 (즉 차이가 있는데) 차이가 없다고 판단하여 채택하는 것이 Type II Error이다. 

**p-value**란 무엇인가? 귀무 가설이 참일 확률을 나타낸다. 즉 p-value가 낮으면 귀무 가설이 거짓일 확률이 높으므로 통계적으로 유의미한 결과라고 판단할 수 있다. 일반적으로 p-value가 0.05보다 낮으면 유의미한 결과라고 판단한다.

왜 귀무가설은 **부정**의 형태로 설정되는가? "차이가 있다"보다 "차이가 없다"가 검증하기 쉽기 때문이다. 차이가 있다는 말은 너무 모호하고 광범위하다. 또한, 과학은 신중하기 때문에 새로운 가설에 대해서는 항상 회의적이다.
**데이터 마이닝의 5단계 프로세스.**

**1단계, 문제 정의 Problem Definition.** 해결하고자 하는 문제를 (구체적, 과학적으로) 정의한다. 예를 들어, "중환자실 산소호흡기에서 이상이 발생하는 문제를 해결하고 싶다." $\rightarrow$ "중환자실 산소 호흡기에서 수집된 breath data 중에서 normal과 abnormal 데이터를 분류하고자 한다."

**2단계, 데이터 수집 및 전처리 Data Collection & Preprocessing.** 문제 해결을 위해 필요한 데이터를 수집하고, 데이터의 품질을 보장하기 위해 결측치 처리, 노이즈 제거, 불완전한 데이터 제거 등 데이터를 전처리하는 과정이다.

**3단계, 데이터 표현 Data Representation.** 수집된 데이터를 데이터 마이닝 알고리즘 적용에 적합한 형태로 변환하는 과정이다. 이 때, 정규화, 이산화, 특징 공학 등이 사용될 수 있다. 경우에 따라서는 raw data가 사용되기도 한다.

**4단계, DM Function/Techniques.** 머신러닝, 통계분석, 시계열분석 등... 질적, 양적 분석 모두 이루어지며, 둘은 상호보완적이다.

**5단계, Evaluation & Interpretation.** 분석 결과의 신뢰성과 유용성을 평가하고, 분석 결과로부터 결론 (지식)을 도출한다.

**Curse of Dimensionality 차원의 저주**란 무엇인가? 다루고자 하는 데이터의 차원이 커짐에 따라 발생하는 문제로, 차원이 커짐에 따라 데이터 공간의 밀도가 기하급수적으로 낮아지고, 이에 따라 기계학습이나 데이터 분석(e.g. clustering)에 필요한 데이터가 희소해지는 현상을 말한다. 또한, 차원이 높아질수록 처리하기 위해 필요한 연산 복잡도가 기하급수적으로 증가하게 된다.
**Dimensionality Reduction 차원 축소** 방법은 무엇이 있는가? PCA, Attribute Subset Selection 등이 있다.
**Principal Component Analysis (PCA)**란 무엇인가? 데이터의 분산을 최대한 보존하면서 데이터의 차원을 축소하는 방법으로, 데이터의 차원이 n이라면, n보다 작은 개수의 직교 벡터(주성분 벡터)를 찾아 이들로 데이터를 표현하는 방법이다.

**Attribute Subset Selection**이란 무엇인가? 중복되거나(redundant) 무관한(irrelevant) 특징을 제거하는 방법이다. 이 때, 각 특징에 대해 통계 검정을 적용하여 각 특징들을 단계적으로 제거하거나 추가할 수 있다.

**Numerosity Reduction**이란 무엇인가? 데이터를 "양"을 줄이는 방법이다. 여기에는 _parametric_ 방법과 _non-parametric_ 방법이 있다. Parametric 방법은 데이터를 잘 표현하는 모델(e.g. regression model)을 찾아 모델의 파라미터로 데이터를 표현하는 방법이다. Non-parametric 방법은 데이터의 대표값을 찾아 데이터를 표현하는 방법이다. 예를 들어, clustering, histogram, sampling 등이 있다.

**Data Compression**은 무엇인가? 데이터를 압축하는 방법으로, 데이터의 종류에 따라 다양한 알고리즘이 존재한다. Lossless 방법은 정보의 손실 없이 압축하는 방법이며 (e.g. string), lossy 방법은 정보의 손실을 어느정도 감수하는 압축 방법이다 (e.g. audio/video).

통계 분석에서 **parametric** 방법과 **non-parametric** 방법은 무엇인가? **Parametric** 방법은 데이터를 잘 표현하는 몇 개의 매개 변수(파라미터)로 데이터를 표현하는 모델을 구성한다. 이 때, 데이터가 특정 분포를 따른다는 가정이 필요하다. **Non-parametric** 방법은 파라미터를 따로 설정하지 않고 데이터 자체만으로 모델을 구성한다. 이 때, 분포에 대한 가정은 필요치 않으나 많은 데이터가 필요하다.

데이터 시각화에서 **어떤 색깔을 선택하는 것이 좋을까?** 첫쨰, **Qualitative** color는 서로 잘 구분되는 "범주형 데이터"를 나타낼 때 유용하다. 둘째, **Sequential** color는 증가하거나 감소하는 순서가 있는 데이터를 나타낼 때 유용하다. 셋째, **Diverging** color는 중간값에서 양극으로 감소하고 증가하는 데이터를 나타낼 때 유용하다.

**상관계수 Correlation coefficient**이란 무엇인가? 두 변수 사이의 상관관계를 나타내는 지표이다. 두 변수 사이의 양의 상관관계가 높다면, 두 변수의 단조 관계 비슷하다 (e.g. 하나가 감소할 떄, 나머지 하나도 감소한다). 음의 상관관계가 높다면, 두 변수의 단조 관계가 반대이다 (e.g. 하나 감소하면 나머지 증가). 상관계수가 0에 가깝다면, 두 변수의 단조 관계가 관련이 없다. 상관계수는 Pearson's correlation과 Spearman's correlation이 대표적이다.

**Pearson's correlation** coefficient는 두 변수의 공분산을 이용해 계산되기 때문에, 데이터가 정규 분포를 따른다는 가정이 필요하다. 반면 **Spearman's correlation** coefficient는 변수의 rank(순서)에 기반하여 계산하기 때문에, 정규 분포 가정이 필요 없다. 두 지표 모두 $\left[-1, 1\right]$의 값을 가진다.

Graph 이론에서, 노드의 **Centrality** (or Prestige)란 무엇인가? 노드의 중심성 즉, 노드가 전체 그래프에서 다른 노드에 비해 얼마나 더 중요한 지를 나타낸다. Degree centrality, closeness centrality, betweenness centrality, eigenvector centrality가 있다.

**Degree centrality**란 degree(이웃 노드의 개수)가 높은 노드를 중요한 노드로 본다.

**Closeness centrality**란 다른 노드와의 평균 거리가 가장 가까운 (또는 가중치가 가장 높은) 노드를 중요한 노드로 본다.

**Betweenness centrality**란 다른 노드들과 많이 연결해주는 노드를 중요한 노드로 본다.

**Eigenvector centrality**란 중요한 노드와 연결된 노드를 중요한 노드로 본다.

자연어 처리에서 텍스트를 나타내는 방법이 무엇이 있는가?

먼저, **Bag of Words**은 텍스트 문서에 등장하는 각 단어들의 등장 빈도를 벡터로 나타낸 것이다. 그러나 vocabulary의 사이즈가 커질수록 희소 벡터가 만들어지고, 단어의 등장 순서나 문맥을 고려하지 않으며, 빈도에 집중하기 때문에 의미 없는 단어(a, the, is 등)에 높은 가중치를 준다.

**TF-IDF**는 term frequency (문서 내에서 단어의 등장 빈도)와 inverse document frequency (document frequence - 특정 단어가 전체 문서중에서 몇개의 문서에서 등장하는지 - 의 역수)의 곱으로 계산된다. 일반적으로 흔히 등장하는 의미 없는 단어들에 대해 낮은 가중치를 주고, 특정 문서에 자주 등장하는 (진짜 중요한) 단어에 가중치를 높게 준다. 그러나 여전히 문맥은 무시하는 단점이 있다.

**N-gram**은 문서 내의 N개의 연속된 단어들을 시퀀스로 묶어서 표현하는 방법이다. 예를 들어, "I love you"에 대한 Bi-gram은 ("I love", "love you")이다. N-gram은 문맥을 고려할 수 있으나, N이 커질수록 발생하는 차원의 저주를 피할 수 없다.

**Word Embedding**은 주로 학습된 딥러닝 모델을 통해 각각의 단어를 임베딩 벡터로 나타내는 방법이다. 각각의 임베딩 벡터들은 임베딩 공간에서 단어들 간의 의미적 관계를 반영하여 표현된다.

**BLEU** 점수란 무엇인가? Bilingual Evaluation Understudy, 모델이 생성한 문장과 사람이 쓴 참조 문장 사이의 유사도를 구하는 지표이다. 두 문장 사이의 n-gram을 비교하여 측정하며, 모델이 생성한 문장의 n-gram 중에 두 문장에 모두 등장하는 n-gram의 비율이다. 주로 기계 번역에서 쓰인다.

**ROUGE**란 무엇인가? 모델의 문장과 참조 문장 사이의 일치도를 평가한다. ROUGE-N은 n-gram을 기준으로, ROUGE-L은 최장공통수열(LCS)를 기준으로, ROUGE-S는 Skip-bigram을 기준으로 측정한다. 주로 텍스트 요약을 평가하는데 쓰인다.,

**ROUGE-N과 BLEU의 차이점**은 무엇인가? BLEU는 Precision 중심, ROUGE-N은 Recall 중심으로 계산한다. 따라서 BLEU는 번역의 정확성을 평가하는데 강점이 있으나 참조 문장의 모든 요소를 포괄하지는 못할 수 있다. ROUGE는 요약의 포괄성을 평가하는데 강점이 있으나 생성된 텍스트가 지나치게 길어질 수 있다. BLEU는 **생성된 텍스트가 분모**에, ROUGE는 **참조 텍스트가 분모**에 온다.


시계열 데이터의 구성 요소는 무엇이 있는가? 첫째, **추세 Trend**는 시간에 따른 데이터가 증가하거나 감소하는 경향을 나타낸다. 둘째, **계절성 Seasonality**는 주로 계절이나 기후와 같이 절대적 시간 주기에 따라 반복되는 정보를 담고 있으며 비교적 짧은 순환 주기를 가진다. Naive한 딥러닝 방법으로 예측하지 못한다. 셋째, **주기성 Cycle**은 불규칙적인 시간 간격에 따라 반복되는 정보를 담고 있으며, 비교적 장기적인 순환 주기를 가진다. 넷째, **불규칙성 Irregularity**은 외부적인 사건에 의해서 발생하는 예측할 수 없는 무작위적인 성분이다.

    

