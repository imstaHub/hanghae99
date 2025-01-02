## Q1) 어떤 task를 선택하셨나요?
> NER 선택!


## Q2) 모델은 어떻게 설계하셨나요? 설계한 모델의 입력과 출력 형태가 어떻게 되나요?
> dataset은 word 단위로 처리가 되어 있어, 가장 긴 문장의 길이는 104여서 input을 104 토큰으로 맞췄습니다.
> label은 총 17개로 되어 있었습니다.
> 출력의 경우 token size x number of labels로 104 x 17 (batch 제외)


## Q3) 어떤 pre-trained 모델을 활용하셨나요?
> 기본과제에서 사용했던 'distilbert' 모델을 사용했습니다.


## Q4) 실제로 pre-trained 모델을 fine-tuning했을 때 loss curve은 어떻게 그려지나요? 그리고 pre-train 하지 않은 Transformer를 학습했을 때와 어떤 차이가 있나요? 
> dataset을 전처리 하느라, 사실상 모델 학습을 다양하게 시도해보지 못했습니다.
> 20 epoch으로 학습을 해봤을 때, loss가 작아지는 것을 확인하였습니다.


### 위의 사항들을 구현하고 나온 결과들을 정리한 보고서를 README.md 형태로 업로드
### 코드 및 실행 결과는 jupyter notebook 형태로 같이 public github repository에 업로드하여 공유해주시면 됩니다(반드시 출력 결과가 남아있어야 합니다!!) 

