# PEFT 실험 비교

### Library

- sentencepiece
- accelerate
- bitsandbytes

### 고려사항

- <|endoftext|>를 중간중간에 넣냐 마냐도 성능에 영향을 미치겠군...
- 

# [DailyDialog dataset]

- 일단 act로만 controll
- Action 분류하는 Bert 학습 시켰음(multi-label classification으로 학습시킨게 찝찝함)
    - 정확도 76.4 (재학습 필요)

## <모델1. Dialogpt>

### 고려사항

- dialogpt size에 대해서 실험할 수 있겠군
- Finetuning 템플릿을 수정해서 실험할 수 있겠군

### Dataset 카테고리 → 일단 act에 대해서만 tuning

act =[inform (1),question (2),directive (3) , commissive (4). 약속 또는 의무

emotion = [no emotion (0),anger (1),disgust (2),fear (3),happiness (4),sadness (5) ,surprise (6).]

### 0. pre_trained

- context를 입력으로 주고 다음 문장을 생성하게 했음. 생성된 문장을 bert에 넣어서 분류해서 정확도 얻음
    - large
    - medium 40.6
    - small

### 1. Finetuning

- loss masking(loss를 구할 애들을 남겨두자)
    - 주어진 문맥,학습 안 해도 되는게 : 0
    - 가려야되는, 학습해야 되는게 1
- Finetuning 템플릿 : 라벨<eos> +context(중간 중간 eos) + reference <eos>
    
    dialog : [’hi’, ‘how are you?’, ‘fine’] act: [inform, question, inform]
    [example1]
    qusetion<eos>hi<eos>맞춰야될꺼(how are you?)<eos>
    loss masking : 은 qusetion<eos>hi<eos> 까지 0, loss 계산해야되는게 how are you?<eos>
    
    [example2]
    inform<eos>hi<eos>how are you?<eos>fine<eos>
    loss masking : 은 qusetion<eos>hi<eos>how are you?<eos> 까지 0, loss 계산해야되는게 fine<eos>
    
    - 근데 label을 reference 바로 앞에 넣어서 할 수도 있겠다. 이게 맞는거같기도 하고.
- 정확도
    - small 7235
- epoch 10했는데 걍 처음부터 loss가 왜 계속 올라가냐 ㅋㅋ

### 2. Prefix tuning

- Tuning 템플릿은 똑같음. model만 바꼈을 뿐,
    - model: virtual token 30개추가
        - 추가할 token 갯수에 따른 실험 가능
- 다만 수렴이 너무 느림
    - early stopping사용 → patient 5
- prefix tuning 학습은 됬는데
    - 9, 60번째 data에 대해서 error가 뜸..
        - → 재우가 해결해줌. max_new_tokens = 500으로, model.config를 찍어봤을 때, n_positions이 1024여서 총 길이가 1024가 넘어가면 error가 발생했음
    - 제외하고 출력해봐도 추론을 다하긴하는데 결과가 이상함
- 정확도 :
    - small -

### 3. Prompt tuning

- prompt tuning init text를 넣을 수 있네. 이걸 넣고 하면 정확도가 올라가긴 할듯 .
    - 일단 안 넣은 걸로.
- setting : virtual token 30개
- loss를 구하는데 있어서 (기존코드에서) logit과 label 갯수가 virtual token 갯수만큼 차이나서 문제가 생김
    - logit[:,30:,] 으로 문제해결…
- 학습 완료 후 추론해보니 아무것도 generate 안 하는 sample이 왤케 많지…. 또 이상하게 생성함 정확도 40 ㅅ
    - 2,3번 prefix, prompt tuning을 huggingface 예제에 있는 방법으로 다시 학습 시켜봐야 될듯,,,
        - 즉, loss를 직접 구하지 않고 label을 이용해서 하는 방법으로. (FT,prefix,prompt 전부 다 )→ hugging face prefix 예제 보고 분석중
- 공용 Inference and eval 코드 만들었고.

Collate_fn 바꾸기 (custom_bert_generate 다바꿔야함) → collate_fn 을 custom_bert_generate 밖으로 빼는게 낳을듯. 

- padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,padding_value = tokenizer.eos_token_id)
- 일단 finetuning_with_no_label에서 바꾸고 돌려놓고 가자 (2/29 새벽 1시)

2/29 할꺼 초록색 

prefix, prompt tuning 결과 기다리는 중 

- 추론 후 평가해보기

학습 방법 초록색으로 바꿔서 다시 학습하기

- preprocessed_datasets2 바꿨음: label도 만들어놓는 방법으로 loss알아서 구할라고
- loss 구하는 방식 바꿨음
- 3/1 prefix, prompt 평가하기

### 4. LoRA

### 3/5

- BERT multi-class로 학습 끝,
    - dialogpt-small 3가지 방법 전부다 정확도 다시 측정
    - dialogpt-medium 학습
        - medium 부터는 turn 4개 넘는거 다 잘르고 학습하기
        - accelrator 구현(batch_size = 20으로) → 학습중  , 정확도 측정하기 ,pretrained_까지 정확도 측정
            - 이거 다하고 large 노트북 만들어서 large 측정하기.
                - large에선 학습할때 context+reference 길이가 100이상 인 애들은 빼는 코드 만들긴했는데, 일단  prefix랑 prompt만 돌림.
                
                ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/aa37adda-a925-4f98-92fd-edf6a1f56d7c/25b6d274-c30d-418b-9336-2e841204bb6c/Untitled.png)
                
            - colab pro 결제해서 a100으로 large 모델 돌림 . 
