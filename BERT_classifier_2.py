import torch

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch.nn.functional as F 
from tensorflow.keras.preprocessing.sequence import pad_sequences

def BERT_classifier(sentence):
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')

    # 모델 불러오기
    model = BertForSequenceClassification.from_pretrained('model2.pt')
    # 사전 훈련 모델로 Tokenizer 생성  
    tokenizer = BertTokenizer.from_pretrained('kykim/bert-kor-base', do_lower_case=False)  

    # 입력 문장
    sentence = ["[CLS] " + str(sentence) + " [SEP]"]

    # 문장 토큰화
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentence]  

    # 패딩
    MAX_LEN = 128
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # 어텐션 마스크 생성
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # 파이토치 텐서로 변환
    user_inputs = torch.tensor(input_ids)
    user_masks = torch.tensor(attention_masks)

    # 배치 및 데이터로더 설정
    BATCH_SIZE = 64
    user_data = TensorDataset(user_inputs, user_masks)
    user_sampler = RandomSampler(user_data)
    user_dataloader = DataLoader(user_data, sampler=user_sampler, batch_size=BATCH_SIZE)

    # 평가모드로 변경
    model.eval()

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(user_dataloader):
        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)
        
        # 배치에서 데이터 추출
        b_input_ids, b_input_mask = batch
        
        # 그래디언트 계산 안함
        with torch.no_grad():     
            # Forward 수행
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
            
        # 예측값 출력
        logits = outputs[0]
        pred = F.softmax(logits).detach().cpu().numpy().argmax()
        return pred