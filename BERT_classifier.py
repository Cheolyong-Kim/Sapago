# ---------------------------------------------------------------------------------------------------
# 분류기 함수
# 입력 : 텍스트 / 출력 : 라벨값(정수 0~5)
# 주의 : AWS에서 돌릴 때는 26번줄 코드의 [0]제거해야함
# ---------------------------------------------------------------------------------------------------

from transformers import TextClassificationPipeline
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification

def BERT_classifier(sentence):
    # 파인-튜닝된 모델 경로
    MODEL_SAVE_PATH = 'data_classification_model/fine-tuned-kykim-bert-base'

    # 모델 불러오기
    loaded_tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)
    loaded_model = TFBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH, id2label={0: 0 , 1: 1, 2: 2, 3: 3, 4: 4, 5: 5})

    text_classifier = TextClassificationPipeline(
        tokenizer=loaded_tokenizer, 
        model=loaded_model, 
        framework='tf',
        return_all_scores=True
    )

    pred = text_classifier(sentence)[0]  # AWS에서 돌릴 때는 [0]제거하기
    output = sorted(pred, key=lambda x: x['score'], reverse=True)[0]['label']
    return output