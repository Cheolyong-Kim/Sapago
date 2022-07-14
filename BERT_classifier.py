from transformers import TextClassificationPipeline
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification

def BERT_classifier(sentence):
    # 파인-튜닝된 모델 경로
    MODEL_SAVE_PATH = '_model/fine-tuned-kykim-bert-base'

    # 모델 불러오기
    loaded_tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)
    loaded_model = TFBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH, id2label={0: 0 , 1: 1, 2: 2})

    text_classifier = TextClassificationPipeline(
        tokenizer=loaded_tokenizer, 
        model=loaded_model, 
        framework='tf',
        return_all_scores=True
    )

    pred = text_classifier(sentence)[0]
    output = sorted(pred, key=lambda x: x['score'], reverse=True)[0]['label']
    return output

print(BERT_classifier('액정 오른쪽 면에 초록색 줄이 생긴 액정 문제 있는 폰이구요,배터리는 84%, 초기화 완료 되었고 케이스 끼고 사용해서 전체적으로 깨끗합니다. 화면 우 상단, 좌 하단에 살짝 흠집 있습니다.'))