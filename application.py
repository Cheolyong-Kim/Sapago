# ---------------------------------------------------------------------------------------------------
# 챗봇 서버 코드
# 사용자가 입력한 텍스트를 모델로 분류, 분류한 값과 입력한 기종을 label_data.csv에서 찾아 평균 가격 도출.
# flask를 활용해 카카오톡 챗봇과 연동
# ---------------------------------------------------------------------------------------------------


import tensorflow as tf
from flask import Flask, request, json, jsonify         
import pandas as pd
import sys
from transformers import TextClassificationPipeline
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification





application = Flask(__name__)

base_data = pd.read_csv('label_data.csv')
base_data.title = base_data.title.str.replace('mini', '미니', case= False)
base_data.title = base_data.title.str.replace('pro', '프로', case= False)
base_data.title = base_data.title.str.replace('ult', '울트라', case= False)
base_data.title = base_data.title.str.replace('\+', '플러스')
base_data.title = base_data.title.str.replace('max', '맥스', case= False)
base_data.title = base_data.title.str.replace(' ', '')


con_dfs = [base_data[base_data.title.str.contains('프로') == True],
                base_data[base_data.title.str.contains('플러스') == True],
                    base_data[base_data.title.str.contains('울트라') == True],
                    base_data[base_data.title.str.contains('맥스') == True], 
                    base_data[base_data.title.str.contains('미니') == True]]
contain_data = pd.concat(con_dfs)


vanila_data = base_data.drop(base_data[base_data.title.str.contains('플러스') == True].index)
vanila_data.drop(vanila_data[vanila_data.title.str.contains('프로') == True].index, inplace= True)
vanila_data.drop(vanila_data[vanila_data.title.str.contains('맥스') == True].index, inplace= True)
vanila_data.drop(vanila_data[vanila_data.title.str.contains('울트라') == True].index, inplace= True)
vanila_data.drop(vanila_data[vanila_data.title.str.contains('미니') == True].index, inplace= True)

# 파인-튜닝된 모델 경로
MODEL_SAVE_PATH = '/workspace/Final_Sapago/modelzip/data_classification_model/end_model/fine-tuned-kykim-bert-base'

# 모델 불러오기
loaded_tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)
loaded_model = TFBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH, id2label={0: 0 , 1: 1, 2: 2, 3: 3, 4: 4, 5: 5})


text_classifier = TextClassificationPipeline(
    tokenizer=loaded_tokenizer, 
    model=loaded_model, 
    framework='tf',
    return_all_scores=True
)


def BERT_classifier(sentence):  #1.43 s

    pred = text_classifier(sentence)
    output = sorted(pred, key=lambda x: x['score'], reverse=True)[0]['label']
    return output

def remove_outlier(df):
    q1 = df['price'].quantile(0.25)
    q3 = df['price'].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df.loc[(df['price'] > fence_low) & (df['price'] < fence_high)]
    return df_out



@application.route("/", methods=['POST'])

def Message():
    content = request.get_json()
    user_product = content['userRequest']['utterance']
    product_data_list = user_product.split(':')
    product_name = product_data_list[0].replace(' ', '')
    product_status = product_data_list[1]
    status_label = BERT_classifier(product_status)
    con_search_data = contain_data[(contain_data['title'].str.contains(product_name, case = False) == True) & (contain_data.label == status_label)]
    van_search_data = vanila_data[(vanila_data['title'].str.contains(product_name, case = False) == True) & (vanila_data.label == status_label)]
    if status_label == 0:
        status_image_url = 'https://ifh.cc/g/MtJcOt.jpg'  
    elif status_label == 1:
        status_image_url = 'https://ifh.cc/g/sh7ccR.jpg'
    elif status_label == 2:
        status_image_url = 'https://ifh.cc/g/RN3c2n.jpg'
    elif status_label == 3:
        status_image_url = 'https://ifh.cc/g/nN7gKA.jpg'
    elif status_label == 4:
        status_image_url = 'https://ifh.cc/g/72z5nT.jpg'
    else :
        status_image_url = 'https://ifh.cc/g/q0CzRR.jpg'
        
    if ('프로'in product_name) & (status_label !=5):       
        remove_outlier_dataframe = remove_outlier(con_search_data)
        if len(remove_outlier_dataframe) == 0:
            dataSend ={
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": f"데이터에 해당 상태의 상품이 존재하지않습니다."
                    }
                }
            ]
       }
    }
        elif len(remove_outlier_dataframe) != 0:
            product_mean_price = round(remove_outlier_dataframe['price'].mean())
            comma_product_mean_price = format(product_mean_price, ',')
            dataSend ={
      "version": "2.0",
      "template": {
        "outputs": [
          {
            "basicCard": {
              "title": f"{product_data_list[0]}",
              "description": f"{comma_product_mean_price}원",
              "thumbnail": {
                "imageUrl": status_image_url
              }
            }
          }
        ]
      }
    }
    
    elif ('플러스' in product_name) & (status_label !=5):
        remove_outlier_dataframe = remove_outlier(con_search_data)
        if len(remove_outlier_dataframe) == 0:
            dataSend ={
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": f"데이터에 해당 상태의 상품이 존재하지않습니다."
                    }
                }
            ]
       }
    }
        elif len(remove_outlier_dataframe) != 0:
            product_mean_price = round(remove_outlier_dataframe['price'].mean())
            comma_product_mean_price = format(product_mean_price, ',')
            dataSend ={
      "version": "2.0",
      "template": {
        "outputs": [
          {
            "basicCard": {
              "title": f"{product_data_list[0]}",
              "description": f"{comma_product_mean_price}원",
              "thumbnail": {
                "imageUrl": status_image_url
              }
            }
          }
        ]
      }
    }
    
    elif ('울트라' in product_name) & (status_label !=5):
        remove_outlier_dataframe = remove_outlier(con_search_data)
        if len(remove_outlier_dataframe) == 0:
            dataSend ={
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": f"데이터에 해당 상태의 상품이 존재하지않습니다."
                    }
                }
            ]
       }
    }
        elif len(remove_outlier_dataframe) != 0:
            product_mean_price = round(remove_outlier_dataframe['price'].mean())
            comma_product_mean_price = format(product_mean_price, ',')
            dataSend ={
      "version": "2.0",
      "template": {
        "outputs": [
          {
            "basicCard": {
              "title": f"{product_data_list[0]}",
              "description": f"{comma_product_mean_price}원",
              "thumbnail": {
                "imageUrl": status_image_url
              }
            }
          }
        ]
      }
    }
    
    elif ('맥스' in product_name) & (status_label !=5):
        remove_outlier_dataframe = remove_outlier(con_search_data)
        if len(remove_outlier_dataframe) == 0:
            dataSend ={
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": f"데이터에 해당 상태의 상품이 존재하지않습니다."
                    }
                }
            ]
       }
    }
        elif len(remove_outlier_dataframe) != 0:
            product_mean_price = round(remove_outlier_dataframe['price'].mean())
            comma_product_mean_price = format(product_mean_price, ',')
            dataSend ={
      "version": "2.0",
      "template": {
        "outputs": [
          {
            "basicCard": {
              "title": f"{product_data_list[0]}",
              "description": f"{comma_product_mean_price}원",
              "thumbnail": {
                "imageUrl": status_image_url
              }
            }
          }
        ]
      }
    }

    elif ('미니' in product_name) & (status_label !=5):
        remove_outlier_dataframe = remove_outlier(con_search_data)
        if len(remove_outlier_dataframe) == 0:
            dataSend ={
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": f"데이터에 해당 상태의 상품이 존재하지않습니다."
                    }
                }
            ]
       }
    }
        elif len(remove_outlier_dataframe) != 0:
            product_mean_price = round(remove_outlier_dataframe['price'].mean())
            comma_product_mean_price = format(product_mean_price, ',')
            dataSend ={
      "version": "2.0",
      "template": {
        "outputs": [
          {
            "basicCard": {
              "title": f"{product_data_list[0]}",
              "description": f"{comma_product_mean_price}원",
              "thumbnail": {
                "imageUrl": status_image_url
              }
            }
          }
        ]
      }
    }
        
    elif status_label == 5:
        dataSend ={
  "version": "2.0",
  "template": {
    "outputs": [
      {
        "basicCard": {
          "title": f"분류불가",
          "description": f"제품상태를 상세하게 입력해주세요.",
          "thumbnail": {
            "imageUrl": status_image_url
          }
        }
      }
    ]
  }
}
    
    else:
        remove_outlier_dataframe = remove_outlier(van_search_data)
        if len(remove_outlier_dataframe) == 0:
            dataSend ={
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": f"데이터에 해당 상태의 상품이 존재하지않습니다."
                    }
                }
            ]
       }
    }
        elif (len(remove_outlier_dataframe) != 0) & (status_label !=5):
            product_mean_price = round(remove_outlier_dataframe['price'].mean())
            comma_product_mean_price = format(product_mean_price, ',')
            dataSend ={
      "version": "2.0",
      "template": {
        "outputs": [
          {
            "basicCard": {
              "title": f"{product_data_list[0]}",
              "description": f"{comma_product_mean_price}원",
              "thumbnail": {
                "imageUrl": status_image_url
              }
            }
          }
        ]
      }
    }

    return jsonify(dataSend)

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=int(sys.argv[1]))
