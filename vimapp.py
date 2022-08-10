# ---------------------------------------------------------------------------------------------------
# 실시간 크롤링 함수
# 설정한 시간이 되면 정해진 사이트를 크롤링하여 데이터 수집, 수집한 데이터를 모델로 라벨링한 뒤 기존 데이터프레임에 붙임.
# ---------------------------------------------------------------------------------------------------

import tensorflow as tf
from transformers import TextClassificationPipeline
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification

import os
import schedule
import time
import requests
import json
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import re
import numpy as np



MODEL_SAVE_PATH = 'modelzip/data_classification_model/end_model/fine-tuned-kykim-bert-base'
loaded_tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)
loaded_model = TFBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH, id2label={0: 0 , 1: 1, 2: 2, 3: 3, 4: 4, 5: 5})

text_classifier = TextClassificationPipeline(
    tokenizer=loaded_tokenizer, 
    model=loaded_model, 
    framework='tf',
    return_all_scores=True
)

def label(label_data):
    label_data['label']=np.nan
    predicted_label_list = []
    for text in label_data['desc']:
        # predict
        preds_list = text_classifier(text)

        sorted_preds_list = sorted(preds_list[0], key=lambda x: x['score'], reverse=True)
        predicted_label_list.append(sorted_preds_list[0]['label']) # label

    label_data['label'] = predicted_label_list
    return label_data

query_phone = ['%B0%B6%B7%B0%BD%C3+s22', '%B0%B6%B7%B0%BD%C3+S8', '%B0%B6%B7%B0%BD%C3+S8%2B', '%B0%B6%B7%B0%BD%C3s9', '%B0%B6%B7%B0%BD%C3s9%2B', '%B0%B6%B7%B0%BD%C3s10', '%B0%B6%B7%B0%BD%C3s10%2B', '%B0%B6%B7%B0%BD%C3s10e',
                '%B0%B6%B7%B0%BD%C3+s20', '%B0%B6%B7%B0%BD%C3S20%2B', '%B0%B6%B7%B0%BD%C3s20%BF%EF%C6%AE%B6%F3', '%B0%B6%B7%B0%BD%C3s21', '%B0%B6%B7%B0%BD%C3s21%2B', '%B0%B6%B7%B0%BD%C3s21%BF%EF%C6%AE%B6%F3',
                '%B0%B6%B7%B0%BD%C3+S22', '%B0%B6%B7%B0%BD%C3s22%2B', '%B0%B6%B7%B0%BD%C3s22%BF%EF%C6%AE%B6%F3', '%B0%B6%B7%B0%BD%C3s105g', '%B0%B6%B7%B0%BD%C3%B3%EB%C6%AE9',
                '%B0%B6%B7%B0%BD%C3%B3%EB%C6%AE10', '%B0%B6%B7%B0%BD%C3%B3%EB%C6%AE10%2B', '%B0%B6%B7%B0%BD%C3%B3%EB%C6%AE20', '%B0%B6%B7%B0%BD%C3%B3%EB%C6%AE20%BF%EF%C6%AE%B6%F3', 
                '%BE%C6%C0%CC%C6%F911', '%BE%C6%C0%CC%C6%F913', '%BE%C6%C0%CC%C6%F913mini', '%BE%C6%C0%CC%C6%F98', '%BE%C6%C0%CC%C6%F98%2B', '%BE%C6%C0%CC%C6%F98X',
                '%BE%C6%C0%CC%C6%F911pro', '%BE%C6%C0%CC%C6%F911promax', '%BE%C6%C0%CC%C6%F912', '%BE%C6%C0%CC%C6%F912mini', '%BE%C6%C0%CC%C6%F912pro', 
                '%BE%C6%C0%CC%C6%F912promax', '%BE%C6%C0%CC%C6%F913pro3', '%BE%C6%C0%CC%C6%F913promax', '%BE%C6%C0%CC%C6%F9se', '%BE%C6%C0%CC%C6%F9se2',
                '%BE%C6%C0%CC%C6%F9XR', '%BE%C6%C0%CC%C6%F9XS', '%BE%C6%C0%CC%C6%F9xsmax',
              '%C8%AB%B9%CC%B3%EB%C6%AE10&',
              '%C8%AB%B9%CC%B3%EB%C6%AE10+%C7%C1%B7%CE&',
              '%C8%AB%B9%CC%B3%EB%C6%AE11&',
              '%C8%AB%B9%CC%B3%EB%C6%AE11+%C7%C1%B7%CE&',
              'V40&',
              'V50&']
query_tablet = ['%BE%C6%C0%CC%C6%D0%B5%E5+%C7%C1%B7%CE+12.9+5%BC%BC%B4%EB', '%B0%B6%B7%B0%BD%C3+%C5%C7+s6+lite', '%BE%C6%C0%CC%C6%D0%B5%E5%C7%C1%B7%CE+11%C7%FC+3%BC%BC%B4%EB',
                           '%BE%C6%C0%CC%C6%D0%B5%E5%C7%C1%B7%CE+11%C7%FC+2%BC%BC%B4%EB', '%BE%C6%C0%CC%C6%D0%B5%E5%C7%C1%B7%CE+11%C7%FC+1%BC%BC%B4%EB',
                            '%BE%C6%C0%CC%C6%D0%B5%E5%C7%C1%B7%CE5%BC%BC%B4%EB', '%BE%C6%C0%CC%C6%D0%B5%E5+%C7%C1%B7%CE+4%BC%BC%B4%EB', '%BE%C6%C0%CC%C6%D0%B5%E5+%BF%A1%BE%EE+5%BC%BC%B4%EB'
                            '%BE%C6%C0%CC%C6%D0%B5%E5+%BF%A1%BE%EE+4%BC%BC%B4%EB', '%BE%C6%C0%CC%C6%D0%B5%E5+%BF%A1%BE%EE+3%BC%BC%B4%EB', '%BE%C6%C0%CC%C6%D0%B5%E5+%B9%CC%B4%CF+6%BC%BC%B4%EB',
                            '%BE%C6%C0%CC%C6%D0%B5%E5+%B9%CC%B4%CF+5%BC%BC%B4%EB', '%BE%C6%C0%CC%C6%D0%B5%E5+%C7%C1%B7%CE+6%BC%BC%B4%EB', '%BE%C6%C0%CC%C6%D0%B5%E58%BC%BC%B4%EB',
                            '%BE%C6%C0%CC%C6%D0%B5%E57%BC%BC%B4%EB', '%BE%C6%C0%CC%C6%D0%B5%E56%BC%BC%B4%EB', '%B0%B6%B7%B0%BD%C3%C5%C7s8%BF%EF%C6%AE%B6%F3',
                            '%B0%B6%B7%B0%BD%C3%C5%C7s8%2B', '%B0%B6%B7%B0%BD%C3%C5%C7s8', '%B0%B6%B7%B0%BD%C3%C5%C7s7%2B', '%B0%B6%B7%B0%BD%C3%C5%C7s7'
                            '%B0%B6%B7%B0%BD%C3%C5%C7s6', '%B0%B6%B7%B0%BD%C3%C5%C7s6+%B6%F3%C0%CC%C6%AE', '%B0%B6%B7%B0%BD%C3%C5%C7s4', '%B0%B6%B7%B0%BD%C3%C5%C7a10.5',
                            '%B0%B6%B7%B0%BD%C3%C5%C7a10.1', '%B0%B6%B7%B0%BD%C3%C5%C7a8', '%B0%B6%B7%B0%BD%C3%C5%C7a8.4', '%B0%B6%B7%B0%BD%C3%C5%C7+a8.0',
                            '%B0%B6%B7%B0%BD%C3%C5%C7+a7', '%B0%B6%B7%B0%BD%C3%C5%C7+a7+%B6%F3%C0%CC%C6%AE', '%B0%B6%B7%B0%BD%C3%C5%C7+a+with+spen']


# 번개장터 크롤링, 전처리 코드 (아이템 리스트 추가 필요)
def bunjang_crawling(category):
    pid = [] # 제품 아이디
    for page in range(1):
        url = f'https://api.bunjang.co.kr/api/1/find_v2.json?order=date&n=96&page={page}&req_ref=search&q={category}&stat_device=w&stat_category_required=1&version=4'
        response = requests.get(url)
        datas = response.json()['list']

        ids = [data['pid'] for data in datas] 
        pid.extend(ids)# 추가
    items=[]
    for i in pid: # 제품 아이디 하나씩 훑기
        url = f'https://api.bunjang.co.kr/api/1/product/{i}/detail_info.json?version=4'
        response = requests.get(url) # 리퀘스트
        try:
            details = response.json()['item_info'] #제이슨 파일을 그대로 가져오되 카테고리 이름, 페이옵션을 삭제
            details.pop('category_name')
            details.pop('pay_option')
            items.append(details)# 가져오면 딕셔너리 형태임
        except:
            print('error')
    df = pd.DataFrame(items)
    return df

def bunjang_preprocess(df):
    bunjang_df = df[['name','price','location','description_for_detail','status','update_time','uid']]
    bunjang_df = bunjang_df.rename({'name':'title','location':'region','description_for_detail':'desc','uid':'seller'},axis='columns')

    # 전처리
    bunjang_df.reset_index(drop = True)
    drop_list = bunjang_df.query('desc.str.contains("매입|삽니다|구매|최고가|전기종")', engine='python').index
    drop_bunjang_df = bunjang_df.drop(drop_list)
    drop_bunjang_df.price = drop_bunjang_df.price.astype('float')
    drop_bunjang_df.reset_index(inplace= True)
    drop_bunjang_df.drop('index', axis= 1, inplace= True)
    return drop_bunjang_df

def bunjang_update_data():
    for i in range(len(item_list)):
        df = bunjang_crawling(item_list[i])
        update_bunjang = bunjang_preprocess(df)
        update_bunjang.drop_duplicates(['title', 'price',  'desc'], inplace= True)
        update_bunjang = label(update_bunjang)
        if i == 0:
            label_update_bunjang = update_bunjang
            continue
        elif i != 0:
            total_bunjang_df = pd.concat([label_update_bunjang, update_bunjang])
    return total_bunjang_df, print('번개장터 update!')


#중고나라 크롤링, 전처리 코드 (쿼리 리스트 항목 추가필요, 데이터 따로 저장됨)
def junggo_phone_crawling():
    link = []
    data = []
    contents_list = []
    junggo = []
    for i in range(len(query_phone)):
        for page in range(1): #원하는 페이지만큼 입력
            menuid = 339
            # 메뉴아이디 749 = 테블릿, 339 = 휴대폰
            url = f"https://cafe.naver.com/ArticleSearchList.nhn?search.clubid=10050146&search.menuid={menuid}&search.media=0&search.searchdate=all&search.exact=&search.include=&userDisplay=50&search.exclude=&search.onSale=1&search.option=3&search.sortBy=date&search.searchBy=1&search.searchBlockYn=0&search.includeAll=&search.query={query_phone[i]}&search.viewtype=title&search.page={1+page}"

            html = requests.get(url)
            soup = BeautifulSoup(html.text, 'html.parser')

    #     contents = soup.select('.inner_list .article') #단순 페이지 링크 구하는 태그

        id_get = soup.select('.inner_number') #판매글의 아이디 구하기

        for i in id_get: #판매글의 아이디를 통해 판매 정보을 포함한 jason 링크 리스트 만들기
            contents_list.append(f'https://apis.naver.com/cafe-web/cafe-articleapi/v2/cafes/10050146/articles/{i.text}?query=&menuId={menuid}&useCafeId=true&requestFrom=A')

    # json 파일에서 제목, 가격, 판매자, 설명 추출해 리스트로 저장
    for url in contents_list:
        response = requests.get(url)
        data = response.json()
        try:
            title = data["result"]["article"]['subject']
            price = data["result"]["saleInfo"]['price']
            #seller = data["result"]["article"]['writer']['nick']
            desc = BeautifulSoup(data["result"]["article"]["contentHtml"], 'html.parser').text
            #타임스탬프
            #date = datetime.fromtimestamp(data['result']['article']['writeDate']/1000).strftime('%y-%m-%d')

            junggo.append({'title' : title, 'price' : price, 'desc' : desc})

        except:
            print('error')
            continue    
    junggo = pd.DataFrame(junggo)
    return junggo

def junggo_tablet_crawling():
    link = []
    data = []
    contents_list = []
    junggo = []
    for i in range(len(query_tablet)):
        for page in range(1): #원하는 페이지만큼 입력   
            menuid = 749
            url = f"https://cafe.naver.com/ArticleSearchList.nhn?search.clubid=10050146&search.menuid={menuid}&search.media=0&search.searchdate=all&search.exact=&search.include=&userDisplay=50&search.exclude=&search.onSale=1&search.option=3&search.sortBy=date&search.searchBy=1&search.searchBlockYn=0&search.includeAll=&search.query={query_tablet[i]}&search.viewtype=title&search.page={1+page}"

            html = requests.get(url)
            soup = BeautifulSoup(html.text, 'html.parser')

        #     contents = soup.select('.inner_list .article') #단순 페이지 링크 구하는 태그

            id_get = soup.select('.inner_number') #판매글의 아이디 구하기

            for i in id_get: #판매글의 아이디를 통해 판매 정보을 포함한 jason 링크 리스트 만들기
                contents_list.append(f'https://apis.naver.com/cafe-web/cafe-articleapi/v2/cafes/10050146/articles/{i.text}?query=&menuId={menuid}&useCafeId=true&requestFrom=A')

    # json 파일에서 제목, 가격, 판매자, 설명 추출해 리스트로 저장
    for url in contents_list:
        response = requests.get(url)
        data = response.json()
        try:
            title = data["result"]["article"]['subject']
            price = data["result"]["saleInfo"]['price']
            #seller = data["result"]["article"]['writer']['nick']
            desc = BeautifulSoup(data["result"]["article"]["contentHtml"], 'html.parser').text
            #타임스탬프
            #date = datetime.fromtimestamp(data['result']['article']['writeDate']/1000).strftime('%y-%m-%d')

            junggo.append({'title' : title, 'price' : price, 'desc' : desc})

        except:
            print('error')
            continue    
    junggo = pd.DataFrame(junggo)
    return junggo

def junggo_preprocess(junggo):
    junggo.reset_index(drop = True)
    junggo['desc'] = junggo['desc'] \
    .str[:255] \
    .replace({'위 중고나라 앱 다운로드 링크를 통해 앱 다운로드하여 판매 및 구매 게시글을 등록해 주세요':'', '중고나라에서 천원 상품권 드리는 거 아시나요 더보기 버튼을 눌러 확인해 주세요':'', '중고나라에서 스타벅스 드리는 거 아시나요 더보기 버튼을 눌러 확인해 주세요' : ''}, regex=True)
    junggo = junggo[junggo['desc'].str.strip().astype(bool)]
    junggo = junggo.drop_duplicates(['desc'], keep = 'first', ignore_index = True)
    junggo = junggo[~junggo['title'].str.contains('매입|삽니다|최고가')]
    junggo = junggo[~junggo['desc'].str.contains('매입|삽니다|최고가')]
    junggo.reset_index(drop = True, inplace = True)
    junggo.price = junggo.price.astype('int')

    junggo.reset_index(drop = True, inplace = True)
    
    return junggo

def Update_junggo():
    junggo_phone_df = junggo_phone_crawling()
    junggo_tablet_df = junggo_tablet_crawling()
    junggo_df = pd.concat([junggo_phone_df, junggo_tablet_df])
    junggo_df = label(junggo_df)
    junggo = junggo_preprocess(junggo_df)
    junggo.drop_duplicates(['title', 'price',  'desc'], inplace= True)
    #total_df.drop(['Unnamed: 0'], axis= 1,inplace= True)
    return junggo, print('중고나라 업데이트 완료!')

def dangn_preprocess(df):
    try:
        dangn_df = df
        dangn_df.reset_index(drop = True)
        dangn_df['desc'] = dangn_df['desc'].str.strip() \
        

        dangn_df = dangn_df[dangn_df['desc'].str.strip().astype(bool)]
        dangn_df['price'] = dangn_df['price'].str.replace("만",'0000')
        dangn_df['price'] = dangn_df['price'].str.replace(r'[^0-9]', '', regex=True)
        drop_list = dangn_df.query('desc.str.contains("매입|삽니다|구매|최고가|전기종|구함|구합니다|구해요|교환")', engine='python').index
        dangn_df = dangn_df.drop(drop_list)
        dangn_df['price']=dangn_df['price'].replace('^ +','')
        dangn_df['price']=dangn_df['price'].replace('',np.nan) 

        drop_list = dangn_df.query('desc.str.contains("매입|삽니다|구매|최고가|전기종|구함|구합니다|구해요|교환|잠금|풀어주세요")', engine='python').index
        dangn_df = dangn_df.drop(drop_list)



        dangn_df=dangn_df.dropna(how='any') 

        dangn_df.price = dangn_df.price.astype('float')

        q1= dangn_df['price'].quantile(0.25)
        q3= dangn_df['price'].quantile(0.75)
        iqr=q3-q1
        condition=dangn_df['price']>q3+1.5*iqr

        drop_price_list = dangn_df[condition].index

        dangn_df.drop(drop_price_list, inplace= True)

        dangn_df.reset_index(inplace= True)

        dangn_df.drop('index', axis= 1, inplace= True)
        dangn_df= dangn_df[['title','price','desc']]
    except:
        print("error")
    return dangn_df

def dangn_update_data(item_list):
    category = []

    for i in range(1):
        for j in range(len(item_list)):
            url =f"https://www.daangn.com/search/{item_list[j]}/more/flea_market?page={i}"
            resp = requests.get(url)
            soup = BeautifulSoup(resp.content , "html.parser")
            getitem = soup.select(".flea-market-article")

            for item in getitem:   
                try:
                    category.append({
                         'title' : item.select('.article-title')[0].text.strip(),
                         'price' : item.select('.article-price')[0].text.strip(),
                         'region' : item.select('.article-region-name')[0].text.strip(),
                         'desc' : item.select('.article-content')[0].text.strip()
                     })
                except:
                    print('error')
    df = pd.DataFrame(category)
    df = dangn_preprocess(df)
    dangn_update_df = label(df)
    
    return dangn_update_df

def save_data_frame():
    update_junggo_df = Update_junggo()
    update_bunjang_df = bunjang_update_data()
    dangn_update_df = dangn_update_data(item_list)
    junggo_bunjang_df = pd.concat([update_junggo_df, update_bunjang_df])
    total_update_data = pd.concat([junggo_bunjang_df, dangn_update_df])
    total_update_data.drop(total_update_data[total_update_data.title.str.contains('교신')== True].index, inplace= True)
    total_update_data.drop(total_update_data[total_update_data.price < 10000].index, inplace = True)
    total_update_data.drop(total_update_data[total_update_data.title.str.contains('삽니다')== True].index, inplace= True)
    basic_data = pd.read_csv('label_data.csv', encoding= 'utf-8')
    save_data = pd.concat([basic_data, total_update_data])
    save_data.drop_duplicates(['title', 'price',  'desc'], inplace= True)
    return save_data.to_csv('라벨링끝 정제.csv')
    
item_list = ['갤럭시s8+',  
'갤럭시s8',
'갤럭시s9+',
'갤럭시s9',
'갤럭시s10',
'갤럭시s10+',
'갤럭시s105g',
'갤럭시s10e',
'갤럭시s20',
'갤럭시s20+',
'갤럭시s20울트라',
'갤럭시s21',
'갤럭시s21+',
'갤럭시s21울트라',
'갤럭시s22',
'갤럭시s22+',
'갤럭시s22울트라',
'아이폰8',
'아이폰8+',
'아이폰8X',
'아이폰XR',
'아이폰XS',
'아이폰XSMAX',
'아이폰 11',
'아이폰11 pro',
'아이폰11proMax',
'아이폰se',
'아이폰12mini',
'아이폰12',
'아이폰12pro',
'아이폰12promax',
'아이폰 13',
'아이폰 13mini',
'아이폰se2',
'아이폰13pro3',
'아이폰13promax',
'갤럭시노트9',
'갤럭시노트10',
'갤럭시노트10+',
'갤럭시노트20',
'갤럭시노트20울트라',

'갤럭시탭 s4',
'갤럭시탭 s6',
'갤럭시탭 s6 라이트',
'갤럭시탭 s7',
'갤럭시탭 s7+',
'갤럭시탭 s8',
'갤럭시탭 s8+',
'갤럭시탭 s8울트라',
'갤럭시탭 a10.5',
'갤럭시탭 a8.0',
'갤럭시탭 a with spen',
'갤럭시탭 a10.1',
'갤럭시탭 a8.4',
'갤럭시탭 a7',
'갤럭시탭 a7 라이트',
'갤럭시탭 a8',
'아이패드프로 5세대',
'아이패드프로 4세대',
'아이패드프로 11형 1세대',
'아이패드프로 11형 2세대',
'아이패드프로 11형 3세대',
'아이패드미니 6세대',
'아이패드미니 5세대',
'아이패드에어 5세대',
'아이패드에어 4세대',
'아이패드에어 3세대',
'아이패드 8세대',
'아이패드 7세대',
'아이패드 6세대',
'아이패드 프로 6세대',
'V40', 'V50', '홍미노트10',
'홍미노트10프로',
'홍미노트11',
'홍미노트11프로'
]

data_base = pd.read_csv('sapago_live_data.csv', encoding= 'utf-8')

schedule.every().day.at("09:35").do(save_data_frame)

while True:
    schedule.run_pending()
    time.sleep(1)
