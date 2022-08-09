# ---------------------------------------------------------------------------------------------------
# 텍스트 데이터 전처리 함수
# 입력 : 데이터 프레임 / 출력 : 새로 만들어진(전처리된) 데이터 프레임
# 주의 : 데이터 프레임에서 텍스트 열의 이름은 desc로 맞춰줄 것
# ---------------------------------------------------------------------------------------------------


import re

def delete_special_character(sentence):
    sentence = re.sub('[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$&\\\=\(\'\"\♥\♡\※\■\▼\§\★\●\⊙\＆\＠\☆\○\◎\◇\◆\■\△\▲\▽\→\←\↑\↓\↔\◁\◀\▷\▶\♤\♠\♧\♣\◈\▣\☎]','',str(sentence))
    sentence = re.sub('\n*\n', '\n', sentence)

    return sentence

def deEmojify(sentence):
    return sentence.encode('euc-kr', 'ignore').decode('euc-kr')

def preprocessor(dataframe):
    new_df = dataframe
    new_df['desc'] = new_df['desc'].apply(delete_special_character)
    new_df['desc'] = new_df['desc'].apply(deEmojify)

    return new_df
