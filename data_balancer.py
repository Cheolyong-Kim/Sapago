# ---------------------------------------------------------------------------------------------------
# 데이터 프레임의 라벨 별 데이터 수를 EDA를 활용해 일정하게 맞춰주는 함수
# 입력 : 데이터 프레임 / 출력 : 새로 만들어진 데이터 프레임
# 주의 : 데이터 프레임에서 텍스트 열의 이름은 desc로 라벨 열의 이름은 label로 맞춰줄 것, 같은 파일 경로에 EDA.py가 존재해야함
# ---------------------------------------------------------------------------------------------------

import pandas as pd
from EDA import EDA
import random
import itertools

def balancing(df, alpha_rs=0.05, p_rd=0.05, num_aug=8):
    copy_df = df
    max_len = max(df['label'].value_counts())
    
    label_num = df.nunique().label
    
    for i in range(label_num):
        p_df = df[df['label']==i]
        sentence_list = p_df['desc'].to_list()
        add_len = max_len - len(sentence_list)
        
        new_sentence_list = []
        for sentence in sentence_list:
            new_sentence_list.append(list(set(EDA(sentence, alpha_rs=alpha_rs, p_rd=p_rd, num_aug=num_aug))))
        new_sentence_list = list(itertools.chain.from_iterable(new_sentence_list))
        random.shuffle(new_sentence_list)
        new_df = pd.DataFrame(new_sentence_list[:add_len], columns=['desc'])
        new_df['label'] = i
        copy_df = pd.concat([copy_df, new_df])
        copy_df = copy_df.drop_duplicates(subset='desc')

    return copy_df