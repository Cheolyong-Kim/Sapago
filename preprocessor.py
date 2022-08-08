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
