import sys
from bm25_model import BM25Okapi
from bm25_config import BMConfig
import numpy as np
import jieba
import time
import re
import json

start_time = time.time()

np.random.seed(10)
config = BMConfig()


def load_corpus(base_list):
    busi_ques = []
    for item in base_list:
        busi_ques.append(item)
    return busi_ques


def load_stop_words(stop_word_path):
    file = open(stop_word_path, 'r', encoding='utf-8')
    stop_words = file.readlines()
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words


stop_words = load_stop_words(config.stopwords_path)


def clean_text(text):
    text = re.sub("[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】《》“”！，。？、~@#￥%……&*（）]+",
                  '',
                  str(text).lower())
    if not text:
        return np.nan
    return text


def clean_seg(text):
    text = clean_text(text)
    if type(text) is not str:
        return []
    else:
        words = jieba.lcut(text)
        return ''.join([word for word in words if word not in stop_words])


class BM25Recall(object):
    def __init__(self, qa_df):
        self.qa_df = qa_df
        self.base_list = [item['question'] for item in qa_df]
        self.busi_ques = load_corpus(self.base_list)
        self.tokenizer = clean_seg
        self.bm25_busi = BM25Okapi(self.busi_ques, self.tokenizer)
        # self.bm25_chat = BM25Okapi(self.chat_ques, self.tokenizer)

    def recall(self, query, top_n=10):
        return self.bm25_busi.get_top_n(query, self.qa_df, n=top_n)


if __name__ == "__main__":
    a = [{'question': '阿里巴巴开发商', 'answer': 'alipay-开发商-阿里巴巴'},
         {'question': '阿里巴巴创立单位', 'answer': '阿里学院-创立单位-阿里巴巴'},
         {'question': '阿里巴巴所属公司', 'answer': '阿里妈妈-所属公司-阿里巴巴'},
         {'question': '阿里巴巴开发商', 'answer': '千牛[阿里巴巴集团卖家工作台]-开发商-阿里巴巴'},
         {'question': '阿里巴巴单位', 'answer': '诚信通-单位-阿里巴巴'},
         {'question': '阿里巴巴公司', 'answer': '贸易通-公司-阿里巴巴'},
         {'question': '阿里巴巴企业', 'answer': 'B2C-企业-阿里巴巴'},
         {'question': '阿里巴巴开发商', 'answer': '淘宝助理-开发商-阿里巴巴'}]
    model = BM25Recall(a)
    with open("redis.json", 'r') as f:
        b = json.load(f)
    model2 = BM25Recall(b)

    print(model.recall("助理", 1))
    print(model2.recall("客户信息验证页签名错误", 1))
