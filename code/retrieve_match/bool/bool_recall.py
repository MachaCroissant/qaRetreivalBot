import pandas as pd
from bool_config import BoolConfig
from bool_model import BoolSearch
import numpy as np
import jieba
import re

config = BoolConfig()


def load_corpus(config):
    print("\nLoading the dataset ... \n")
    qa_df = pd.read_excel(config.corpus_path)
    qa_df["question"] = qa_df["question"].apply(str)
    qa_df["answer"] = qa_df["answer"].apply(str)
    qa_dict = qa_df.to_dict(orient="records")
    return qa_dict


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


class BoolRecall(object):
    def __init__(self, qa_dict):
        self.tokenizer = clean_seg
        self.qa_dict = qa_dict
        self.bool_qa = BoolSearch(self.qa_dict, self.tokenizer)

    def recall(self, query, top_n=10):
        return self.bool_qa.get_top_n(query, n=top_n)


if __name__ == "__main__":
    qa_dict = load_corpus(config)
    busi_ques = ["账户状态不正常", "申请为什么一直发送中", "一直黑屏", "充不进电怎么办"]
    model_busi = BoolRecall(qa_dict)
    for q1 in busi_ques:
        print(model_busi.recall(q1))
