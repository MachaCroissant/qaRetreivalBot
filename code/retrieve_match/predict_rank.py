import sys
import os
import pathlib
import pandas as pd
import json

root = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent

bm25_root = os.path.join(root, 'code/retrieve_match/bm25')
bool_root = os.path.join(root, 'code/retrieve_match/bool')
simbert_root = os.path.join(root, 'code/retrieve_match/simbert')
config_root = os.path.join(root, 'code/retrieve_match/model_config')


sys.path.extend([bm25_root, bool_root, simbert_root, config_root])

from bm25_recall import BM25Recall
from bool_recall import BoolRecall
from retrieval_bunny import RetrievalSim
from model_config import Config

cf = Config()


class Rank(object):
    def __init__(self, qa_dict):
        self.bm25_pred = BM25Recall(qa_dict).recall
        self.bool_pred = BoolRecall(qa_dict).recall

        self.flag = "normal"
        self.duplicate_q = ['start']    # 初始化以下，保证总有一个query存在

    def get_answer(self, query, top_n=10, threshold=0.5):
        if query == self.duplicate_q[-1]:
            self.duplicate_q.append(query)
            top_n_one, top_n_recall_sort = cf.duplicate_response(self.duplicate_q)
            return top_n_one, top_n_recall_sort
        else:
            self.duplicate_q = []
            self.duplicate_q.append(query)

        if self.flag == "normal":
            top_n_one, top_n_recall_sort = cf.norma_ans(query, self.bm25_pred, self.bool_pred, RetrievalSim, top_n)

        if top_n_one["sim_rate"] < threshold:
            top_n_recall_sort = []
            top_n_one = {'question': '无法解答', 'answer': cf.cannot_ans, 'sim_rate': 0.0}
            return top_n_one, top_n_recall_sort

        return top_n_one, top_n_recall_sort


def match(text):
    text = text.lower()
    hello_words = ["hi", "你好", "您好", "嗨", "hello", "Hello", " ",
                   "哈哈", "哈罗", "哈啰", "哈喽", "很高心认识你","你是谁",
                   "是谁", "您是谁"]
    if len(text) <= 1 or text in hello_words:
        return cf.hello_ans


if __name__ == "__main__":
    data_path = os.path.join(root, 'data/output.xlsx')
    questions = ["系统怎么总是提示密码错误", "网络连接不稳定", "草稿箱中的内容是否能够长期保存在云端"]
    qa_df = pd.read_excel(data_path)
    qa_df["question"] = qa_df["question"].apply(str)
    qa_df["answer"] = qa_df["answer"].apply(str)
    qa_dict = qa_df.to_dict(orient="records")
    ranker = Rank(qa_dict)

    for query in questions:
        match_answer = ranker.get_answer(query)
        print("\nThe question matched is %s\n\n" % str(match_answer))
