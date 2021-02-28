import os
import pathlib
import numpy as np

root = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent


class Config:
    def __init__(self):
        self.msg_dic = {
            '000': '成功',
            '100': '模型预测失败',
            '200': '请求格式错误',
            '300': '没有输入内容',
        }
        self.duplicate_txt = ["咦，怎么又是这句呀？",
                              "咱们好好说话行不行呀？",
                              "这个问题我刚刚才回答过哦，请换个问题考考我。",
                              "我回答过了哦。",
                              "我们换个话题继续聊吧！",
                              "换个问题来问我吧！",
                              "咦，你怎么老是一句话呀？",
                              "你刚刚问过这个问题啦！"]

        self.cannot_ans = "不好意思，我太笨啦，暂时无法帮助您解决问题，您可以留言咨询。"
        self.hello_ans = "您好，我是成电助手清清，请问有什么可以帮到您的？"

    def norma_ans(self, query, bm25_pred, bool_pred, retrieval_sim, top_n):
        """粗排使用bm25和bool的结果，继续用simbert做问句与粗排结果的相似性匹配

        :param query: 问句
        :param bm25_pred: 调用bm25筛选qa对
        :param bool_pred: 调用bool筛选qa对
        :param retrieval_sim: 从bool和bm25的筛选结果进一步做相似度匹配
        :param top_n: 筛选前多少个相似的问句
        :return:
        """
        bm25_qa = bm25_pred(query, top_n)
        # print("bm25匹配到的问题为:")
        # for _, item in enumerate(bm25_qa):
        #     print(item)
        # print("\n")

        bool_qa = bool_pred(query, top_n)
        # print("bool匹配到的问题为:")
        # for _, item in enumerate(bool_qa):
        #     print(item)
        # print("\n")

        match_qa = [i for i in bm25_qa if i not in bool_qa] + bool_qa
        # print("Match_Qa is:")
        # for _, item in enumerate(match_qa):
        #     print(item)
        # print("\n")

        sim_pred = retrieval_sim(match_qa)
        sim_qa = sim_pred.most_similar(query)

        top_n_one = sim_qa[0]
        top_n_recall_sort = sim_qa[1:6]

        return top_n_one, top_n_recall_sort

    def duplicate_response(self, duplicate_q):
        if len(len(duplicate_q) >= 3):
            res = np.random.choice(self.duplicate_txt)
            top_n_one = {'question': '重复文字', 'answer': res, 'sim_rate': 0.0}
            return top_n_one, []