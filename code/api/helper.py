import sys
import pathlib
import os
import pandas as pd
root = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
rank_path = os.path.join(root, 'code/retrieve_match')
data_path = os.path.join(root, 'data/output.xlsx')
sys.path.append(rank_path)
from predict_rank import Rank


msg_dic = {'200': '正常', '300': '请求格式错误', '400': '模型预测失败'}

qa_df = pd.read_excel(data_path)
qa_df["question"] = qa_df["question"].apply(str)
qa_df["answer"] = qa_df["answer"].apply(str)
qa_dict = qa_df.to_dict(orient="records")


class Server:
    def __init__(self):
        self.predict = Rank(qa_dict).get_answer

    def request_parse(self, app_data):
        request_id = app_data["request_id"]
        text = app_data["query"]
        return request_id, text

    def get_result(self, data):
        code = '200'
        try:
            request_id, text = self.request_parse(data)
        except Exception as e:
            print('error info: {}'.format(e))
            code = '300'
            request_id = "None"
        try:
            if code == '200':
                answer, top_n_recall_sort = self.predict(text)
            elif code == '300':
                answer = '不好意思，我太笨啦，不能理解你说的问题'
        except Exception as e:
            print('error info: {}'.format(e))
            answer = '不好意思，我太笨啦，不能理解你说的问题'
            code = '400'

        result = {'answer': answer["answer"], 'code': code,
                  'message': msg_dic[code], 'request_id': request_id}
        return result


if __name__ == '__main__':
    server = Server()
    data = {"request_id": "ExamServer", "query": "pad黑屏了怎么办啊"}
    print("\n The result is ", server.get_result(data))
    data = {"request_id": "ExamServer", "query": "pad黑屏了怎么办啊"}
    print("\n The result is ", server.get_result(data))
    data = {"request_id": "ExamServer", "query": "pad黑屏了怎么办啊"}
    print("\n The result is ", server.get_result(data))