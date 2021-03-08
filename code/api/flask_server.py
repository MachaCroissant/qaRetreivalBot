from flask import Flask, request, make_response
from flask_cors import CORS
from helper import Server
import json
import time
app = Flask(__name__)
CORS(app)

myServer = Server()


@app.route('/', methods=['GET'])
def testfunc():
    question = str(request.args['ques'])
    query_dict = {"request_id": 'ExamServer', "query": question}
    try:
        result = myServer.get_result(query_dict)
        return result
    except KeyError:
        return f'输入无效！'


# @app.route('/QA', methods=['POST', 'OPTIONS'])
# def model_server(request):
#     try:
#         json_bytes = request.body
#         json_string = json_bytes.decode('utf-8')
#         json_dict = json.loads(json_string)
#         start_time = time.time()
#         result = myServer.get_result(json_dict)
#         print('耗时：', time.time()-start_time)
#     except Exception as e:
#         result = {"code": 400, "message": "预测失败", "Error": e}
#     return make_response(result)


if __name__ == '__main__':
    app.run(debug=True)