from sanic import Sanic
from sanic import response
import json
import time
from sanic.exceptions import NotFound
from sanic.response import text
from helper import Server

myServer = Server()

app = Sanic(__name__)
ip, port = "127.0.0.1", 9010


@app.exception(NotFound)
async def url_404(request, excep):
    return response.json({"Error": excep})


@app.route('/', methods=['GET'])
async def testfunc(request):
    return text("你想从我这里获得什么消息呢？")


@app.route('/QA', methods=['POST'])
async def model_server(request):
    try:
        json_bytes = request.body
        json_string = json_bytes.decode('utf-8')
        json_dict = json.loads(json_string)
        start_time = time.time()
        result = myServer.get_result(json_dict)
        print('耗时：', time.time()-start_time)
    except Exception as e:
        result = {"code": 400, "message": "预测失败", "Error": e}
    return response.json(result)


if __name__ == '__main__':
    app.run(host=ip, port=port, debug=True)