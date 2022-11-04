from flask import Flask, request
from flask_restx import Api, Resource
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

api = Api(app)


model = pickle.load(open('lm.pkl','rb'))

# 입력 형태에 맞게 만들어주는 함수
def _create_input(_quarter, _code, _ser_code, _n):
    return [np.array(pd.Series({
        'quarter': _quarter,
        'code': _code,
        'ser_code': _ser_code,
        'n': _n
    }))]

# 입력 받은 값으로 예측한 결과값을 출력
def predict_value(_quarter, _code, _ser_code, _n):
    data_input = _create_input(_quarter, _code, _ser_code, _n)
    result_value = model.predict(data_input)

    return result_value[0]

# 클라이언트 요청 처리함수 > 클라이언트가 지정변수 값을 담아서 서버가 인식할 수 있게 보내줌
@api.route('/api')
class getAllData(Resource):
    @api.response(200, 'Success')
    @api.response(500, 'Failed')
    def get(self):
        quarter = float(request.args.get('quarter'))
        code = float(request.args.get('code'))
        ser_code = float(request.args.get('ser_code'))
        n = float(request.args.get('n'))
        
        
        return {
            "result": predict_value(quarter, code, ser_code, n)
        }
        
app.run(debug=True, host='0.0.0.0', port=80)
# 이 파일을 직접 실행할 경우에 main에 있는 걸 실행 
# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=80)