from flask import Flask, request
from flask_restx import Api, Resource
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

api = Api(app)


model = pickle.load(open('lm.pkl','rb'))

# 입력 형태에 맞게 만들어주는 함수
def _create_input(_s_facil_no_of_supmkt, _s_facil_no_of_bank, _s_facil_no_of_subway, _s_facil_no_of_dept, _s_facil_no_of_bus):
    return [np.array(pd.Series({
        's_facil_no_of_supmkt': _s_facil_no_of_supmkt,
        's_facil_no_of_bank': _s_facil_no_of_bank,
        's_facil_no_of_subway': _s_facil_no_of_subway,
        's_facil_no_of_dept': _s_facil_no_of_dept,
        's_facil_no_of_bus': _s_facil_no_of_bus
    }))]

# 입력 받은 값으로 예측한 결과값을 출력
def predict_value(_s_facil_no_of_supmkt, _s_facil_no_of_bank, _s_facil_no_of_subway, _s_facil_no_of_dept, _s_facil_no_of_bus):
    data_input = _create_input(_s_facil_no_of_supmkt, _s_facil_no_of_bank, _s_facil_no_of_subway, _s_facil_no_of_dept, _s_facil_no_of_bus)
    result_value = model.predict(data_input)

    return result_value[0]

# 클라이언트 요청 처리함수 > 클라이언트가 지정변수 값을 담아서 서버가 인식할 수 있게 보내줌
@api.route('/api')
class getAllData(Resource):
    @api.response(200, 'Success')
    @api.response(500, 'Failed')
    def get(self):
        s_facil_no_of_supmkt = float(request.args.get('s_facil_no_of_supmkt'))
        s_facil_no_of_bank = float(request.args.get('s_facil_no_of_bank'))
        s_facil_no_of_subway = float(request.args.get('s_facil_no_of_subway'))
        s_facil_no_of_dept = float(request.args.get('s_facil_no_of_dept'))
        s_facil_no_of_bus = float(request.args.get('s_facil_no_of_bus'))
             
        return {
            "result": predict_value(s_facil_no_of_supmkt, s_facil_no_of_bank, s_facil_no_of_subway, s_facil_no_of_dept, s_facil_no_of_bus)
        }
        
app.run(debug=False, host='0.0.0.0', port=5000)
# 이 파일을 직접 실행할 경우에 main에 있는 걸 실행 
# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=80)
