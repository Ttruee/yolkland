from flask import Flask, request, make_response  # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource, Namespace, fields  # Api 구현을 위한 Api 객체 import
import json
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

api = Api(
    app,
    version='1.0',
    title="타이틀",
    description="설명",
    terms_url="/",
    contact="",
    license="MIT",
)


model = pickle.load(open('lm.pkl','rb'))

def _create_input(_quarter, _code, _ser_code, _n):
    return pd.Series({
        'quarter': _quarter,
        'code': _code,
        'ser_code': _ser_code,
        'n': _n
    })

def predict_value(_quarter, _code, _ser_code, _n):
    data_input = _create_input(_quarter, _code, _ser_code, _n)
    print(data_input)

    result_value = model.predict([np.array(data_input)])
    return result_value[0]

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

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=80)