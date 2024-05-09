from flask import Flask
from flask import url_for, request
from markupsafe import escape
from ddz.predict.predict_winrate import PredictWinrate
from ddz.predict.predictor import PredictorManager
import os, time

app = Flask(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

PREDICT_MODEL_PATH = "./checkpoints"

winrate_predictor = None
predictor_manager = PredictorManager()


@app.route('/ai/ddz/create_desk', methods=['POST'])
def create_desk():
    if request.method == 'POST':
        try:
            json_data = request.json
            # print("json data of create desk:", json_data)
            desk_id = json_data['desk_id']
            agents = json_data['agents']
            print("create desk:", desk_id)
            game_config = {}
            for agent in agents:
                position = agent['position']
                level = agent['level']
                cards = agent['cards']
                game_config[position] = {'level': level, 'cards': cards}
            if game_config:
                predictor_manager.create_desk(desk_id, game_config)
                return {
                    "desk_id": desk_id
                }
            else:
                raise RuntimeError("not agent config found.")
        except RuntimeError as err:
            print("create desk error:", err)
            return {
                "error": "create desk error"
            }, 520
    else:
        return 'POST method should be used in create_desk.'


@app.route('/ai/ddz/action', methods=['POST'])
def get_action():
    if request.method == 'POST':
        try:
            json_data = request.json
            desk_id = json_data['desk_id']
            position = json_data['position']
            # start = time.time()
            cards = predictor_manager.get_action(desk_id, position)
            # print("get action time:", time.time() - start)
            return {
                'desk_id': desk_id,
                'position': position,
                'aiPlayCards': cards,
            }
        except RuntimeError as err:
            print("get action error:", err)
            return {
                "error": "get action error."
            }, 520
    else:
        return 'POST method should be used in get_action.'


@app.route('/ai/ddz/step', methods=['POST'])
def step():
    if request.method == 'POST':
        try:
            json_data = request.json
            desk_id = json_data['desk_id']
            position = json_data['position']
            cards = json_data['cards']
            predictor_manager.step(desk_id, position, cards)
            return {
                'desk_id': desk_id,
                'position': position,
            }

        except RuntimeError as err:
            print("step error:", err)
            return {
                "error": "step env error."
            }, 520
    else:
        return 'POST method should be used in step.'


@app.route('/ai/ddz/destroy_desk', methods=['POST'])
def destroy_desk():
    if request.method == 'POST':
        try:
            json_data = request.json
            desk_id = json_data['desk_id']
            print("destroy desk:", desk_id)
            predictor_manager.destroy_desk(desk_id)
            return {
                "desk_id": desk_id
            }
        except RuntimeError as err:
            print("destroy desk error:", err)
            return {
                "error": "destroy desk error."
            }, 520
    else:
        return 'POST method should be used in destroy_desk.'


@app.route('/ai/ddz/predict_dizhu', methods=['POST'])
def predict_dizhu():
    if request.method == 'POST':
        global winrate_predictor
        if not winrate_predictor:
            winrate_predictor = PredictWinrate(PREDICT_MODEL_PATH)
        try:
            json_data = request.json
            cards = json_data['cards']
            should_be_dizhu = winrate_predictor.predict_winrate(cards)
            return {
                "dizhu": should_be_dizhu
            }
        except RuntimeError as err:
            print("predict dizhu error:", err)
            return {
                "error": "predict dizhu error."
            }, 520
    else:
        return 'POST method should be used in destroy_desk.'


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=False, port=9091)
