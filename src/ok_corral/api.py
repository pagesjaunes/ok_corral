import traceback, logging

from flask import Flask, jsonify, Blueprint, request
from flask_restplus import Resource, Api, swagger
from requests import codes as http_codes


from ok_corral.agent_manager import AgentManager
from ok_corral.bandits import BANDIT_AVAILABLES

app = Flask(__name__)

blueprint = Blueprint('api', __name__)

api = Api(blueprint, title='API', description="Api de bandits")

app.register_blueprint(blueprint)

agent_manager = AgentManager()

@api.route('/heartbeat')
class Heart(Resource):
    @staticmethod
    def get():
        """
            Heartbeat
            Est-ce que l'api est en vie ?
        """
        response = {
            'status_code': 200,
            'message': 'Heartbeat'
        }

        return _success(response)


@api.route('/bandit/')
class Bandit(Resource):


    doc_parser = api.parser()
    doc_parser.add_argument('p_user_key', location='args', required=True)
    doc_parser.add_argument('name', location='args', required=False)
    doc_parser.add_argument('type_algorithme', location='args', required=True, choices = BANDIT_AVAILABLES)
    doc_parser.add_argument('nombre_actions', type=int, location='args', required=True)

    @api.expect(doc_parser)
    def post(self):
        """
            Création d'une instance de bandit
        """
        p_user_key = request.args['p_user_key']
        name = request.args['name']
        type_algorithme = request.args['type_algorithme']
        nombre_actions = int(request.args['nombre_actions'])

        print(name,type_algorithme,nombre_actions)
        key = agent_manager.add_bandit(p_user_key,name,type_algorithme,nombre_actions)

        return jsonify(instance_key = key)


    doc_parser = api.parser()
    doc_parser.add_argument('instance_key', location='args', required=True)

    @api.expect(doc_parser)
    def get(self):
        """
        Retourne la décision prise par une instance
        """
        action = agent_manager.get_decision(request.args['instance_key'])

        return jsonify(action = action)


    doc_parser = api.parser()
    doc_parser.add_argument('instance_key', location='args', required=True)
    doc_parser.add_argument('action', type = int, location='args', required=True)
    doc_parser.add_argument('reward', type = float, location='args', required=True)

    @api.expect(doc_parser)
    def put(self):
        """
        Met à jour l'algorithme
        """

        action = agent_manager.observe(request.args['instance_key'], int(request.args['action']), float(request.args['reward']))

        return jsonify(message = "ok")


def _success(response):
    return make_reponse(response, http_codes.OK)


def _failure(exception, http_code=http_codes.SERVER_ERROR):
    try:
        exn = traceback.format_exc(exception)
    except:
        logging.info("EXCEPTION: {}".format(exception))
    try:
        data, code = exception.to_tuple()
        return make_reponse(data, code)
    except:
        try:
            data = exception.to_dict()
            return make_reponse(data, exception.http)
        except Exception as exn:
            return make_reponse(None, http_code)


def make_reponse(p_object=None, status_code: int = 200):
    """
        Fabrique un objet Response à partir d'un p_object et d'un status code
    """
    if p_object is None and status_code == 404:
        p_object = {"status": {"status_content": [{"code": "404 - Not Found", "message": "Resource not found"}]}}

    json_response = jsonify(p_object)
    json_response.status_code = status_code
    json_response.content_type = 'application/json;charset=utf-8'
    json_response.headers['Cache-Control'] = 'max-age=3600'
    return json_response


if __name__ == '__main__':
    app.run(debug=True)                #  Start a development server