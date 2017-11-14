import logging, traceback, json

from flask import Flask, jsonify, Blueprint, request
from flask_restplus import Resource, Api
from requests import codes as http_codes

from ok_corral.bandits import BANDIT_AVAILABLES
from ok_corral.engine.agent_manager import AgentManager, PrivilegeException

app = Flask(__name__)
app.config.SWAGGER_UI_JSONEDITOR = True

blueprint = Blueprint('api', __name__)

api = Api(blueprint, title='API', description="Api de bandits")

app.register_blueprint(blueprint)

agent_manager = AgentManager()


@api.route('/chifoumi')
class Chifoumi(Resource):

    PLAY = "play"

    PIERRE = "pierre"
    FEUILLE = "feuille"
    CISEAUX = "ciseaux"


    doc_parser = api.parser()
    doc_parser.add_argument(PLAY, location='args', required=True, choices=[PIERRE, FEUILLE, CISEAUX])

    @api.expect(doc_parser)
    def get(self):
        """
        Joues avec un bandit
        """

        if request.args[self.PLAY] == self.PIERRE:
            coup = self.FEUILLE

        elif request.args[self.PLAY] == self.FEUILLE:
            coup = self.CISEAUX

        else:
            coup = self.PIERRE

        response = {
            'status_code': 200,
            'message': coup
        }

        return _success(response)


@api.route('/bandit/')
class Bandit(Resource):

    USER_KEY = "user_key"
    NAME = "name"
    TYPE_ALGORITHME = "type_algorithme"
    NB_ACTIONS = "nombre_actions"
    DESC_CONTEXTE = 'description_contexte'
    INST_KEY = "instance_key"
    CONTEXTE = "contexte"
    ACTION = "action"
    REWARD = "reward"
    FILTRE = "filtre"


    HELP_USER_KEY = 'La clé utilisateur'
    HELP_CONTEXTE = "Le contexte (obligatoire pour les bandits contextuels)"
    HELP_FILTRE = "La liste des actions disponibles (ne pas remplir si tout est dispo)"

    doc_parser = api.parser()
    doc_parser.add_argument(USER_KEY, location='args', help=HELP_USER_KEY, required=True)
    doc_parser.add_argument(NAME, location='args', required=False)
    doc_parser.add_argument(TYPE_ALGORITHME, location='args', help="L'algorithme de bandits à utiliser",
                            required=True, choices=BANDIT_AVAILABLES)
    doc_parser.add_argument(NB_ACTIONS, type=int, location='args', required=True)
    doc_parser.add_argument(DESC_CONTEXTE, location='args',
                            help="La description du contexte (obligatoire pour les bandits contextuels)",
                            required=False)

    @api.expect(doc_parser)
    def post(self):
        """
            Création d'une instance de bandit
        """

        try:
            p_user_key = request.args[self.USER_KEY]
            name = request.args[self.NAME]
            type_algorithme = request.args[self.TYPE_ALGORITHME]
            nombre_actions = int(request.args[self.NB_ACTIONS])

            description_contexte = request.args[
                self.DESC_CONTEXTE] if self.DESC_CONTEXTE in request.args else None

            key = agent_manager.add_bandit(p_user_key, name, type_algorithme, nombre_actions,
                                           p_context_description=description_contexte)

            return jsonify(instance_key=str(key))

        except PrivilegeException as e:
            return jsonify(erreur=str(e))

    doc_parser = api.parser()
    doc_parser.add_argument(INST_KEY, location='args', required=True)
    doc_parser.add_argument(CONTEXTE, location='args', help=HELP_CONTEXTE,
                            required=False)
    doc_parser.add_argument(FILTRE, location='args', help=HELP_FILTRE,
                            required=False)
    @api.expect(doc_parser)
    def get(self):
        """
        Retourne la décision prise par une instance
        """
        context = request.args[self.CONTEXTE] if self.CONTEXTE in request.args else None
        filtre = request.args[self.FILTRE] if self.FILTRE in request.args else None
        if filtre is not None:
            filtre = set(int(i_x) for i_x in json.loads(filtre))
        try:
            action = agent_manager.get_decision(request.args[self.INST_KEY], p_context=context, p_filtre = filtre)

            return jsonify(action=str(action))

        except PrivilegeException as e:
            return jsonify(erreur=str(e))

    doc_parser = api.parser()
    doc_parser.add_argument(INST_KEY, location='args', required=True)
    doc_parser.add_argument(ACTION, type=int, location='args', required=True)
    doc_parser.add_argument(REWARD, type=float, location='args', required=True)
    doc_parser.add_argument(CONTEXTE, location='args', help=HELP_CONTEXTE,
                            required=False)

    @api.expect(doc_parser)
    def put(self):
        """
        Met à jour l'algorithme
        """
        contexte = request.args[self.CONTEXTE] if self.CONTEXTE in request.args else None


        try:
            action = agent_manager.observe(request.args[self.INST_KEY], int(request.args[self.ACTION]),
                                           float(request.args[self.REWARD]), p_context=contexte)
            return jsonify(message="ok")

        except PrivilegeException as e:
            return jsonify(erreur=str(e))


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
    app.run(debug=True)  # Start a development server
