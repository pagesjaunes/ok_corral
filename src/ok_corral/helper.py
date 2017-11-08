import json


def deserialize_json(p_json):
    if type(p_json) == str:
        return json.loads(p_json)
    else:
        return p_json


def serialize_json(p_json, p_dump=True):
    return json.dumps(p_json) if p_dump else p_json


"""
if __name__ == '__main__':

    from src.ok_corral.api import api, app

    with app.app_context():

        with open('../../swagger.json', 'w') as outfile:
            json.dump(api.__schema__, outfile)

    La doc est ensuite injectée dans le README.md avec:
    pip install swagger2markdown
"""
