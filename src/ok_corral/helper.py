import json


def deserialize_json(p_json):
    if type(p_json) == str:
        return json.loads(p_json)
    else:
        return p_json


def serialize_json(p_json, p_dump = True):
    return json.dumps(p_json) if p_dump else p_json