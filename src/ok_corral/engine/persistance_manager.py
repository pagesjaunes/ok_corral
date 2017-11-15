import os.path
import sqlite3

from ok_corral.bandits import get_class_from_algo

DATA_BASE_NAME = 'ok_corral.db'

def check_database():
    """

    :return: True si elle existe, False si elle a été crée
    """

    if not os.path.isfile(DATA_BASE_NAME):
        _create_tables()
        return False

    else:
        return True


def _create_tables():

    conn = sqlite3.connect(DATA_BASE_NAME)

    c = conn.cursor()

    # Creation de la table des users
    c.execute('''CREATE TABLE users
                 (key TEXT PRIMARY KEY, name TEXT) WITHOUT ROWID;''')

    # Creation de la table des instances
    c.execute('''CREATE TABLE instances
                 (key TEXT PRIMARY KEY, name TEXT, type TEXT, algorithme TEXT, instance TEXT, owner_key TEXT) WITHOUT ROWID;''')


    conn.commit()

    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.
    conn.close()

def add_user_key_to_database(p_user_key, p_name):

    conn = sqlite3.connect(DATA_BASE_NAME)

    c = conn.cursor()

    c.execute('''INSERT INTO users VALUES (?,?)''',(p_user_key,p_name))

    conn.commit()

def get_user_keys_from_database():

    r_keys = set()

    conn = sqlite3.connect(DATA_BASE_NAME)

    c = conn.cursor()

    for row in c.execute('SELECT key FROM users'):
        r_keys.add(row[0])

    return r_keys



def add_instance_to_database(p_key, p_instance):

    from ok_corral.engine.agent_manager import AgentManager

    conn = sqlite3.connect(DATA_BASE_NAME)

    c = conn.cursor()

    c.execute('''INSERT INTO instances VALUES (?,?,?,?,?,?)''',\
              (p_key,\
               p_instance[AgentManager.NAME],\
               p_instance[AgentManager.TYPE_BANDIT],\
               p_instance[AgentManager.ALGO_NAME],\
               p_instance[AgentManager.INSTANCE].to_json(),\
               p_instance[AgentManager.OWNER_KEY]))

    conn.commit()


def get_instances_from_database():

    from ok_corral.engine.agent_manager import AgentManager

    instances = {}

    conn = sqlite3.connect(DATA_BASE_NAME)

    c = conn.cursor()

    for i_row in c.execute('SELECT * FROM instances'):
        i_key = i_row[0]
        i_name = i_row[1]
        i_type = i_row[2]
        i_algo = i_row[3]
        i_instance = i_row[4]
        i_owner_key = i_row[5]

        instances[i_key] = {AgentManager.INSTANCE: get_class_from_algo(i_algo).from_json(i_instance),\
                            AgentManager.TYPE_BANDIT: i_type,\
                            AgentManager.ALGO_NAME: i_algo,\
                            AgentManager.NAME: i_name,\
                            AgentManager.OWNER_KEY: i_owner_key}

    return instances


def update_instance_in_database(p_key, p_instance):

    from ok_corral.engine.agent_manager import AgentManager

    conn = sqlite3.connect(DATA_BASE_NAME)

    c = conn.cursor()

    c.execute('''UPDATE instances
            SET instance = ?
            WHERE key = ?''',\
          (p_instance[AgentManager.INSTANCE].to_json(), \
           p_key))

    conn.commit()


if __name__ == '__main__':
    #create_tables()
    """
    from ok_corral.agent_manager import SQUELETON_KEY
    add_user_key_to_database(p_user_key=SQUELETON_KEY)
    """
    print(get_instances_from_database())
    pass
