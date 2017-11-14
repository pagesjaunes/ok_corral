# coding=utf-8
import abc
import random


class Agent:
    __metaclass__ = abc.ABCMeta

    """
    Contrat de l'agent
    """

    NOMBRE_BRAS = "nombre_bras"
    CLASS = "class"

    def __init__(self):
        pass

    @abc.abstractmethod
    def select_action(self, *args):
        pass

    @abc.abstractmethod
    def observe(self, *args):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def to_json(self, p_dump):
        return None

    @staticmethod
    @abc.abstractmethod
    def from_json(self, p_json):
        return None


