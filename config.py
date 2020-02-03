import os
import yaml


class Configure:
    __config = dict()

    @classmethod
    def get_config(cls):
        assert len(cls.__config) != 0, 'Configure is must be initialized by init method. config.py -> set_config'
        return cls.__config

    @classmethod
    def init(cls, path):
        with open(path, 'r') as stream:
            default = yaml.safe_load(stream)
        cls.__config.update(default)

        if cls.__config['DATASET']['NAME'] == 'MNIST':
            cls.__config['DATASET']['N_CHANNELS'] = 1
            cls.__config['DATASET']['IM_SIZE'] = 32
        if cls.__config['DATASET']['NAME'] == 'CIFAR10':
            cls.__config['DATASET']['N_CHANNELS'] = 3
            cls.__config['DATASET']['IM_SIZE'] = 32
        if cls.__config['DATASET']['NAME'] == 'CELEBA64':
            cls.__config['DATASET']['N_CHANNELS'] = 3
            cls.__config['DATASET']['IM_SIZE'] = 64

        if not os.path.exists('result'):
            os.mkdir('result')
        if not os.path.exists('save'):
            os.mkdir('save')
        if not os.path.exists(os.path.join('result', cls.__config['NAME'])):
            os.mkdir(os.path.join('result', cls.__config['NAME']))
        if not os.path.exists(os.path.join('save', cls.__config['NAME'])):
            os.mkdir(os.path.join('save', cls.__config['NAME']))


def get_config():
    return Configure.get_config()


def set_config(path):
    Configure.init(path)