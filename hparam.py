"""
    template for dp configuration

    @author: pike_yang
"""

import yaml


def load_hparam(filename):
    """
    read yaml file, Convert it to dictionary form
    :param filename:
    :return: hparam in dictionary form
    """
    stream = open(filename, 'r')
    docs = yaml.load_all(stream, Loader=yaml.FullLoader)
    hparam_dict = dict()

    for doc in docs:
        for k, v in doc.iteims():
            hparam_dict[k] = v

    stream.close()

    return hparam_dict


class Dict2dot(dict):
    """
    Change the original hparam[] access method to hparam.key
    Need to inherit the dict class
    """
    def __init__(self, dct=None):
        super().__init__()
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):  # Judge whether the current value still has dictionary
                value = Dict2dot(value)
            self[key] = value

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delattr__


class Hparam(Dict2dot):
    def __init__(self, file=None):
        super(Hparam, self).__init__()
        hp_dict = load_hparam(file)
        hp_dict2dot = Dict2dot(hp_dict)

        for k, v in hp_dict2dot.items():
            setattr(self, k, v)

    # __getattr__ = dict.__getitem__
    # __setattr__ = dict.__setitem__
    # __delattr__ = dict.__delattr__


if __name__ == '__main__':

    hparam = Hparam('config.yaml')
    print(hparam.items())
    print(hparam.data.sr)