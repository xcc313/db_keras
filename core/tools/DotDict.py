class DotDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

    def to_dot_dict(data):
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    data[k] = DotDict(v)
                    DotDict.to_dot_dict(data[k])
        else:
            return data

        return DotDict(data)
