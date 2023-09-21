class ConfigWrapper(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = ConfigWrapper(value)
        return value