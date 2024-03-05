class Task:
    def __init__(self):
        self.name =self.__class__.__name__.lower()

    def __call__(self):
        raise NotImplementedError
