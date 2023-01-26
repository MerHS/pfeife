default_optimizer = dict(type="adam", lr=1e-3)


class PipeOption:
    """
    Properties:
    'compiler' (str): Type of TorchDynamo compiler (default: 'aot_eager')
    'optimizer' (dict):
        'type' (str): Type of gradient optimizer (default: 'adam')
        other values: injected into kwargs of the optimzier __init__ method
    """

    def __init__(self, **kwargs):
        self.compiler = kwargs.get("compiler", "aot_eager")

        optimizer = kwargs.get("optimizer", default_optimizer)
        self.optimzier_type = optimizer.setdefault("type", "adam")

        opt_kwargs = optimizer.copy()
        del opt_kwargs["type"]

        self.optimizer_kwargs = opt_kwargs
