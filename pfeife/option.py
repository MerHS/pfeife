default_optimizer = dict(type="adam", lr=1e-3)


class PipeOption:
    """
    Properties:
    'compiler' (str): Type of TorchDynamo compiler (default: 'aot_eager')
    'optimizer' (dict):
        'type' (str): Type of gradient optimizer (default: 'adam')
        other values: injected into kwargs of the optimzier __init__ method
    'scheduler' (str): pipeline scheduling strategy (default: gpipe)
    'batch_cnt' (int): number of microbatches (default: 4)
    'device_cnt' (int): number of usable GPU devices (default: 2)
    'stage_cnt' (int): numbers of pipeline stages (default: same as device_cnt)
    """

    def __init__(self, **kwargs):
        self.compiler = kwargs.get("compiler", "aot_eager")
        self.scheduler = kwargs.get("scheduler", "gpipe")
        self.batch_cnt = kwargs.get("batch_cnt", 4)
        self.device_cnt = kwargs.get("device_cnt", 2)
        self.stage_cnt = kwargs.get("stage_cnt", self.device_cnt)

        optimizer = kwargs.get("optimizer", default_optimizer)
        self.optimzier_type = optimizer.setdefault("type", "adam")

        opt_kwargs = optimizer.copy()
        del opt_kwargs["type"]

        self.optimizer_kwargs = opt_kwargs
