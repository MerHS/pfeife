import logging
import torch
import torch.optim as optim
from .utils import set_logger_level

default_optimizer = dict(type="adam", lr=1e-5)


def get_optimizer_cls(optimizer_type: str):
    type_str = optimizer_type.lower()
    if type_str == "adam":
        return optim.Adam
    elif type_str == "adamw":
        return optim.AdamW
    elif type_str == "adagrad":
        return optim.Adagrad
    elif type_str == "adamax":
        return optim.Adamax
    elif type_str == "asgd":
        return optim.ASGD
    elif type_str == "rmsprop":
        return optim.RMSprop
    elif type_str == "rprop":
        return optim.Rprop
    elif type_str == "sgd":
        return optim.SGD
    else:
        print(f"Unknown optimizer type: {optimizer_type}. Use SGD instead.")
        return optim.SGD


class PipeOption:
    """
    Properties:
    'compiler' (str): Type of TorchDynamo compiler (default: None - do not compile further)
    'optimizer' (dict):
        'type' (str): Type of gradient optimizer (default: 'adam')
        other values: injected into kwargs of the optimzier __init__ method

    'device_cnt' (int): number of usable GPU devices (default: 2)
    'batch_size' (int): size of the minibatch. count of the minibatch will be determined automaticaly (default: 4)
    'batch_cnt' (int): count of minibatch (default: device_cnt)
    'loop_cnt' (int): if set to nonzero, loop the forward pass by loop_cnt (default: 0). If zero, find the best loop_cnt automatically.

    'device_map' (list[int]): list of usable device id. use 0 to device_cnt-1 if None
    'device_bench' (str): path to the benchmark result of device communication.
        if there is no such file, PipeManager will run new benchmark and save it to the path.

    'verbosity' (str): 'error', 'warning', 'info', 'debug',
    """

    def __init__(self, **kwargs):
        self.compiler = kwargs.get("compiler", None)
        self.reducer_backend = kwargs.get("reducer_backend", "nccl")
        self.cpu_test = kwargs.get("cpu_test", False)

        self.device_cnt = kwargs.get("device_cnt", 2)
        self.batch_size = kwargs.get("batch_size", 4)
        self.batch_cnt = kwargs.get("batch_cnt", self.device_cnt)

        self.loop_cnt = kwargs.get("loop_cnt", None)
        self.loop_batch = kwargs.get("loop_batch", None)
        self.prefetch_fwd = kwargs.get("prefetch_fwd", None)
        self.split_points = kwargs.get("split_points", None)
        self.bench_seed = kwargs.get("bench_seed", 0)

        self.verbosity = kwargs.get("verbosity", "info")
        self.print_graph = kwargs.get("print_graph", False)
        self.device_map = kwargs.get(
            "device_map", [f"cuda:{i}" for i in range(self.device_cnt)]
        )
        self.device_bench = kwargs.get("device_bench", None)
        self.graph_bench = kwargs.get("graph_bench", None)
        self.scheduler = kwargs.get("scheduler", "single_comm")
        self.partitioner = kwargs.get("partitioner", "beam_single")
        self.linearizer = kwargs.get("linearizer", "none")
        self.use_fsdp = kwargs.get("use_fsdp", False)
        self.mixed_precision = kwargs.get("mixed_precision", None)
        if self.mixed_precision == "fp16":
            self.mixed_precision = torch.float16
        elif self.mixed_precision == "bf16":
            self.mixed_precision = torch.bfloat16

        self.no_side_effect = kwargs.get("no_side_effect", False)
        self.no_io_cache = kwargs.get("no_io_cache", False)
        self.single_comm_thread = kwargs.get("single_comm_thread", False)
        self.no_bench_optimizer = kwargs.get("no_bench_optimizer", False)

        if self.verbosity == "error":
            self.verbosity = logging.ERROR
        elif self.verbosity == "info":
            self.verbosity = logging.INFO
        elif self.verbosity == "debug":
            self.verbosity = logging.DEBUG
        else:
            self.verbosity = logging.WARNING

        optimizer = kwargs.get("optimizer", default_optimizer)
        self.optimizer_type = optimizer["type"]

        optimizer = optimizer.copy()
        del optimizer["type"]
        self.optimizer_kwargs = optimizer

    @staticmethod
    def from_args(args):
        if args.old_ver:
            args.no_side_effect = True
            args.no_io_cache = True
            args.single_comm_thread = True

        if args.cpu:
            args.single_comm_thread = True

        optimizer = {"type": args.optimizer_type, "lr": 1e-3}

        device_map = None
        if args.device_map:
            if ":" in args.device_map:
                [start, end] = args.device_map.strip().split(":")
                device_map = [f"cuda:{i}" for i in range(int(start), int(end))]
            else:
                devices = map(int, args.device_map.strip().split(","))
                device_map = [f"cuda:{i}" for i in devices]

        split_points = None
        if args.split_points:
            split_points = [
                int(t.strip()) for t in args.split_points.strip().split(",")
            ]

        option_kwargs = dict(
            compiler=args.backend,
            reducer_backend=args.reducer_backend,
            device_cnt=args.device_cnt,
            loop_cnt=args.loop_cnt,
            loop_batch=args.loop_batch,
            prefetch_fwd=args.prefetch_fwd,
            split_points=split_points,
            batch_size=args.batch_size,
            batch_cnt=args.batch_cnt,
            use_fsdp=args.use_fsdp,
            mixed_precision=args.mixed_precision,
            optimizer=optimizer,
            scheduler=args.scheduler,
            partitioner=args.partitioner,
            linearizer=args.linearizer,
            device_bench=args.device_bench,
            graph_bench=args.graph_bench,
            verbosity="debug" if args.verbose else "info",
            bench_seed=0,
            cpu_test=args.cpu,
            print_graph=args.print_graph,
            no_bench_optimizer=args.no_bench_optimizer,
            no_side_effect=args.no_side_effect,
            no_io_cache=args.no_io_cache,
            single_comm_thread=args.single_comm_thread,
        )
        if device_map is not None:
            option_kwargs["device_map"] = device_map

        option = PipeOption(**option_kwargs)
        option.set_logger()

        return option

    def set_logger(self):
        set_logger_level(self.verbosity)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--backend",
            help="backend for dynamo compiler",
        )
        parser.add_argument(
            "--reducer_backend",
            default="nccl",
            choices=["nccl", "gloo"],
            help="backend for gradient reducer",
        )
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument(
            "--device_cnt", type=int, default=2, help="Size of pipeline parallelism"
        )
        parser.add_argument("--batch_size", type=int, default=2)
        parser.add_argument("--batch_cnt", type=int, default=4)
        parser.add_argument("--optimizer_type", default="adam")
        parser.add_argument("--use_fsdp", action="store_true")
        parser.add_argument(
            "--mixed_precision",
            choices=["fp16", "bf16"],
            default=None,
        )
        parser.add_argument(
            "--bench_seed",
            type=int,
            help="set seed for benchmarking. if 0, do not set seed.",
        )

        parser.add_argument(
            "--scheduler",
            choices=["single_comm", "beam", "dfs", "bfs", "pipeless"],
            default="single_comm",
        )
        parser.add_argument(
            "--partitioner",
            choices=["beam_single", "beam_2", "random_walk", "same_mem", "same_time"],
            default="beam_single",
        )
        parser.add_argument(
            "--linearizer",
            choices=["save_mem", "short_first", "long_first", "none"],
            default="none",
        )

        parser.add_argument(
            "--device_map", help="select usable devices. e.g. '0:3', '1,2,5,6'"
        )
        parser.add_argument(
            "--device_bench",
            help="path to device benchmark file. If there is no file, create one.",
        )
        parser.add_argument(
            "--graph_bench",
            help="path to ComputationGraph benchmark file. If there is no file, create one.",
        )
        parser.add_argument(
            "--cpu",
            action="store_true",
            help="run on cpu (test only)",
        )

        parser.add_argument(
            "--split_points",
            help="set split points for the pipe graph. e.g. '10,15,20'",
        )
        parser.add_argument(
            "--loop_cnt",
            type=int,
            help="loop the forward pass by loop_cnt. If not set, find the best loop_cnt automatically.",
        )
        parser.add_argument(
            "--loop_batch",
            type=int,
            help="how many micro-batches in a single loop. If not set, find it automatically.",
        )
        parser.add_argument(
            "--prefetch_fwd",
            type=int,
            help="how many forward passese should be prefetched. If not set, find it automatically.",
        )

        parser.add_argument(
            "--no_bench_optimizer",
            action="store_true",
            help="run an optimizer when running computation graph benchmark",
        )
        parser.add_argument(
            "--no_side_effect",
            action="store_true",
            help="turn off dynamo side-effect",
        )
        parser.add_argument(
            "--no_io_cache",
            action="store_true",
            help="do not create io cache and send new tensor every time by tensor.to",
        )
        parser.add_argument(
            "--single_comm_thread",
            action="store_true",
            help="use single communication thread for all devices.",
        )
        parser.add_argument(
            "--old_ver", action="store_true", help="apply the above three actions."
        )
        parser.add_argument("--print_graph", action="store_true")

        return parser
