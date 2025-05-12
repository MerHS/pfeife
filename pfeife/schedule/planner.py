import time
import random
from typing import Optional, List, Dict, Tuple, Set
from collections import defaultdict, deque, namedtuple
from pathlib import Path


import matplotlib.pyplot as plt

from ..graph.computation_graph import CompGraph
from ..graph.pipe_graph import PipeGraph, PipeNode, PipeEdge
from ..device_bench import DeviceBench
from .scheduler import PipeSchedParams, PipeSchedSet
from ..option import PipeOption
from ..utils import fmem

RESERVE_MEM_RATIO = 0.85
RAND_WALK_ITER = 1000
BEAM_ITER = 100
FWD_AVG_MIN = 3.0
T_MAX_EPSILON = 0.01

OptimResult = namedtuple(
    "OptimResult", ["split_points", "time", "peak_mem", "validity"]
)


class SchedNode:
    def __init__(self, duration: float = 0.0):
        self.prev_nodes: List[SchedNode] = []
        self.next_nodes: List[SchedNode] = []
        self.start_time = 0
        self.duration = duration
        self.visited = False

    def add_next(self, node):
        self.next_nodes.append(node)
        node.prev_nodes.append(self)

    def set_start_time(self):
        if self.prev_nodes:
            self.start_time = max(
                prev.start_time + prev.duration for prev in self.prev_nodes
            )

    def serialize(self):
        # print("serializing", self, [(id(n), str(n)) for n in self.prev_nodes])

        return {
            "prev_nodes": [id(n) for n in self.prev_nodes],
            "next_nodes": [id(n) for n in self.next_nodes],
            "start_time": self.start_time,
            "duration": self.duration,
            "visited": self.visited,
            "class": self.__class__.__name__,
        }

    def deserialize(self, serial_obj, node_map: Dict[int, "SchedNode"]):
        self.prev_nodes = [node_map[id] for id in serial_obj["prev_nodes"]]
        self.next_nodes = [node_map[id] for id in serial_obj["next_nodes"]]
        self.start_time = serial_obj["start_time"]
        self.duration = serial_obj["duration"]
        self.visited = serial_obj["visited"]

    def text_next(self):
        return " / ".join([str(n) for n in self.next_nodes])


class TerminalNode(SchedNode):
    def __init__(self, duration: float = 0.0):
        super().__init__(duration)


class SchedCompNode(SchedNode):
    def __init__(
        self,
        rank: int,
        node_id: int,
        batch_id: int,
        is_forward: bool,
        duration: float = 0.0,
    ):
        super().__init__(duration)
        self.rank = rank
        self.node_id = node_id
        self.batch_id = batch_id
        self.is_forward = is_forward

    def __str__(self):
        return f"([{self.rank}] {self.node_id} {self.batch_id}{'+' if self.is_forward else '-'})"

    def serialize(self):
        return {
            **super().serialize(),
            "rank": self.rank,
            "node_id": self.node_id,
            "batch_id": self.batch_id,
            "is_forward": self.is_forward,
        }

    def deserialize(self, serial_obj, node_map):
        super().deserialize(serial_obj, node_map)
        self.rank = serial_obj["rank"]
        self.node_id = serial_obj["node_id"]
        self.batch_id = serial_obj["batch_id"]
        self.is_forward = serial_obj["is_forward"]


class SchedCommNode(SchedNode):
    def __init__(
        self,
        from_node: SchedCompNode,
        to_node: SchedCompNode,
        duration: float = 0.0,
        do_init=True,
    ):
        super().__init__(duration)
        if do_init:
            self.src_comp = from_node
            self.dst_comp = to_node
            self.prev_nodes.append(from_node)
            self.next_nodes.append(to_node)
            from_node.next_nodes.append(self)
            to_node.prev_nodes.append(self)
        else:
            self.src_comp = None
            self.dst_comp = None

    def __str__(self):
        return f"comm {self.src_comp} -> {self.dst_comp} ({self.duration} ms)"

    def serialize(self):
        return {
            **super().serialize(),
            "src_comp": id(self.src_comp),
            "dst_comp": id(self.dst_comp),
        }

    def deserialize(self, serial_obj, node_map):
        super().deserialize(serial_obj, node_map)
        self.src_comp = node_map[serial_obj["src_comp"]]
        self.dst_comp = node_map[serial_obj["dst_comp"]]


class DependencyGraph:
    def __init__(self, sched_params: PipeSchedParams):
        self.sched_params = sched_params
        self.source_node = TerminalNode()
        self.sink_node = TerminalNode()

        self.rank_nodes: List[List[SchedCompNode]] = [
            [] for _ in range(sched_params.device_cnt)
        ]
        self.comm_nodes: Dict[Tuple[SchedCompNode, SchedCompNode], SchedCommNode] = (
            dict()
        )

        # populated after calc_schedule
        self.channel_comms: Dict[Tuple[int, int], List[SchedCommNode]] = defaultdict(
            list
        )

        # node_id, batch_id, is_fwd
        self.comp_dict: Dict[Tuple[int, int, bool], SchedCompNode] = dict()

        # communication nodes that are not a direct sequence of the linear computation
        # i.e. not (rank -> rank+1)
        self._nonstd_comms: List[SchedCommNode] = []

        self._build_default_graph()

    def serialize(self):
        kv_dict = {
            "sched_params": self.sched_params,
            "source_node": id(self.source_node),
            "sink_node": id(self.sink_node),
            "rank_nodes": [[id(n) for n in rn] for rn in self.rank_nodes],
            "comm_nodes": {
                (id(k), id(v)): id(n) for (k, v), n in self.comm_nodes.items()
            },
            "channel_comms": {
                k: [id(n) for n in v] for k, v in self.channel_comms.items()
            },
            "comp_dict": {(k[0], k[1], k[2]): id(v) for k, v in self.comp_dict.items()},
            "_nonstd_comms": [id(n) for n in self._nonstd_comms],
        }

        node_set = set([self.source_node, self.sink_node])
        for rn in self.rank_nodes:
            for n in rn:
                node_set.add(n)
        i = 0
        for c in self.comm_nodes.values():
            node_set.add(c)
            # print("comm", i, c.visited, c, id(c), [str(n) for n in c.prev_nodes], [(id(n), str(n)) for n in c.next_nodes])
            i += 1
        i = 0
        for c in self.comp_dict.values():
            node_set.add(c)
            # print("comp", i, c.visited, c, id(c), [str(n) for n in c.prev_nodes], [str(n) for n in c.next_nodes])
            i += 1
        i = 0
        for c in self._nonstd_comms:
            node_set.add(c)
            # print("non-std", i, c.visited, c, id(c), [str(n) for n in c.prev_nodes], [str(n) for n in c.next_nodes])
            i += 1

        node_dict = {id(n): n.serialize() for n in node_set}

        return kv_dict, node_dict

    @staticmethod
    def deserialize(kv_dict, node_dict_serial):
        self = DependencyGraph(kv_dict["sched_params"])

        node_dict = dict()
        for nid, serial_obj in node_dict_serial.items():
            if serial_obj["class"] == "TerminalNode":
                node = TerminalNode(serial_obj["duration"])
            elif serial_obj["class"] == "SchedCompNode":
                node = SchedCompNode(
                    serial_obj["rank"],
                    serial_obj["node_id"],
                    serial_obj["batch_id"],
                    serial_obj["is_forward"],
                    serial_obj["duration"],
                )
            elif serial_obj["class"] == "SchedCommNode":
                node = SchedCommNode(None, None, serial_obj["duration"], do_init=False)

            node_dict[nid] = node

        for nid, serial_obj in node_dict_serial.items():
            node = node_dict[nid]
            node.deserialize(serial_obj, node_dict)

        self.source_node = node_dict[kv_dict["source_node"]]
        self.sink_node = node_dict[kv_dict["sink_node"]]
        self.rank_nodes = [
            [node_dict[nid] for nid in rn] for rn in kv_dict["rank_nodes"]
        ]
        self.comm_nodes = {
            (node_dict[k], node_dict[v]): node_dict[n]
            for (k, v), n in kv_dict["comm_nodes"].items()
        }
        self.channel_comms = defaultdict(list)
        for k, v in kv_dict["channel_comms"].items():
            self.channel_comms[k] = [node_dict[nid] for nid in v]

        self.comp_dict = {
            (k[0], k[1], k[2]): node_dict[nid]
            for k, nid in kv_dict["comp_dict"].items()
        }
        self._nonstd_comms = [node_dict[nid] for nid in kv_dict["_nonstd_comms"]]

        return self

    def _build_default_graph(self):
        params = self.sched_params
        total_node = params.node_cnt

        # create and order rank bound computations
        for rank, rank_work in enumerate(params.rank_works):
            rank_nodes = self.rank_nodes[rank]
            for node_idx, batch_id, is_forward in rank_work:
                curr_node = SchedCompNode(rank, node_idx, batch_id, is_forward)
                rank_nodes.append(curr_node)
                self.comp_dict[(node_idx, batch_id, is_forward)] = curr_node

            opt_node = TerminalNode()
            rank_nodes.append(opt_node)
            opt_node.add_next(self.sink_node)

            if rank == 0:
                comp_nodes = [self.source_node, *rank_nodes]
            else:
                comp_nodes = rank_nodes

            for prev, curr in zip(comp_nodes, comp_nodes[1:]):
                prev.add_next(curr)

        # create communications
        for rank, comp_nodes in enumerate(self.rank_nodes):
            for curr_node in comp_nodes:
                if isinstance(curr_node, TerminalNode):
                    continue

                node_idx = curr_node.node_id
                is_forward = curr_node.is_forward
                batch_id = curr_node.batch_id

                if is_forward and node_idx < total_node - 1:
                    next_node = self.comp_dict[(node_idx + 1, batch_id, True)]
                    fwd_comm_node = SchedCommNode(curr_node, next_node)
                    self.comm_nodes[(curr_node, next_node)] = fwd_comm_node

                elif not is_forward and node_idx > 0:
                    prev_node = self.comp_dict[(node_idx - 1, batch_id, False)]
                    bwd_comm_node = SchedCommNode(curr_node, prev_node)
                    self.comm_nodes[(curr_node, prev_node)] = bwd_comm_node

        def norm_chan(r1, r2):
            if r1 > r2:
                return (r2, r1)
            return (r1, r2)

        # order of rank bound communications
        for comp_nodes in self.rank_nodes:
            prev_fwd = None
            prev_bwd = None
            for curr_node in comp_nodes:
                if isinstance(curr_node, TerminalNode):
                    continue

                if curr_node.is_forward:
                    if prev_fwd is None:
                        prev_fwd = curr_node
                        continue
                    prev_node = prev_fwd
                    prev_fwd = curr_node
                else:
                    if prev_bwd is None:
                        prev_bwd = curr_node
                        continue
                    prev_node = prev_bwd
                    prev_bwd = curr_node

                for prev_comm in prev_node.next_nodes:
                    if not isinstance(prev_comm, SchedCommNode):
                        continue
                    prev_channel = norm_chan(
                        prev_comm.src_comp.rank, prev_comm.dst_comp.rank
                    )
                    for curr_comm in curr_node.next_nodes:
                        if not isinstance(curr_comm, SchedCommNode):
                            continue
                        curr_channel = norm_chan(
                            curr_comm.src_comp.rank,
                            curr_comm.dst_comp.rank,
                        )
                        if prev_channel == curr_channel:
                            prev_comm.add_next(curr_comm)
                            break

    def add_nonstd_comm(self, src_node_id, dst_node_id, duration=0.0):
        batch_cnt = self.sched_params.batch_cnt
        prev_fwd = None
        prev_bwd = None
        for batch_id in range(batch_cnt):
            src_fwd = self.comp_dict[(src_node_id, batch_id, True)]
            dst_fwd = self.comp_dict[(dst_node_id, batch_id, True)]
            src_bwd = self.comp_dict[(src_node_id, batch_id, False)]
            dst_bwd = self.comp_dict[(dst_node_id, batch_id, False)]

            if (src_fwd, dst_fwd) in self.comm_nodes:
                break

            fwd_comm = SchedCommNode(src_fwd, dst_fwd, duration)
            bwd_comm = SchedCommNode(dst_bwd, src_bwd, duration)
            self._nonstd_comms.append(fwd_comm)
            self._nonstd_comms.append(bwd_comm)
            self.comm_nodes[(src_fwd, dst_fwd)] = fwd_comm
            self.comm_nodes[(dst_bwd, src_bwd)] = bwd_comm

            # add dependency between same dst_rank & different node

            for out_comm in src_fwd.next_nodes:
                if not isinstance(out_comm, SchedCommNode):
                    continue
                if out_comm.dst_comp.rank != dst_fwd.rank:
                    continue
                if out_comm.dst_comp.node_id < dst_fwd.node_id:
                    out_comm.add_next(fwd_comm)

            for out_comm in dst_bwd.next_nodes:
                if not isinstance(out_comm, SchedCommNode):
                    continue
                if out_comm.dst_comp.rank != src_bwd.rank:
                    continue
                if out_comm.dst_comp.node_id > src_bwd.node_id:
                    out_comm.add_next(bwd_comm)

            if prev_fwd is not None:
                prev_fwd.add_next(fwd_comm)
            prev_fwd = fwd_comm

            if prev_bwd is not None:
                prev_bwd.add_next(bwd_comm)
            prev_bwd = bwd_comm

    def clear_nonstd_comm(self):
        for comm in self._nonstd_comms:
            src = comm.src_comp
            dst = comm.dst_comp
            src.next_nodes.remove(comm)
            dst.prev_nodes.remove(comm)

            del self.comm_nodes[(src, dst)]

        self._nonstd_comms.clear()

        for rank_nodes in self.rank_nodes:
            for node in rank_nodes:
                node.visited = False
                node.start_time = 0
        for comm_node in self.comm_nodes.values():
            comm_node.visited = False
            comm_node.start_time = 0

        self.source_node.visited = False
        self.sink_node.visited = False

    def calc_schedule(self):
        initial_node = self.source_node

        comp_nodes = self.rank_nodes
        channel_comms: Dict[Tuple[int, int], List[SchedCommNode]] = defaultdict(list)
        all_nodes = []

        for nodes in comp_nodes:
            all_nodes.extend(nodes)
        all_nodes.extend(self.comm_nodes.values())

        def calc_longest_path_len(initial_nodes: List[SchedNode]):
            # calculate the length of the longest path as if we have infinite number of communication channels
            q = deque(initial_nodes)
            not_visited = [node for node in all_nodes if not node.visited]
            max_time = 0
            while q:
                node = q.popleft()
                if node.visited:
                    continue
                can_visit = True
                min_time = 0
                for prev in node.prev_nodes:
                    if prev.visited is False:
                        can_visit = False
                        break
                    min_time = max(min_time, prev.start_time + prev.duration)
                if can_visit:
                    node.visited = True
                    node.start_time = min_time
                    max_time = max(max_time, min_time + node.duration)
                    q.extend(node.next_nodes)

            for node in not_visited:
                node.visited = False

            return max_time

        q = deque([initial_node])
        while q:
            node = q.popleft()
            if node.visited:
                continue

            can_visit = True
            min_time = 0
            for prev in node.prev_nodes:
                if prev.visited is False:
                    can_visit = False
                    q.append(prev)  # re-insert prev non-visited node
                min_time = max(min_time, prev.start_time + prev.duration)

            if can_visit:
                if isinstance(node, SchedCommNode):
                    rank_f, rank_t = node.src_comp.rank, node.dst_comp.rank
                    chan_ranks = (min(rank_f, rank_t), max(rank_f, rank_t))
                    channel = channel_comms[chan_ranks]
                    node.start_time = min_time

                    if len(channel) == 0:
                        channel.append(node)
                    else:
                        overlap = -1
                        for i, comm in enumerate(channel):  # check overlap
                            if (
                                comm.start_time < node.start_time + node.duration
                                and node.start_time < comm.start_time + comm.duration
                            ):
                                overlap = i
                                break

                        # found 'node' and 'comm_over' overlaps
                        if overlap >= 0:
                            comm_over = channel[overlap]

                            for c in channel[overlap:]:
                                c.visited = False
                                for nn in c.next_nodes:
                                    nn.visited = False

                            comm_over.add_next(node)
                            time_orig = calc_longest_path_len([comm_over, node, *q])
                            comm_over.next_nodes.pop()
                            node.prev_nodes.pop()

                            node.add_next(comm_over)
                            time_inter = calc_longest_path_len([node, comm_over, *q])
                            node.next_nodes.pop()
                            comm_over.prev_nodes.pop()

                            if time_orig < time_inter or (
                                time_orig == time_inter
                                and comm_over.start_time <= node.start_time
                            ):  # original order wins
                                comm_over.add_next(node)
                                comm_over.set_start_time()
                                node.set_start_time()
                                channel = channel[: overlap + 1]
                                channel.append(node)
                            else:
                                node.add_next(comm_over)
                                node.set_start_time()
                                comm_over.set_start_time()
                                channel = channel[:overlap]
                                channel.extend((node, comm_over))

                            comm_over.visited = True
                            q.extend(comm_over.next_nodes)
                        else:
                            channel.append(node)

                    channel_comms[chan_ranks] = channel
                else:
                    node.start_time = min_time
                node.visited = True
                q.extend(node.next_nodes)

        # max_time = max(
        #     [max(n.start_time + n.duration for n in nodes) for nodes in comp_nodes]
        # )

        max_time = self.sink_node.start_time

        self.channel_comms = channel_comms
        self.max_time = max_time


class PipeGraphGenerator:
    def __init__(
        self,
        comp_graph: CompGraph,
        device_bench: Optional[DeviceBench],
        option: PipeOption,
    ):
        self.comp_graph = comp_graph
        self.device_bench = device_bench
        self.option = option

        self.comp_adj = self._get_comp_adjacency()
        self._scan_comp_time()

    def _scan_comp_time(self):
        self.comp_time_scan = [0.0]
        self.back_time_scan = [0.0]
        self.optim_time_scan = [0.0]

        fb_ratio = self.comp_graph.fb_ratio
        comp_init = 0
        back_init = 0
        optim_init = 0

        for cnode in self.comp_graph.nodes:
            comp_init += cnode.comp_time
            back_init += cnode.comp_time * fb_ratio
            optim_init += cnode.optim_time

            self.comp_time_scan.append(comp_init)
            self.back_time_scan.append(back_init)
            self.optim_time_scan.append(optim_init)

    def print_sched(self, graph, sched, dist):
        for dev_id, dev_sched in enumerate(sched):
            for node_sched in dev_sched:
                node_id, batch_id, is_forward = node_sched
                node = graph.nodes[node_id]
                start = dist[node_sched]
                time = node.comp_time if is_forward else node.back_time

                print(dev_id, node_id, batch_id, is_forward, start, time)

    def save_graph_png(
        self, graph, sched, batch_cnt, device_cnt, file_name="graph.png"
    ):
        # TODO: set save path
        path = Path(f"{file_name}")

        _, dist = graph.get_value(sched, batch_cnt, return_dist=True)

        fig, ax = plt.subplots()

        ax.set_yticks(range(1, device_cnt + 1))
        ax.set_yticklabels([f"dev {i}" for i in range(1, device_cnt + 1)])
        color = ["tab:blue", "tab:orange", "tab:red", "tab:green"]

        for dev_id, dev_sched in enumerate(sched):
            dev = dev_id + 1
            for node_sched in dev_sched:
                node_id, batch_id, is_forward = node_sched
                node = graph.nodes[node_id]

                start = dist[node_sched]
                time = node.comp_time if is_forward else node.back_time
                ax.broken_barh(
                    [(start, time)],
                    (dev - 0.15, 0.3),
                    linewidth=1,
                    facecolors=color[batch_id % 4],
                    edgecolor="black",
                )
                ax.text(
                    x=start + time / 2,
                    y=dev,
                    s=f"{node_id}_{batch_id}",
                    ha="center",
                    size=7,
                    va="center",
                    color="white",
                )

        save_path = str(path.absolute())
        plt.savefig(save_path)
        print(f"schedule saved to {save_path}")

    def gen_dep_graph(self, sched_params, split_points):
        dep_graph = DependencyGraph(sched_params)
        return dep_graph

    def gen_graph(self, batch_cnt: int, max_mem: List[int], split_points=None):
        # print(f"generating pipe graph: fb ratio {self.comp_graph.fb_ratio}")

        device_cnt = len(max_mem)

        if split_points is not None:
            init_graph = self._draw_graph(split_points, device_cnt, batch_cnt)
            prefetch_fwd = (
                self.option.prefetch_fwd if self.option.prefetch_fwd is not None else 0
            )
            loop_batch = (
                self.option.loop_batch
                if self.option.loop_batch is not None
                else device_cnt
            )

            sched_params = PipeSchedParams(
                device_cnt, batch_cnt, len(split_points) + 1, loop_batch, prefetch_fwd
            )

            curr_time = init_graph.get_value(sched_params.rank_works, batch_cnt)
            peak_mem = init_graph.get_peak_memory(sched_params.rank_works)
            print(
                f"split_points: {split_points}, expect time: {curr_time}, schedule: {sched_params}"
            )

            mem_info = [
                f"dev {i} - idle mem: {fmem(init_graph.device_mem[i])} peak: {fmem(m)}"
                for (i, m) in enumerate(peak_mem)
            ]
            print("\n".join(mem_info))

            # self.print_sched(graph, pipe_sched.sched, dist)
            dep_graph = self.gen_dep_graph(sched_params, split_points)

            return init_graph, sched_params, dep_graph

        if self.option.loop_cnt:
            fused_node_cnts = [self.option.loop_cnt * device_cnt]
        else:
            fused_node_cnts = [i * device_cnt for i in range(1, 4)]

        min_opt_result: OptimResult = None
        min_sched_param = None
        min_dep_graph = None

        prefetch_fwd = self.option.prefetch_fwd
        loop_batch = self.option.loop_batch
        reserve_mem = [m * RESERVE_MEM_RATIO for m in max_mem]

        for loop_idx, fused_node_cnt in enumerate(fused_node_cnts):
            if fused_node_cnt + 4 > len(self.comp_graph.nodes):
                print(
                    f"too small number of nodes: {len(self.comp_graph.nodes)} comp nodes for {fused_node_cnt} fused nodes"
                )
                break

            if self.option.scheduler == "dfs":
                scheds = [
                    PipeSchedParams(
                        device_cnt, batch_cnt, fused_node_cnt, device_cnt, 0
                    )
                ]
            else:
                sched_set = PipeSchedSet(device_cnt, batch_cnt, fused_node_cnt)
                scheds = [pipe_sched for pipe_sched in sched_set.scheds]

            last_time = 0
            last_point = None
            last_sets = None
            for sched_params in scheds:
                if (
                    prefetch_fwd is not None
                    and sched_params.prefetch_fwd != prefetch_fwd
                ):
                    continue
                if loop_batch is not None and sched_params.loop_batch != loop_batch:
                    continue

                if last_point is None:
                    init_graph, split_points = self._get_initial_split_points(
                        sched_params, reserve_mem
                    )
                else:
                    init_graph = self._draw_graph(last_point, device_cnt, batch_cnt)
                    split_points = last_point

                if init_graph is None:
                    break

                if len(split_points) == 0:
                    continue

                print(f"initial split points for {fused_node_cnt} nodes:", split_points)

                if (
                    loop_idx != 0 and self.option.loop_cnt is None
                ):  # loop is automatically created
                    mean_comp_time = sum(n.comp_time for n in init_graph.nodes) / len(
                        init_graph.nodes
                    )
                    if mean_comp_time <= FWD_AVG_MIN:
                        print(
                            f"average computation time of the nodes is too small ({mean_comp_time}ms <= {FWD_AVG_MIN}ms)"
                        )
                        print(f"skip looped schedule for {fused_node_cnt} nodes")
                        continue

                print(f"test {sched_params}")
                opt_start = time.perf_counter()
                opt_dep_graph, opt_result = self._optimize_split(
                    init_graph, split_points, sched_params, max_mem
                )

                validity = opt_result.validity
                new_time = opt_result.time

                last_time = min_opt_result.time if min_opt_result is not None else 0

                if validity and (
                    min_opt_result is None or new_time < min_opt_result.time
                ):
                    min_opt_result = opt_result
                    min_sched_param = sched_params
                    min_dep_graph = opt_dep_graph
                    last_point = min_opt_result.split_points

                    last_sets = (
                        min_opt_result,
                        min_sched_param,
                        min_dep_graph,
                        last_point,
                    )
                elif not validity:
                    print(f"Memory capacity exceeded: {sched_params}")
                    no_graph = self._draw_graph(split_points, device_cnt, batch_cnt)
                    pmem = no_graph.get_peak_memory(sched_params.rank_works)
                    mem_info = [
                        f"dev {i} - idle mem: {fmem(no_graph.device_mem[i])} peak: {fmem(m)}"
                        for (i, m) in enumerate(pmem)
                    ]
                    print("\n".join(mem_info))

                opt_end = time.perf_counter()
                print(f"elapsed time: {opt_end - opt_start:.2f}s, result: {new_time}")

                # if new schedule does not improve more than 1% then step.
                criterion = new_time < last_time and (last_time - new_time) / last_time < 0.01
                if min_opt_result is not None and criterion:
                    if last_sets is not None:
                        min_opt_result, min_sched_param, min_dep_graph, last_point = (
                            last_sets
                        )
                    print(f"End of improvement for loop {loop_idx + 1}")
                    break

        if min_sched_param is not None:
            sched = min_sched_param
            split_opt = min_opt_result.split_points
            min_graph = self._draw_graph(split_opt, device_cnt, batch_cnt)
            if min_dep_graph is None:
                curr_time = min_graph.get_value(sched.rank_works, batch_cnt)
                peak_mem = min_graph.get_peak_memory(sched.rank_works)
            else:
                curr_time = min_dep_graph.max_time
                peak_mem = min_graph.get_peak_memory(
                    sched.rank_works
                )  # TODO: use min_dep_graph

            print(
                f"split points: {split_opt}, expect time: {min_opt_result.time}, schedule: {sched}"
            )
            if peak_mem is not None:
                mem_info = [
                    f"dev {i} - idle mem: {fmem(min_graph.device_mem[i])} peak: {fmem(m)}"
                    for (i, m) in enumerate(peak_mem)
                ]
                print("\n".join(mem_info))

            # self.print_sched(min_graph, sched.sched, dist)
        else:
            raise Exception(
                "There is no valid split points. Maybe there is not enough nodes or weights are too large"
            )

        return min_graph, min_sched_param, min_dep_graph

    def _get_comp_adjacency(self):
        # return adj list of computation graph
        # first node: source (in_nodes) / last node: sink (out_nodes)
        cgraph = self.comp_graph
        adj = [[] for _ in range(len(cgraph.nodes) + 2)]

        for n in cgraph.in_nodes:
            adj[0].append([cgraph.nodes.index(n) + 1, 0])

        for idx, n in enumerate(cgraph.nodes):
            adj_idx = adj[idx + 1]
            for e in n.out_edges:
                adj_idx.append([cgraph.nodes.index(e.end_node) + 1, e.weight])

        for n, _ in cgraph.out_nodes:
            adj[cgraph.nodes.index(n) + 1].append([len(cgraph.nodes) + 1, 0])

        # summation weights on same edge
        for idx, adj_list in enumerate(adj):
            edge_dict = defaultdict(int)
            for next_idx, weight in adj_list:
                edge_dict[next_idx] += weight

            adj[idx] = [[k, v] for k, v in edge_dict.items()]

        rev_adj = [[] for _ in range(len(cgraph.nodes) + 2)]
        for idx, adj_list in enumerate(adj):
            for next_idx, weight in adj_list:
                rev_adj[next_idx].append([idx, weight])

        return adj, rev_adj

    def _valuate_points(self, points, device_cnt, sched, *, return_dist=False):
        device_bench = self.device_bench
        batch_cnt = self.option.batch_cnt
        graph = dict()

        sink = (-1, -1, False)
        graph[sink] = []

        cgraph_len = len(self.comp_graph.nodes)
        all_points = [0, *points, cgraph_len]
        point_pidxs = [-1 for _ in range(cgraph_len + 2)]  # cgraph + source + sink
        point_ranks = [-1 for _ in range(cgraph_len + 2)]

        comp_times = []
        back_times = []
        optim_times = [0 for _ in range(device_cnt)]
        pgraph_len = len(points) + 1
        out_weights = [[] for _ in range(pgraph_len)]
        in_weights = [[] for _ in range(pgraph_len)]

        pnode_ranks = [i % device_cnt for i in range(pgraph_len)]
        pnode_ranks[-1] = -1

        pidx = 0
        curr_rank = 0

        for idx, (start, end) in enumerate(zip(all_points, all_points[1:])):
            dev_idx = idx % device_cnt
            comp_times.append(self.comp_time_scan[end] - self.comp_time_scan[start])
            back_times.append(self.back_time_scan[end] - self.back_time_scan[start])
            optim_times[dev_idx] += (
                self.optim_time_scan[end] - self.optim_time_scan[start]
            )

            for idx in range(start + 1, end + 1):
                point_pidxs[idx] = pidx
                point_ranks[idx] = curr_rank
            pidx += 1
            curr_rank = pidx % device_cnt

        out_adj = self.comp_adj[0]
        send_dict = defaultdict(int)

        for idx, out_list in enumerate(out_adj[1:-1]):
            idx = idx + 1
            for out_idx, weight in out_list:
                if (
                    point_ranks[idx] == point_ranks[out_idx]
                    or point_ranks[out_idx] == -1
                ):
                    continue

                in_pidx = point_pidxs[idx]
                out_pidx = point_pidxs[out_idx]
                send_dict[(in_pidx, out_pidx)] += weight

        for (in_pidx, out_pidx), weight in send_dict.items():
            out_weights[in_pidx].append((out_pidx, weight))
            in_weights[out_pidx].append((in_pidx, weight))

        # dependency between nodes
        for pidx in range(pgraph_len):
            for bidx in range(batch_cnt):
                # Forward
                f_vert = (pidx, bidx, True)
                b_vert = (pidx, bidx, False)

                f_edges = []
                b_edges = []

                for out_idx, weight in out_weights[pidx]:
                    send_time = (
                        device_bench.send_time_rank(
                            pnode_ranks[pidx], pnode_ranks[out_idx], weight
                        )
                        if device_bench is not None
                        else 0
                    )
                    time = comp_times[pidx] + send_time
                    next_node = (out_idx, bidx, True)
                    f_edges.append((next_node, time))

                for in_idx, weight in in_weights[pidx]:
                    send_time = (
                        device_bench.send_time_rank(
                            pnode_ranks[in_idx], pnode_ranks[pidx], weight
                        )
                        if device_bench is not None
                        else 0
                    )
                    time = back_times[pidx] + send_time
                    next_node = (in_idx, bidx, False)
                    b_edges.append((next_node, time))

                graph[f_vert] = f_edges
                graph[b_vert] = b_edges

        # dependency between schedules within a single device
        for dev_idx, rank_sched in enumerate(sched):
            for i in range(len(rank_sched) - 1):
                prev_v = rank_sched[i]
                next_v = rank_sched[i + 1]

                if prev_v[2]:
                    comp_time = comp_times[prev_v[0]]
                    graph[prev_v].append((next_v, comp_time))
                else:
                    back_time = back_times[prev_v[0]]
                    graph[prev_v].append((next_v, back_time))

            last_v = rank_sched[-1]
            back_time = back_times[last_v[0]]
            optim_time = optim_times[dev_idx]

            graph[last_v].append((sink, back_time + optim_time))

        # for n, k in graph.items():
        #     print(n, k)

        # Step 1: Topological sorting
        visited = {node: False for node in graph}
        stack = []

        def topological_sort(node, visited, stack):
            visited[node] = True
            if node in graph:
                for neighbor, _ in graph[node]:
                    if not visited[neighbor]:
                        topological_sort(neighbor, visited, stack)
            stack.insert(0, node)

        for node in graph:
            if not visited[node]:
                topological_sort(node, visited, stack)

        start_node = (0, 0, True)

        # Step 2: Initialize distances
        distances = {node: float("-inf") for node in graph}
        distances[start_node] = 0

        # Step 3: Traverse the graph
        predecessors = {}
        for node in stack:
            if node in graph:
                for neighbor, weight in graph[node]:
                    new_distance = distances[node] + weight
                    if new_distance > distances[neighbor]:
                        distances[neighbor] = new_distance
                        predecessors[neighbor] = node

        # Step 4: Find the longest path
        longest_distance = distances[sink]

        if return_dist:
            return longest_distance, distances
        else:
            return longest_distance

    def _optimize_split(
        self, graph: PipeGraph, split_points, sched, batch_cnt, max_mem: List[int]
    ):
        # iteratively upgrade split points by random walk
        max_mem = []
        dev_cnt = len(max_mem)

        reserve_mem = [m * RESERVE_MEM_RATIO for m in max_mem]

        if self.option.partitioner == "same_mem":
            peak_mem = graph.get_peak_memory(sched)
            validity = all([m < mx for (m, mx) in zip(peak_mem, reserve_mem)])
            return (
                graph,
                split_points,
                graph.get_value(sched, batch_cnt),
                peak_mem,
                validity,
            )
        elif self.option.partitioner == "same_time":
            cgraph = graph.comp_graph
            fwd_times = self.comp_time_scan
            node_cnt = len(split_points) + 1
            sum_time = fwd_times[-1]
            mean_time = sum_time / node_cnt
            new_points = []
            curr_point = 0

            for idx in range(len(cgraph.nodes)):
                time_diff = fwd_times[idx] - fwd_times[curr_point]
                if time_diff >= mean_time:
                    curr_point = idx
                    new_points.append(curr_point)
                    node_cnt -= 1
                    sum_time -= time_diff
                    mean_time = sum_time / node_cnt
                if len(new_points) >= len(split_points):
                    break

            len_diff = len(split_points) - len(new_points)
            if len_diff > 0:
                node_len = len(graph.nodes)
                new_points.extend(list(range(node_len - len_diff, node_len)))

            graph = self._draw_graph(new_points, dev_cnt, batch_cnt)
            peak_mem = graph.get_peak_memory(sched)
            validity = all([m < mx for (m, mx) in zip(peak_mem, reserve_mem)])

            return (
                graph,
                new_points,
                graph.get_value(sched, batch_cnt),
                peak_mem,
                validity,
            )

        curr_time = graph.get_value(sched, batch_cnt)
        peak_mem = graph.get_peak_memory(sched)
        validity = all([m < mx for (m, mx) in zip(peak_mem, reserve_mem)])

        # print(
        #     f"initial time for {len(split_points) + 1} nodes:",
        #     curr_time,
        #     [f"{fmem(m)}/{fmem(mx)}" for (m, mx) in zip(peak_mem, max_mem)],
        # )

        opt_graph = graph
        node_len = len(self.comp_graph.nodes)
        split_len = len(split_points)

        # TODO: should we make this 1/8 of the avg len of slices?
        min_gap = max(1, node_len // ((split_len + 1) * 8))

        if node_len <= split_len + 1:
            return opt_graph

        for try_iter in range(RAND_WALK_ITER):
            new_split_points = split_points.copy()

            # randomly select a split point
            found_next = False
            for trial in range(split_len * 3):
                idx = random.randint(0, split_len - 1)
                point = split_points[idx]
                min_point = split_points[idx - 1] if idx > 0 else 1
                max_point = (
                    split_points[idx + 1] if idx < split_len - 1 else node_len - 2
                )

                if min_point + 1 >= point and max_point - 1 <= point:
                    continue

                # move max 3 nodes per step
                min_point = min(max(min_point + min_gap, point - 10), point)
                max_point = max(min(max_point - min_gap, point + 10), point)

                if min_point >= max_point:
                    continue

                next_point = random.randint(min_point, max_point - 1)
                if next_point >= point:
                    next_point += 1
                found_next = True
                break

            if not found_next:
                print(f"next point not found: {split_points}")
                break

            new_split_points[idx] = next_point

            new_graph = self._draw_graph(new_split_points, dev_cnt, batch_cnt)
            new_time = new_graph.get_value(sched, batch_cnt)
            new_peak_mem = new_graph.get_peak_memory(sched)
            new_validity = all([m < mx for (m, mx) in zip(new_peak_mem, reserve_mem)])

            # if not new_validity:
            #     print(
            #         f"broken at {try_iter}:",
            #         new_time,
            #         [f"{fmem(m)}/{fmem(mx)}" for (m, mx) in zip(new_peak_mem, max_mem)],
            #     )

            if not (validity and (not new_validity)) and new_time < curr_time:
                curr_time = new_time
                peak_mem = new_peak_mem
                split_points = new_split_points
                opt_graph = new_graph
                validity = new_validity or False

                # print(
                #     f"new time {try_iter} / {'valid' if new_validity else 'invalid'}:",
                #     curr_time,
                #     [f"{fmem(m)}/{fmem(mx)}" for (m, mx) in zip(peak_mem, max_mem)],
                # )
                # print(new_path)
                print("new:", new_split_points, node_len, curr_time)

        return opt_graph, split_points, curr_time, peak_mem, validity

    def _dfs_schedule(self, graph: PipeGraph, batch_cnt: int, device_cnt: int):
        nodes = graph.nodes
        batch_cnt = batch_cnt
        device_cnt = device_cnt

        sched = [[] for _ in range(device_cnt)]

        rank_fwd = [[] for _ in range(device_cnt)]
        rank_bwd = [[] for _ in range(device_cnt)]

        for batch_start in range(0, batch_cnt, device_cnt):
            batch_end = min(batch_start + device_cnt, batch_cnt)
            for node in nodes:
                fwd = rank_fwd[node.rank]
                for batch_id in range(batch_start, batch_end):
                    fwd.append((node.idx, batch_id))

            for node in reversed(nodes):
                bwd = rank_bwd[node.rank]
                for batch_id in range(batch_start, batch_end):
                    bwd.append((node.idx, batch_id))

        node_len = len(nodes)
        for rank in range(device_cnt):
            rank_sched = sched[rank]
            fwd = rank_fwd[rank]
            bwd = rank_bwd[rank]

            curr_bwd = 0
            for fwd_idx, fwd_batch in fwd:
                rank_sched.append((fwd_idx, fwd_batch, True))
                if fwd_batch >= device_cnt or (fwd_batch + 1 >= node_len - fwd_idx):
                    bwd_idx, bwd_batch = bwd[curr_bwd]
                    rank_sched.append((bwd_idx, bwd_batch, False))
                    curr_bwd += 1

            for bwd_idx, bwd_batch in bwd[curr_bwd:]:
                rank_sched.append((bwd_idx, bwd_batch, False))

        return sched

    def _get_initial_split_points(self, sched_params: PipeSchedParams, reserve_mem):
        slice_cnt = sched_params.node_cnt
        device_cnt = sched_params.device_cnt
        batch_cnt = sched_params.batch_cnt
        sched = sched_params.rank_works

        if slice_cnt <= 0:
            return []

        split_points = []

        mem_sum = 0
        for cnode in self.comp_graph.nodes:
            mem_diff = cnode.mem_diff() * device_cnt
            mem_sum += mem_diff

        mem_avg = mem_sum // slice_cnt
        curr_mem = 0

        # Simple bucket based slicing
        has_weight = False
        for idx, cnode in enumerate(self.comp_graph.nodes):
            mem_diff = cnode.mem_diff() * device_cnt
            has_weight = has_weight or cnode.weight > 0
            if mem_avg < curr_mem + mem_diff and has_weight:
                split_points.append(idx + 1)
                curr_mem = 0
                has_weight = False

            curr_mem += mem_diff

        len_nodes = len(self.comp_graph.nodes)
        points = split_points[: slice_cnt - 1]

        # if there is no sufficient number of points due to the lack of weight,
        # force split all
        if len(points) < slice_cnt - 1:
            print(f"No valid split points: not enough number of nodes ({len_nodes})")
            avg_dist = max(len_nodes // slice_cnt, 1)
            points = [i * avg_dist for i in range(1, slice_cnt)]
            return None, points

        # do not split output only node
        last_point = len_nodes - 1
        last_idx = len(points) - 1
        while points[last_idx] == last_point and last_idx >= 0:
            points[last_idx] -= 1
            last_point -= 1
            last_idx -= 1

        if last_idx < 0:
            # just split by node counts
            points = [(i * len_nodes) // slice_cnt for i in range(1, slice_cnt)]

        orig_split_points = points.copy()
        graph = self._draw_graph(points, device_cnt, batch_cnt)
        temp_peak_mem = graph.get_peak_memory(sched)
        validity = all([m < mx for (m, mx) in zip(temp_peak_mem, reserve_mem)])
        reduce_loop_cnt = 0

        while validity is False and reduce_loop_cnt < 10:
            reduce_loop_cnt += 1

            diff = points[0] * 0.08
            round_diff = max(round(diff), 1)
            points[0] = max(1, points[0] - round_diff)

            for idx in range(1, len(points)):
                pos = 0.9 * (len(points) - idx) / len(points) + 0.1
                round_diff = round(diff * pos)
                points[idx] = max(points[idx - 1] + 1, points[idx] - round_diff)

            graph = self._draw_graph(points, device_cnt, batch_cnt)
            temp_peak_mem = graph.get_peak_memory(sched)
            validity = all([m < mx for (m, mx) in zip(temp_peak_mem, reserve_mem)])

        if not validity:
            graph = self._draw_graph(orig_split_points, device_cnt, batch_cnt)
            return graph, orig_split_points

        return graph, points

    def _draw_graph(self, split_points, device_cnt: int, batch_cnt: int):
        pipe_graph = PipeGraph(self.comp_graph)

        points = split_points.copy() + [len(self.comp_graph.nodes)]
        pidx = 0
        curr_point = points[pidx]

        for idx in range(len(points)):
            pipe_graph.nodes.append(PipeNode(idx))

        for idx, cnode in enumerate(self.comp_graph.nodes):
            if idx >= curr_point:
                pidx += 1
                curr_point = points[pidx]
            curr_node = pipe_graph.nodes[pidx]

            if len(cnode.placeholders) > 0:
                pipe_graph.input_nodes.append(curr_node)
            elif len(cnode.outputs) > 0:
                pipe_graph.output_nodes.append(curr_node)

            curr_node.nodes.append(cnode)
            pipe_graph.back_dict[cnode] = curr_node

        for idx, pnode in enumerate(pipe_graph.nodes):
            pnode.idx = idx

        # edge generation
        edge_dict: Dict[Tuple[PipeNode, PipeNode], PipeEdge] = dict()
        back_dict = pipe_graph.back_dict
        for pnode in pipe_graph.nodes:
            for cnode in pnode.nodes:
                for edge in cnode.out_edges:
                    out_pnode = back_dict[edge.end_node]
                    if out_pnode != pnode:
                        if (pnode, out_pnode) in edge_dict.keys():
                            pedge = edge_dict[pnode, out_pnode]
                        else:
                            pedge = PipeEdge(pnode, out_pnode)
                            pipe_graph.edges.append(pedge)
                            edge_dict[pnode, out_pnode] = pedge
                            pedge.assign()
                        pedge.comp_edges.append(edge)

        for i, pnode in enumerate(pipe_graph.nodes):
            pnode.rank = i % device_cnt

        for pnode in pipe_graph.nodes:
            pnode.compute_mem(fb_ratio=self.comp_graph.fb_ratio)

        # TODO: shared param
        params = self.comp_graph.param_set.main_params
        param_dict: Dict[int, List[PipeNode]] = defaultdict(list)
        device_params: Dict[int, Set[int]] = defaultdict(set)
        device_mem = [0 for _ in range(device_cnt)]

        for pnode in pipe_graph.nodes:
            for param in pnode.params:
                param_dict[param].append(pnode)
                device_params[pnode.rank].add(param)

        for dev_id, param_set in device_params.items():
            for param in param_set:
                # TODO: non-adam
                device_mem[dev_id] += (
                    params[param][0].bytes * 4
                )  # weight + grad + optimizer (2)

        for i, pnode in enumerate(pipe_graph.nodes):
            dev = pnode.rank
            for in_edge in pnode.in_edges:
                device_mem[dev] += (
                    in_edge.weight * batch_cnt * 3
                )  # input + grad + temp_input for in-place op
            for out_edge in pnode.out_edges:
                device_mem[dev] += out_edge.weight * 2 * batch_cnt  # output + grad

            # for in_edge in pnode.in_edges:
            #     device_mem[dev] += (
            #         in_edge.weight * batch_cnt
            #     )  # input + grad + temp_input for in-place op
            # for out_edge in pnode.out_edges:
            #     device_mem[dev] += out_edge.weight * batch_cnt  # output + grad
            for in_node in pnode.placeholders:
                device_mem[dev] += in_node.meta["_weight"] * batch_cnt  # input only
            # for out_node in pnode.outputs:
            #     device_mem[dev] += out_node.meta["_weight"] * 2  # output + grad

        pipe_graph.device_mem = device_mem

        # TODO: proper topo sort
        # pipe_graph.topo_sort()

        if self.device_bench is not None:
            for pedge in pipe_graph.edges:
                pedge.send_time = self.device_bench.send_time_rank(
                    pedge.start_node.rank, pedge.end_node.rank, pedge.weight
                )

        return pipe_graph


class BeamSearchGenerator(PipeGraphGenerator):
    def __init__(
        self,
        comp_graph: CompGraph,
        device_bench: Optional[DeviceBench],
        option: PipeOption,
    ):
        super().__init__(comp_graph, device_bench, option)

    def _optimize_split(
        self,
        init_graph: PipeGraph,
        init_split_points,
        sched_params: PipeSchedParams,
        max_mem: List[int],
    ):
        # iteratively upgrade split points by random walk
        reserve_mem = [m * RESERVE_MEM_RATIO for m in max_mem]

        device_len = sched_params.device_cnt
        batch_cnt = sched_params.batch_cnt
        sched = sched_params.rank_works

        curr_time = init_graph.get_value(sched, batch_cnt)
        peak_mem = init_graph.get_peak_memory(sched)
        validity = all([m < mx for (m, mx) in zip(peak_mem, reserve_mem)])

        opt_graph = init_graph
        node_len = len(self.comp_graph.nodes)
        split_len = len(init_split_points)

        # TODO: should we make this 1/8 of the avg len of slices?
        min_gap = max(1, node_len // ((split_len + 1) * 8))

        beam = [[curr_time, init_split_points, peak_mem, validity, False]]

        if node_len <= split_len + 1:
            return None, None

        def local_opt(split_points, curr_time):
            split_ranges = []
            new_beam = []
            min_2_time = curr_time

            for idx, point in enumerate(split_points):
                point = split_points[idx]
                min_point = split_points[idx - 1] if idx > 0 else 1
                max_point = (
                    split_points[idx + 1] if idx < split_len - 1 else node_len - 2
                )

                if min_point + 1 >= point and max_point - 1 <= point:
                    split_ranges.append(None)
                    continue

                # move max 3 nodes per step
                min_point = min(max(min_point + min_gap, point - 15), point)
                max_point = max(min(max_point - min_gap, point + 15), point)

                if min_point >= max_point:
                    split_ranges.append(None)
                else:
                    split_ranges.append((min_point, max_point, point))

            def append_beam(new_points):
                nonlocal min_2_time, new_beam

                new_time = self._valuate_points(new_points, device_len, sched)

                if len(new_beam) == 2 and new_time >= min_2_time:
                    return

                new_graph = self._draw_graph(new_points, device_len, batch_cnt)
                new_peak_mem = new_graph.get_peak_memory(sched)
                new_validity = all(
                    [m < mx for (m, mx) in zip(new_peak_mem, reserve_mem)]
                )
                if not new_validity:
                    return

                new_item = [
                    new_time,
                    new_points,
                    new_peak_mem,  # peak mem
                    new_validity,  # validity
                    False,
                ]

                if len(new_beam) == 0:
                    new_beam.append(new_item)
                elif len(new_beam) == 1:
                    if new_beam[0][0] < new_item[0]:
                        new_beam.append(new_item)
                    else:
                        new_beam.insert(0, new_item)
                    min_2_time = new_beam[1][0]
                elif new_time < min_2_time:
                    if new_time < new_beam[0][0]:
                        new_beam[1] = new_beam[0]
                        new_beam[0] = new_item
                    else:
                        new_beam[1] = new_item

                    min_2_time = new_beam[1][0]

            # move just once
            for idx, rng in enumerate(split_ranges):
                if rng is None:
                    continue

                min_point, max_point, point = rng

                for new_point in range(min_point, max_point):
                    new_points = split_points.copy()

                    if new_point >= point:
                        new_point += 1
                    new_points[idx] = new_point

                    append_beam(new_points)

            grids = [1, *split_points, node_len - 2]
            diffs = []
            for i in range(len(grids) - 1):
                diffs.append(grids[i + 1] - grids[i])

            # move all by ratio
            for idx, rng in enumerate(split_ranges):
                if rng is None:
                    continue

                min_point, max_point, point = rng
                left_dist = point
                right_dist = (node_len - 1) - point if point < node_len - 1 else 1

                for new_point in range(min_point, max_point):
                    new_points = []
                    new_diffs = []

                    if new_point >= point:
                        new_point += 1

                    left_ratio = new_point / left_dist
                    right_ratio = ((node_len - 1) - new_point) / right_dist

                    for i, diff in enumerate(diffs):
                        if i <= idx:
                            new_diffs.append(diff * left_ratio)
                        else:
                            new_diffs.append(diff * right_ratio)

                    curr_pos = 1
                    last_pos = 1
                    found = True
                    for diff in new_diffs[:-1]:
                        curr_pos += diff
                        pos = round(curr_pos)
                        if pos == last_pos:
                            pos += 1
                        if pos >= node_len - 1:
                            pos -= node_len - 2
                        if pos <= last_pos:
                            found = False
                            break

                        last_pos = pos
                        new_points.append(last_pos)

                    if found:
                        append_beam(new_points)

            return new_beam

        for try_iter in range(BEAM_ITER):
            new_beam = beam.copy()

            for beam_item in beam:
                if beam_item is None:
                    continue
                (
                    curr_time,
                    split_points,
                    peak_mem,
                    validity,
                    is_optimal,
                ) = beam_item

                if is_optimal:
                    continue

                next_beam = local_opt(split_points, curr_time)

                found = False
                for next_item in next_beam:
                    if len(beam) < 2 or next_item[0] < beam[1][0]:
                        new_beam.append(next_item)
                        found = True
                if not found:
                    beam_item[4] = True

            new_beam = sorted(new_beam, key=lambda x: x[0])
            beam = new_beam[:2]

            beam_0 = f"{beam[0][0]} / {beam[0][1]} {beam[0][4]}" if beam[0] else "None"
            if len(beam) >= 2:
                beam_1 = (
                    f"{beam[1][0]} / {beam[1][1]} {beam[1][4]}" if beam[1] else "None"
                )
                print(f"beam {try_iter} - {beam_0} - {beam_1}")
                if beam[0] and beam[0][4] and beam[1] and beam[1][4]:
                    break
            else:
                print(f"beam {try_iter} - {beam_0}")
                if beam[0] and beam[0][4]:
                    break

        split_points, curr_time, peak_mem, validity = (
            beam[0][1],
            beam[0][0],
            beam[0][2],
            beam[0][3],
        )

        opt_result = OptimResult(split_points, curr_time, peak_mem, validity)

        return None, opt_result


class OptTimer:
    def __init__(self):
        self.start_time = dict()
        self.end_time = dict()
        self.durations = defaultdict(int)
        self.counts = defaultdict(int)

    def start(self, name):
        self.start_time[name] = time.time()

    def end(self, name):
        self.end_time[name] = time.time()
        self.durations[name] += self.end_time[name] - self.start_time[name]
        self.counts[name] += 1

    def print(self):
        print("Opt Timer:")
        for name in self.durations:
            print(
                f"{name}: total {self.durations[name]}, mean {self.durations[name] / self.counts[name]}, count {self.counts[name]}"
            )


class BeamSingleGenerator(BeamSearchGenerator):
    def __init__(
        self,
        comp_graph: CompGraph,
        device_bench: Optional[DeviceBench],
        option: PipeOption,
    ):
        super().__init__(comp_graph, device_bench, option)

    def gen_dep_graph(self, sched_params, split_points):
        dep_graph = super().gen_dep_graph(sched_params, split_points)
        self._refine_dep_graph(dep_graph, split_points)
        return dep_graph

    def _optimize_split(
        self,
        init_graph: PipeGraph,
        init_split_points,
        sched_params: PipeSchedParams,
        max_mem: List[int],
    ):
        # iteratively upgrade split points by random walk
        reserve_mem = [m * RESERVE_MEM_RATIO for m in max_mem]

        device_len = sched_params.device_cnt
        batch_cnt = sched_params.batch_cnt
        sched = sched_params.rank_works

        curr_time = init_graph.get_value(sched, batch_cnt)
        peak_mem = init_graph.get_peak_memory(sched)
        validity = all([m < mx for (m, mx) in zip(peak_mem, reserve_mem)])

        node_len = len(self.comp_graph.nodes)
        split_len = len(init_split_points)

        # TODO: should we make this 1/8 of the avg len of slices?
        min_gap = max(1, node_len // ((split_len + 1) * 8))

        dep_graph = DependencyGraph(sched_params)
        beam = [[curr_time, init_split_points, peak_mem, validity, False]]

        if node_len <= split_len + 1:
            print("No valid split points: not enough number of nodes")
            return None, None, None

        # do not search previously searched points at the last step
        prev_point_set = set()

        def local_opt(split_points, curr_time, prev_point_set):
            point_set = set()
            split_ranges = []
            new_beam = []
            min_2_time = curr_time

            for idx, point in enumerate(split_points):
                point = split_points[idx]
                min_point = split_points[idx - 1] if idx > 0 else 1
                max_point = (
                    split_points[idx + 1] if idx < split_len - 1 else node_len - 2
                )

                if min_point + 1 >= point and max_point - 1 <= point:
                    split_ranges.append(None)
                    continue

                # move max 3 nodes per step
                min_point = min(max(min_point + min_gap, point - 10), point)
                max_point = max(min(max_point - min_gap, point + 10), point)

                if min_point >= max_point:
                    split_ranges.append(None)
                else:
                    split_ranges.append((min_point, max_point, point))

            def append_beam(new_points):
                pt_tuple = tuple(new_points)
                point_set.add(pt_tuple)
                if pt_tuple in prev_point_set:
                    return

                nonlocal min_2_time, new_beam, dep_graph

                # timer.start("gen dep graph")
                # dep_graph = DependencyGraph(sched_params)
                # timer.end("gen dep graph")
                # timer.start("refine dep graph")
                # self._refine_dep_graph(dep_graph, new_points)
                # timer.end("refine dep graph")
                # new_time = dep_graph.max_time

                # timer.start("valuate points")
                new_time = self._valuate_points(new_points, device_len, sched)
                # timer.end("valuate points")

                if len(new_beam) == 2 and new_time >= min_2_time:
                    return

                new_graph = self._draw_graph(new_points, device_len, batch_cnt)
                new_peak_mem = new_graph.get_peak_memory(sched)
                new_validity = all(
                    [m < mx for (m, mx) in zip(new_peak_mem, reserve_mem)]
                )
                if not new_validity:
                    return

                new_item = [
                    new_time,
                    new_points,
                    new_peak_mem,  # peak mem
                    new_validity,  # validity
                    False,
                ]

                if len(new_beam) == 0:
                    new_beam.append(new_item)
                elif len(new_beam) == 1:
                    if new_beam[0][0] < new_item[0]:
                        new_beam.append(new_item)
                    else:
                        new_beam.insert(0, new_item)
                    min_2_time = new_beam[1][0]
                elif new_time < min_2_time:
                    if new_time < new_beam[0][0]:
                        new_beam[1] = new_beam[0]
                        new_beam[0] = new_item
                    else:
                        new_beam[1] = new_item

                    min_2_time = new_beam[1][0]

            # move just once
            for idx, rng in enumerate(split_ranges):
                if rng is None:
                    continue

                min_point, max_point, point = rng

                for new_point in range(min_point, max_point):
                    new_points = split_points.copy()

                    if new_point >= point:
                        new_point += 1
                    new_points[idx] = new_point

                    append_beam(new_points)

            grids = [1, *split_points, node_len - 2]
            diffs = []
            for i in range(len(grids) - 1):
                diffs.append(grids[i + 1] - grids[i])

            # move all by ratio
            for idx, rng in enumerate(split_ranges):
                if rng is None:
                    continue

                min_point, max_point, point = rng
                left_dist = point
                right_dist = (node_len - 1) - point if point < node_len - 1 else 1

                for new_point in range(min_point, max_point):
                    new_points = []
                    new_diffs = []

                    if new_point >= point:
                        new_point += 1

                    left_ratio = new_point / left_dist
                    right_ratio = ((node_len - 1) - new_point) / right_dist

                    for i, diff in enumerate(diffs):
                        if i <= idx:
                            new_diffs.append(diff * left_ratio)
                        else:
                            new_diffs.append(diff * right_ratio)

                    curr_pos = 1
                    last_pos = 1
                    found = True
                    for diff in new_diffs[:-1]:
                        curr_pos += diff
                        pos = round(curr_pos)
                        if pos == last_pos:
                            pos += 1
                        if pos >= node_len - 1:
                            pos -= node_len - 2
                        if pos <= last_pos:
                            found = False
                            break

                        last_pos = pos
                        new_points.append(last_pos)

                    if found:
                        append_beam(new_points)

            return new_beam, point_set

        for try_iter in range(BEAM_ITER):
            new_beam = beam.copy()

            for beam_item in beam:
                if beam_item is None:
                    continue
                (
                    curr_time,
                    split_points,
                    peak_mem,
                    validity,
                    is_optimal,
                ) = beam_item

                if is_optimal:
                    continue

                next_beam, prev_point_set = local_opt(
                    split_points, curr_time, prev_point_set
                )

                found = False
                for next_item in next_beam:
                    if len(beam) < 2 or next_item[0] < beam[1][0]:
                        new_beam.append(next_item)
                        found = True
                if not found:
                    beam_item[4] = True

            new_beam = sorted(new_beam, key=lambda x: x[0])
            beam = new_beam[:2]

            beam_0 = f"{beam[0][0]} / {beam[0][1]} {beam[0][4]}" if beam[0] else "None"
            if len(beam) >= 2:
                beam_1 = (
                    f"{beam[1][0]} / {beam[1][1]} {beam[1][4]}" if beam[1] else "None"
                )
                print(f"beam {try_iter} - {beam_0} - {beam_1}")
                if beam[0] and beam[0][4] and beam[1] and beam[1][4]:
                    break
            else:
                print(f"beam {try_iter} - {beam_0}")
                if beam[0] and beam[0][4]:
                    break

        split_points, curr_time, peak_mem, validity = (
            beam[0][1],
            beam[0][0],
            beam[0][2],
            beam[0][3],
        )

        dep_graph = DependencyGraph(sched_params)
        self._refine_dep_graph(dep_graph, split_points)

        opt_result = OptimResult(split_points, curr_time, peak_mem, validity)

        return dep_graph, opt_result

    def _refine_dep_graph(self, dep_graph: DependencyGraph, points):
        dep_graph.clear_nonstd_comm()

        device_bench = self.device_bench
        batch_cnt = self.option.batch_cnt
        device_cnt = self.option.device_cnt

        cgraph_len = len(self.comp_graph.nodes)
        all_points = [0, *points, cgraph_len]
        point_pidxs = [-1 for _ in range(cgraph_len + 2)]  # cgraph + source + sink
        point_ranks = [-1 for _ in range(cgraph_len + 2)]

        comp_times = []
        back_times = []
        optim_times = [0 for _ in range(device_cnt)]
        pgraph_len = len(points) + 1
        out_weights = [[] for _ in range(pgraph_len)]
        in_weights = [[] for _ in range(pgraph_len)]

        pnode_ranks = [i % device_cnt for i in range(pgraph_len)]
        pnode_ranks[-1] = -1

        pidx = 0
        curr_rank = 0

        for idx, (start, end) in enumerate(zip(all_points, all_points[1:])):
            dev_idx = idx % device_cnt
            comp_times.append(self.comp_time_scan[end] - self.comp_time_scan[start])
            back_times.append(self.back_time_scan[end] - self.back_time_scan[start])
            optim_times[dev_idx] += (
                self.optim_time_scan[end] - self.optim_time_scan[start]
            )

            for idx in range(start + 1, end + 1):
                point_pidxs[idx] = pidx
                point_ranks[idx] = curr_rank
            pidx += 1
            curr_rank = pidx % device_cnt

        out_adj = self.comp_adj[0]
        send_dict = defaultdict(int)

        for idx, out_list in enumerate(out_adj[1:-1]):
            idx = idx + 1
            for out_idx, weight in out_list:
                if (
                    point_ranks[idx] == point_ranks[out_idx]
                    or point_ranks[out_idx] == -1
                ):
                    continue

                in_pidx = point_pidxs[idx]
                out_pidx = point_pidxs[out_idx]
                send_dict[(in_pidx, out_pidx)] += weight

        for (in_pidx, out_pidx), weight in send_dict.items():
            out_weights[in_pidx].append((out_pidx, weight))
            in_weights[out_pidx].append((in_pidx, weight))

        for pidx in range(pgraph_len):
            for bidx in range(batch_cnt):
                fwd_node = dep_graph.comp_dict[pidx, bidx, True]
                bwd_node = dep_graph.comp_dict[pidx, bidx, False]
                fwd_node.duration = comp_times[pidx]
                bwd_node.duration = back_times[pidx]

                for out_idx, weight in out_weights[pidx]:
                    send_time = (
                        device_bench.send_time_rank(
                            pnode_ranks[pidx], pnode_ranks[out_idx], weight
                        )
                        if device_bench is not None
                        else 0
                    )

                    to_node = dep_graph.comp_dict[out_idx, bidx, True]
                    if (fwd_node, to_node) in dep_graph.comm_nodes:
                        comm = dep_graph.comm_nodes[(fwd_node, to_node)]
                        comm.duration = send_time
                    else:
                        dep_graph.add_nonstd_comm(
                            fwd_node.node_id, to_node.node_id, send_time
                        )

                for in_idx, weight in in_weights[pidx]:
                    send_time = (
                        device_bench.send_time_rank(
                            pnode_ranks[in_idx], pnode_ranks[pidx], weight
                        )
                        if device_bench is not None
                        else 0
                    )
                    back_node = dep_graph.comp_dict[in_idx, bidx, False]
                    if (bwd_node, back_node) in dep_graph.comm_nodes:
                        comm = dep_graph.comm_nodes[(bwd_node, back_node)]
                        comm.duration = send_time
                    else:
                        dep_graph.add_nonstd_comm(
                            bwd_node.node_id, back_node.node_id, send_time
                        )

        for rank, rank_nodes in enumerate(dep_graph.rank_nodes):
            optim_time = optim_times[rank]
            rank_nodes[-1].duration = optim_time

        dep_graph.calc_schedule()

        return dep_graph


def get_pipe_class(partitioner):
    if partitioner == "beam_2":
        return BeamSearchGenerator
    elif partitioner == "beam_single":
        return BeamSingleGenerator
    else:
        return PipeGraphGenerator
