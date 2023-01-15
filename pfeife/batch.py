import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map


def send_value(value, device):
    def tensor_map(v):
        if torch.is_tensor(v):
            return v.to(device)
        else:
            return v

    return tree_map(tensor_map, value)


def shard_dict_of_args(args_dict, batch_spilt):
    # Stage 1+2: flatten and shard/replicate
    # args_sharded_replicated : [num args, num flat values, num chunks]
    args_sharded_replicated = {}
    arg_specs = []

    for arg_key, arg in args_dict.items():
        flat, spec = tree_flatten(arg)
        arg_specs.append(spec)

        sharded_arg_flat = []

        for v in zip(flat):
            if torch.is_tensor(v):
                # TODO: do not force split first dimension
                chunk_tensors = torch.tensor_split(v, batch_spilt, 0)
                sharded_arg_flat.append(chunk_tensors)
            else:
                sharded_arg_flat.append([v] * batch_spilt)

        args_sharded_replicated[arg_key] = sharded_arg_flat

    # chunks_flat : [num chunks, num args, num flat values]
    chunks_flat = []
    for chunk_idx in range(batch_spilt):
        chunk_args = {}
        for key, arg in args_sharded_replicated.items():
            arg_single_chunk = []
            for v_flat in arg:
                arg_single_chunk.append(v_flat[chunk_idx])
            chunk_args[key] = arg_single_chunk
        chunks_flat.append(chunk_args)

    # args_split : [num chunks, num args]
    args_split = []

    for chunk in chunks_flat:
        per_chunk_args = {}
        assert len(arg_specs) == len(chunk)
        for (key, arg), arg_spec in zip(chunk.items(), arg_specs):
            per_chunk_args[key] = tree_unflatten(arg, arg_spec)
        args_split.append(per_chunk_args)

    return args_split


def split_args_kwargs(args, kwargs, batch_split):
    # Given `args` and `kwargs`, we want to yield a set of `chunks` args and kwargs such that
    # the constituent Tensor values have been sharded/replicated according to the `args_chunk_spec`
    # and `kwargs_chunk_spec` specifications. The steps are as follows:
    #
    # 1. Use pytree.tree_flatten to flatten each arg and its spec into nto a 1d array of values.
    #    To use a running example: suppose our inputs look like
    #
    #       args = ([A, [B, C]], D) args_spec = ([None, [None, TensorChunkSpec]], None)
    #       (kwargs not shown but it's a similar process)
    #
    #    Then for this step we would end up with
    #
    #       args = ([A, B, C], D) args_spec = ([None, None, TensorChunkSpec], None)
    #
    # 2. Shard or replicate the arguments subject to the policy in the spec. Suppose chunks = 2
    #
    #       args = ([[A, A], [B, B], [C_1, C_2]], [D, D])
    #
    # 3. Rotate the nesting order such that chunks are the outer dimension
    #
    #       args_chunks = [
    #           ([A, B, C_1], D),
    #           ([A, B, C_2], D),
    #       ]
    #
    # 4. Unflatten each chunk according to the spec
    #
    #       args_chunks = [
    #           ([A, [B, C_1]], D),
    #           ([A, [B, C_2]], D),
    #       ]

    # TODO: _debug_mask_minibatches

    args_split_dict = shard_dict_of_args(dict(enumerate(args)), batch_split)
    kwargs_split = shard_dict_of_args(kwargs, batch_split)

    args_split = []
    for chunk_args in args_split_dict:
        args_split.append(tuple(chunk_args[i] for i in range(len(chunk_args))))

    return args_split, kwargs_split
