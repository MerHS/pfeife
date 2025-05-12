"""
Copied from PiPPy: 
Copyright (c) 2022 Meta Platforms, Inc. and its affiliates.  
All rights reserved.  
"""

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map


def send_value(value, device):
    def tensor_map(v):
        if torch.is_tensor(v):
            return v.to(device)
        else:
            return v

    return tree_map(tensor_map, value)


def _is_leaf(v):
    if isinstance(v, tuple) or isinstance(v, dict):
        return False
    elif isinstance(v, list):
        return not any(torch.is_tensor(x) for x in v)
    else:
        return True


def shard_dict_of_args(args_dict, batch_spilt):
    # Stage 1+2: flatten and shard/replicate
    # args_sharded_replicated : [num args, num flat values, num chunks]
    args_sharded_replicated = {}
    arg_specs = []

    for arg_key, arg in args_dict.items():
        flat, spec = tree_flatten(arg, is_leaf=_is_leaf)
        arg_specs.append(spec)

        sharded_arg_flat = []

        for v in flat:
            if torch.is_tensor(v):
                # TODO: do not force split first dimension
                chunk_tensors = torch.tensor_split(v, batch_spilt, 0)
                sharded_arg_flat.append(chunk_tensors)
            elif isinstance(v, list):
                # split v into batch_split number of chunks
                split_size = (len(v) + batch_spilt - 1) // batch_spilt
                chunks = [
                    v[i * split_size : (i + 1) * split_size] for i in range(batch_spilt)
                ]
                sharded_arg_flat.append(chunks)
            else:
                sharded_arg_flat.append([v] * batch_spilt)
                # sharded_arg_flat.append(v)

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

    if args is None:
        args_split = [(None,) for _ in range(batch_split)]
    else:
        args_split_dict = shard_dict_of_args(dict(enumerate(args)), batch_split)
        args_split = []
        for chunk_args in args_split_dict:
            args_split.append(tuple(chunk_args[i] for i in range(len(chunk_args))))

    if kwargs is None:
        kwargs_split = [dict() for _ in range(batch_split)]
    else:
        kwargs_split = shard_dict_of_args(kwargs, batch_split)

    return args_split, kwargs_split
