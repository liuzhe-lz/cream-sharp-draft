import json
from pathlib import Path
import struct

import torch

def export(model, path='export'):
    Path(path, 'weights').mkdir(parents=True, exist_ok=True)

    arch = _get_model_arch(model, 'root')
    Path(path, 'architecture.json').write_text(json.dumps(arch, indent=4))

    meta = []

    for name, tensor in model.named_parameters(recurse=True):
        tensor_meta = {
            'name': name,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype)
        }
        meta.append(tensor_meta)

        tensor_list = tensor.view(-1).tolist()
        fmt = str(len(tensor_list)) + _dtype_to_format[tensor.dtype]
        tensor_bytes = struct.pack(fmt, *tensor_list)
        Path(path, 'weights', name).write_bytes(tensor_bytes)

    Path(path, 'weights_meta.json').write_text(json.dumps(meta, indent=4))

def patch(arch):
    if arch['type'] == 'Conv2d':
        kwargs = arch['kwargs']
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple)):
                assert len(value) == 2 and value[0] == value[1]
                kwargs[key] = value[0]
        kwargs['dilation'] = kwargs.get('dilation', 1)
        kwargs['padding'] = kwargs.get('padding', 0)
        kwargs['groups'] = kwargs.get('groups', 1)
        kwargs['bias'] = kwargs.get('bias', False)

    elif arch['type'] == 'Sequential':
        arch['children'] = list(arch['children'].values())

    return arch

def _get_model_arch(model, name):
    extra_repr = model.extra_repr()
    if extra_repr:
        code = f'_get_args({extra_repr})'
        args, kwargs = eval(code)
    else:
        args = None
        kwargs = None

    children = {}
    for key, module in model._modules.items():
        children[key] = _get_model_arch(module, '.'.join([name, key]))

    arch = { 'name': name, 'type': model._get_name() }
    if args is not None:
        arch['args'] = args
        arch['kwargs'] = kwargs
    if children:
        arch['children'] = children
    return patch(arch)

def _get_args(*args, **kwargs):
    return args, kwargs

_dtype_to_format = {
    torch.float32: 'f',
    torch.float64: 'd',
    torch.uint8: 'B',
    torch.int8: 'b',
    torch.int16: 'h',
    torch.int32: 'i',
    torch.int64: 'q',
    torch.bool: '?',
}
