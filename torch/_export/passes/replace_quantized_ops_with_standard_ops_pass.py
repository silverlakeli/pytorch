# mypy: allow-untyped-defs
import operator

import torch
import torch.export._trace
from torch._ops import OpOverload
from torch.ao.quantization.fx._decomposed import (
    dequantize_per_channel,
    dequantize_per_tensor,
    quantize_per_tensor,
)
from torch.ao.quantization.utils import calculate_qmin_qmax


_TORCH_DTYPE_TO_ENUM = {
    torch.uint8: 0,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 5,
    torch.float32: 6,
    torch.float64: 7,
    torch.complex32: 8,
    torch.complex64: 9,
    torch.complex128: 10,
    torch.bool: 11,
    torch.qint8: 12,
    torch.quint8: 13,
    torch.bfloat16: 15,
}

_TORCH_ENUM_TO_DTYPE = {value: key for key, value in _TORCH_DTYPE_TO_ENUM.items()}

def enum_to_dtype(val):
    if isinstance(val, torch.dtype):
        return val
    dtype = _TORCH_ENUM_TO_DTYPE[val]
    if dtype == torch.quint8:
        return torch.uint8
    elif dtype == torch.qint8:
        return torch.int8
    return dtype


def fx_enum_to_dtype(gm, val):
    return gm.graph.call_function(
        enum_to_dtype,
        (val,)
    )


def get_quantized(
    val,
    scale,
    zero_point,
    qmin,
    qmax,
    dtype,
):
    quantize_per_tensor(
        val,
        scale,
        zero_point,
        qmin,
        qmax,
        dtype,
    )


def fx_get_quantized(
    gm,
    val_node,
    scale_node,
    zero_point_node,
    qmin_node,
    qmax_node,
    dtype_node,
    qscheme,
):
    return gm.graph.call_function(
        quantize_per_tensor,
        (
            val_node,
            scale_node,
            zero_point_node,
            qmin_node,
            qmax_node,
            dtype_node,
        ),
    )


def get_dequantized(
    val,
    scale,
    zero_point,
    qmin,
    qmax,
    dtype,
    axis,
    qscheme,
):
    if qscheme is torch.per_tensor_affine:
        return dequantize_per_tensor(
            val,
            scale,
            zero_point,
            qmin,
            qmax,
            dtype,
        )
    elif qscheme is torch.per_channel_affine:
        return dequantize_per_channel(
            val,
            scale,
            zero_point,
            axis,
            qmin,
            qmax,
            dtype,
        )
    else:
        raise RuntimeError(f"Unsupported dequantization scheme: {qscheme}")


def fx_get_dequantized(
    gm,
    val_node,
    scale_node,
    zero_point_node,
    qmin_node,
    qmax_node,
    dtype_node,
    axis_node,
    qscheme,
):
    if qscheme is torch.per_tensor_affine:
        return gm.graph.call_function(
            dequantize_per_tensor,
            (
                val_node,
                scale_node,
                zero_point_node,
                qmin_node,
                qmax_node,
                dtype_node,
            ),
        )
    elif qscheme is torch.per_channel_affine:
        return gm.graph.call_function(
            dequantize_per_channel,
            (
                val_node,
                scale_node,
                zero_point_node,
                axis_node,
                qmin_node,
                qmax_node,
                dtype_node,
            )
        )
    else:
        raise RuntimeError(f"Unsupported dequantization scheme: {qscheme}")


def get_qmin_qmax(dtype):
    return calculate_qmin_qmax(None, None, False, dtype, False)


def fx_get_qmin_qmax(gm, dtype_node):
    q_min_max_node = gm.graph.call_function(
        calculate_qmin_qmax, (None, None, False, dtype_node, False)
    )
    qmin_node = gm.graph.call_function(operator.getitem, (q_min_max_node, 0))
    qmax_node = gm.graph.call_function(operator.getitem, (q_min_max_node, 1))
    return qmin_node, qmax_node


def get_mod_attr(mod, attr_name):
    for attr in attr_name.split("."):
        mod = getattr(mod, attr)
    return mod


def get_dequantize_param_by_index(gm, param_node, index=0, dequant=True):
    """Directly inline tensor from a get_attr fx node."""
    assert isinstance(param_node, torch.fx.Node)
    assert param_node.op == "get_attr"

    attr_name = param_node.target
    mod = get_mod_attr(gm, attr_name)
    qtensor = mod.unpack()[index]

    # Manual conversion because qint8 is not used anymore.
    if qtensor.dtype in [torch.qint8, torch.quint8]:
        tensor = qtensor.int_repr()
    else:
        tensor = qtensor

    if dequant:
        qscheme = qtensor.qscheme()
        if qscheme == torch.per_channel_affine:
            scale, zero_point, axis = (
                qtensor.q_per_channel_scale(),
                qtensor.q_per_channel_zero_points(),
                qtensor.q_per_channel_axis(),
            )
        else:
            scale, zero_point, axis = qtensor.q_scale(), qtensor.q_zero_point(), None
        dtype = tensor.dtype
        qmin, qmax = get_qmin_qmax(dtype)
        return get_dequantized(
            tensor, scale, zero_point, qmin, qmax, dtype, axis, qscheme
        )
    return tensor


def fx_transform_quantized_op_to_standard_op(gm, node):
    opname, args = node.target._opname, node.args
    if opname == "conv2d":
        scale_node, zero_point_node = args[2], args[3]
        param_0 = get_dequantize_param_by_index(gm, args[1])
        param_1 = get_dequantize_param_by_index(gm, args[1], index=1, dequant=False)
        op_res_node = gm.graph.call_function(
            torch.ops.aten.conv2d, (args[0], param_0, param_1)
        )
    elif opname == "conv2d_relu":
        scale_node, zero_point_node = args[2], args[3]
        param_0 = get_dequantize_param_by_index(gm, args[1])
        param_1 = get_dequantize_param_by_index(gm, args[1], index=1, dequant=False)
        op_res_node = gm.graph.call_function(
            torch.ops.aten.conv2d, (args[0], param_0, param_1)
        )
        op_res_node = gm.graph.call_function(torch.ops.aten.relu, (op_res_node,))
    else:
        raise RuntimeError(f"Unsupported quantized op during transformation: {opname}")

    def _get_dtype(t):
        return t.dtype

    gm.graph.inserting_before(node)
    # dtype_node = gm.graph.call_function(_get_dtype, (op_res_node,))
    dtype_node = torch.int8
    qmin_node, qmax_node = fx_get_qmin_qmax(gm, dtype_node)

    q_fx_node = fx_get_quantized(
        gm,
        op_res_node,
        scale_node,
        zero_point_node,
        qmin_node,
        qmax_node,
        dtype_node,
        torch.per_tensor_affine,
    )
    dq_fx_node = fx_get_dequantized(
        gm,
        q_fx_node,
        scale_node,
        zero_point_node,
        qmin_node,
        qmax_node,
        dtype_node,
        None,
        torch.per_tensor_affine,
    )
    return dq_fx_node


def replace_quantized_ops_with_standard_ops(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        if isinstance(node.target, OpOverload):
            namespace, opname = node.target.namespace, node.target._opname
            if namespace == "quantized":
                fx_node = fx_transform_quantized_op_to_standard_op(gm, node)
                node.replace_all_uses_with(fx_node)
                gm.graph.erase_node(node)
            elif namespace == "aten" and opname == "quantize_per_tensor":
                inp_node, scale_node, zero_point_node, dtype_node = node.args
                gm.graph.inserting_before(node)
                dtype_node = fx_enum_to_dtype(gm, dtype_node)
                qmin_node, qmax_node = fx_get_qmin_qmax(gm, dtype_node)
                q_fx_node = fx_get_quantized(
                    gm,
                    inp_node,
                    scale_node,
                    zero_point_node,
                    qmin_node,
                    qmax_node,
                    dtype_node,
                    torch.per_tensor_affine,
                )
                dq_fx_node = fx_get_dequantized(
                    gm,
                    q_fx_node,
                    scale_node,
                    zero_point_node,
                    qmin_node,
                    qmax_node,
                    dtype_node,
                    None,
                    torch.per_tensor_affine,
                )
                node.replace_all_uses_with(dq_fx_node)
                gm.graph.erase_node(node)
            elif namespace == "aten" and opname == "dequantize":
                # Dequantized value should be populated after each operator
                # already, so we can directly pass that.
                pass
    
    # Post-processing again to remove get_attr node on ScriptObjects.
    attr_names = set()
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            mod = get_mod_attr(gm, node.target)
            if isinstance(mod, torch.ScriptObject):
                gm.graph.erase_node(node)
                attr_names.add(node.target)

    for attr_name in attr_names:
        pmod = get_mod_attr(gm, ".".join(attr_name.split(".")[:-1]))
        delattr(pmod, attr_name.split(".")[-1])
