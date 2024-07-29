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


def get_quantized(
    gm,
    val,
    scale,
    zero_point,
    qmin,
    qmax,
    dtype,
    qscheme,
):
    if qscheme is torch.per_tensor_affine:
        quantize_per_tensor(
            val,
            scale,
            zero_point,
            qmin,
            qmax,
            dtype,
        )
    else:
        raise RuntimeError(f"Unsupported quantization scheme: {qscheme}")


def fx_get_quantized(
    gm,
    val_node,
    scale_node,
    zero_point_node,
    qmin_node,
    qmax_node,
    dtype_node,
    qscheme_node,
):
    return gm.graph.call_function(
        get_quantized,
        (
            val_node,
            scale_node,
            zero_point_node,
            qmin_node,
            qmax_node,
            dtype_node,
            qscheme_node,
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
    qscheme_node,
):
    return gm.graph.call_function(
        get_dequantized,
        (
            val_node,
            scale_node,
            zero_point_node,
            qmin_node,
            qmax_node,
            dtype_node,
            axis_node,
            qscheme_node,
        ),
    )


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

    dtype_node = gm.graph.call_function(_get_dtype, (op_res_node,))
    qmin_node, qmax_node = fx_get_qmin_qmax(gm, dtype_node)

    q_fx_node = fx_get_dequantized(
        gm,
        op_res_node,
        scale_node,
        zero_point_node,
        qmin_node,
        qmax_node,
        dtype_node,
        None,
        torch.per_tensor_affine,
    )
    dq_fx_node = fx_get_quantized(
        gm,
        q_fx_node,
        scale_node,
        zero_point_node,
        qmin_node,
        qmax_node,
        dtype_node,
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
                qmin_node, qmax_node = fx_get_qmin_qmax(gm, dtype_node)
                q_fx_node = fx_get_dequantized(
                    gm,
                    inp_node,
                    scale_node,
                    zero_point_node,
                    qmin_node,
                    qmax_node,
                    dtype_node,
                    None,
                    torch.per_tensor_affine,
                )
                dq_fx_node = fx_get_quantized(
                    gm,
                    q_fx_node,
                    scale_node,
                    zero_point_node,
                    qmin_node,
                    qmax_node,
                    dtype_node,
                    torch.per_tensor_affine,
                )
                node.replace_all_uses_with(dq_fx_node)
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
            
