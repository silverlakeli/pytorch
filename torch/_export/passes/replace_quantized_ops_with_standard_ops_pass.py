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


def fx_get_qmin_qmax(gm, dtype_node):
    q_min_max_node = gm.graph.call_function(
        calculate_qmin_qmax, (None, None, False, dtype_node, False)
    )
    qmin_node = gm.graph.call_function(operator.getitem, (q_min_max_node, 0))
    qmax_node = gm.graph.call_function(operator.getitem, (q_min_max_node, 1))
    return qmin_node, qmax_node


def fx_get_dequantize_param(gm, param_node, index=0, dequant=True):
    def _get_tensor(param, index):
        return param.unpack()[index]

    def _get_q_scheme(param, index):
        return param.unpack()[index].qscheme()

    def _get_scale(param, index):
        qscheme = param.unpack()[index].qscheme()
        if qscheme == torch.per_channel_affine:
            return param.unpack()[index].q_per_channel_scales()
        return param.unpack()[index].q_scale()

    def _get_zero_point(param, index):
        qscheme = param.unpack()[index].qscheme()
        if qscheme == torch.per_channel_affine:
            return param.unpack()[index].q_per_channel_zero_points()
        return param.unpack()[index].q_zero_point()

    def _get_axis(param, index):
        qscheme = param.unpack()[index].qscheme()
        if qscheme == torch.per_channel_affine:
            return param.unpack()[index].q_per_channel_axis()
        return None

    def _get_dtype(param, index):
        return param.unpack()[index].dtype

    if dequant:
        tensor_node = gm.graph.call_function(_get_tensor, (param_node, index))
        q_scheme_node = gm.graph.call_function(_get_q_scheme, (param_node, index))
        scale_node = gm.graph.call_function(_get_scale, (param_node, index))
        zero_point_node = gm.graph.call_function(_get_zero_point, (param_node, index))
        axis_node = gm.graph.call_function(_get_axis, (param_node, index))
        dtype_node = gm.graph.call_function(_get_dtype, (param_node, index))
        qmin_node, qmax_node = fx_get_qmin_qmax(gm, dtype_node)

        return gm.graph.call_function(
            get_dequantized,
            (
                tensor_node,
                scale_node,
                zero_point_node,
                qmin_node,
                qmax_node,
                dtype_node,
                axis_node,
                q_scheme_node,
            ),
        )
    else:
        return gm.graph.call_function(_get_tensor, (param_node, index))


def fx_transform_quantized_op_to_standard_op(gm, node):
    opname, args = node.target._opname, node.args
    if opname == "conv2d":
        scale_node, zero_point_node = args[2], args[3]
        param_node_0 = fx_get_dequantize_param(gm, args[1])
        param_node_1 = fx_get_dequantize_param(gm, args[1], index=1, dequant=False)
        op_res_node = gm.graph.call_function(
            torch.ops.aten.conv2d, (args[0], param_node_0, param_node_1)
        )
    elif opname == "conv2d_relu":
        scale_node, zero_point_node = args[2], args[3]
        param_node_0 = fx_get_dequantize_param(gm, args[1])
        param_node_1 = fx_get_dequantize_param(gm, args[1], index=1, dequant=False)
        op_res_node = gm.graph.call_function(
            torch.ops.aten.conv2d, (args[0], param_node_0, param_node_1)
        )
        op_res_node = gm.graph.call_function(
            torch.ops.aten.relu, (op_res_node,)
        )
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
