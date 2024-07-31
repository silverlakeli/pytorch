# not for land

import csv
import gc
import os
import time
from typing import Callable

import pandas as pd

import torch
import torch.utils.benchmark as benchmark
from torch.profiler import profile, ProfilerActivity, record_function

from .vasiliy_debug_extract_subgraphs import summary_headers

# don't truncate long fields
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_columns', None)  

bytes_in_gb = 1024 * 1024 * 1024

def benchmark_torch_function_in_microseconds(
    func: Callable,
    *args,
    **kwargs,
) -> float:

    if True:
        # warmup
        for _ in range(2):
            func(*args, **kwargs)
        t0 = benchmark.Timer(
            stmt="func(*args, **kwargs)",
            globals={"args": args, "kwargs": kwargs, "func": func},
        )
        return t0.blocked_autorange().median * 1e6

    if False:
        n_warmup = 3
        n_iter = 10

        for _ in range(n_warmup):
            func(*args, **kwargs)

        t0 = time.time()
        for _ in range(n_iter):
            func(*args, **kwargs)
        t1 = time.time()
        return (t1 - t0) / n_iter * 1e6


def profile_to_file(target_file, func, *args, **kwargs):
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    # warm up
    for _ in range(2):
        func(*args, **kwargs)
        torch.cuda.synchronize()
    with profile(activities=activities) as prof:
        for _ in range(3):
            func(*args, **kwargs)
            torch.cuda.synchronize()
    prof.export_chrome_trace(target_file)


def fwd_and_bwd(m, args):
    outs = m(*args)
    if isinstance(outs, tuple):
        outs = torch.cat([*outs], dim=0)
    else:
        outs = torch.cat([outs], dim=0)
    outs.sum().backward()
    torch.cuda.synchronize()

# TODO reuse input and weight to save memory
@torch.inference_mode
def bench_fwd_bwd_gemms(M, K, N):
    # fwd: in (M, K) @ w_t (K, N) -> out (M, N)
    # bwd 1: grad_out (M, N) @ w (N, K) -> grad_in (M, K)
    # bwd 2: grad_out_t (N, M) @ in (M, K) -> grad_w (N, K)

    input = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    weight = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    grad_out = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')

    input_fp8 = input.to(torch.float8_e4m3fn)
    weight_fp8 = weight.to(torch.float8_e4m3fn)
    grad_out_fp8 = grad_out.to(torch.float8_e5m2)

    scale_a = torch.tensor([1.0], device='cuda')
    scale_b = torch.tensor([1.0], device='cuda')
    
    fwd_time_bf16 = benchmark_torch_function_in_microseconds(
        torch.mm,
        input, weight.t()
    )

    fwd_time_fp8 = benchmark_torch_function_in_microseconds(
        torch._scaled_mm,
        input_fp8, weight_fp8.t(),
        scale_a, scale_b, out_dtype=torch.bfloat16,
    )

    grad_in_time_bf16 = benchmark_torch_function_in_microseconds(
        torch.mm,
        grad_out, weight,
    )

    grad_in_time_fp8 = benchmark_torch_function_in_microseconds(
        torch._scaled_mm,
        grad_out_fp8, weight_fp8.t().contiguous().t(),
        scale_a, scale_b, out_dtype=torch.bfloat16,
    )

    grad_w_time_bf16 = benchmark_torch_function_in_microseconds(
        torch.mm,
        grad_out.t(), input,
    )

    grad_w_time_fp8 = benchmark_torch_function_in_microseconds(
        torch._scaled_mm,
        grad_out_fp8.t().contiguous(), input_fp8.t().contiguous().t(),
        scale_a, scale_b, out_dtype=torch.bfloat16,
    )

    # print(f'out bf16: {fwd_time_bf16:.2f}, fp8: {fwd_time_fp8:.2f}')
    # print(f'grad_in bf16: {grad_in_time_bf16:.2f}, fp8: {grad_in_time_fp8:.2f}')
    # print(f'grad_w bf16: {grad_w_time_bf16:.2f}, fp8: {grad_w_time_fp8:.2f}')
    total_bf16 = fwd_time_bf16 + grad_in_time_bf16 + grad_w_time_bf16
    total_fp8 = fwd_time_fp8 + grad_in_time_fp8 + grad_w_time_fp8
    # print(f'fp8 speedup: {total_bf16/total_fp8}')
    # TODO add dual gemms here

    del input, weight, grad_out, input_fp8, weight_fp8, grad_out_fp8

    return total_bf16, total_fp8, total_bf16 / total_fp8


def get_mkn(inputs, m):
    # hack: assume that the first input with rank 2 is the linear input
    # TODO fix it! 
    first_linear_input = None
    for input in inputs:
        if len(input.size()) == 2:
            first_linear_input = input
            break
    assert first_linear_input is not None, 'unsupported'
    M, K = first_linear_input.shape
    # We know m.0 is the first linear because of how we constructed this 
    # subgraph in the extraction code.
    linear_mod = getattr(m, '0')
    K2, N = linear_mod.in_features, linear_mod.out_features
    assert K == K2, 'unexpected K'
    return M, K, N


def analyze_subgraphs(
    target_folder: str,
    extracted_bsz: int,
    target_bsz: int,
) -> None:
    """
    Assumes folder structure:

        target_folder/
          debug_logs.txt
          summary.csv
          subgraph_with_inputs_0.pt
          ...
          subgraph_with_inputs_(n-1).pt

    Does the following:
    * load each subgraph in bf16
    * increase batch size to target_batch_size
    * benchmark fw+bw for each and record the runtime, display a table comparing
      the relative runtime of each in bf16
    """
    summary_filename = os.path.join(target_folder, 'summary.csv')
    # summary_df = pd.read_csv(os.path.join(target_folder, 'summary.csv'))
    # print()
    # print(summary_df)

    summary_rows = []
    with open(summary_filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            summary_rows.append(row)

    # add names of metrics we are about to collect to headers
    # TODO: adding new metrics is brittle because we need to hand track the
    # indices, there is a definitely a better way to do this
    # Note: not loading directly in a dataframe here because we are going to
    # modify rows inplace and that seemed annoying with dataframes, but there
    # is probably a solution. If someone figures that out, doing `df` throughout
    # this file is cleaner.
    summary_rows[0].extend([
        'gemm_time_bf16', 'gemm_time_fp8', 'gemm_time_speedup', 
        'time_eager_us', 'time_compile_us',
    ])

    # [1:] to skip header row
    for row_idx, row in enumerate(summary_rows[1:]):
        subgraph_idx = row[1]
        subgraph_fname = f'subgraph_with_inputs_{subgraph_idx}.pt'
        print(f'benchmarking {subgraph_fname}')
        subgraph_fname = os.path.join(target_folder, subgraph_fname)

        m, inputs = torch.load(subgraph_fname, weights_only=False)

        # adjust each input's bsz to target_bsz
        # enable grad
        def resize_input_and_enable_grad(t):
            if len(t.shape) > 1:
                old_first_dim, old_rest = t.size()[0], t.size()[1:]
                new_first_dim = old_first_dim // extracted_bsz * target_bsz
                new_shape = (new_first_dim, *old_rest)
                t = torch.randn(*new_shape, dtype=t.dtype, device=t.device, requires_grad=True)
                # t.resize_(new_first_dim, *old_rest).random_(-1000, 1000).div_(1000.0)
            else:
                # assume that rank 1 tensors do not depend on batch size
                t.requires_grad_(True)
                pass
            return t

        inputs = [resize_input_and_enable_grad(t) for t in inputs]

        # estimate memory used by inputs, params, grads
        input_gb = 0
        for inp in inputs:
            input_gb += (inp.numel() * inp.element_size()) / bytes_in_gb
        model_gb = sum(p.numel() * p.element_size() / bytes_in_gb for p in m.parameters())
        grad_gb = input_gb + model_gb
        total_gb = input_gb + model_gb + grad_gb
        # print(f'param mem estimate (GB): input {input_gb} model {model_gb} grad {grad_gb} total {total_gb}')

        # benchmark gemm time in bf16 vs fp8
        # TODO add dual gemm support here
        M, K, N = get_mkn(inputs, m)
        gemm_time_bf16, gemm_time_fp8, gemm_time_speedup = bench_fwd_bwd_gemms(M, K, N)
        row.extend([gemm_time_bf16, gemm_time_fp8, gemm_time_speedup])

        time_eager_us = benchmark_torch_function_in_microseconds(fwd_and_bwd, m, inputs)
        profile_file_eager = os.path.join(target_folder, f'profile_{subgraph_idx}_eager.json')
        profile_to_file(profile_file_eager, fwd_and_bwd, m, inputs)
        row.append(time_eager_us)

        m = torch.compile(m)
        time_compile_us = benchmark_torch_function_in_microseconds(fwd_and_bwd, m, inputs)
        profile_file_compile = os.path.join(target_folder, f'profile_{subgraph_idx}_compile.json')
        profile_to_file(profile_file_compile, fwd_and_bwd, m, inputs)
        row.append(time_compile_us)

        # need to manually delete grad from inputs, otherwise it would survive
        # and eventually OOM for large problem sizes
        for inp in inputs:
            del inp.grad
        del m, inputs
        gc.collect()
        torch.cuda.empty_cache()

        if row_idx == 1:
            break

    # convert to pandas df for easy printing and aggregate manipulations
    summary_df = pd.DataFrame(summary_rows[1:], columns=summary_rows[0])

    # calculate total subgraph time and each row's contribution to it
    total_time_us = summary_df['time_compile_us'].sum()

    summary_df['pct_time'] = summary_df['time_compile_us'] / total_time_us
    print(summary_df)

    # summary_df = summary_df.join(time_df)
    # print(summary_df)

    print('done')
