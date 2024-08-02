# mypy: allow-untyped-defs
from __future__ import annotations

import os
from typing import List, Optional

from .. import config
from ..virtualized import V
from .common import TensorArg


class DebugPrinter:
    def get_debug_filtered_kernel_names(self) -> List[str]:
        return [
            x.strip()
            for x in os.environ.get(
                "AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT",
                ",".join(V.graph.all_codegen_kernel_names),
            )
            .lower()
            .split(",")
        ]

    def codegen_intermediate_tensor_value_printer(
        self,
        args_to_print,
        kernel_name,
        before_launch=True,
        arg_types: Optional[List[str]] = None,
    ) -> None:
        wrapper = V.graph.wrapper_code

        # when invoking this codegen_intermediate_tensor_value_printer function, we already assured that the AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER
        # env var is set to 1, so we can directly use get method for filtered kernel info here
        filtered_kernel_names_to_print = []
        if V.graph.cpp_wrapper:
            filtered_kernel_names_to_print = self.get_debug_filtered_kernel_names()

        for i, arg in enumerate(args_to_print):
            if arg_types is not None and not isinstance(arg_types[i], TensorArg):
                continue
            if (
                len(filtered_kernel_names_to_print) > 0
                and kernel_name not in filtered_kernel_names_to_print
            ):
                continue
            launch_prefix = "before_launch" if before_launch else "after_launch"
            if V.graph.cpp_wrapper:
                if config.abi_compatible:
                    wrapper.writeline(
                        f'aoti_torch_print_tensor_handle({arg}, "{launch_prefix} - {kernel_name} - {arg}");'
                    )
                else:
                    # TODO: add non-abi compatible mode debug printing info
                    pass
            else:
                line = f"print('{launch_prefix} {kernel_name} - {arg}', {arg})"
                wrapper.writeline(line)
