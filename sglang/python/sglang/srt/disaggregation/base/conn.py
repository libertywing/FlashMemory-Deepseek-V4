from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import DisaggregationMode


class KVArgs:
    engine_rank: int
    kv_data_ptrs: List[int]
    kv_data_lens: List[int]
    kv_item_lens: List[int]
    aux_data_ptrs: List[int]
    aux_data_lens: List[int]
    aux_item_lens: List[int]
    state_data_ptrs: List[int]
    state_data_lens: List[int]
    state_item_lens: List[int]
    state_type: str  # "none", "mamba", "swa"
    # for mamba state different tp slice transfer
    state_dim_per_tensor: List[int]  # dimension to slice for each state tensor
    ib_device: str
    ib_traffic_class: str
    gpu_id: int
    # for different tp
    decode_tp_size: int
    kv_head_num: int
    page_size: int
    # for pp prefill
    prefill_pp_size: int
    pp_rank: int
    prefill_start_layer: int
    # for system dp
    system_dp_rank: int
    # Path-P-B: number of leading kv_data_ptrs entries that are c4 buffers redirected
    # to a host (DRAM) mirror instead of GPU. 0 = off (all kv_data_ptrs are VRAM).
    # When >0, register/send treat the first c4_dram_layer_num entries as DRAM dst.
    c4_dram_layer_num: int = 0
    # Path-I: number of index-K buffers (grp 1, following the c4 group) redirected to a
    # host mirror. Legacy contiguous-prefix hint; superseded by kv_dram_mask when present.
    index_dram_layer_num: int = 0
    # Path-I SELECTIVE: authoritative per-position DRAM (host-mirror) mask over kv_data_ptrs,
    # built by get_contiguous_buf_infos. True == this buffer's dst is a host mirror (DRAM);
    # False == VRAM. Handles the NON-CONTIGUOUS DRAM set (only offload indexer layers → mirror,
    # target layers stay VRAM interleaved). Empty => fall back to the prefix counts above.
    kv_dram_mask: List[bool] = []


class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class BaseKVManager(ABC):
    """Base class for managing transfer states"""

    @abstractmethod
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ): ...


class BaseKVSender(ABC):

    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ): ...

    @abstractmethod
    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """
        Set req's index metadata locally or notify the decoder server about the kv indices length and aux index.
        """
        ...

    @abstractmethod
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        """
        Send the kv cache at the given kv indices and the extra cache/state at the given indices to the decoder server.
        """
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer.
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails.
        """
        ...


class BaseKVReceiver(ABC):

    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ): ...

    @abstractmethod
    def init(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        """
        Set req's index metadata locally or notify the prefill server about the kv indices, aux index, and state_indices.
        """
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer.
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails.
        """
        ...

    def clear(self):
        """
        Clear any internal states.
        """
        pass

    def abort(self):
        """
        Abort the current transfer.
        """
        pass


class BaseKVBootstrapServer(ABC):
    @abstractmethod
    def __init__(self, host: str, port: int): ...
