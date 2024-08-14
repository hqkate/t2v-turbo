import logging
import mindspore as ms
from mindspore import mint
from mindspore.communication.management import get_group_size, get_rank, init

from mindone.utils.seed import set_random_seed


logger = logging.getLogger(__name__)


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    jit_level: str = "O0",
    global_bf16: bool = False,
    debug: bool = False,
    dtype: ms.dtype = ms.float32,
):
    """
    Initialize MindSpore environment.

    Args:
        mode: MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        seed: The seed value for reproducibility. Default is 42.
        distributed: Whether to enable distributed training. Default is False.
    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)
    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    if debug and mode == ms.GRAPH_MODE:  # force PyNative mode when debugging
        logger.warning("Debug mode is on, switching execution mode to PyNative.")
        mode = ms.PYNATIVE_MODE

    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
        ms.reset_auto_parallel_context()

        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
    else:
        device_num = 1
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
            pynative_synchronize=debug,
        )

    try:
        if jit_level in ["O0", "O1", "O2"]:
            ms.set_context(jit_config={"jit_level": jit_level})
        else:
            logger.warning(
                f"Unsupport jit_level: {jit_level}. The framework automatically selects the execution method"
            )
    except Exception:
        logger.warning(
            "The current jit_level is not suitable because current MindSpore version or mode does not match,"
            "please ensure the MindSpore version >= ms2.3_0615, and use GRAPH_MODE."
        )

    if global_bf16:
        ms.set_context(ascend_config={"precision_mode": "allow_mix_precision_bf16"})
        logger.info("Using precision_mode: allow_mix_precision_bf16")
    elif dtype == ms.bfloat16:
        ms.set_context(ascend_config={"precision_mode": "allow_fp32_to_bf16"})
        logger.info("Using precision_mode: allow_fp32_to_bf16")
    elif dtype == ms.float16:
        ms.set_context(ascend_config={"precision_mode": "allow_mix_precision_fp16"})
        logger.info("Using precision_mode: allow_mix_precision_fp16")

    return rank_id, device_num
