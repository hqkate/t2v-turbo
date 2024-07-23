import os, sys
import mindspore as ms
import numpy as np
from mindspore import context

# GRAPH_MODE 0
# PYNATIVE_MODE 1

context.set_context(mode=0, device_target="CPU", device_id=1)

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))


from utils.lora_handler import LoraHandler


use_unet_lora = True
lora_manager = LoraHandler(
    version="cloneofsimo",
    use_unet_lora=use_unet_lora,
    save_for_webui=True,
    unet_replace_modules=["UNetModel"],
)

