import os
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union, List
from dataclasses import dataclass, asdict

import mindspore as ms
from mindspore import ops, nn
from mindspore.dataset import transforms, vision

from lvdm.modules.encoders.clip import CLIPModel, CLIPTokenizer, parse, support_list


# Image processing
CLIP_RESIZE = vision.Resize((224, 224), interpolation=vision.Inter.BICUBIC)
CLIP_NORMALIZE = vision.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)
CENTER_CROP = vision.CenterCrop(224)

ViCLIP_NORMALIZE = vision.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# Constants
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def load_clip_model(arch, pretrained_ckpt_path, dtype):
    """
    Load CLIP model.

    Args:
        arch (str): Model architecture.
        pretrained_ckpt_path (str): Path of the pretrained checkpoint.
    Returns:
        model (CLIPModel): CLIP model.
    """

    config_path = support_list[arch.lower()]
    config = parse(config_path, pretrained_ckpt_path)
    config.dtype = dtype
    model = CLIPModel(config)
    return model


def get_pick_score_fn(precision="fp32"):
    """
    Loss function for PICK SCORE
    """
    print("Loading PICK SCORE model")

    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval()
    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model.requires_grad_(False)
    if precision == "fp16":
        model.to_float(ms.float16)

    def score_fn(image_inputs: ms.Tensor, text_inputs: str, return_logits=False):
        image_transforms = transforms.Compose(
            [
                CLIP_RESIZE,
                CENTER_CROP,
                CLIP_NORMALIZE,
            ]
        )
        pixel_values = image_transforms(image_inputs)

        # embed
        image_embs = model.get_image_features(pixel_values=pixel_values)
        image_embs = image_embs / ops.norm(image_embs, dim=-1, keepdim=True)

        with ms._no_grad():
            preprocessed = processor(
                text=text_inputs,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            text_embs = model.get_text_features(**preprocessed)
            text_embs = text_embs / ops.norm(text_embs, dim=-1, keepdim=True)

        # Get predicted scores from model(s)
        score = (text_embs * image_embs).sum(-1)
        if return_logits:
            score = score * model.logit_scale.exp()
        return score

    return score_fn


def get_hpsv2_fn(precision="amp"):
    precision = "amp" if precision == "no" else precision
    assert precision in ["bf16", "fp16", "amp", "fp32"]

    # model, _, preprocess_val = create_model_and_transforms(
    #     "ViT-H-14",
    #     "checkpoints/HPS_v2.1_compressed.pt",
    #     precision=precision,
    #     device="cpu",
    #     jit=False,
    #     force_quick_gelu=False,
    #     force_custom_text=False,
    #     force_patch_dropout=False, 
    #     force_image_size=None,
    #     pretrained_image=False,
    #     image_mean=None,
    #     image_std=None,
    #     light_augmentation=True,
    #     aug_cfg={},
    #     output_dict=True,
    #     with_score_predictor=False,
    #     with_region_predictor=False,
    # )
    model = load_clip_model("ViT-H-14", "checkpoints/HPS_v2.1_compressed.pt")
    preprocess_val = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=None,
        std=None,
        resize_longest_max=True,
    )
    tokenizer = CLIPTokenizer("ViT-H-14", pad_token="!")

    model.set_train(False)
    model.requires_grad_(False) # TODO!!!

    # gets vae decode as input
    def score_fn(
        image_inputs: ms.Tensor, text_inputs: List[str], return_logits=False
    ):
        # Process pixels and multicrop
        for t in preprocess_val.transforms[2:]:
            image_inputs = ops.stack([t(img) for img in image_inputs])

        if isinstance(text_inputs[0], str):
            text_inputs = tokenizer(text_inputs)

        # embed
        image_features = model.encode_image(image_inputs, normalize=True)

        with ms._no_grad():
            text_features = model.encode_text(text_inputs, normalize=True)

        hps_score = (image_features * text_features).sum(-1)
        if return_logits:
            hps_score = hps_score * model.logit_scale.exp()
        return hps_score

    return score_fn


def get_img_reward_fn(precision="fp32"):
    # pip install image-reward
    import ImageReward as RM

    model = RM.load("ImageReward-v1.0")
    model.eval()
    model.requires_grad_(False)

    rm_preprocess = transforms.Compose(
        [
            vision.Resize(224, interpolation=vision.Inter.BICUBIC),
            vision.CenterCrop(224),
            CLIP_NORMALIZE,
        ]
    )

    # gets vae decode as input
    def score_fn(
        image_inputs: ms.Tensor, text_inputs: List[str], return_logits=False
    ):
        del return_logits
        if precision == "fp16":
            model.to_float(ms.float16)

        image = rm_preprocess(image_inputs)
        text_input = model.blip.tokenizer(
            text_inputs,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )
        rewards = model.score_gard(
            text_input.input_ids, text_input.attention_mask, image
        )
        return -ops.relu(-rewards + 2).squeeze(-1)

    return score_fn


class ResizeCropMinSize(nn.Cell):

    def __init__(self, min_size, interpolation=vision.Inter.BICUBIC, fill=0):
        super().__init__()
        if not isinstance(min_size, int):
            raise TypeError(f"Size should be int. Got {type(min_size)}")
        self.min_size = min_size
        self.interpolation = interpolation
        self.fill = fill 
        self.random_crop = vision.RandomCrop((min_size, min_size))

    def construct(self, img):
        if isinstance(img, ms.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size
        scale = self.min_size / float(min(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = vision.Resize(new_size, self.interpolation)(img)
            img = self.random_crop(img)
        return img


@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float]]] = None
    interpolation: Optional[str] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False


class ResizeMaxSize(nn.Cell):

    def __init__(self, max_size, interpolation=vision.Inter.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def construct(self, img):
        if isinstance(img, ms.Tensor):
            height, width = img.shape[1:]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = vision.Resize(new_size, self.interpolation)(img)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = ops.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img


class MaskAwareNormalize(nn.Cell):
    def __init__(self, mean, std):
        super().__init__()
        self.normalize = vision.Normalize(mean=mean, std=std)

    def construct(self, tensor):
        if tensor.shape[0] == 4:
            return ops.cat([self.normalize(tensor[:3]), tensor[3:]], axis=0)
        else:
            return self.normalize(tensor)


def _convert_to_rgb_or_rgba(image):
    if image.mode == 'RGBA':
        return image
    else:
        return image.convert('RGB')


def image_transform(
    image_size: int,
    is_train: bool,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    resize_longest_max: bool = False,
    fill_color: int = 0,
    aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()
    normalize = MaskAwareNormalize(mean=mean, std=std)
    if is_train:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        use_timm = aug_cfg_dict.pop('use_timm', False)
        if use_timm:
            assert False, "not tested for augmentation with mask"
            from timm.data import create_transform  # timm can still be optional
            if isinstance(image_size, (tuple, list)):
                assert len(image_size) >= 2
                input_size = (3,) + image_size[-2:]
            else:
                input_size = (3, image_size, image_size)
            # by default, timm aug randomly alternates bicubic & bilinear for better robustness at inference time
            aug_cfg_dict.setdefault('interpolation', 'random')
            aug_cfg_dict.setdefault('color_jitter', None)  # disable by default
            train_transform = create_transform(
                input_size=input_size,
                is_training=True,
                hflip=0.,
                mean=mean,
                std=std,
                re_mode='pixel',
                **aug_cfg_dict,
            )
        else:
            train_transform = vision.Compose([
                _convert_to_rgb_or_rgba,
                vision.ToTensor(),
                vision.RandomResizedCrop(
                    image_size,
                    scale=aug_cfg_dict.pop('scale'),
                    interpolation=vision.Inter.BICUBIC,
                ),
                normalize,
            ])
            if aug_cfg_dict:
                warnings.warn(f'Unused augmentation cfg items, specify `use_timm` to use ({list(aug_cfg_dict.keys())}).')
        return train_transform
    else:
        transforms = [
            _convert_to_rgb_or_rgba,
            vision.ToTensor(),
        ]
        if resize_longest_max:
            transforms.extend([
                ResizeMaxSize(image_size, fill=fill_color)
            ])
        else:
            transforms.extend([
                vision.Resize(image_size, interpolation=vision.Inter.BICUBIC),
                vision.CenterCrop(image_size),
            ])
        transforms.extend([
            normalize,
        ])
        return vision.Compose(transforms)



def get_vi_clip_score_fn(rm_ckpt_dir: str, precision="amp", n_frames=8):
    assert n_frames == 8
    from viclip import get_viclip

    model_dict = get_viclip("l", rm_ckpt_dir)
    vi_clip = model_dict["viclip"]
    vi_clip.eval()
    vi_clip.requires_grad_(False)
    if precision == "fp16":
        vi_clip.to(ms.float16)

    viclip_resize = ResizeCropMinSize(224)

    def score_fn(image_inputs: ms.Tensor, text_inputs: str):
        # Process pixels and multicrop
        b, t = image_inputs.shape[:2]
        image_inputs = image_inputs.view(b * t, *image_inputs.shape[2:])
        pixel_values = ViCLIP_NORMALIZE(viclip_resize(image_inputs))
        pixel_values = pixel_values.view(b, t, *pixel_values.shape[1:])
        video_features = vi_clip.get_vid_feat_with_grad(pixel_values)

        with ms._no_grad():
            text_features = vi_clip.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        score = (video_features * text_features).sum(-1)
        return score

    return score_fn


def get_intern_vid2_score_fn(rm_ckpt_dir: str, precision="amp", n_frames=8):
    from intern_vid2.demo_config import Config, eval_dict_leaf
    from intern_vid2.demo_utils import setup_internvideo2

    config = Config.from_file("intern_vid2/configs/internvideo2_stage2_config.py")
    config = eval_dict_leaf(config)
    config["inputs"]["video_input"]["num_frames"] = n_frames
    config["inputs"]["video_input"]["num_frames_test"] = n_frames
    config["model"]["vision_encoder"]["num_frames"] = n_frames

    config["model"]["vision_encoder"]["pretrained"] = rm_ckpt_dir
    config["pretrained_path"] = rm_ckpt_dir

    vi_clip, tokenizer = setup_internvideo2(config)
    vi_clip.set_train(False)
    vi_clip.requires_grad_(False)
    if precision == "fp16":
        vi_clip.to(ms.float16)

    viclip_resize = ResizeCropMinSize(224)

    def score_fn(image_inputs: ms.Tensor, text_inputs: str):
        # Process pixels and multicrop
        b, t = image_inputs.shape[:2]
        image_inputs = image_inputs.view(b * t, *image_inputs.shape[2:])
        pixel_values = ViCLIP_NORMALIZE(viclip_resize(image_inputs))

        pixel_values = pixel_values.view(b, t, *pixel_values.shape[1:])
        video_features = vi_clip.get_vid_feat_with_grad(pixel_values)

        with ms._no_grad():
            text = tokenizer(
                text_inputs,
                padding="max_length",
                truncation=True,
                max_length=40,
                return_tensors="pt",
            )
            _, text_features = vi_clip.encode_text(text)
            text_features = vi_clip.text_proj(text_features)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        score = (video_features * text_features).sum(-1)
        return score

    return score_fn


def get_reward_fn(reward_fn_name: str, **kwargs):
    if reward_fn_name == "pick":
        return get_pick_score_fn(**kwargs)
    elif reward_fn_name == "hpsv2":
        return get_hpsv2_fn(**kwargs)
    elif reward_fn_name == "img_reward":
        return get_img_reward_fn(**kwargs)
    elif reward_fn_name == "vi_clip":
        return get_vi_clip_score_fn(**kwargs)
    elif reward_fn_name == "vi_clip2":
        return get_intern_vid2_score_fn(**kwargs)
    else:
        raise ValueError("Invalid reward_fn_name")