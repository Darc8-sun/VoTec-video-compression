"""
Video codec test utility module.

Provides padding, simulated timesteps, model loading, dataset creation and
other helpers used by the test_codec.py codec test script.
"""

import math
import os

import cv2
import torch
import torch.nn.functional as F

from wan.modules.vae import WanVAE
from wan.modules.model import WanModel
from Vtc_compressor.new_igc import LatentCodec
from Vtc_module.Prior_refinement import Prior_refine_DIT
from Vtc_utils.yuv_video_dataset import YUVVideoDataset


# ========================= Tensor operations =========================

def pad_to_mult64(x, multiple=64):
    """
    Pad the H/W dimensions of a 5D tensor [B, C, T, H, W] to a multiple of `multiple`.
    Symmetric padding; any extra row/column is assigned to the bottom/right.

    Returns:
        (padded_tensor, pad_info) - pad_info is used by unpad_tensor to restore the shape.
    """
    H, W = x.shape[3], x.shape[4]
    new_H = math.ceil(H / multiple) * multiple
    new_W = math.ceil(W / multiple) * multiple

    pad_h, pad_w = new_H - H, new_W - W
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

    x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    pad_info = {
        'pad_top': pad_top, 'pad_bottom': pad_bottom,
        'pad_left': pad_left, 'pad_right': pad_right,
    }
    return x_padded, pad_info


def unpad_tensor(x, pad_info):
    """Remove the padding added by pad_to_mult64, restoring the tensor to its original H/W size."""
    pt = pad_info['pad_top']
    pb = pad_info['pad_bottom']
    pl = pad_info['pad_left']
    pr = pad_info['pad_right']

    H_end = x.size(3) - pb if pb > 0 else x.size(3)
    W_end = x.size(4) - pr if pr > 0 else x.size(4)

    return x[:, :, :, pt:H_end, pl:W_end]


# ========================= Codec helpers =========================

def compute_simulated_timesteps(c_hat, ori_latent):
    """
    Compute simulated timesteps from the difference between the compressed and original latents.

    Args:
        c_hat: list of tensors, each with shape [C, T, H, W]
        ori_latent: tensor [B, C, T, H, W]

    Returns:
        list of int, range [0, 1000]
    """
    timesteps = []
    for i in range(len(c_hat)):
        diff = c_hat[i] - ori_latent[i]
        c_std = torch.norm(diff) / (diff.numel() ** 0.5)
        t = c_std / (1 + c_std)
        if not torch.isfinite(t):
            t = torch.tensor(0.5, device=t.device, dtype=t.dtype)
        timesteps.append(int(torch.clamp(torch.round(t * 1000), min=0, max=1000)))
    return timesteps


def compute_seq_len(c_hat_sample, wan_flow):
    """
    Compute the sequence length from the latent shape and the model patch_size.

    Args:
        c_hat_sample: a single latent tensor [C, T, H, W]
        wan_flow: WanModel instance used to obtain patch_size

    Returns:
        int: sequence length
    """
    shape = c_hat_sample.size()
    patch_size = getattr(wan_flow, 'patch_size',
                         getattr(wan_flow.config, 'patch_size', (1, 2, 2)))
    return math.ceil(
        (shape[2] * shape[3]) / (patch_size[1] * patch_size[2]) * shape[1]
    )


# ========================= Flow-state prediction =========================

def Predict_v2x(v_pre, c_hat, timesteps):

    # 若输入已经是 Tensor，则直接 batch 并行计算
    if isinstance(v_pre, torch.Tensor) and isinstance(c_hat, torch.Tensor):
        timesteps = torch.as_tensor(
            timesteps, dtype=v_pre.dtype, device=v_pre.device)
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        sigma_t = timesteps.view(-1, 1, 1, 1, 1) / 1000.0
        return c_hat - sigma_t * v_pre

    # 兼容旧版 list 输入
    x_list = []
    for i in range(len(v_pre)):
        time = torch.tensor(timesteps[i], device=v_pre[i].device)
        sigma_t = time / 1000.0
        x0_pred = c_hat[i] - sigma_t * v_pre[i]
        x_list.append(x0_pred)
    return x_list

# ========================= Video saving =========================

def save_tensor_as_mp4(tensor: torch.Tensor, save_path: str, fps: int = 25):
    """
    Save a float tensor of shape [1, T, C, H, W] as an mp4 file.
    Handles both [-1, 1] and [0, 1] value ranges automatically.
    """
    video = tensor.squeeze(0)
    if video.min() < 0:
        video = (video + 1.0) / 2.0
    video = video.clamp(0, 1)
    video = (video * 255).byte().cpu().numpy()

    T, C, H, W = video.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    for i in range(T):
        frame_bgr = video[i].transpose(1, 2, 0)[:, :, ::-1]
        writer.write(frame_bgr)

    writer.release()
    print(f"Video saved to: {save_path}")


# ========================= Checkpoint utilities =========================

def resolve_checkpoint(ckpt_dir, ckpt_path):
    """Resolve the checkpoint path: prefer ckpt_path, otherwise look for best.pt under ckpt_dir."""
    if ckpt_path is not None:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Specified checkpoint does not exist: {ckpt_path}")
        return ckpt_path
    if ckpt_dir is not None:
        candidate = os.path.join(ckpt_dir, "best.pt")
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(
            f"No best.pt found under checkpoint directory {ckpt_dir}; "
            f"please specify the file with --ckpt_path."
        )
    raise ValueError("Either --ckpt_dir or --ckpt_path must be provided.")


def infer_model_name(ckpt_dir, ckpt_path, model_name):
    """Infer the model name, preferring the explicitly specified model_name."""
    if model_name is not None:
        return model_name
    if ckpt_dir is not None:
        return os.path.basename(os.path.normpath(ckpt_dir))
    return os.path.basename(os.path.dirname(os.path.normpath(ckpt_path)))


# ========================= Dataset configuration =========================

# Default location of the YUV test sets, relative to the repository root.
# Override it with the `dataset_root` field in the YAML config.
DEFAULT_DATASET_ROOT = "./datasets"

# Dataset name -> (relative sub-directory under dataset_root, geometry).
# `subdir` is joined with dataset_root at runtime; the remaining keys are
# forwarded verbatim to YUVVideoDataset.
DATASET_CONFIGS = {
    "UVG": {
        "subdir": "UVG",
        "width": 1920, "height": 1080, "num_frames": 96,
        "crop_size": (1080, 1920),
    },
    "HEVC_CLASSB": {
        "subdir": "HEVC-B",
        "width": 1920, "height": 1080, "num_frames": 96,
        "crop_size": (1080, 1920),
    },
    "MCL_JCV": {
        "subdir": "MCL",
        "width": 1920, "height": 1080, "num_frames": 96,
        "crop_size": (1080, 1920),
    },
    "HEVC_CLASSC": {
        "subdir": "HEVC-C",
        "width": 832, "height": 480, "num_frames": None,
        "crop_size": (480, 832),
    },
}


def create_dataset(dataset_name, dataset_root=None):
    """
    Create a YUVVideoDataset instance from the dataset name.

    Args:
        dataset_name: one of DATASET_CONFIGS.keys()
        dataset_root: root directory holding the per-dataset sub-folders.
            Falls back to DEFAULT_DATASET_ROOT ("./datasets") when None.
            The final path is os.path.join(dataset_root, subdir).

    Raises:
        ValueError: unknown dataset_name
        FileNotFoundError: the resolved directory does not exist
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset name: {dataset_name}; "
                         f"available: {list(DATASET_CONFIGS.keys())}")

    cfg = dict(DATASET_CONFIGS[dataset_name])
    subdir = cfg.pop("subdir")
    root = dataset_root if dataset_root else DEFAULT_DATASET_ROOT
    video_dir = os.path.join(root, subdir)

    if not os.path.isdir(video_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {video_dir}\n"
            f"Place the raw .yuv files under '{root}/{subdir}/' or set "
            f"'dataset_root' in the YAML config (see README 'Dataset Preparation')."
        )

    return YUVVideoDataset(
        video_dir=video_dir,
        pixel_format='yuv420p',
        bit_depth=8,
        normalize=True,
        value_range=(-1, 1),
        **cfg,
    )


# ========================= Model loading =========================

def load_models(args, device, use_lora=False, lora_rank=8, lora_alpha=1.0):
    """
    Load all models required for encoding and decoding.

    Args:
        args: parameter object containing vae_path, flow_model_path, ckpt_dir, ckpt_path
        device: torch device
        use_lora: whether to inject LoRA into WanFlow
        lora_rank: LoRA rank parameter (only effective when use_lora=True)
        lora_alpha: LoRA alpha parameter (only effective when use_lora=True)

    Returns:
        dict: dictionary containing vae, wan_flow, compressor, dit_refine
    """
    # --- VAE ---
    vae = WanVAE(z_dim=16, vae_pth=args.vae_path,
                 dtype=torch.float32, device=device)
    vae.model.eval()
    for param in vae.model.parameters():
        param.requires_grad = False

    # --- WanFlow (diffusion prior) ---
    wan_flow = WanModel.from_pretrained(args.flow_model_path)
    wan_flow.eval().requires_grad_(False).to(device)

    # --- LoRA injection (optional) ---
    flow_model = wan_flow  # model reference actually used for inference
    if use_lora:
        from Vtc_module.lora import WanFlowLoRA
        flow_lora = WanFlowLoRA(wan_flow, rank=lora_rank, alpha=lora_alpha)
        flow_lora = flow_lora.to(device)


    # --- Compressor ---
    compressor = LatentCodec(N=256, latent_ch=16, y_ch=128,
                          cb_size=[8192, 2048]).to(device)

    # --- DiT Refine ---
    dit_refine = Prior_refine_DIT(
        dim=1536, ffn_dim=8192,
        num_heads=16, num_layers=32,
    ).to(device)

    # --- Load checkpoint ---
    ckpt = torch.load(
        resolve_checkpoint(args.ckpt_dir, args.ckpt_path),
        map_location=device,
    )
    dit_refine.load_state_dict(ckpt['dit_refine_state_dict'])
    dit_refine.eval()
    compressor.load_state_dict(ckpt['compressor_state_dict'])
    compressor.eval()

    if use_lora:
        flow_lora.load_state_dict(ckpt['flow_lora_state_dict'])
        flow_lora.eval()

    return {
        'vae': vae,
        'wan_flow': flow_model,
        'compressor': compressor,
        'dit_refine': dit_refine,
    }


# ========================= Frame-group codec =========================

def decode_group(
    frames, group_idx, num_pred_groups,
    vae, wan_flow, compressor, dit_refine,
    context, seq_len, context_path, device,
    cache_frame, last_frame_lat, cache_hat,
):
    """
    Run encode -> compress -> prior prediction -> refine -> decode for a single frame group.

    On the first call, the text context is loaded from context_path and seq_len is
    computed; subsequent calls reuse the already computed context and seq_len.

    Args:
        frames: full frame sequence [B, C, T, H, W] (already padded)
        group_idx: index of the current frame group (0 is the I-frame group)
        num_pred_groups: number of P-frame groups
        context: list of text contexts; None on the first call (loaded inside the function)
        seq_len: sequence length; None on the first call (computed inside the function)
        context_path: path to the text context tensor file
        cache_frame: last frame pixels of the previous group (needed by P-frame groups)
        last_frame_lat: last frame latent of the previous group (needed by P-frame groups)
        cache_hat: refined latent cache of the previous group (needed by P-frame groups)

    Returns:
        (x_out, bits, frame_2_bits, context, seq_len,
         cache_frame, last_frame_lat, cache_hat)
        - frame_2_bits: number of bits of frame_2 in the P-frame group (0 for the I-frame group)
        - context / seq_len: loaded/computed conditioning context (for external caching)
    """
    is_iframe = (group_idx == 0)

    # --- Frame slicing ---
    if is_iframe:
        x = frames[:, :, :9, :, :].to(device)
    else:
        x = frames[:, :, 8 * group_idx + 1:8 * group_idx + 9, :, :].to(device)
        x = torch.cat([cache_frame, x], dim=2)

    # --- VAE encoding ---
    latent_x = vae.encode_(x)

    # --- Compression ---
    if is_iframe:
        output_dict = compressor.forward(latent_x)
    else:
        output_dict = compressor.forward(
            latent_x, mode='Group2', x0_GT=last_frame_lat.squeeze(2),
        )

    # --- Bit counting ---
    bits = sum(
        output_dict[k]["bit"].sum().item()
        for k in output_dict if k.startswith("frame_")
    )
    frame_2_bits = (
        output_dict["frame_2"]["bit"].sum().item()
        if "frame_2" in output_dict else 0
    )

    # --- Simulated timesteps ---
    c_hat = list(output_dict['xhat_T'])
    timesteps = compute_simulated_timesteps(c_hat, latent_x)
    t = torch.tensor(timesteps, device=device)

    # --- First call: compute seq_len and load context ---
    if context is None:
        seq_len = compute_seq_len(c_hat[0], wan_flow)
        ctx_tensor = torch.load(context_path)
        context = [ctx_tensor.float().to(device)] * latent_x.size(0)
    arg_c = {'context': context, 'seq_len': seq_len}

    # --- Flow prior prediction ---
    c_hat_t = [c * (1 - t0 / 1000.0) for c, t0 in zip(c_hat, t)]
    v_pre = wan_flow(c_hat_t, t=t, **arg_c)

    # --- Denoising + DiT refinement ---
    x_pre = predict_v2x(v_pre, c_hat_t, timesteps)
    refine_kwargs = arg_c if is_iframe else {**arg_c, 'x_cache': cache_hat}
    x_refine = dit_refine(c_hat, x_pre, t, **refine_kwargs)

    # --- VAE decoding ---
    x_out = vae.decode_(x_refine)

    # --- Update caches (used for the next P-frame group) ---
    cache_frame = x[:, :, -1, :, :].unsqueeze(2)
    last_frame_lat = vae.encode_(x_out[:, :, -1, :, :].unsqueeze(2))
    cache_hat = [c.detach() for c in x_refine]

    # Drop the first frame of a P-frame group (it comes from the previous group's cached frame)
    if not is_iframe:
        x_out = x_out[:, :, 1:, :, :]

    return x_out, bits, frame_2_bits, context, seq_len, \
        cache_frame, last_frame_lat, cache_hat
