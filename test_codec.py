import argparse
import os

import torch
import yaml
from tqdm import tqdm

from codec_utils import (
    create_dataset,
    decode_group,
    infer_model_name,
    load_models,
    pad_to_mult64,
    save_tensor_as_mp4,
    unpad_tensor,
)

from Vtc_utils.utils_tool import adaptive_instance_normalization


# Required fields in YAML config
REQUIRED_FIELDS = [
    'dataset_name', 'ckpt_path', 'vae_path', 'flow_model_path',
    'context_path', 'num_pred_groups', 'output_dir', 'device',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="FLVC video compression/reconstruction test script. "
                    "All parameters are provided via a YAML config file."
    )
    parser.add_argument("--config", type=str, default="./test_codec_config.yaml",
                        help="Path to YAML config file")
    return parser.parse_args()


def load_config(config_path):
    """Load and validate a YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    missing = [k for k in REQUIRED_FIELDS if k not in config]
    if missing:
        raise ValueError(f"YAML config is missing required fields: {missing}")

    return config


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Parse config
    dataset_name = cfg['dataset_name']
    dataset_root = cfg.get('dataset_root')
    ckpt_dir = cfg.get('ckpt_dir')
    ckpt_path = cfg['ckpt_path']
    model_name = cfg.get('model_name')
    vae_path = cfg['vae_path']
    flow_model_path = cfg['flow_model_path']
    context_path = cfg['context_path']
    num_pred_groups = cfg['num_pred_groups']
    output_dir = cfg['output_dir']
    save_video = cfg.get('save_video', False)
    video_save_idx = cfg.get('video_save_idx', 0)
    device = cfg['device']
    use_lora = cfg.get('use_lora', False)
    lora_rank = cfg.get('lora_rank', 8)
    lora_alpha = cfg.get('lora_alpha', 1.0)
    use_ain = cfg.get('use_ain', False)
    model_name = infer_model_name(ckpt_dir, ckpt_path, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # ========================= Print Config =========================
    print("=" * 60)
    print("Test Configuration")
    print("=" * 60)
    print(f"Dataset:         {dataset_name}")
    print(f"Dataset root:    {dataset_root or './datasets'}")
    print(f"Checkpoint:      {ckpt_path}")
    print(f"Model name:      {model_name}")
    print(f"Output dir:      {output_dir}")
    print(f"num_pred_groups: {num_pred_groups}")
    print(f"use_lora:        {use_lora}")
    if use_lora:
        print(f"lora_rank:       {lora_rank}")
        print(f"lora_alpha:      {lora_alpha}")
    print("=" * 60)

    # ========================= Data & Models =========================
    visual_dataset = create_dataset(dataset_name, dataset_root=dataset_root)

    # Wrap YAML config as SimpleNamespace for load_models
    from types import SimpleNamespace
    model_args = SimpleNamespace(
        vae_path=vae_path, flow_model_path=flow_model_path,
        ckpt_dir=ckpt_dir, ckpt_path=ckpt_path,
    )
    models = load_models(model_args, device,
                        use_lora=use_lora,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha)

    vae = models['vae']
    wan_flow = models['wan_flow']
    compressor = models['compressor']
    dit_refine = models['dit_refine']

    # ========================= Codec Loop =========================
    group_size = 9 + num_pred_groups * 8

    context = None
    seq_len = None

    for num in tqdm(range(len(visual_dataset))):
        data_dict = visual_dataset[num]
        y = data_dict['frames'].unsqueeze(0)  # [1, C, T, H, W]

        y = torch.cat([y, y[:, :, -1:, :, :].repeat(1, 1, 4, 1, 1)], dim=2)

        num_groups = y.size(2) // group_size
        y_padded, pad_info = pad_to_mult64(y)

        # ---------- Per-group encode/decode ----------
        out_frames = []
        cache_frame = None
        last_frame_lat = None
        cache_hat = None
        bits_all = 0.0
        for i in tqdm(range(num_groups), leave=False):
            group_frames = y_padded[:, :, group_size * i: group_size * (i + 1), :, :]

            for g in range(1 + num_pred_groups):
                with torch.inference_mode(), torch.cuda.amp.autocast():
                    x_out, _bits, _f2_bits, context, seq_len, \
                        cache_frame, last_frame_lat, cache_hat = decode_group(
                        frames=group_frames, group_idx=g,
                        num_pred_groups=num_pred_groups,
                        vae=vae, wan_flow=wan_flow,
                        compressor=compressor, dit_refine=dit_refine,
                        context=context, seq_len=seq_len,
                        context_path=context_path, device=device,
                        cache_frame=cache_frame,
                        last_frame_lat=last_frame_lat,
                        cache_hat=cache_hat,
                    )
                    out_frames.append(x_out.cpu())
                    bits_all +=_bits
                    if i == num_groups - 1 and g == num_pred_groups:
                        bits_cut = _f2_bits

            torch.cuda.empty_cache()

        # ---------- Concatenate & restore ----------
        y_out = unpad_tensor(torch.cat(out_frames, dim=2), pad_info)
        recon_rgb = y_out[:, :, :96, :, :].permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]


        #####
        orig_rgb = y[:, :, 0:96, :, :].permute(0, 2, 1, 3, 4)
        bits_ain=0.0
        if use_ain:
            with torch.inference_mode():
                B, T, C, H, W = recon_rgb.shape
                recon_rgb_4d = recon_rgb.reshape(B * T, C, H, W)
                orig_rgb_4d = orig_rgb.reshape(B * T, C, H, W)

                recon_rgb,bits_ain = adaptive_instance_normalization(
                    recon_rgb_4d, orig_rgb_4d,bits=8
                )
                recon_rgb=recon_rgb.reshape(B, T, C, H, W)
        # Save video (optional)
        bpp = (bits_all-bits_cut+bits_ain) / (orig_rgb.size(1) * orig_rgb.size(3) * orig_rgb.size(4))
        # ---------- Save outputs ----------
        # Save decoded tensor
        tensor_path = os.path.join(
            output_dir,
            f"{dataset_name}_{model_name}_video{num}_bpp{bpp:.6f}_decoded.pt",
        )
        torch.save(recon_rgb, tensor_path)
        print(f"[Video {num}] Decoded tensor saved: {tensor_path}")


        if save_video and num == video_save_idx:
            video_path = os.path.join(
                output_dir,
                f"{dataset_name}_{model_name}_video{num}_GP_{num_groups}.mp4",
            )
            save_tensor_as_mp4(recon_rgb, video_path, fps=25)


        # Clear per-video cache
        cache_frame = None
        torch.cuda.empty_cache()

    print(f"\nCodec complete! Processed {len(visual_dataset)} videos, results saved to: {output_dir}")


if __name__ == "__main__":
    main()
