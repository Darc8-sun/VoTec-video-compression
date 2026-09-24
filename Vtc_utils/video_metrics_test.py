# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict


class VideoQualityMetrics:
    """
    Video quality assessment metrics.

    Supported metrics:
    - PSNR (Peak Signal-to-Noise Ratio)
    - LPIPS (Learned Perceptual Image Patch Similarity)
    - DISTS (Deep Image Structure and Texture Similarity)
    """

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the metrics calculator.

        Args:
            device: computation device, defaults to cuda (if available)
        """
        self.device = device
        self.lpips_model = None
        self.dists_model = None
        self._init_lpips()
        self._init_dists()

    def _ensure_5d(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensure the input tensor is 5D (B, T, C, H, W).

        Args:
            tensor: input tensor

        Returns:
            5D tensor
        """
        if tensor.dim() == 4:
            # (T, C, H, W) -> (1, T, C, H, W)
            return tensor.unsqueeze(0)
        elif tensor.dim() == 5:
            return tensor
        else:
            raise ValueError(f"Input tensor must be 4D or 5D, got {tensor.dim()}")

    def calculate_psnr(self,
                       pred: torch.Tensor,
                       target: torch.Tensor,
                       data_range: float = 2.0) -> torch.Tensor:
        """
        Compute PSNR (Peak Signal-to-Noise Ratio).

        Defined in: VideoQualityMetrics class, this file.

        Args:
            pred: predicted video, shape (B, T, C, H, W) or (T, C, H, W), value range [-1, 1]
            target: target video, same shape as pred, value range [-1, 1]
            data_range: data range, default 2.0 (from -1 to 1)

        Returns:
            PSNR values, shape (B,), one value per batch item

        Example:
            >>> pred = torch.rand(2, 10, 3, 256, 448) * 2 - 1  # (B=2, T=10, C=3, H=256, W=448)
            >>> target = torch.rand(2, 10, 3, 256, 448) * 2 - 1
            >>> psnr = metrics.calculate_psnr(pred, target)
            >>> print(psnr.shape)  # torch.Size([2])
            >>> print(psnr)  # tensor([25.3, 26.1])
        """
        pred = self._ensure_5d(pred).to(self.device)
        target = self._ensure_5d(target).to(self.device)

        B, T, C, H, W = pred.shape

        # Reshape (B, T, C, H, W) to (B, T*C*H*W)
        pred_flat = pred.reshape(B, -1)
        target_flat = target.reshape(B, -1)

        # Compute MSE
        mse = F.mse_loss(pred_flat, target_flat, reduction='none').mean(dim=1)

        # Avoid log(0)
        mse = torch.clamp(mse, min=1e-10)

        # Compute PSNR
        max_val = data_range
        psnr = 10 * torch.log10((max_val ** 2) / mse)

        return psnr

    def _init_lpips(self):
        """Initialize the LPIPS models."""
        if self.lpips_model is None:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_model.eval()
            self.lpips_model_V = lpips.LPIPS(net='vgg').to(self.device)
            self.lpips_model_V.eval()

    def calculate_lpips(self,
                        pred: torch.Tensor,
                        target: torch.Tensor,
                        frame_batch_size: int = 8) -> torch.Tensor:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity).

        Defined in: VideoQualityMetrics class, this file.

        Args:
            pred: predicted video, shape (B, T, C, H, W) or (T, C, H, W), value range [-1, 1]
            target: target video, same shape as pred, value range [-1, 1]
            frame_batch_size: number of frames processed per batch to save memory

        Returns:
            LPIPS values, shape (B,), one value per batch item (averaged over all frames)

        Example:
            >>> pred = torch.rand(2, 10, 3, 256, 448) * 2 - 1
            >>> target = torch.rand(2, 10, 3, 256, 448) * 2 - 1
            >>> lpips = metrics.calculate_lpips(pred, target)
            >>> print(lpips.shape)  # torch.Size([2])
            >>> print(lpips)  # tensor([0.15, 0.18])
        """

        pred = self._ensure_5d(pred).to(self.device)
        target = self._ensure_5d(target).to(self.device)

        B, T, C, H, W = pred.shape

        lpips_scores = []

        with torch.no_grad():
            for b in range(B):
                frame_scores = []

                # Process frames in chunks to save memory
                for t_start in range(0, T, frame_batch_size):
                    t_end = min(t_start + frame_batch_size, T)

                    # Extract the current chunk of frames: (batch_frames, C, H, W)
                    pred_frames = pred[b, t_start:t_end]  # (t_end-t_start, C, H, W)
                    target_frames = target[b, t_start:t_end]

                    # LPIPS expects input in [0, 1] or [-1, 1];
                    # the lpips library handles normalization internally
                    if pred_frames.min() < 0:
                        # Convert from [-1, 1] to [0, 1]
                        pred_frames = (pred_frames + 1) / 2
                        target_frames = (target_frames + 1) / 2

                    # Compute LPIPS
                    dist = self.lpips_model(pred_frames, target_frames, normalize=True)
                    frame_scores.append(dist.mean().item())

                lpips_scores.append(np.mean(frame_scores))

        return torch.tensor(lpips_scores, device=self.device)

    def calculate_lpips_VGG(self,
                            pred: torch.Tensor,
                            target: torch.Tensor,
                            frame_batch_size: int = 8) -> torch.Tensor:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity) with the VGG backbone.

        Defined in: VideoQualityMetrics class, this file.

        Args:
            pred: predicted video, shape (B, T, C, H, W) or (T, C, H, W), value range [-1, 1]
            target: target video, same shape as pred, value range [-1, 1]
            frame_batch_size: number of frames processed per batch to save memory

        Returns:
            LPIPS values, shape (B,), one value per batch item (averaged over all frames)

        Example:
            >>> pred = torch.rand(2, 10, 3, 256, 448) * 2 - 1
            >>> target = torch.rand(2, 10, 3, 256, 448) * 2 - 1
            >>> lpips = metrics.calculate_lpips(pred, target)
            >>> print(lpips.shape)  # torch.Size([2])
            >>> print(lpips)  # tensor([0.15, 0.18])
        """

        pred = self._ensure_5d(pred).to(self.device)
        target = self._ensure_5d(target).to(self.device)

        B, T, C, H, W = pred.shape

        lpips_scores = []

        with torch.no_grad():
            for b in range(B):
                frame_scores = []

                # Process frames in chunks to save memory
                for t_start in range(0, T, frame_batch_size):
                    t_end = min(t_start + frame_batch_size, T)

                    # Extract the current chunk of frames: (batch_frames, C, H, W)
                    pred_frames = pred[b, t_start:t_end]  # (t_end-t_start, C, H, W)
                    target_frames = target[b, t_start:t_end]

                    # LPIPS expects input in [0, 1] or [-1, 1];
                    # the lpips library handles normalization internally
                    if pred_frames.min() < 0:
                        # Convert from [-1, 1] to [0, 1]
                        pred_frames = (pred_frames + 1) / 2
                        target_frames = (target_frames + 1) / 2

                    # Compute LPIPS
                    dist = self.lpips_model_V(pred_frames, target_frames, normalize=True)
                    frame_scores.append(dist.mean().item())

                lpips_scores.append(np.mean(frame_scores))

        return torch.tensor(lpips_scores, device=self.device)

    def _init_dists(self):
        """Initialize the DISTS model."""
        if self.dists_model is None:
            from DISTS_pytorch import DISTS
            self.dists_model = DISTS().to(self.device)
            self.dists_model.eval()

    def calculate_dists(self,
                        pred: torch.Tensor,
                        target: torch.Tensor,
                        frame_batch_size: int = 8) -> torch.Tensor:
        """
        Compute DISTS (Deep Image Structure and Texture Similarity).

        Defined in: VideoQualityMetrics class, this file.

        Args:
            pred: predicted video, shape (B, T, C, H, W) or (T, C, H, W), value range [-1, 1]
            target: target video, same shape as pred, value range [-1, 1]
            frame_batch_size: number of frames processed per batch to save memory

        Returns:
            DISTS values, shape (B,), one value per batch item (averaged over all frames)

        Example:
            >>> pred = torch.rand(2, 10, 3, 256, 448) * 2 - 1
            >>> target = torch.rand(2, 10, 3, 256, 448) * 2 - 1
            >>> dists = metrics.calculate_dists(pred, target)
            >>> print(dists.shape)  # torch.Size([2])
            >>> print(dists)  # tensor([0.12, 0.14])
        """

        pred = self._ensure_5d(pred).to(self.device)
        target = self._ensure_5d(target).to(self.device)

        B, T, C, H, W = pred.shape

        dists_scores = []

        with torch.no_grad():
            for b in range(B):
                frame_scores = []

                # Process frames in chunks to save memory
                for t_start in range(0, T, frame_batch_size):
                    t_end = min(t_start + frame_batch_size, T)

                    # Extract the current chunk of frames: (batch_frames, C, H, W)
                    pred_frames = pred[b, t_start:t_end]
                    target_frames = target[b, t_start:t_end]

                    # DISTS expects input in [0, 1]
                    if pred_frames.min() < 0:
                        pred_frames = (pred_frames + 1) / 2
                        target_frames = (target_frames + 1) / 2

                    # Compute DISTS
                    dist = self.dists_model(pred_frames, target_frames)
                    frame_scores.append(dist.mean().item())

                dists_scores.append(np.mean(frame_scores))

        return torch.tensor(dists_scores, device=self.device)

    def calculate_all_metrics(self,
                              pred: torch.Tensor,
                              target: torch.Tensor,
                              frame_batch_size: int = 8) -> Dict[str, torch.Tensor]:
        """
        Compute all video quality assessment metrics.

        Args:
            pred: predicted video, shape (B, T, C, H, W) or (T, C, H, W), value range [-1, 1]
            target: target video, same shape as pred, value range [-1, 1]
            frame_batch_size: number of frames processed per batch to save memory

        Returns:
            Dict containing all metrics

        Example:
            >>> pred = torch.rand(2, 10, 3, 256, 448) * 2 - 1
            >>> target = torch.rand(2, 10, 3, 256, 448) * 2 - 1
            >>> results = metrics.calculate_all_metrics(pred, target)
            >>> print(results)
            {
                'psnr': tensor([25.3, 26.1]),
                'lpips': tensor([0.15, 0.18]),
                'lpips_v': tensor([0.20, 0.22]),
                'dists': tensor([0.12, 0.14])
            }
        """
        results = {
            'psnr': self.calculate_psnr(pred, target),
            'lpips': self.calculate_lpips(pred, target, frame_batch_size),
            'lpips_v': self.calculate_lpips_VGG(pred, target, frame_batch_size),
            'dists': self.calculate_dists(pred, target, frame_batch_size),
        }

        return results

    def calculate_all_metrics_chunked(self,
                                      pred: torch.Tensor,
                                      target: torch.Tensor,
                                      clip_length: int = 24,
                                      frame_batch_size: int = 8) -> Dict[str, torch.Tensor]:
        """
        Compute all video quality assessment metrics in time chunks,
        avoiding loading a long video into GPU memory all at once.

        Args:
            pred: predicted video, shape (B, T, C, H, W) or (T, C, H, W), value range [-1, 1]
            target: target video, same shape as pred, value range [-1, 1]
            clip_length: number of frames per chunk; smaller values reduce peak GPU memory
            frame_batch_size: number of frames processed per batch to save memory

        Returns:
            Dict of metrics with the same keys as calculate_all_metrics
        """
        pred = self._ensure_5d(pred)
        target = self._ensure_5d(target)
        B, T, C, H, W = pred.shape

        mse_sums = [0.0] * B
        mse_counts = [0] * B
        lpips_scores = [[] for _ in range(B)]
        lpips_V_scores = [[] for _ in range(B)]
        dists_scores = [[] for _ in range(B)]
        total_frames = 0

        for start in range(0, T, clip_length):
            end = min(start + clip_length, T)
            n_frames = end - start
            total_frames += n_frames

            # Move only the current chunk to GPU, free it right after use
            pred_chunk = pred[:, start:end].to(self.device)
            target_chunk = target[:, start:end].to(self.device)

            # 1) PSNR: accumulate per-batch MSE so the result matches
            #    computing it over the whole video at once
            for b in range(B):
                diff = pred_chunk[b] - target_chunk[b]
                mse_sums[b] += float((diff ** 2).sum().item())
                mse_counts[b] += diff.numel()

            # 2) LPIPS / DISTS: reuse the existing methods on the current chunk
            chunk_results = self.calculate_all_metrics(
                pred_chunk, target_chunk, frame_batch_size=frame_batch_size
            )

            for b in range(B):
                lpips_scores[b].append(float(chunk_results['lpips'][b].item()) * n_frames)
                lpips_V_scores[b].append(float(chunk_results['lpips_v'][b].item()) * n_frames)
                dists_scores[b].append(float(chunk_results['dists'][b].item()) * n_frames)

            del pred_chunk, target_chunk, chunk_results
            torch.cuda.empty_cache()

        # Aggregate PSNR (weighted MSE -> PSNR)
        psnr_list = []
        for b in range(B):
            mse = mse_sums[b] / max(mse_counts[b], 1)
            mse = max(mse, 1e-10)
            psnr_list.append(10 * torch.log10(torch.tensor((2.0 ** 2) / mse)))

        results = {
            'psnr': torch.stack(psnr_list).to(self.device),
            'lpips': torch.tensor([sum(lpips_scores[b]) / total_frames for b in range(B)], device=self.device),
            'lpips_v': torch.tensor([sum(lpips_V_scores[b]) / total_frames for b in range(B)], device=self.device),
            'dists': torch.tensor([sum(dists_scores[b]) / total_frames for b in range(B)], device=self.device),
        }

        return results


def test_video_metrics():
    """
    Test the video quality assessment metrics.

    Input example:
        - batch_size = 2
        - num frames = 5
        - channels = 3 (RGB)
        - height = 256
        - width = 448
        - value range = [-1, 1]
    """
    print("=" * 60)
    print("Video Quality Metrics Test")
    print("=" * 60)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create the evaluator
    metrics = VideoQualityMetrics(device=device)

    # Generate test data
    # Shape: (B, T, C, H, W), value range: [-1, 1]
    B, T, C, H, W = 2, 5, 3, 256, 448

    print(f"\nGenerated test data:")
    print(f"  - Batch size (B): {B}")
    print(f"  - Num frames (T): {T}")
    print(f"  - Channels (C): {C}")
    print(f"  - Height (H): {H}")
    print(f"  - Width (W): {W}")
    print(f"  - Value range: [-1, 1]")
    print(f"  - Input shape: ({B}, {T}, {C}, {H}, {W})")

    # Original video (reference)
    original_video = torch.rand(B, T, C, H, W) * 2 - 1

    # Compressed video (distorted) - add noise to simulate compression artifacts
    compressed_video = original_video + torch.randn_like(original_video) * 0.1
    # Clamp to [-1, 1]
    compressed_video = torch.clamp(compressed_video, -1, 1)

    print("\n" + "-" * 60)
    print("Computing metrics...")
    print("-" * 60)

    # Compute all metrics
    results = metrics.calculate_all_metrics(compressed_video, original_video)

    # Print results
    print("\nEvaluation results:")
    print("-" * 60)
    for metric_name, value in results.items():
        print(f"  {metric_name.upper()}:")
        print(f"    Batch 0: {value[0].item():.4f}")
        print(f"    Batch 1: {value[1].item():.4f}")
        print(f"    Mean: {value.mean().item():.4f}")

    print("\n" + "=" * 60)
    print("Test finished!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    test_video_metrics()
