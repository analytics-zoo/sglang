# SPDX-License-Identifier: Apache-2.0

import itertools
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig


def apply_vision_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)
    return torch.view_as_real(x_complex * freqs).flatten(-2).to(x.dtype)


class OnyxVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dtype: torch.dtype):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=True, dtype=dtype)
        self.k_proj = nn.Linear(dim, dim, bias=True, dtype=dtype)
        self.v_proj = nn.Linear(dim, dim, bias=True, dtype=dtype)
        self.o_proj = nn.Linear(dim, dim, bias=True, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        sequence_length: int,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = x.shape[0] // sequence_length
        shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
        query = apply_vision_rotary_emb(self.q_proj(x).view(shape), freqs_cis)
        key = apply_vision_rotary_emb(self.k_proj(x).view(shape), freqs_cis)
        value = self.v_proj(x).view(shape)
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
        )
        output = output.transpose(1, 2).contiguous().view(x.shape[0], -1)
        return self.o_proj(output)


class OnyxVisionMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dtype: torch.dtype):
        super().__init__()
        self.c_fc = nn.Linear(dim, hidden_dim, bias=True, dtype=dtype)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class OnyxVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, dtype=dtype)
        self.attn = OnyxVisionAttention(dim, num_heads, dtype)
        self.ln_2 = nn.LayerNorm(dim, dtype=dtype)
        self.mlp = OnyxVisionMLP(dim, mlp_hidden_dim, dtype)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, sequence_length, dim = x.shape
        flat = x.view(batch_size * sequence_length, dim)
        flat = flat + self.attn(
            self.ln_1(flat),
            sequence_length,
            freqs_cis,
            attention_mask,
        )
        flat = flat + self.mlp(self.ln_2(flat))
        return flat.view(batch_size, sequence_length, dim)


class OnyxVisionEncoder(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        latent_dim = config.vision_latent_dim
        patch_dim = (
            config.vision_patch_temporal * 3 * config.vision_patch_size**2
        )
        self.latent_dim = latent_dim
        self.patch_size = config.vision_patch_size
        self.patch_temporal = config.vision_patch_temporal
        self.downsample_factor = config.vision_downsample_factor
        self.sparse_attention_factor = config.vision_sparse_attention_factor
        self.pos_emb_grid_h = config.vision_pos_emb_grid_h
        self.pos_emb_grid_w = config.vision_pos_emb_grid_w
        self.head_dim = latent_dim // config.vision_heads

        self.conv1_linear = nn.Linear(
            patch_dim, latent_dim, bias=False, dtype=dtype
        )
        self.positional_embedding_vlm = nn.Parameter(
            torch.zeros(
                config.vision_pos_emb_grid_h
                * config.vision_pos_emb_grid_w,
                latent_dim,
                dtype=dtype,
            )
        )
        self.ln_pre = nn.LayerNorm(latent_dim, dtype=dtype)
        self.transformer = nn.ModuleList(
            [
                OnyxVisionBlock(
                    latent_dim,
                    config.vision_heads,
                    int(config.vision_mlp_ratio * latent_dim),
                    dtype,
                )
                for _ in range(config.vision_layers)
            ]
        )
        self.ln_post = nn.LayerNorm(latent_dim, dtype=dtype)

    def _make_2d_rope(
        self, grid_h: int, grid_w: int, device: torch.device
    ) -> torch.Tensor:
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(
                    0,
                    half_dim,
                    2,
                    dtype=torch.float32,
                    device=device,
                )[: half_dim // 2]
                / half_dim
            )
        )
        idx_h = torch.arange(1, grid_h + 1, dtype=torch.float32, device=device)
        idx_w = torch.arange(1, grid_w + 1, dtype=torch.float32, device=device)
        freq_h = torch.outer(
            idx_h.unsqueeze(1).expand(-1, grid_w).reshape(-1), inv_freq
        )
        freq_w = torch.outer(
            idx_w.unsqueeze(0).expand(grid_h, -1).reshape(-1), inv_freq
        )
        freq = torch.cat([freq_w, freq_h], dim=-1)
        return torch.view_as_complex(
            torch.stack([torch.cos(freq), torch.sin(freq)], dim=-1)
        )

    def _get_pos_emb(
        self, grid_h: int, grid_w: int, device: torch.device
    ) -> torch.Tensor:
        dtype = self.positional_embedding_vlm.dtype
        pos_emb = (
            self.positional_embedding_vlm.view(
                self.pos_emb_grid_h,
                self.pos_emb_grid_w,
                self.latent_dim,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        ys = torch.linspace(
            -1 + 1 / grid_h,
            1 - 1 / grid_h,
            grid_h,
            device=device,
            dtype=dtype,
        )
        xs = torch.linspace(
            -1 + 1 / grid_w,
            1 - 1 / grid_w,
            grid_w,
            device=device,
            dtype=dtype,
        )
        grid = torch.stack(torch.meshgrid(ys, xs, indexing="xy"), dim=-1)
        grid = grid.reshape(-1, 2)[None, None]
        return F.grid_sample(
            pos_emb, grid, mode="bilinear", align_corners=False
        )[0, :, 0, :].T

    def _pixel_shuffle_downsample(
        self, x: torch.Tensor, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        factor = self.downsample_factor
        permutation = torch.arange(grid_h * grid_w, device=x.device)
        permutation = permutation.view(
            grid_h // factor, factor, grid_w // factor, factor
        )
        permutation = permutation.permute(0, 2, 1, 3).reshape(-1)
        flat = x.squeeze(0)[permutation]
        output_tokens = (grid_h // factor) * (grid_w // factor)
        return (
            flat.view(output_tokens, factor * factor, self.latent_dim)
            .permute(0, 2, 1)
            .contiguous()
            .view(output_tokens, self.latent_dim * factor * factor)
            .unsqueeze(0)
        )

    def _get_sparse_permutation(
        self, grid_h: int, grid_w: int, device: torch.device
    ) -> tuple[torch.Tensor, List[int]]:
        block_h = self.pos_emb_grid_h
        block_w = self.pos_emb_grid_w
        pad_h = math.ceil(grid_h / block_h) * block_h
        pad_w = math.ceil(grid_w / block_w) * block_w
        indices = torch.arange(grid_h * grid_w, device=device)
        indices = F.pad(
            indices.view(grid_h, grid_w),
            (0, pad_w - grid_w, 0, pad_h - grid_h),
            value=-1,
        ).flatten()
        indices = indices.view(
            pad_h // block_h, block_h, pad_w // block_w, block_w
        )
        indices = indices.permute(0, 2, 1, 3).reshape(-1)
        valid = (indices != -1).view(-1, block_h * block_w)
        return indices[indices != -1], valid.sum(dim=1).tolist()

    @staticmethod
    def _make_block_diagonal_mask(
        lengths: List[int], device: torch.device
    ) -> torch.Tensor:
        total = sum(lengths)
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        offset = 0
        for length in lengths:
            mask[offset : offset + length, offset : offset + length] = True
            offset += length
        return mask

    @staticmethod
    def _compute_grid_size(
        image_width: int,
        image_height: int,
        patch_size: int,
        max_tokens: int,
    ) -> tuple[int, int, int]:
        patch_h = image_height / patch_size
        patch_w = image_width / patch_size
        ratio = patch_w / patch_h if patch_h > 0 else 1.0
        if patch_h * patch_w > max_tokens:
            patch_h = (max_tokens / ratio) ** 0.5
            patch_w = patch_h * ratio
        candidates = list(
            set(
                itertools.product(
                    [math.floor(patch_h), math.ceil(patch_h)],
                    [math.floor(patch_w), math.ceil(patch_w)],
                )
            )
        )
        candidates = [
            (height, width)
            for height, width in candidates
            if height >= 1
            and width >= 1
            and height * width <= max_tokens
        ]
        if not candidates:
            candidates = [(max(1, round(patch_h)), max(1, round(patch_w)))]
        patch_h, patch_w = min(
            candidates,
            key=lambda size: abs(
                size[0] / size[1] - image_height / image_width
            ),
        )
        return (
            patch_h * patch_size,
            patch_w * patch_size,
            patch_h * patch_w,
        )

    def compute_image_size(
        self, image_width: int, image_height: int
    ) -> tuple[int, int, int]:
        patch_size = self.patch_size * self.downsample_factor
        return self._compute_grid_size(
            image_width, image_height, patch_size, 4096
        )

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        if not images:
            return torch.empty(
                0,
                self.latent_dim * self.downsample_factor**2,
                device=self.conv1_linear.weight.device,
                dtype=self.conv1_linear.weight.dtype,
            )

        device = self.conv1_linear.weight.device
        dtype = self.conv1_linear.weight.dtype
        all_hidden_states = []
        all_freqs = []
        sparse_lengths: List[int] = []
        global_lengths = []
        image_metadata = []

        for image in images:
            image = image.to(device=device, dtype=dtype)
            if image.ndim == 3:
                image = image.unsqueeze(0)
            _, channels, height, width = image.shape
            grid_h, grid_w = (
                height // self.patch_size,
                width // self.patch_size,
            )
            num_tokens = grid_h * grid_w
            if channels != 3:
                raise ValueError(
                    "Onyx SGLang vision currently supports images only; "
                    f"received {channels} channels"
                )

            patches = image.unfold(
                2, self.patch_size, self.patch_size
            ).unfold(3, self.patch_size, self.patch_size)
            patches = patches.contiguous().view(
                1,
                channels,
                grid_h,
                grid_w,
                self.patch_size,
                self.patch_size,
            )
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            patches = patches.unsqueeze(3).expand(
                -1, -1, -1, self.patch_temporal, -1, -1, -1
            )
            hidden_states = self.conv1_linear(
                patches.reshape(1, num_tokens, -1)
            )
            hidden_states = hidden_states + self._get_pos_emb(
                grid_h, grid_w, device
            ).unsqueeze(0).to(dtype)
            hidden_states = self.ln_pre(
                hidden_states.view(-1, self.latent_dim)
            ).view(1, -1, self.latent_dim)
            freqs_cis = self._make_2d_rope(grid_h, grid_w, device)

            sparse_permutation = None
            if self.sparse_attention_factor > 1:
                sparse_permutation, lengths = self._get_sparse_permutation(
                    grid_h, grid_w, device
                )
                hidden_states = hidden_states[:, sparse_permutation]
                freqs_cis = freqs_cis[sparse_permutation]
                sparse_lengths.extend(lengths)

            all_hidden_states.append(hidden_states.squeeze(0))
            all_freqs.append(freqs_cis)
            global_lengths.append(num_tokens)
            image_metadata.append(
                (grid_h, grid_w, num_tokens, sparse_permutation)
            )

        hidden_states = torch.cat(all_hidden_states, dim=0).unsqueeze(0)
        freqs_cis = torch.cat(all_freqs, dim=0)
        global_mask = (
            self._make_block_diagonal_mask(global_lengths, device)
            if len(images) > 1
            else None
        )
        sparse_mask = (
            self._make_block_diagonal_mask(sparse_lengths, device)
            if sparse_lengths
            else None
        )

        for layer_id, block in enumerate(self.transformer):
            is_global = (
                layer_id == len(self.transformer) - 1
                or (layer_id + 1) % self.sparse_attention_factor == 0
            )
            hidden_states = block(
                hidden_states,
                freqs_cis,
                global_mask if is_global or not sparse_lengths else sparse_mask,
            )

        all_features = []
        offset = 0
        for grid_h, grid_w, num_tokens, sparse_permutation in image_metadata:
            image_hidden_states = hidden_states[
                :, offset : offset + num_tokens
            ]
            offset += num_tokens
            if sparse_permutation is not None:
                inverse = torch.empty_like(sparse_permutation)
                inverse[sparse_permutation] = torch.arange(
                    len(sparse_permutation), device=device
                )
                image_hidden_states = image_hidden_states[:, inverse]
            image_hidden_states = self.ln_post(
                image_hidden_states.view(-1, self.latent_dim)
            ).view(1, -1, self.latent_dim)
            all_features.append(
                self._pixel_shuffle_downsample(
                    image_hidden_states, grid_h, grid_w
                ).squeeze(0)
            )
        return torch.cat(all_features, dim=0)


class OnyxVisionAdapter(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.c_fc = nn.Linear(
            config.vision_output_dim,
            config.vision_adapter_dim,
            bias=False,
            dtype=dtype,
        )
        self.c_proj = nn.Linear(
            config.vision_adapter_dim,
            config.vision_adapter_dim,
            bias=False,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.c_proj(F.gelu(self.c_fc(x))))
