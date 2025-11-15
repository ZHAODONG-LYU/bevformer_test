
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 debug_bev_query=False,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self._debug_print_count = 0  # rate limit for optional debug
        self.debug_bev_query = debug_bev_query
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)
        indexes = []
        # Distance-based sampling with fixed per-camera budgets (hardware friendly).
        # Radii expressed in BEV cell units (e.g., 200x200 grid).
        R_INNER, R_MID = 40.0, 80.0
        TARGET_QUERIES_PER_CAM = 5000
        KEEP_RATIO_MID = 0.75
        KEEP_RATIO_OUTER = 0.10
        # num_query == bev_h * bev_w, assume square grid for coordinate recovery
        grid_size = int(num_query ** 0.5)
        cx, cy = (grid_size - 1) / 2.0, (grid_size - 1) / 2.0

        for i, mask_per_img in enumerate(bev_mask):
            # keep the same visible index derivation as before
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)

            if index_query_per_img.numel() == 0:
                indexes.append(index_query_per_img)
                continue

            # map linear indices -> (y, x) on BEV grid
            ys = (index_query_per_img // grid_size).to(dtype=torch.float32)
            xs = (index_query_per_img % grid_size).to(dtype=torch.float32)
            dist2 = (xs - cx) ** 2 + (ys - cy) ** 2

            mask_inner = dist2 <= (R_INNER ** 2)
            mask_mid = (dist2 > (R_INNER ** 2)) & (dist2 <= (R_MID ** 2))
            mask_outer = dist2 > (R_MID ** 2)

            idx_inner = index_query_per_img[mask_inner]
            idx_mid = index_query_per_img[mask_mid]
            idx_outer = index_query_per_img[mask_outer]

            dist_inner = dist2[mask_inner]
            dist_mid = dist2[mask_mid]
            dist_outer = dist2[mask_outer]

            # Sort by distance (closer first)
            if idx_inner.numel() > 0:
                order_inner = torch.argsort(dist_inner, descending=False)
                idx_inner = idx_inner[order_inner]
            if idx_mid.numel() > 0:
                order_mid = torch.argsort(dist_mid, descending=False)
                idx_mid = idx_mid[order_mid]
            if idx_outer.numel() > 0:
                order_outer = torch.argsort(dist_outer, descending=False)
                idx_outer = idx_outer[order_outer]

            # Fixed budget allocation
            available_total = index_query_per_img.numel()
            target_keep = min(TARGET_QUERIES_PER_CAM, available_total)
            remaining = target_keep

            # Inner: keep all (priority 1)
            keep_inner = min(idx_inner.numel(), remaining)
            remaining -= keep_inner

            # Mid: keep 75% (priority 2)
            def _preferred_keep(total, ratio):
                if total == 0 or ratio <= 0:
                    return 0
                keep = int(total * ratio)
                if keep == 0 and total > 0:
                    keep = 1
                return min(keep, total)

            keep_mid = min(_preferred_keep(idx_mid.numel(), KEEP_RATIO_MID), remaining)
            remaining -= keep_mid

            # Outer: keep 10% (priority 3)
            keep_outer = min(_preferred_keep(idx_outer.numel(), KEEP_RATIO_OUTER), remaining)
            remaining -= keep_outer

            # Backfill to reach target_keep if still have budget
            if remaining > 0 and idx_mid.numel() > keep_mid:
                extra_mid = min(idx_mid.numel() - keep_mid, remaining)
                keep_mid += extra_mid
                remaining -= extra_mid
            if remaining > 0 and idx_outer.numel() > keep_outer:
                extra_outer = min(idx_outer.numel() - keep_outer, remaining)
                keep_outer += extra_outer
                remaining -= extra_outer
            if remaining > 0 and idx_inner.numel() > keep_inner:
                extra_inner = min(idx_inner.numel() - keep_inner, remaining)
                keep_inner += extra_inner
                remaining -= extra_inner

            # Collect final selection
            selected = []
            if keep_inner > 0:
                selected.append(idx_inner[:keep_inner])
            if keep_mid > 0:
                selected.append(idx_mid[:keep_mid])
            if keep_outer > 0:
                selected.append(idx_outer[:keep_outer])

            if selected:
                sampled_index_query_per_img = torch.cat(selected, dim=0)
            else:
                sampled_index_query_per_img = index_query_per_img.new_empty(0)

            indexes.append(sampled_index_query_per_img)
        max_len = max([len(each) for each in indexes])

        # Optional debug print for effective BEV queries:
        # Enabled when debug_bev_query=True (via cfg), or auto-enabled for NuScenes-mini.
        def _contains_nuscenes_mini(meta):
            try:
                if isinstance(meta, dict):
                    # common fields to check directly first
                    for k in ('data_root', 'ann_file', 'img_filename', 'filename', 'ori_filename'):
                        if k in meta and isinstance(meta[k], str):
                            s = meta[k].lower()
                            if ('nuscenes-mini' in s) or ('v1.0-mini' in s) or ('/mini/' in s) or s.endswith('-mini'):
                                return True
                    # some datasets store version explicitly
                    for k in ('version', 'nus_version', 'dataset_version'):
                        if k in meta and isinstance(meta[k], str):
                            s = meta[k].lower()
                            if ('v1.0-mini' in s) or ('mini' in s):
                                return True
                    for v in meta.values():
                        if _contains_nuscenes_mini(v):
                            return True
                elif isinstance(meta, (list, tuple)):
                    for v in meta:
                        if _contains_nuscenes_mini(v):
                            return True
                elif isinstance(meta, str):
                    s = meta.lower()
                    return ('nuscenes-mini' in s) or ('v1.0-mini' in s) or ('/mini/' in s) or s.endswith('-mini')
            except Exception:
                return False
            return False

        is_mini = False
        try:
            img_metas = kwargs.get('img_metas', None)
            if img_metas is not None and len(img_metas) > 0:
                # img_metas is a list with length == bs
                # check the first meta as representative
                is_mini = _contains_nuscenes_mini(img_metas[0])
        except Exception:
            is_mini = False

        # only print on rank0 (or non-distributed) to ensure visibility with multi-GPU
        def _is_rank0():
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    return dist.get_rank() == 0
            except Exception:
                pass
            return True

        # Unconditional debug printing (rate-limited, rank0 only), for all datasets
        force_debug = True
        if (force_debug and self._debug_print_count < 3 and _is_rank0()):
            try:
                per_cam_total = [int(mask_per_img[0].sum(-1).nonzero().numel()) for mask_per_img in bev_mask]
                per_cam_kept = [int(idx.numel()) for idx in indexes]
                kept_sum = int(sum(per_cam_kept))
                total_sum = int(sum(per_cam_total))
                ratio = (kept_sum / max(total_sum, 1)) if total_sum > 0 else 0.0
                msg = (f"[BEV DEBUG] kept_queries_per_cam={per_cam_kept} / total_per_cam={per_cam_total} "
                       f"=> kept_sum={kept_sum}, total_sum={total_sum}, ratio={ratio:.3f}, max_len_batch={max_len}")
                # stdout/stderr to ensure visibility with DDP, plus logger for logs
                try:
                    sys.stderr.write(msg + "\n")
                    sys.stderr.flush()
                except Exception:
                    pass
                try:
                    logging.getLogger(__name__).warning(msg)
                except Exception:
                    pass
            except Exception:
                pass
            self._debug_print_count += 1

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #

        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output
