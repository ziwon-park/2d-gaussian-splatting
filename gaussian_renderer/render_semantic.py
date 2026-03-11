"""
Multi-pass semantic rendering using the existing 3-channel 2DGS rasterizer.

Since the diff-surfel-rasterization CUDA kernel is hardcoded to 3 output channels,
we render semantic logits in multiple passes of 3 channels each, using override_color.
This approach requires NO CUDA modifications and is fully differentiable.
"""

import math
import torch
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel


def render_semantic(viewpoint_camera, pc: GaussianModel, pipe, bg_color_rgb: torch.Tensor,
                    scaling_modifier=1.0):
    """
    Render semantic logits map via multi-pass 3-channel rasterization.

    Args:
        viewpoint_camera: Camera object.
        pc: GaussianModel with _semantic_logits initialized.
        pipe: PipelineParams.
        bg_color_rgb: Background color tensor [3,] (used for geometry setup).
        scaling_modifier: Scale modifier.

    Returns:
        dict with:
            'semantic_logits': (C, H, W) raw logits
            'semantic_probs':  (C, H, W) softmax probabilities
            'semantic_labels': (H, W) argmax class labels
    """
    assert pc.get_semantic_logits is not None, "Semantic logits not initialized"

    semantic_logits = pc.get_semantic_logits  # (N, C)
    num_classes = semantic_logits.shape[1]

    # Pad to multiple of 3
    remainder = num_classes % 3
    if remainder != 0:
        pad_size = 3 - remainder
        padding = torch.zeros(semantic_logits.shape[0], pad_size,
                            device=semantic_logits.device, dtype=semantic_logits.dtype)
        semantic_logits_padded = torch.cat([semantic_logits, padding], dim=1)
    else:
        semantic_logits_padded = semantic_logits
        pad_size = 0

    num_passes = semantic_logits_padded.shape[1] // 3
    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)

    rendered_channels = []

    for pass_idx in range(num_passes):
        start_ch = pass_idx * 3
        end_ch = start_ch + 3
        override_color = semantic_logits_padded[:, start_ch:end_ch]  # (N, 3)

        # Use zero background for logit rendering
        bg = torch.zeros(3, device="cuda")

        rendered_pass = _render_pass(
            viewpoint_camera, pc, pipe, bg,
            scaling_modifier, override_color
        )
        rendered_channels.append(rendered_pass)  # (3, H, W)

    # Concatenate all passes
    all_channels = torch.cat(rendered_channels, dim=0)  # (num_passes*3, H, W)

    # Remove padding channels
    if pad_size > 0:
        all_channels = all_channels[:num_classes]

    semantic_probs = torch.softmax(all_channels, dim=0)
    semantic_labels = all_channels.argmax(dim=0)

    return {
        'semantic_logits': all_channels,
        'semantic_probs': semantic_probs,
        'semantic_labels': semantic_labels,
    }


def _render_pass(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color):
    """Single 3-channel render pass with override_color."""
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    ) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=override_color,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return rendered_image  # (3, H, W)
