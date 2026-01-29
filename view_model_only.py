import sys
import torch
import numpy as np
from scene.gaussian_model import GaussianModel
from argparse import ArgumentParser
from arguments import PipelineParams
from gaussian_renderer import render, network_gui
from utils.image_utils import render_net_image
from scene.cameras import MiniCam
import math

def create_default_camera(width=800, height=600, fov=60):
    """Create a default camera for viewing"""
    fovy = math.radians(fov)
    fovx = fovy * width / height
    znear = 0.01
    zfar = 100.0
    
    # Default view looking at origin from z=3
    world_view_transform = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 3, 1]
    ], dtype=torch.float32).cuda()
    
    # Simple projection matrix
    tan_half_fovy = math.tan(fovy / 2)
    tan_half_fovx = math.tan(fovx / 2)
    
    proj = torch.zeros(4, 4).cuda()
    proj[0, 0] = 1 / tan_half_fovx
    proj[1, 1] = 1 / tan_half_fovy
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = 1
    proj[3, 2] = -(zfar * znear) / (zfar - znear)
    
    full_proj_transform = world_view_transform @ proj
    
    return MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)

def view_model(model_path, pipe, iteration):
    # Load gaussians
    gaussians = GaussianModel(3)  # sh_degree=3
    
    # Find and load the PLY file
    import os
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        # Try to find any iteration
        pc_dir = os.path.join(model_path, "point_cloud")
        if os.path.exists(pc_dir):
            iterations = [d for d in os.listdir(pc_dir) if d.startswith("iteration_")]
            if iterations:
                iteration = int(iterations[-1].split("_")[1])
                ply_path = os.path.join(pc_dir, f"iteration_{iteration}", "point_cloud.ply")
                print(f"Using iteration {iteration}")
    
    print(f"Loading PLY from: {ply_path}")
    gaussians.load_ply(ply_path)
    print(f"Loaded {gaussians.get_xyz.shape[0]} gaussians")
    
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
    
    print("Waiting for viewer connection on port 6009...")
    print("Run SIBR_remoteGaussian_app to connect")
    
    while True:
        with torch.no_grad():
            if network_gui.conn == None:
                network_gui.try_connect(render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(render_pkg, render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0]
                    }
                    network_gui.send(net_image_bytes, model_path, metrics_dict)
                except Exception as e:
                    print(f'Viewer disconnected: {e}')
                    network_gui.conn = None

if __name__ == "__main__":
    parser = ArgumentParser(description="View 2DGS model without dataset")
    pp = PipelineParams(parser)
    parser.add_argument('-m', '--model_path', type=str, required=True, help="Path to trained model")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--iteration', type=int, default=30000)
    args = parser.parse_args(sys.argv[1:])
    
    print(f"Viewing model: {args.model_path}")
    network_gui.init(args.ip, args.port)
    
    view_model(args.model_path, pp.extract(args), args.iteration)
