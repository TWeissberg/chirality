from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings, MeshRasterizer
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer.lighting import PointLights
import torch
import math
import sys
import trimesh
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor)


def get_colored_depth_maps(raw_depths,H,W):
    import matplotlib
    import matplotlib.cm as cm
    cmap = cm.get_cmap('Greys')
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    depth_images = []
    for i in range(raw_depths.size()[0]):
        d = raw_depths[i]
        dmax = torch.max(d) ; dmin = torch.min(d)
        d = (d-dmin)/(dmax-dmin)
        flat_d = d.view(1,-1).cpu().detach().numpy()
        flat_colors = mapper.to_rgba(flat_d)
        depth_colors = np.reshape(flat_colors,(H,W,4))[:,:,:3]
        np_image = depth_colors*255
        np_image = np_image.astype('uint8')
        depth_images.append(np_image)

    return torch.from_numpy(np.stack(depth_images))


@torch.no_grad()
def run_rendering(device, mesh, mesh_vertices, num_views, H, W, add_angle_azi=0, add_angle_ele=0, use_normal_map=False,return_images=True, focal_length = 1, scaling_factor = 0.65):
    pointclouds = Pointclouds(points=[mesh_vertices],features=[torch.ones(mesh_vertices.shape).float().cuda()])
    bbox = pointclouds.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    distance = torch.sqrt((bb_diff * bb_diff).sum())
    distance *= scaling_factor
    steps = int(math.sqrt(num_views))
    end = 360 - 360/steps
    elevation = torch.linspace(start = 0 , end = end , steps = steps).repeat(steps) + add_angle_ele
    azimuth = torch.linspace(start = 0 , end = end , steps = steps)
    azimuth = torch.repeat_interleave(azimuth, steps) + add_angle_azi
    bbox_center = bbox_center.unsqueeze(0)
    rotation, translation = look_at_view_transform(
        dist=distance, azim=azimuth, elev=elevation, device=device, at=bbox_center
    )
    camera = PerspectiveCameras(R=rotation, T=translation, focal_length = focal_length, device=device)

    #rasterizer
    rasterization_settings = PointsRasterizationSettings(
        image_size=H,
        radius = 0.02,
        points_per_pixel = 1,
        bin_size = 0,
        max_points_per_bin = 0
    )

    #render pipeline
    rasterizer = PointsRasterizer(cameras=camera, raster_settings=rasterization_settings)
    camera_centre = camera.get_camera_center()
    batch_renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )
    batch_points = pointclouds.extend(num_views)
    fragments = rasterizer(batch_points)
    raw_depth = fragments.zbuf

    if not return_images:
        return None,None,camera,raw_depth
    else:
        list_depth_images = get_colored_depth_maps(raw_depth,H,W)
        # print(raw_depth.shape)
        # print(list_depth_images_np[0].shape)
        # from PIL import Image
        # Image.fromarray(raw_depth[0, :, :, 0].cpu().numpy().astype('uint8'), mode = "L").save("depth.png")
        # Image.fromarray(list_depth_images_np[0]).save("depth_image.png")
        return list_depth_images,None,camera,raw_depth


def batch_render(device, mesh, mesh_vertices, num_views, H, W, use_normal_map=False,return_images=True, focal_length = 1, scaling_factor = 0.65):
    trials = 0
    add_angle_azi = 0
    add_angle_ele = 0
    while trials < 5:
        try:
            return run_rendering(device, mesh, mesh_vertices, num_views, H, W, add_angle_azi=add_angle_azi, add_angle_ele=add_angle_ele, use_normal_map=use_normal_map,return_images=return_images, focal_length = focal_length, scaling_factor = scaling_factor)
        except torch.linalg.LinAlgError as e:
            trials += 1
            print("lin alg exception at rendering, retrying ", trials)
            add_angle_azi = 0.1#torch.randn(1)  # Make it deterministic
            add_angle_ele = 0.1#torch.randn(1)  # Make it deterministic
            continue
