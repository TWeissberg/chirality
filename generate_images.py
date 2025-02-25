import torch
from diff3f import get_features_per_vertex
from time import time
from utils import convert_mesh_container_to_torch_mesh, cosine_similarity, double_plot, get_colors, generate_colors
from dataloaders.mesh_container import MeshContainer
from diffusion import init_pipe
from dino import init_dino
from functional_map import compute_surface_map
from dino import get_dino_features
from pytorch3d.ops import ball_query, knn_points
import pytorch3d

import open3d as o3d
import PIL

device = torch.device('cuda')
#torch.cuda.set_device(device)
num_views = 100   #FIXME: Should be 100
H = 512
W = 512
num_images_per_prompt = 1
tolerance = 0.01
random_seed = 42
use_normal_map = True   #?

import random
from diff3f import VERTEX_GPU_LIMIT, arange_pixels
from render import batch_render
from tqdm import tqdm
from diffusion import add_texture_to_render
import numpy as np
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from torchvision import transforms as tfs

def generate_images(device, pipe, m, prompt):
    mesh = convert_mesh_container_to_torch_mesh(m, device=device, is_tosca=False)
    mesh_vertices = mesh.verts_list()[0]
    if mesh_vertices is None:
        mesh_vertices = mesh.verts_list()[0]
    if len(mesh_vertices) > 15000:#VERTEX_GPU_LIMIT:
        #assert False
        samples = random.sample(range(len(mesh_vertices)), 10000)
        maximal_distance = torch.cdist(mesh_vertices[samples], mesh_vertices[samples]).max()
    else:
        maximal_distance = torch.cdist(mesh_vertices, mesh_vertices).max()  # .cpu()
    ball_drop_radius = maximal_distance * tolerance
    batched_renderings, normal_batched_renderings, camera, depth = batch_render(
        device, mesh, mesh.verts_list()[0], num_views, H, W, use_normal_map, focal_length = 2., scaling_factor = 1.
    )
    
    if use_normal_map:
        normal_batched_renderings = normal_batched_renderings.cpu()
    batched_renderings = batched_renderings.cpu()
    pixel_coords = arange_pixels((H, W), invert_y_axis=True)[0]
    pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
    camera = camera.cpu()
    normal_map_input = None
    depth = depth.cpu()
    torch.cuda.empty_cache()

    textured_images = []
    sd_creation_features = []
    for idx in tqdm(range(len(batched_renderings)), disable = False):
        print(camera[idx].get_camera_center())
        diffusion_input_img = (
            batched_renderings[idx, :, :, :3].cpu().numpy() * 255
        ).astype(np.uint8)
        if use_normal_map:
            normal_map_input = normal_batched_renderings[idx]
        depth_map = depth[idx, :, :, 0].unsqueeze(0).to(device)
        diffusion_output = add_texture_to_render(
            pipe,
            diffusion_input_img,
            depth_map,
            prompt,
            normal_map_input=normal_map_input,
            use_latent=False,
            num_images_per_prompt=num_images_per_prompt,
            return_image=True
        )

        sd_creation_features.append(diffusion_output[0])
        textured_images.append(diffusion_output[1][0].convert('RGB'))
    sd_creation_features = torch.stack(sd_creation_features)
    return [to_pil_image(x.permute(2, 0, 1)).convert('RGB') for x in batched_renderings], textured_images, camera, depth, sd_creation_features


# source_path = Path("./data/")

if __name__ == "__main__":
    # import sys
    # n = sys.argv[1]
    # print("Generating for", sys.argv[1])
    # source_mesh = MeshContainer().load_from_file(Path("./meshes") / (sys.argv[1] + ".obj"))

    # untextured_images, textured_images, pixel_to_vertex = generate_images(device, pipe, source_mesh, "cat")#sys.argv[1])

    # for i, im in enumerate(textured_images):
    #     im.save("./images/" + str(i) + ".png")

    import glob
    import os
    from pathlib import Path
    import sys
    data_path = os.path.expanduser(f"/lustre/scratch/data/tweissbe_hpc-becos-all/{sys.argv[1]}/{sys.argv[2]}/{sys.argv[3]}/")

    print(data_path)

    mesh_file_0 = glob.glob(data_path + "0_*.off")
    mesh_file_1 = glob.glob(data_path + "1_*.off")
    assert len(mesh_file_0) == 1 and len(mesh_file_1) == 1

    from prompt import get_prompt
    mesh_0 = MeshContainer().load_from_file(mesh_file_0[0])
    mesh_1 = MeshContainer().load_from_file(mesh_file_1[0])

    pipe = init_pipe(device)

    def check_or_save(data, path):
        if os.path.exists(path):  # Debug to check if deterministic
            y = torch.load(path)
            if not isinstance(data, list):
                data = [data]
                y = [y]
            for i in range(len(data)):
                if isinstance(data[i], PIL.Image.Image):
                    diff = pil_to_tensor(data[i]) - pil_to_tensor(y[i])
                    print(path, ":", torch.count_nonzero(diff))
                    #assert torch.count_nonzero(diff) <= 2
                elif isinstance(data[i], pytorch3d.renderer.cameras.PerspectiveCameras):
                    print(path, ":", torch.allclose(data[i].get_camera_center(), y[i].get_camera_center()))
                    # assert torch.allclose(data[i].get_camera_center(), y[i].get_camera_center())
                else:
                    # print(torch.max(torch.abs(data[i] - y[i])))
                    print(path, ":", torch.allclose(data[i], y[i]))
                    # assert torch.allclose(data[i], y[i])
        else:
            torch.save(data, path)

    untextured_images, textured_images, cameras, depth, sd_creation_features = generate_images(device, pipe, mesh_0, get_prompt(mesh_file_0[0]))
    folder_path = data_path + "0_images/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # torch.save(untextured_images, folder_path + "untextured.pt")
    # torch.save(textured_images, folder_path + "textured.pt")
    # torch.save(cameras, folder_path + "cameras.pt")
    # torch.save(depth, folder_path + "depth.pt")
    # torch.save(sd_creation_features, folder_path + "generation_sd_features.pt")
    check_or_save(cameras, folder_path + "cameras.pt")
    check_or_save(depth, folder_path + "depth.pt")
    check_or_save(untextured_images, folder_path + "untextured.pt")
    check_or_save(textured_images, folder_path + "textured.pt")
    check_or_save(sd_creation_features, folder_path + "generation_sd_features.pt")
    untextured_images, textured_images, cameras, depth, sd_creation_features = generate_images(device, pipe, mesh_1, get_prompt(mesh_file_1[0]))
    folder_path = data_path + "1_images/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # torch.save(untextured_images, folder_path + "untextured.pt")
    # torch.save(textured_images, folder_path + "textured.pt")
    # torch.save(cameras, folder_path + "cameras.pt")
    # torch.save(depth, folder_path + "depth.pt")
    # torch.save(sd_creation_features, folder_path + "generation_sd_features.pt")
    check_or_save(cameras, folder_path + "cameras.pt")
    check_or_save(depth, folder_path + "depth.pt")
    check_or_save(untextured_images, folder_path + "untextured.pt")
    check_or_save(textured_images, folder_path + "textured.pt")
    check_or_save(sd_creation_features, folder_path + "generation_sd_features.pt")


# if __name__ == "__main__":
#     # import sys
#     # n = sys.argv[1]
#     # print("Generating for", sys.argv[1])
#     # source_mesh = MeshContainer().load_from_file(Path("./meshes") / (sys.argv[1] + ".obj"))

#     # untextured_images, textured_images, pixel_to_vertex = generate_images(device, pipe, source_mesh, "cat")#sys.argv[1])

#     # for i, im in enumerate(textured_images):
#     #     im.save("./images/" + str(i) + ".png")

#     import glob
#     import os
#     from pathlib import Path
#     import sys
#     import re

#     def get_prompt(mesh_file):
#         file_name = mesh_file.split("/")[-1][:-4]
#         object_name = re.sub(r'[0-9]+', '', file_name)
#         return object_name
    
#     # def get_prompt(mesh_file):
#     #     file_name = mesh_file.split("/")[-1][:-4]
#     #     return {
#     #         "tr_reg_095": "human",
#     #         "tr_reg_097": "human",
#     #         "tr_reg_098": "human",
#     #         "tr_reg_099": "human",
#     #         "banana": "banana",
#     #         "car": "car",
#     #         "chair": "chair",
#     #         "hammer": "hammer",
#     #         "simple_mesh": "banana",
#     #         "cat-4": "cat",
#     #         "cat-23": "cat",
#     #         "david-10": "human",
#     #         "horse-11": "horse",
#     #         "michael-34": "human",
#     #     }[file_name]

#     data_path = os.path.expanduser(f"/lustre/scratch/data/tweissbe_hpc-becos/{sys.argv[1]}/")

#     mesh_files = glob.glob(data_path + "*.off")
#     mesh_files.sort()

#     mesh_file = mesh_files[int(sys.argv[2])]

#     print(data_path, get_prompt(mesh_file))

#     mesh = MeshContainer().load_from_file(mesh_file)

#     # vertices = mesh.vert - np.mean(mesh.vert, 0)
#     # R = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(np.array([90, 0, 0])))
#     # vertices = np.dot(vertices, R)
#     # assert np.allclose(np.mean(vertices, 0), np.array([0., 0., 0.]))

#     # mesh = MeshContainer(vertices, mesh.face)

#     print(mesh.vert.shape, np.mean(mesh.vert, axis = 0))

#     mesh.vert

#     pipe = init_pipe(device)

#     untextured_images, textured_images, cameras, depth, sd_creation_features = generate_images(device, pipe, mesh, get_prompt(mesh_file))
#     folder_path = mesh_file[:-4] + "_images/"
#     print(folder_path)
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#     torch.save(untextured_images, folder_path + "untextured.pt")
#     torch.save(textured_images, folder_path + "textured.pt")
#     torch.save(cameras, folder_path + "cameras.pt")
#     torch.save(depth, folder_path + "depth.pt")
#     torch.save(sd_creation_features, folder_path + "generation_sd_features.pt")