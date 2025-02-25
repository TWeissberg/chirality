import torch
from PIL import Image
from torchvision.utils import make_grid
import numpy as np
from diffusion import add_texture_to_render
from dino import get_dino_features
from render import batch_render
from pytorch3d.ops import ball_query
from tqdm import tqdm
from time import time
import random


FEATURE_DIMS = 1280+768 # diffusion unet + dino
VERTEX_GPU_LIMIT = 100000000000


def arange_pixels(
    resolution=(128, 128),
    batch_size=1,
    subsample_to=None,
    invert_y_axis=False,
    margin=0,
    corner_aligned=True,
    jitter=None,
):
    h, w = resolution
    n_points = resolution[0] * resolution[1]
    uh = 1 if corner_aligned else 1 - (1 / h)
    uw = 1 if corner_aligned else 1 - (1 / w)
    if margin > 0:
        uh = uh + (2 / h) * margin
        uw = uw + (2 / w) * margin
        w, h = w + margin * 2, h + margin * 2

    x, y = torch.linspace(-uw, uw, w), torch.linspace(-uh, uh, h)
    if jitter is not None:
        dx = (torch.ones_like(x).uniform_() - 0.5) * 2 / w * jitter
        dy = (torch.ones_like(y).uniform_() - 0.5) * 2 / h * jitter
        x, y = x + dx, y + dy
    x, y = torch.meshgrid(x, y)
    pixel_scaled = (
        torch.stack([x, y], -1)
        .permute(1, 0, 2)
        .reshape(1, -1, 2)
        .repeat(batch_size, 1, 1)
    )

    if subsample_to is not None and subsample_to > 0 and subsample_to < n_points:
        idx = np.random.choice(
            pixel_scaled.shape[1], size=(subsample_to,), replace=False
        )
        pixel_scaled = pixel_scaled[:, idx]

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.0

    return pixel_scaled


def get_features_per_vertex(
    device,
    pipe,
    dino_model,
    mesh,
    prompt,
    num_views=100,
    H=512,
    W=512,
    tolerance=0.01,
    use_latent=False,
    use_normal_map=True,
    num_images_per_prompt=1,
    mesh_vertices=None,
    return_image=True,
    bq=True,
    prompts_list=None,
):
    t1 = time()
    if mesh_vertices is None:
        mesh_vertices = mesh.verts_list()[0]
    if len(mesh_vertices) > VERTEX_GPU_LIMIT:
        samples = random.sample(range(len(mesh_vertices)), 10000)
        maximal_distance = torch.cdist(mesh_vertices[samples], mesh_vertices[samples]).max()
    else:
        maximal_distance = torch.cdist(mesh_vertices, mesh_vertices).max()  # .cpu()
    ball_drop_radius = maximal_distance * tolerance
    batched_renderings, normal_batched_renderings, camera, depth = batch_render(
        device, mesh, mesh.verts_list()[0], num_views, H, W, use_normal_map, focal_length = 2., scaling_factor = 1.
    )
    print("Rendering complete")
    if use_normal_map:
        normal_batched_renderings = normal_batched_renderings.cpu()
    batched_renderings = batched_renderings.cpu()
    pixel_coords = arange_pixels((H, W), invert_y_axis=True)[0]
    pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
    camera = camera.cpu()
    normal_map_input = None
    depth = depth.cpu()
    torch.cuda.empty_cache()
    ft_per_vertex = torch.zeros((len(mesh_vertices), FEATURE_DIMS)).half() #.to(device)
    ft_per_vertex_count = torch.zeros((len(mesh_vertices), 1)).half() #.to(device)
    for idx in tqdm(range(len(batched_renderings))):
        dp = depth[idx].flatten().unsqueeze(1)
        xy_depth = torch.cat((pixel_coords, dp), dim=1)
        indices = xy_depth[:, 2] != -1
        #print(torch.count_nonzero(indices))
        xy_depth = xy_depth[indices]
        world_coords = (
            camera[idx].unproject_points(
                xy_depth, world_coordinates=True, from_ndc=True
            )  # .cpu()
        ).to(device)
        diffusion_input_img = (
            batched_renderings[idx, :, :, :3].cpu().numpy() * 255
        ).astype(np.uint8)
        if use_normal_map:
            normal_map_input = normal_batched_renderings[idx]
        depth_map = depth[idx, :, :, 0].unsqueeze(0).to(device)
        if prompts_list is not None:
            prompt = random.choice(prompts_list)
        # depth_cmp = torch.load("depth.pt")
        # normal_cmp = torch.load("normal.pt")
        # img_cmp = torch.load("inp_image.pt")
        # from torchvision.transforms.functional import to_pil_image, pil_to_tensor
        # print("Depth", torch.max(torch.abs(depth_cmp.cuda() - depth_map)))
        # print("Normal", torch.max(torch.abs(normal_cmp - normal_map_input)))
        # print(torch.max(torch.abs(torch.from_numpy(img_cmp) - torch.from_numpy(diffusion_input_img))))
        # print(torch.count_nonzero(torch.abs(torch.from_numpy(img_cmp) - torch.from_numpy(diffusion_input_img))))
        diffusion_output = add_texture_to_render(
            pipe,
            diffusion_input_img,
            depth_map,
            prompt,
            normal_map_input=normal_map_input,
            use_latent=use_latent,
            num_images_per_prompt=num_images_per_prompt,
            return_image=return_image
        )
        # print(depth_map)
        # print(prompt)
        # print(normal_map_input)
        # print(use_latent)
        # print(num_images_per_prompt)
        # print(return_image)
        aligned_dino_features = get_dino_features(device, dino_model, diffusion_output[1][0], grid)

        # cmp_image_textured = torch.load("/lustre/scratch/data/tweissbe_hpc-becos-all/full_full/train/0/0_images/textured.pt")
        # cmp_image_untextured = torch.load("/lustre/scratch/data/tweissbe_hpc-becos-all/full_full/train/0/0_images/untextured.pt")
        # print(pil_to_tensor(diffusion_output[1][0].convert('RGB')).shape, pil_to_tensor(cmp_image_textured[idx]).shape)
        # print(torch.max(torch.abs(pil_to_tensor(diffusion_output[1][0].convert('RGB')) - pil_to_tensor(cmp_image_textured[idx]))))
        # print(torch.count_nonzero(torch.abs(pil_to_tensor(diffusion_output[1][0].convert('RGB')) - pil_to_tensor(cmp_image_textured[idx]))))
        # print((batched_renderings[idx, :, :, :3].permute(2, 0, 1) * 255).byte().shape, pil_to_tensor(cmp_image_untextured[idx]).shape)
        # print(torch.max(torch.abs((batched_renderings[idx, :, :, :3].permute(2, 0, 1) * 255).byte() - pil_to_tensor(cmp_image_untextured[idx]))))
        # print(torch.count_nonzero(torch.abs((batched_renderings[idx, :, :, :3].permute(2, 0, 1) * 255).byte() - pil_to_tensor(cmp_image_untextured[idx]))))
        # cmp = torch.load("test.pt")
        # print("dino", torch.max(torch.abs(aligned_dino_features -  cmp)))

        # aligned_dino_features_2 = get_dino_features(device, dino_model, cmp_image_textured[idx], grid)
        # print("dino2", torch.max(torch.abs(aligned_dino_features_2 -  aligned_dino_features)))

        # cmp_ft = torch.load("/lustre/scratch/data/tweissbe_hpc-becos-all/full_full/train/0/0_images/generation_sd_features.pt")
        # print("cmp_ft", torch.max(torch.abs(cmp_ft[idx] - diffusion_output[0])))
        # print("cmp_ft", torch.count_nonzero(torch.abs(cmp_ft[idx] - diffusion_output[0]) > 1e-3))

        aligned_features = None
        with torch.no_grad():
            ft = torch.nn.Upsample(size=(H,W), mode="bilinear")(diffusion_output[0].unsqueeze(0)).to(device)
            ft_dim = ft.size(1)
            aligned_features = torch.nn.functional.grid_sample(
                ft, grid, align_corners=False
            ).reshape(1, ft_dim, -1)
            aligned_features = torch.nn.functional.normalize(aligned_features, dim=1)
        # this is feature per pixel in the grid
        aligned_features = torch.hstack([aligned_features*0.5, aligned_dino_features*0.5])

        # cmp_aligned = torch.load(f"diff3f_{idx}.pt")
        # print("cmp_aligned", torch.max(torch.abs(cmp_aligned.cpu() - aligned_features[0].cpu())))
        # print("cmp_aligned", torch.count_nonzero(torch.abs(cmp_aligned.cpu() - aligned_features[0].cpu()) > 1e-3))

        features_per_pixel = aligned_features[0, :, indices].cpu()
        # map pixel to vertex on mesh
        if bq:
            queried_indices = (
                ball_query(
                    world_coords.unsqueeze(0),
                    mesh_vertices.unsqueeze(0),
                    K=100,
                    radius=ball_drop_radius,
                    return_nn=False,
                )
                .idx[0]
                .cpu()
            )
            # print("Nonfound", torch.count_nonzero(torch.count_nonzero(queried_indices > -1, dim = 1) == 0))
            mask = queried_indices != -1
            repeat = mask.sum(dim=1)
            # cmp_repeat = torch.load("repeat.pt")
            # cmp_repeat = cmp_repeat[cmp_repeat > 0]
            # print(len(repeat), len(cmp_repeat))
            # print(repeat, cmp_repeat)
            # print(torch.max(torch.abs(repeat[repeat > 0].cpu() - cmp_repeat.cpu())))
            # pixel_to_vertices = torch.load("pixel_to_vertices.pt")
            # print("Pixel", torch.max(torch.abs(pixel_to_vertices[indices].cpu() - queried_indices.cpu())))
            # print("Pixel", torch.count_nonzero(torch.abs(pixel_to_vertices[indices].cpu() - queried_indices.cpu())))
            # cmp_queried_indices = torch.load(f"p2v_{idx}.pt")
            # cmp_indices = torch.load(f"indices_{idx}.pt")
            # print(cmp_queried_indices.shape, queried_indices.shape)
            # print(indices.shape, cmp_indices.shape)
            # print(torch.max(torch.abs(cmp_queried_indices - queried_indices)))
            # print(torch.max(torch.abs(indices.int() - cmp_indices.int())))

            ft_per_vertex_count[queried_indices[mask]] += 1
            torch.use_deterministic_algorithms(True)
            ft_per_vertex[queried_indices[mask]] += features_per_pixel.repeat_interleave(
                repeat, dim=1
            ).T #.to(device = device)
            torch.use_deterministic_algorithms(False)
        else:
            distances = torch.cdist(
            world_coords, mesh_vertices, p=2
            )
            closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
            ft_per_vertex[closest_vertex_indices] += features_per_pixel.T
            ft_per_vertex_count[closest_vertex_indices] += 1

        # cmp_ft_per_vertex = torch.load(f"ft_per_vertex_{idx}.pt")
        # print(torch.max(torch.abs(ft_per_vertex.cpu() - cmp_ft_per_vertex.cpu())))
        # print(torch.count_nonzero(torch.abs(ft_per_vertex.cpu() - cmp_ft_per_vertex.cpu()) > 1e-3))

    idxs = (ft_per_vertex_count != 0)[:, 0]
    ft_per_vertex[idxs, :] = ft_per_vertex[idxs, :] / ft_per_vertex_count[idxs, :]
    missing_features = len(ft_per_vertex_count[ft_per_vertex_count == 0])
    print("Number of missing features: ", missing_features)
    print("Copied features from nearest vertices")

    if missing_features > 0:
        filled_indices = ft_per_vertex_count[:, 0] != 0
        missing_indices = ft_per_vertex_count[:, 0] == 0
        distances = torch.cdist(
            mesh_vertices[missing_indices], mesh_vertices[filled_indices], p=2
        )
        closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
        ft_per_vertex[missing_indices, :] = ft_per_vertex[filled_indices][
            closest_vertex_indices, :
        ]
    t2 = time() - t1
    t2 = t2 / 60
    print("Time taken in mins: ", t2)
    return ft_per_vertex

if __name__ == "__main__":
    import glob
    import os
    from pathlib import Path
    import sys
    data_path = os.path.expanduser(f"/lustre/scratch/data/tweissbe_hpc-becos/all/{sys.argv[1]}/{sys.argv[2]}/")

    print(data_path)

    mesh_file_0 = glob.glob(data_path + "0_*.off")
    mesh_file_1 = glob.glob(data_path + "1_*.off")
    assert len(mesh_file_0) == 1 and len(mesh_file_1) == 1

    from prompt import get_prompt
    from dataloaders.mesh_container import MeshContainer
    from diffusion import init_pipe
    from dino import init_dino
    mesh_0 = MeshContainer().load_from_file(mesh_file_0[0])
    mesh_1 = MeshContainer().load_from_file(mesh_file_1[0])

    pipe = init_pipe("cuda")
    dino_model = init_dino("cuda")

    from utils import convert_mesh_container_to_torch_mesh

    mesh_0 = convert_mesh_container_to_torch_mesh(mesh_0, device="cuda", is_tosca=False)
    ft_per_vertex = get_features_per_vertex("cuda", pipe, dino_model, mesh_0, get_prompt(mesh_file_0[0]), num_views=100)

    torch.save(ft_per_vertex.cpu(), "./test_ft.pt")