import torch

import os
from PIL import Image
import random

from torchvision import transforms as tfs

from dino import init_dino, get_dino_features
from diff3f import VERTEX_GPU_LIMIT, arange_pixels
from transformers import CLIPProcessor, CLIPVisionModel

from tqdm import tqdm

from dataloaders.mesh_container import MeshContainer

import glob

from diffusion import init_pipe
from pathlib import Path

from pytorch3d.ops import ball_query
from utils import convert_mesh_container_to_torch_mesh

from stablediffusion import *
H = 512
W = 512

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
#device_id = "cuda:0" if use_cuda else "cpu"
device_id = "cuda"
lr = 5e-4
num_images = 100


def init_clip(device_id):
	model = CLIPVisionModel.from_pretrained(
		"openai/clip-vit-large-patch14",
		#attn_implementation="flash_attention_2",
		device_map=device_id,
		#torch_dtype=torch.float,
	)

	model = model.eval()
	return model

@torch.no_grad
def get_clip_features(device, clip_model, img, grid):
	transform = tfs.Compose(
		[
			tfs.Resize((224, 224)),
			tfs.ToTensor(),
			tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		]
	)
	img = transform(img)[:3].unsqueeze(0).to(device)
	features = clip_model(pixel_values = img, output_hidden_states = True)
	#print(len(features["hidden_states"]))
	features = features["hidden_states"][-1][:, :-1].half()
	patch_size = 14
	h, w = int(img.shape[2] / patch_size), int(img.shape[3] / patch_size)

	dim = features.shape[-1]
	features = features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
	features = torch.nn.functional.grid_sample(
		features, grid, align_corners=False
	).reshape(1, dim, -1)
	features = torch.nn.functional.normalize(features, dim=1)
	return features

@torch.no_grad
def get_sd_features(device, sd_model, img, grid):
    features = process_features_and_mask(sd_model[0], sd_model[1], img, mask = True, raw = True)
    del features["s2"]
    feature_list = []
    for k in features.keys():
        feature_list.append(
            torch.nn.functional.grid_sample(features[k].half(), grid, align_corners = False)
        )
        # print(features[k].shape)
    features = torch.cat(feature_list, dim = 1)
    dim = features.shape[1]
    features = features.reshape(1, dim, -1)
    features = torch.nn.functional.normalize(features, dim=1)
    return features

def calculate_pixel_to_vertices(mesh, cameras, depth, tolerance = 0.01, bq = True):
    mesh_vertices = mesh.verts_list()[0]
    if len(mesh_vertices) > 15000:#VERTEX_GPU_LIMIT:
        #assert False
        samples = random.sample(range(len(mesh_vertices)), 10000)
        maximal_distance = torch.cdist(mesh_vertices[samples], mesh_vertices[samples]).max()
    else:
        maximal_distance = torch.cdist(mesh_vertices, mesh_vertices).max()  # .cpu()
    ball_drop_radius = maximal_distance * tolerance
    pixel_coords = arange_pixels((H, W), invert_y_axis=True)[0]
    pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
    pixel_to_vertices = []
    for idx in tqdm(range(len(depth))):
        dp = depth[idx].flatten().unsqueeze(1)
        xy_depth = torch.cat((pixel_coords, dp), dim=1)
        indices = xy_depth[:, 2] != -1
        #xy_depth = xy_depth[indices]
        world_coords = (
            cameras[idx].unproject_points(
                xy_depth, world_coordinates=True, from_ndc=True
            )  # .cpu()
        ).to(device_id)
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
            queried_indices[~indices] = -1
            pixel_to_vertices.append(queried_indices)
        else:
            distances = torch.cdist(
            world_coords, mesh_vertices, p=2
            )
            queried_indices = torch.argmin(distances, dim=1).cpu().unsqueeze(1)
            queried_indices[~indices] = -1
            pixel_to_vertices.append(queried_indices)
    return torch.stack(pixel_to_vertices)


def generate_features(shape_path, dino_model, clip_model, sd_model):
    mesh = MeshContainer().load_from_file(shape_path)
    #torch.use_deterministic_algorithms(True)
    import open3d as o3d

    # vertices = mesh.vert - np.mean(mesh.vert, 0)
    # R = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(np.array([90, 0, 0])))
    # vertices = np.dot(vertices, R)
    # assert np.allclose(np.mean(vertices, 0), np.array([0., 0., 0.]))

    # mesh = MeshContainer(vertices, mesh.face)

    mesh = convert_mesh_container_to_torch_mesh(mesh, device = device_id, is_tosca = False)

    index = shape_path.split("/")[-1][0] 
    main_path = "/".join(shape_path.split("/")[:-1]) + "/"
    images_path = main_path + f"{index}_images/"
    # images_path = mesh_file[:-4] + "_images/"
    # untextured_images = torch.load(images_path + "untextured.pt")
    textured_images = torch.load(images_path + "textured.pt")
    cameras = torch.load(images_path + "cameras.pt")
    depth = torch.load(images_path + "depth.pt")
    sd_creation_f = torch.load(images_path + "generation_sd_features.pt")


    # pixel_to_vertex, indices = calculate_pixel_to_vertices(mesh, cameras, depth, bq = True)
    queried_indices = calculate_pixel_to_vertices(mesh, cameras, depth, bq = True)


    hflip = tfs.functional.hflip

    def hflip_indices(x):
        # Input : torch.Size([9, 262144, 100])
        n = x.shape[0]
        return hflip(x.permute(0, 2, 1).reshape(n, -1, H, W)).reshape(n, -1, H*W).permute(0, 2, 1)

    diff3f_features, _ = features_diff3f(mesh, textured_images, queried_indices, dino_model, sd_creation_f)

    # DEBUG
    ##############################################
    if random.randint(0, 999) == 0:
        from diff3f import get_features_per_vertex

        pipe = init_pipe("cuda")
        # dino_model = init_dino("cuda")

        from prompt import get_prompt
        diff3f_compare = get_features_per_vertex("cuda", pipe, dino_model, mesh, get_prompt(shape_path))

        print("Sanity", diff3f_features.shape, diff3f_compare.shape)
        print(torch.max(torch.abs(diff3f_compare.cpu() - diff3f_features.cpu())))
        print(torch.count_nonzero(torch.abs(diff3f_compare.cpu() - diff3f_features.cpu()) > 1e-3))
    ##############################################


    # shape_features, num_features = features_from_images(mesh, untextured_images, queried_indices, get_dino_features, dino_model, dino_feature_dim)
    # shape_features_flip, num_features_flip = features_from_images(mesh, [hflip(x) for x in untextured_images], hflip_indices(queried_indices), get_dino_features, dino_model, dino_feature_dim)
    # untextured_dino_features = torch.stack([shape_features, shape_features_flip])
    # assert (num_features == num_features_flip).all()

    shape_features, num_features = features_from_images(mesh, textured_images, queried_indices, get_dino_features, dino_model, dino_feature_dim)
    shape_features_flip, num_features_flip = features_from_images(mesh, [hflip(x) for x in textured_images], hflip_indices(queried_indices), get_dino_features, dino_model, dino_feature_dim)
    textured_dino_features = torch.stack([shape_features, shape_features_flip])
    assert (num_features == num_features_flip).all()

    # shape_features, num_features = features_from_images(mesh, untextured_images, queried_indices, get_clip_features, clip_model, clip_feature_dim)
    # shape_features_flip, num_features_flip = features_from_images(mesh, [hflip(x) for x in untextured_images], hflip_indices(queried_indices), get_clip_features, clip_model, clip_feature_dim)
    # untextured_clip_features = torch.stack([shape_features, shape_features_flip])
    # assert (num_features == num_features_flip).all()

    # shape_features, num_features = features_from_images(mesh, textured_images, queried_indices, get_clip_features, clip_model, clip_feature_dim)
    # shape_features_flip, num_features_flip = features_from_images(mesh, [hflip(x) for x in textured_images], hflip_indices(queried_indices), get_clip_features, clip_model, clip_feature_dim)
    # textured_clip_features = torch.stack([shape_features, shape_features_flip])
    # assert (num_features == num_features_flip).all()

    # shape_features, num_features = features_from_images(mesh, untextured_images, queried_indices, get_sd_features, sd_model, sd_feature_dim)
    # shape_features_flip, num_features_flip = features_from_images(mesh, [hflip(x) for x in untextured_images], hflip_indices(queried_indices), get_sd_features, sd_model, sd_feature_dim)
    # untextured_sd_features = torch.stack([shape_features, shape_features_flip])
    # assert (num_features == num_features_flip).all()

    shape_features, num_features = features_from_images(mesh, textured_images, queried_indices, get_sd_features, sd_model, sd_feature_dim)
    shape_features_flip, num_features_flip = features_from_images(mesh, [hflip(x) for x in textured_images], hflip_indices(queried_indices), get_sd_features, sd_model, sd_feature_dim)
    textured_sd_features = torch.stack([shape_features, shape_features_flip])
    assert (num_features == num_features_flip).all()
    

    features = {
          "diff3f_features": diff3f_features,
        #   "untextured_dino_features": untextured_dino_features,
          "textured_dino_features": textured_dino_features,
        #   "untextured_clip_features": untextured_clip_features,
        #   "textured_clip_features": textured_clip_features,
        #   "untextured_sd_features": untextured_sd_features,
          "textured_sd_features": textured_sd_features,
          "num_features": num_features
    }

    # feature_path = mesh_file[:-4] + "_features/"
    feature_path = main_path + f"/{index}_features/"

    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    for k in features.keys():
        torch.save(features[k], feature_path + f"{k}.pt")

    
    
def features_diff3f(mesh, images, queried_indices, dino_model, sd_creation_features):
    mesh_vertices = mesh.verts_list()[0]
    num_vert = len(mesh_vertices)
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device_id).reshape(1, H, W, 2).half()
    #pixel_to_vertices = pixel_to_vertices.to(device = device_id)

    feature_dim = 2048
    ft_per_vertex = torch.zeros(num_vert, feature_dim).half().to(device = device_id)
    ft_per_vertex_count = torch.zeros(num_vert, 1).half().to(device = device_id)

    for n in tqdm(range(len(images)), "Generating features"):
        p2v = queried_indices[n]#.to(device = device_id)
        indices = torch.count_nonzero(queried_indices[n] != -1, dim = 1) > 0
        p2v = p2v[indices]
        _feature = get_dino_features("cuda", dino_model, images[n], grid.to(device = device_id))
        ft = torch.nn.Upsample(size=(H,W), mode="bilinear")(sd_creation_features[n].unsqueeze(0)).to(device_id)
        ft_dim = ft.size(1)
        sd_features = torch.nn.functional.grid_sample(
            ft, grid, align_corners=False
        ).reshape(1, ft_dim, -1)
        sd_features = torch.nn.functional.normalize(sd_features, dim=1)
        diff3f_features = torch.hstack([sd_features * 0.5, _feature*0.5]).squeeze(0)
        diff3f_features = diff3f_features[:, indices]
        mask = p2v != -1
        repeat = mask.sum(dim=1)
        ft_per_vertex_count[p2v[mask]] += 1
        torch.use_deterministic_algorithms(True)
        ft_per_vertex[p2v[mask]] += diff3f_features.repeat_interleave(
            repeat.to(device = device_id), dim=1
        ).T#.cpu()
        torch.use_deterministic_algorithms(False)

    idxs = (ft_per_vertex_count != 0)[:, 0]
    ft_per_vertex[idxs, :] = ft_per_vertex[idxs, :] / ft_per_vertex_count[idxs, :]

    missing_features = len(ft_per_vertex_count[ft_per_vertex_count == 0])

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
    return ft_per_vertex.cpu(), ft_per_vertex_count.cpu()


def features_from_images(mesh, images, queried_indices, get_features, model, feature_dim):
    mesh_vertices = mesh.verts_list()[0]
    num_vert = len(mesh_vertices)
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device_id).reshape(1, H, W, 2).half()

    ft_per_vertex = torch.zeros(num_vert, feature_dim).half().to(device = device_id)
    ft_per_vertex_count = torch.zeros(num_vert, 1).half().to(device = device_id)

    for n in tqdm(range(len(images)), "Generating features"):
        p2v = queried_indices[n]#.to(device = device_id)
        indices = torch.count_nonzero(queried_indices[n] != -1, dim = 1) > 0
        p2v = p2v[indices]
        _feature = get_features("cuda", model, images[n], grid.to(device = device_id)).squeeze(0)
        # _feature = torch.nn.functional.normalize(_feature, dim=0) #FIXME: Normalize or not
        assert _feature.shape[1] == H * W
        _feature = _feature[:, indices]
        mask = p2v != -1
        repeat = mask.sum(dim=1)
        ft_per_vertex_count[p2v[mask]] += 1
        torch.use_deterministic_algorithms(True)
        ft_per_vertex[p2v[mask]] += _feature.repeat_interleave(
            repeat.to(device = device_id), dim=1
        ).T
        torch.use_deterministic_algorithms(False)

    idxs = (ft_per_vertex_count != 0)[:, 0]
    ft_per_vertex[idxs, :] = ft_per_vertex[idxs, :] / ft_per_vertex_count[idxs, :]

    missing_features = len(ft_per_vertex_count[ft_per_vertex_count == 0])

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
    return ft_per_vertex.cpu(), ft_per_vertex_count.cpu()

if __name__ == "__main__":
    import sys
    dino_model = init_dino("cuda")
    dino_feature_dim = 768
    clip_model = init_clip("cuda")
    clip_feature_dim = 1024
    sd_model = load_model(diffusion_ver='v1-5', image_size=512, num_timesteps=50, block_indices=[2,5,8,11])
    sd_feature_dim = 3200

    batch_size = 5

    for i in tqdm(range(batch_size)):
        data_path = os.path.expanduser(f"/lustre/scratch/data/tweissbe_hpc-becos-all/{sys.argv[1]}/{sys.argv[2]}/{int(sys.argv[3])*batch_size + i}/")
        print("Working on", data_path)

        mesh_file_0 = glob.glob(data_path + "0_*.off")
        mesh_file_1 = glob.glob(data_path + "1_*.off")
        assert len(mesh_file_0) == 1 and len(mesh_file_1) == 1
        #print(type(sd_model[0]))
        with torch.no_grad():
            generate_features(mesh_file_0[0], dino_model, clip_model, sd_model)
            generate_features(mesh_file_1[0], dino_model, clip_model, sd_model)

# if __name__ == "__main__":
#     import sys
#     dino_model = init_dino("cuda")
#     dino_feature_dim = 768
#     clip_model = init_clip("cuda")
#     clip_feature_dim = 1024
#     sd_model = load_model(diffusion_ver='v1-5', image_size=512, num_timesteps=50, block_indices=[2,5,8,11])
#     sd_feature_dim = 3200

#     batch_size = 1

#     data_path = os.path.expanduser(f"/lustre/scratch/data/tweissbe_hpc-becos/{sys.argv[1]}/")

#     mesh_files = glob.glob(data_path + "*.off")
#     mesh_files.sort()

#     mesh_file = mesh_files[int(sys.argv[2])]

#     #mesh = MeshContainer().load_from_file(mesh_file)

#     generate_features(mesh_file, dino_model, clip_model, sd_model)

#     # for i in tqdm(range(batch_size)):
#     #     data_path = os.path.expanduser(f"/lustre/scratch/data/tweissbe_hpc-becos/{sys.argv[1]}/{sys.argv[2]}/{int(sys.argv[3])*batch_size + i}/")

#     #     mesh_file_0 = glob.glob(data_path + "0_*.off")
#     #     mesh_file_1 = glob.glob(data_path + "1_*.off")
#     #     assert len(mesh_file_0) == 1 and len(mesh_file_1) == 1
#     #     #print(type(sd_model[0]))
#     #     with torch.no_grad():
#     #         generate_features(mesh_file_0[0], dino_model, clip_model, sd_model)
#     #         generate_features(mesh_file_1[0], dino_model, clip_model, sd_model)