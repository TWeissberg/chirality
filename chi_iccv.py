import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt

from dataclasses import dataclass


import os
from PIL import Image
import random

from tqdm import tqdm

import pyvista as pv
from dataloaders.mesh_container import MeshContainer
import torchvision

import argparse
from pathlib import Path
import glob

from prompt import get_prompt

device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs = 2, help="Path to training data and training category")
parser.add_argument("--test", nargs = 2, help="Path to test data and test category")
parser.add_argument("--pretrained", help="Path to pretrained model")
parser.add_argument("--save_path", help="Path where to save resulting model. Ignored if not --train")
args = parser.parse_args()

@dataclass
class Config:
    feature_type: tuple = ("textured_sd_features", "textured_dino_features")
    invertible: bool = False
    lambda_dis: float = 9500**0.5
    lambda_fifty: float = 0.013666331402393064
    lambda_inv: float = 2.3085700200712207
    lambda_l1: float = 276.92404142567483
    new_feature_dim: int = 1
    num_layers: int = 2
    data: list = tuple(args.train) if args.train is not None else None
    seed: int = 0
    activation: str = "norm"
config = Config()

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)
print(config)


def plot_and_save_html(V, F, assignment, name, keep = False):
	plotter = pv.Plotter(shape=(1, 1), off_screen=True)
	F = np.hstack([np.full((F.shape[0], 1), 3), F]).flatten()
	mesh = pv.PolyData(V, F)

	colors = plt.get_cmap("viridis") #create_colormap(assignment) # color map blue vs red per point (colors should have same shape as V) 

	plotter.add_mesh(mesh, scalars=colors(assignment), lighting=True, metallic=False, smooth_shading=True)

	plotter.camera_position = 'xy'
	plotter.link_views()
	plotter.render()
	# path_html = "./plot.html"
	if keep:
		path = name + ".html"
		plotter.export_html(path)
		plotter.close()
	else:
	    plotter.show()

class BecosSymmetryDataset(Dataset):
    def __init__(self, data_dir, feature_type, num = None, category = "all"):
        self.data_dir = data_dir
        pairs = sorted(glob.glob(str(data_dir / "*")))
        self.feature_type = feature_type
        shape_files = []
        feature_files = []
        chirality_files = []
        print(f"Searching {len(pairs)} pairs")
        for path in tqdm(pairs, "Finding shape and feature files 0"):
            shape_file0 = glob.glob(str(Path(path) / "0_*.off"))
            prompt0 = get_prompt(shape_file0[0])
            if (category == "human" and prompt0 != "human") or (category == "animal" and prompt0 == "human"):
                continue
            feature_files0 = [tuple([glob.glob(str(Path(path) / f"0_features/{ft}.pt"))[0] for ft in feature_type])]
            chirality_file0 = glob.glob(str(Path(path) / f"0_chirality.txt")) + glob.glob(str(Path(path) / f"0_annotation.npy"))
            if len(shape_file0) == 1 and len(chirality_file0) == 1:
                assert len(feature_files0[0]) == len(config.feature_type)
                shape_files += shape_file0
                feature_files += feature_files0
                chirality_files += chirality_file0

        for path in tqdm(pairs, "Finding shape and feature files 1"):
            shape_file1 = glob.glob(str(Path(path) / "1_*.off"))
            prompt1 = get_prompt(shape_file1[0])
            if (category == "human" and prompt1 != "human") or (category == "animal" and prompt1 == "human"):
                continue
            feature_files1 = [tuple([glob.glob(str(Path(path) / f"1_features/{ft}.pt"))[0] for ft in feature_type])]
            chirality_file1 = glob.glob(str(Path(path) / f"1_chirality.txt")) + glob.glob(str(Path(path) / f"1_annotation.npy"))
            if len(shape_file1) == 1 and len(chirality_file1) == 1:
                assert len(feature_files1[0]) == len(config.feature_type)
                shape_files += shape_file1
                feature_files += feature_files1
                chirality_files += chirality_file1

        if category == "all":
            assert len(shape_files) == 2 * len(pairs)
        self.shape_files = shape_files[:num] if num is not None else shape_files
        self.feature_files = feature_files[:num] if num is not None else feature_files
        self.chirality_files = chirality_files[:num] if num is not None else chirality_files

    def __len__(self):
        assert len(self.shape_files) == len(self.feature_files)
        return len(self.shape_files)

    def __getitem__(self, idx):
        shape = MeshContainer().load_from_file(self.shape_files[idx])
        features = torch.cat([torch.nn.functional.normalize(torch.load(x), 2) / len(self.feature_files[idx]) for x in self.feature_files[idx]], 2)
        chirality_info = torch.from_numpy(np.loadtxt(self.chirality_files[idx])).bool() if self.chirality_files[idx].endswith(".txt") else torch.from_numpy(np.load(self.chirality_files[idx])).bool()
        return shape, features, chirality_info
    
        
def collate_fn(batch):
    return batch[0]

num = None
if args.train is not None:
    train_path = Path(os.path.expanduser(args.train[0])) / "train"
    train_data = BecosSymmetryDataset(train_path, config.feature_type, category = args.train[1], num = num)
    print("Size of the training data:", len(train_data))
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn = collate_fn)

if args.test is not None:
    test_path = Path(os.path.expanduser(args.test[0])) / "test"
    test_data = BecosSymmetryDataset(test_path, config.feature_type, category = args.test[1], num = num)
    print("Size of the testing data:", len(test_data))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn = collate_fn)
    
    val_path = Path(os.path.expanduser(args.test[0])) / "val"
    val_data = BecosSymmetryDataset(val_path, config.feature_type, category = args.test[1], num = num)
    print("Size of the valdiation data:", len(val_data))
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn = collate_fn)

num_examples = None
target_iterations = 20000

feature_size_dict = {
    "textured_dino_features": 768,
    "textured_clip_features": 1024,
    "textured_sd_features": 3200,
    "untextured_dino_features": 768,
    "untextured_clip_features": 1024,
    "untextured_sd_features": 3200,
}
feature_size = sum(feature_size_dict[ft] for ft in config.feature_type)


A = torch.nn.Parameter((torch.eye(feature_size) + 0.1 * torch.rand(feature_size, feature_size)).to(device = device))
if config.num_layers == 0:
    forward_model = torch.nn.Identity()
    backward_model = torch.nn.Identity()
    optim = torch.optim.Adam([A], lr = 0.001)
else:
    forward_model = torchvision.ops.MLP(feature_size, [feature_size] * config.num_layers).to(device = device)
    backward_model = torchvision.ops.MLP(feature_size, [feature_size] * config.num_layers).to(device = device)
    optim = torch.optim.Adam(list(forward_model.parameters()) + list(backward_model.parameters()) + [A], lr = 0.001)


total_iterations = 1
def run_epoch(forward_model, backward_model, A, loader, optim, num_examples = None, do_evaluate = False):
    avg_loss = 0
    avg_dissimilarity_loss = 0
    avg_similarity_loss = 0
    avg_invertibility_loss = 0
    avg_l1_loss = 0
    avg_fifty_loss = 0
    avg_accuracy = 0

    i = 0
    for mesh, both_features, chirality in tqdm(loader, total = num_examples):
        if num_examples is not None and i > num_examples:
            break
        chirality = chirality.to(device = device)
        both_features = both_features.to(device = device).float()
        both_features = torch.nn.functional.normalize(both_features, dim = -1)
        
        if not config.invertible or config.num_layers == 0:
            forward_features = forward_model(both_features)
            backward_features = backward_model(forward_features)
            backward_features = torch.nn.functional.normalize(backward_features, dim = -1)
            invertibility_loss = torch.linalg.norm(both_features - backward_features) / (both_features.shape[1])**0.5
        else:
            forward_features = torch.stack([forward_model(both_features[0])[0], forward_model(both_features[1])[0]])
            invertibility_loss = torch.tensor([0]).to(device = device)

        transformed_features = forward_features @ A
        transformed_features = torch.nn.functional.normalize(transformed_features, dim = -1)

        chirality_feature, non_chirality_feature = transformed_features[0, :].split([config.new_feature_dim, transformed_features.shape[2] - config.new_feature_dim], -1)
        chirality_feature_flip, non_chirality_feature_flip = transformed_features[1, :].split([config.new_feature_dim, transformed_features.shape[2] - config.new_feature_dim], -1)

        if config.new_feature_dim == 2:
            chirality_feature = torch.softmax(chirality_feature, dim = -1)
            chirality_feature_flip = torch.softmax(chirality_feature_flip, dim = -1)

        dissimilarity_loss = - torch.linalg.norm(chirality_feature - chirality_feature_flip) / (len(chirality_feature)**0.5)
        similarity_loss = torch.linalg.norm(non_chirality_feature - non_chirality_feature_flip)

        if config.new_feature_dim == 1:
            fifty_loss = torch.abs(torch.mean(chirality_feature)) / torch.max(torch.abs(chirality_feature)) + torch.abs(torch.mean(chirality_feature_flip)) / torch.max(torch.abs(chirality_feature_flip))
        elif config.new_feature_dim == 2:
            fifty_loss = torch.linalg.norm(torch.mean(chirality_feature[:, 0]) - torch.mean(chirality_feature[:, 1])) + torch.linalg.norm(torch.mean(chirality_feature_flip[:, 0]) - torch.mean(chirality_feature_flip[:, 1]))
        else:
            raise Exception()

        faces = torch.from_numpy(mesh.face)
        edges = torch.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = torch.unique(torch.sort(edges, dim = -1).values, dim = 0)
        l1_loss = torch.linalg.norm(chirality_feature[edges[:, 0]] - chirality_feature[edges[:, 1]], ord = 1) / len(edges) + torch.linalg.norm(chirality_feature_flip[edges[:, 0]] - chirality_feature_flip[edges[:, 1]], ord = 1) / len(edges)

        loss = config.lambda_dis * dissimilarity_loss \
                + config.lambda_l1 * l1_loss \
                    + config.lambda_inv * invertibility_loss \
                        + config.lambda_fifty * fifty_loss

        # Calculate accuracy
        if config.new_feature_dim == 1:
            assignment = (chirality_feature[:, 0] > 0).flatten()
        elif config.new_feature_dim == 2:
            assignment = chirality_feature[:, 0] > 0.5
        # Our assignment is up to permutation
        accuracy = torch.mean((assignment == chirality).float())#max([torch.mean((assignment == chirality).float()), torch.mean(((~assignment) == chirality).float())])
        if optim is not None:
            loss.backward()
            optim.step()
            optim.zero_grad()

        avg_loss += loss.item()
        avg_dissimilarity_loss += dissimilarity_loss.item()
        avg_similarity_loss += similarity_loss.item()
        avg_invertibility_loss += invertibility_loss.item()
        avg_l1_loss += l1_loss.item()
        avg_fifty_loss += fifty_loss.item()
        avg_accuracy += accuracy.item()

        global total_iterations
        if do_evaluate and total_iterations % 2000 == 0:
            with torch.no_grad():
                for set_name in val_loaders:
                    val_result = run_epoch(forward_model, backward_model, A, val_loaders[set_name], None)
                    val_result = {f"val_{set_name}_{k}": v for k, v in val_result.items()}
                    print(val_result)
        if optim is not None:
            total_iterations += 1
        i = i + 1

        if optim is not None and total_iterations > target_iterations:
            break

    length = i# len(loader) if num_examples is None else min(len(loader), num_examples)

    return {"loss": avg_loss /length,
    "dissimilarity_loss": avg_dissimilarity_loss / length,
    "similarity_loss": avg_similarity_loss / length,
    "invertibility_loss": avg_invertibility_loss / length,
    "l1_loss": avg_l1_loss / length,
    "fifty_loss": avg_fifty_loss / length,
    "accuracy": max(avg_accuracy /length, 1 - avg_accuracy / length)}

def visualize(forward_model, backward_model, A, loader, num = 4, keep = True, harsh = False):
    plotter = pv.Plotter(shape=(1, num), off_screen=True)
    for i, dat in enumerate(loader):
        if i >= num:
            break

        mesh, both_features, chirality = dat
        both_features = both_features.to(device = device).float()
        both_features = torch.nn.functional.normalize(both_features, dim = -1)

        if not config.invertible or config.num_layers == 0:
            forward_features = forward_model(both_features)
        else:
            forward_features = torch.stack([forward_model(both_features[0])[0], forward_model(both_features[1])[0]])

        transformed_features = forward_features @ A
        transformed_features = torch.nn.functional.normalize(transformed_features, dim = -1)

        chirality_feature, _ = transformed_features[0, :].split([config.new_feature_dim, transformed_features.shape[2] - config.new_feature_dim], -1)

        chirality_feature = chirality_feature[:, 0].detach().cpu().numpy()
        if harsh:
            if config.new_feature_dim == 1:
                chirality_feature = chirality_feature > 0
            else:
                chirality_feature = chirality_feature > 0.5

        V, F = mesh.vert, mesh.face
        F = np.hstack([np.full((F.shape[0], 1), 3), F]).flatten()
        mesh = pv.PolyData(V, F)

        colors = plt.get_cmap("viridis")

        plotter.subplot(0, i)
        plotter.add_mesh(mesh, scalars=colors(chirality_feature))#, lighting=True, metallic=False, smooth_shading=True)

    plotter.camera_position = 'xy'
    plotter.render()
    if keep:
        path = "test.html"
        plotter.export_html(path)
        plotter.close()
    else:
        plotter.show()

if args.pretrained is not None:
    A = torch.load(os.path.join(args.pretrained, "A.pt"))
    forward_model.load_state_dict(torch.load(os.path.join(args.pretrained, "forward.pt")))
    if os.path.exists(os.path.join(args.pretrained, "backward.pt")):
        backward_model.load_state_dict(torch.load(os.path.join(args.pretrained, "backward.pt")))

if args.train is not None:
    while total_iterations < target_iterations:
        train_result = run_epoch(forward_model, backward_model, A, train_loader, optim, num_examples, do_evaluate = True)
        train_result = {"train_"+k: v for k, v in train_result.items()}
        print(total_iterations, train_result)


if args.test is not None:
    with torch.no_grad():
        test_result = run_epoch(forward_model, backward_model, A, test_loader, None)
        test_result = {f"test_{k}": v for k, v in test_result.items()}
        print(test_result)
    
    
        # visualize(forward_model, backward_model, A, test_loader, keep = False)
        # visualize(forward_model, backward_model, A, test_loader, keep = False, harsh = True)

        with torch.no_grad():
            val_result = run_epoch(forward_model, backward_model, A, val_loader, None)
        val_result = {"val_"+k: v for k, v in val_result.items()}
        print(val_result)

if args.save_path is not None and args.train is not None:
    torch.save(A, os.path.join(args.save_path, "A.pt"))
    torch.save(forward_model.state_dict(), os.path.join(args.save_path, "forward.pt"))
    if backward_model is not None:
        torch.save(backward_model.state_dict(), os.path.join(args.save_path, "backward.pt"))