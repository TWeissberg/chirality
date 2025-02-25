import torch
from torch import nn
import torchvision
    
class ChiralityInvertible(nn.Module):
	def __init__(self, feature_size, num_layers):
		super(ChiralityInvertible, self).__init__()
		import FrEIA.framework as Ff
		import FrEIA.modules as Fm
		from torch import nn

		def subnet_fc(c_in, c_out):
			return nn.Sequential(nn.Linear(c_in, feature_size), nn.ReLU(),
								nn.Linear(feature_size,  c_out))

		forward_model = Ff.SequenceINN(feature_size)
		for k in range(num_layers):
			forward_model.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)    #?
		self.forward_model = forward_model
		self.backward_model = None


class ChiralityMLP(nn.Module):
	def __init__(self, feature_size, num_layers):
		super(ChiralityMLP, self).__init__()

		if num_layers == 0:
			self.forward_model = torch.nn.Identity()
			self.backward_model = torch.nn.Identity()
		else:
			self.forward_model = torchvision.ops.MLP(feature_size, [feature_size] * num_layers)
			self.backward_model = torchvision.ops.MLP(feature_size, [feature_size] * num_layers)

class ChiralityDisentangler(nn.Module):
	def __init__(self, feature_size, num_layers, model_type, normalization, chirality_dim = 1, force_orthogonal = False, skip_connection = False):
		super(ChiralityDisentangler, self).__init__()
		
		assert model_type in ["invertible", "mlp"]
		if model_type == "invertible":
			self.m = ChiralityInvertible(feature_size, num_layers)
		elif model_type == "mlp":
			self.m = ChiralityMLP(feature_size, num_layers)

		if force_orthogonal:
			self.A = nn.utils.parametrizations.orthogonal(nn.Linear(feature_size, feature_size, bias = False), orthogonal_map = "cayley")
		else:
			self.A = nn.Linear(feature_size, feature_size, bias = False)

		assert normalization in ["matrix", "tanh", "before", "beforeAndAfter"]
		self.normalization = normalization

		self.new_feature_dim = chirality_dim
		self.skip_connection = skip_connection
	

	def forward(self, x, return_backward = False):
		x = self.m.forward_model(x) + x if self.skip_connection else self.m.forward_model(x)

		forward_features = x.clone()
		
		backward_features = None
		if return_backward:
			backward_features = self.m.backward_model(x) + x if self.skip_connection else self.m.backward_model(x)
		
		if self.normalization == "before" or self.normalization == "beforeAndAfter":
			x = torch.nn.functional.normalize(x, dim = -1)

		x = self.A(x)
		
		if self.normalization == "matrix":
			x = torch.nn.functional.normalize(x, dim = -1)
		elif self.normalization == "tanh":
			x = torch.tanh(x)

		chiral, non_chiral = x.split([self.new_feature_dim, x.shape[2] - self.new_feature_dim], -1)

		if self.normalization == "beforeAndAfter":
			non_chiral = torch.nn.functional.normalize(non_chiral, dim = -1)

		if return_backward:
			return chiral, non_chiral, forward_features, backward_features
		else:
			return chiral, non_chiral, forward_features

if __name__ == "__main__":
	A = torch.load("best_full/A.h5")
	forward = torch.load("best_full/forward.h5")
	backward = torch.load("best_full/backward.h5")

	model = ChiralityDisentangler(len(A), 2, "mlp", "matrix", 1, False)
	with torch.no_grad():
		model.A.weight = torch.nn.Parameter(A.T)
	model.m.forward_model.load_state_dict(forward)
	model.m.backward_model.load_state_dict(backward)

	torch.save(model, "best_full/model.pt")