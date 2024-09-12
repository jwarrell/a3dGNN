from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
import torch
import copy
import pickle
import dimenet
import AffinityClustering
from torch_geometric.nn import GraphConv,TransformerConv,GCNConv
from sklearn import metrics
import numpy as np
from networkx.utils import open_file
from dig.threedgraph.method import SphereNet, DimeNetPP
from torch.autograd import Variable
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

def swish(x):
    return x * torch.sigmoid(x)

@open_file(0, mode='rb')
def read_gpickle(path):
    """Read graph object in Python pickle format.

    Pickles are a serialized byte stream of a Python object [1]_.
    This format will preserve Python objects used as nodes or edges.

    Parameters
    ----------
    path : file or string
       File or filename to write.
       Filenames ending in .gz or .bz2 will be uncompressed.

    Returns
    -------
    G : graph
       A NetworkX graph

    Examples
    --------
    # >>> G = nx.path_graph(4)
    # >>> nx.write_gpickle(G, "test.gpickle")
    # >>> G = nx.read_gpickle("test.gpickle")

    References
    ----------
    .. [1] http://docs.python.org/library/pickle.html
    """
    return pickle.load(path)

class my_spherenet(torch.nn.Module):
    def __init__(self, energy_and_force=False, cutoff=5.0, num_layers=4,
                  hidden_channels=128, out_channels=1, int_emb_size=64,
                  basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                  num_spherical=3, num_radial=6, envelope_exponent=5,
                  num_before_skip=1, num_after_skip=2, num_output_layers=3):
        super(my_spherenet, self).__init__()
        self.spherenet = SphereNet(energy_and_force=False, cutoff=cutoff, num_layers=num_layers,
                                   hidden_channels=hidden_channels, out_channels=out_channels,
                                   int_emb_size=int_emb_size, basis_emb_size_dist=basis_emb_size_dist,
                                   basis_emb_size_angle=basis_emb_size_angle,
                                   basis_emb_size_torsion=basis_emb_size_torsion,
                                   out_emb_channels=out_emb_channels, num_spherical=num_spherical,
                                   num_radial=num_radial, envelope_exponent=envelope_exponent,
                                   num_before_skip=num_before_skip, num_after_skip=num_after_skip,
                                   num_output_layers=num_output_layers)

    def forward(self, x):
        x.z = x.x.long()
        x.pos = x.pos.float()
        # breakpoint()
        return self.spherenet(x)

class my_dimenet(torch.nn.Module):
    def __init__(self, input_feature_dim, energy_and_force=False, cutoff=5.0, num_layers=4, hidden_channels=128,
                 out_channels=1, int_emb_size=64, basis_emb_size=8, out_emb_channels=256,
                 num_spherical=7, num_radial=6, envelope_exponent=5, num_before_skip=1,
                 num_after_skip=2, num_output_layers=3, act=swish, output_init='GlorotOrthogonal'):
        super(my_dimenet, self).__init__()
        self.dimenet = dimenet.DimeNetPP(input_feature_dim=input_feature_dim, energy_and_force=energy_and_force,
                                         cutoff=cutoff, num_layers=num_layers, hidden_channels=hidden_channels,
                                         out_channels=out_channels, int_emb_size=int_emb_size,
                                         basis_emb_size=basis_emb_size, out_emb_channels=out_emb_channels,
                                         num_spherical=num_spherical, num_radial=num_radial,
                                         envelope_exponent=envelope_exponent, num_before_skip=num_before_skip,
                                         num_after_skip=num_after_skip, num_output_layers=num_output_layers,
                                         act=act, output_init=output_init)

    def forward(self, x):
        # x.z = x.x.long()
        x.z = x.x.float()
        x.pos = x.pos.float()
        return self.dimenet(x)

def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    m = torch.nn.Linear(in_features, out_features, bias)

    if weight_init == 'kaiming':
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    elif weight_init == 'xavier':
        torch.nn.init.xavier_uniform_(m.weight, gain)
    else:
        raise NotImplemented

    if bias:
        torch.nn.init.zeros_(m.bias)

    return m
class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_model, nhead, kv_size=None, q_size=None, dropout=0.,
                 gain=1., epsilon=1e-8, inp_competition=False):
        '''
        :param d_model:
        :param nhead:
        :param kv_size:
        :param q_size:
        :param dropout:
        :param gain:
        :param inp_competition: for slot attention, if True, add slot competition
        '''
        super().__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        if kv_size is None:
            kv_size = d_model
        if q_size is None:
            q_size = d_model
        self.nhead = nhead

        self.attn_dropout = torch.nn.Dropout(dropout)
        self.output_dropout = torch.nn.Dropout(dropout)

        self.proj_q = linear(q_size, d_model, bias=False)
        self.proj_k = linear(kv_size, d_model, bias=False)
        self.proj_v = linear(kv_size, d_model, bias=False)
        self.proj_o = linear(d_model, q_size, bias=False, gain=gain)

        self.epsilon = epsilon
        self.inp_competition = inp_competition
        if inp_competition:
            self.attn_inp_softmax = torch.nn.Identity()

    def forward(self, q, k, v, attn_mask=None, weight=None):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """
        # breakpoint()
        q = rearrange(self.proj_q(q), 'b n_q (h d) -> b h n_q d', h=self.nhead)
        k = rearrange(self.proj_k(k), 'b n_k (h d) -> b h n_k d', h=self.nhead)
        v = rearrange(self.proj_v(v), 'b n_v (h d) -> b h n_v d', h=self.nhead)

        q = q * (q.shape[-1] ** (-0.5))

        attn = torch.einsum('...qd,...kd->...qk', q, k)

        # breakpoint()
        if weight is not None:
            attn = attn * weight[:, None]

        if attn_mask is not None:
            # breakpoint()
            attn = attn.masked_fill(attn_mask[:, None], float('-inf'))

        if self.inp_competition:
            attn = rearrange(attn, '... q k -> ... k q') # transpose such that logging will be easy, not functional property
            attn = F.softmax(attn, dim=-1)
            self.attn_inp_softmax((attn, 'cross'))
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            attn = rearrange(attn, '... k q -> ... q k')
        else:
            attn = F.softmax(attn, dim=-1)
            # breakpoint()
            attn = self.attn_dropout(attn)

        output = torch.einsum('...qk,...kd->...qd', attn, v)
        output = rearrange(output, 'b h n_q d -> b n_q (h d)')
        output = self.proj_o(output)
        output = self.output_dropout(output)
        return output

class my_transformer(torch.nn.Module):
    def __init__(self, input_feature_dim, hidden_dim):
        super(my_transformer, self).__init__()
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(input_feature_dim, hidden_dim),
            torch.nn.CELU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.transformer1 = MultiHeadAttention(d_model=hidden_dim, nhead=4)
        self.transformer2 = MultiHeadAttention(d_model=hidden_dim, nhead=4)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.CELU(),
            torch.nn.Linear(hidden_dim, 2)
        )

    def forward(self, data):
        # breakpoint()
        x = data.x
        batch = data.batch
        max_length = torch.bincount(batch).max()
        batch_size = len(torch.unique(batch))
        x = self.input_net(x)
        feature_dim = x.shape[-1]

        input_reshaped = x.new_zeros(batch_size, max_length, feature_dim)
        mask = x.new_zeros(batch_size, max_length, max_length)
        for i in range(batch_size):
            batch_x = x[batch == i]
            input_reshaped[i, :batch_x.shape[0]] = batch_x
            # breakpoint()
            # mask[i, batch_x.shape[0]:] = 1
            mask[i, :, batch_x.shape[0]:] = 1
        # breakpoint()
        mask = mask.to(bool)
        # breakpoint()
        x = self.transformer1(input_reshaped, input_reshaped, input_reshaped, attn_mask=mask)
        x = self.transformer2(x, x, x, attn_mask=mask)
        x = self.classifier(x.mean(1))
        return x


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,input_dim,num_classes,num_layers):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv_c=torch.nn.ModuleList()
        self.bn_c=torch.nn.ModuleList()
        for _ in range(int(num_layers)):
            self.conv_c.append(GCNConv(hidden_channels, hidden_channels))
            self.bn_c.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edges, batch):
        x = self.conv1(x, edges)
        x = self.bn1(x)
        x = x.relu()
        if len(self.conv_c) > 0:
            for index,conv_c_i in enumerate(self.conv_c):
                x = conv_c_i(x,edges)
                x = self.bn_c[index](x)
                x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.lin(x)
        return x

class preLin():
    def __init__(self, input_dim, output_dim, init_sigma, init_mu, device):
        if init_mu == 'identity':
            # assert input_dim == output_dim
            self.W_mu = torch.zeros(input_dim, output_dim).to(device)
            self.b_mu = torch.zeros(output_dim).to(device)
            # self.W_mu[torch.arange(input_dim), torch.arange(output_dim)] = 1
        else:
            self.W_mu = (torch.ones(input_dim, output_dim) * init_mu).to(device)
            self.b_mu = (torch.ones(output_dim) * init_mu).to(device)
        self.sigma = init_sigma
        self.current_W = None
        self.current_b = None
        self.device = device
    def sample(self):
        # breakpoint()
        self.current_W = torch.normal(mean=self.W_mu, std=self.sigma).to(self.device)
        self.current_b = torch.normal(mean=self.b_mu, std=self.sigma).to(self.device)
        return self.current_W, self.current_b
    def forward(self, x):
        return torch.matmul(x, self.current_W) + self.current_b

    def set_sigma(self, sigma):
        self.sigma = sigma
    def update(self, samples, objective_func):
        # samples is a list, each element is a tuple (sample_W, sample_b)
        # objective_func is a torch tensor in shape (Ns,)
        # breakpoint()
        objective_func = objective_func.to(self.device)
        sample_W = torch.stack([i[0] for i in samples], dim=0).to(self.device)
        sample_b = torch.stack([i[1] for i in samples], dim=0).to(self.device)

        _, n1, n2 = sample_W.shape

        self.W_mu = (sample_W * objective_func.unsqueeze(-1).unsqueeze(-1)).sum(0)
        self.W_mu = self.W_mu / objective_func.sum()
        self.b_mu = (sample_b * objective_func.unsqueeze(-1)).sum(0)
        self.b_mu = self.b_mu / objective_func.sum()
        tmp_mu = torch.cat([self.W_mu, self.b_mu.unsqueeze(0)], dim=0)
        tmp_sample = torch.cat([sample_W, sample_b.unsqueeze(1)], dim=1)
        sigma_tmp = (((tmp_sample - tmp_mu.unsqueeze(0)) ** 2) * objective_func.unsqueeze(-1).unsqueeze(-1)).sum()
        sigma_tmp = (sigma_tmp / ((n1+1) * n2 * objective_func.sum())).sqrt()
        self.sigma = sigma_tmp
        # breakpoint()

class prelinear_twolayers():
    def __init__(self, input_dim, output_dim, init_sigma, init_mu, device):
        self.layer1 = preLin(input_dim, output_dim, init_sigma, init_mu, device)
        self.layer2 = preLin(output_dim, output_dim, init_sigma, init_mu, device)
        self.celu = torch.nn.CELU()
        self.sigma = init_sigma

    def sample(self):
        layer1_W, layer1_b = self.layer1.sample()
        layer2_W, layer2_b = self.layer2.sample()
        return (layer1_W, layer1_b), (layer2_W, layer2_b)

    def forward(self, x):
        # breakpoint()
        x = self.layer1.forward(x)
        x = self.celu(x)
        x = self.layer2.forward(x)
        return x

    def set_sigma(self, sigma):
        self.layer1.set_sigma(sigma)
        self.layer2.set_sigma(sigma)
        self.sigma = sigma

    def update(self, samples, objective_func):
        # breakpoint()
        sample1 = [s[0] for s in samples]
        sample2 = [s[1] for s in samples]
        self.layer1.update(sample1, objective_func)
        self.layer2.update(sample2, objective_func)
        self.sigma = min(self.layer1.sigma, self.layer2.sigma)

class data_weight_var():
    def __init__(self, data_num, sigma, device):
        self.weight_mean = torch.ones(data_num).to(device)
        self.weight_sigma = (torch.ones_like(self.weight_mean) * sigma).to(device)
        self.current_weight = None
        self.device = device

    def forward(self, idx):
        return self.current_weight[idx]

    def sample(self):
        self.current_weight = torch.normal(mean=self.weight_mean, std=self.weight_sigma).to(self.device)
        self.current_weight[self.current_weight < 0] = 0
        return self.current_weight
        # breakpoint()
        # self.current_weight[self.current_weight > 1] = 1
    def update(self, sample, objective_func):
        # sample Ns x num_data
        # objective_func Ns
        # breakpoint()
        objective_func = objective_func.to(self.device)

        self.weight_mean = (sample * objective_func.unsqueeze(-1)).sum(0)
        self.weight_mean = self.weight_mean / objective_func.sum()
        weight_sigma = (sample - self.weight_mean.unsqueeze(0)) ** 2
        weight_sigma = (weight_sigma * objective_func.unsqueeze(-1)).sum(0)
        self.weight_sigma = (weight_sigma / objective_func.sum()).sqrt()

class data_weight(torch.nn.Module):
    def __init__(self, data_num):
        super(data_weight, self).__init__()
        # self.weight = torch.nn.parameter.Parameter(torch.zeros(data_num), requires_grad=True).to(device)
        self.register_parameter(name='weight', param=torch.nn.parameter.Parameter(torch.zeros(data_num)))
        self.weight.requires_grad = True
        self.data_num = data_num

    def forward(self, idx):
        return self.weight[idx]

    def reset(self):
        self.weight = torch.nn.parameter.Parameter(torch.zeros(self.data_num), requires_grad=True).to(self.device)

def weighted_cross_entropy(input, target, weight, calculate_weight_grad=False):
    loss = torch.nn.functional.cross_entropy(input, target, reduction='none')
    loss = (loss.flatten() * weight.flatten()).mean()
    if calculate_weight_grad:
        g = torch.autograd.grad(loss, weight)
    else:
        g = None
    return loss, g

class GTN(torch.nn.Module):
    def __init__(self, hidden_channels,input_dim,num_classes,num_layers,train_data_num):
        super(GTN, self).__init__()
        self.conv1 = TransformerConv(input_dim, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv_c=torch.nn.ModuleList()
        self.bn_c=torch.nn.ModuleList()
        for _ in range(int(num_layers)):
            self.conv_c.append(TransformerConv(hidden_channels, hidden_channels))
            self.bn_c.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lin = Linear(hidden_channels, num_classes)
        # self.emb_layer = emb_layer(input_dim)
        self.image_weights = torch.ones(train_data_num, requires_grad=True)
        self.params = [hidden_channels,input_dim,num_classes,num_layers,train_data_num]

    def get_image_weights(self):
        return [self.image_weights]

    def to_device(self, device):
        self.image_weights = self.image_weights.to(device).detach().requires_grad_(True)

    def loss(self, images, gt_classes, partition, weight_idx=None, calculate_weight_grad=False):
        scores = self(images)
        if partition == 'train':

            # print(self.image_weights.shape)
            weight = self.image_weights[weight_idx]

            # if partition == 'train':
            #     breakpoint()

            loss, g = weighted_cross_entropy(scores, gt_classes, weight, calculate_weight_grad)

            if calculate_weight_grad:
                # breakpoint()
                g_complete = torch.zeros_like(self.image_weights)
                g_complete[weight_idx] = g[0]
                return loss, g_complete
            return loss, g
        # print(scores, gt_classes)
        # print()
        return F.cross_entropy(scores, gt_classes), None

    def new(self):
        model_new = GTN(*self.params)
        for x, y in zip(model_new.get_image_weights(), self.get_image_weights()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, data):
        # x = self.emb_layer(x)
        x, edges, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edges)
        x = self.bn1(x)
        x = x.relu()
        if len(self.conv_c) > 0:
            for index,conv_c_i in enumerate(self.conv_c):
                x = conv_c_i(x,edges)
                x = self.bn_c[index](x)
                x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.lin(x)
        return x


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels,input_dim,num_classes,num_layers):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv_c=torch.nn.ModuleList()
        self.bn_c=torch.nn.ModuleList()
        for _ in range(int(num_layers)):
            self.conv_c.append(GraphConv(hidden_channels, hidden_channels))
            self.bn_c.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edges, batch):
        x = self.conv1(x, edges)
        x = self.bn1(x)
        x = x.relu()
        if len(self.conv_c) > 0:
            for index,conv_c_i in enumerate(self.conv_c):
                x = conv_c_i(x,edges)
                x = self.bn_c[index](x)
                x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.lin(x)
        return x

class emb_layer(torch.nn.Module):
    def __init__(self, dim):
        super(emb_layer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.CELU(),
            torch.nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.layer(x)

def train(model,train_loader, optimizer,criterion,device,drop_ratio=None,
          data_weight=None, train_loader2=None, data_weight_optimizer=None,
          lam=None, weight_trainer=None, scheduler=None):
    model.train()
    # if data_weight is None or (data_weight is not None and train_loader2 is None):
    #     train_loader2 = train_loader
    # breakpoint()
    train_loader2_iter = iter(train_loader2)
    for data in train_loader:  # Iterate in batches over the training dataset.
        # breakpoint()
        try:
            data2 = next(train_loader2_iter)
        except:
            train_loader2_iter = iter(train_loader2)
            data2 = next(train_loader2_iter)
        if drop_ratio is not None:
            keep_edges = torch.tensor([True] * data.edge_index.shape[-1])
            drop_ratio = np.random.random() * drop_ratio
            remove_index = np.random.choice(data.edge_index.shape[-1],
                                            size=int(data.edge_index.shape[-1] * drop_ratio),
                                            replace=False)
            # breakpoint()
            is_peptide = data.x[:, -1]
            can_drop = (is_peptide[data.edge_index[0]] == 0) & (is_peptide[data.edge_index[1]] == 0)
            keep_edges[remove_index] = False
            keep_edges = keep_edges | (~can_drop)
            data.edge_index = data.edge_index[:, keep_edges]
            # drop_index
        if isinstance(model, my_spherenet) or isinstance(model, my_dimenet) or isinstance(model, my_transformer):
            data.x = data.x.to(device)
            # data.edge_index = data.edge_index.to(device)
            data.pos = torch.cat([torch.from_numpy(i) for i in data.pos], dim=0).to(device)
            data.batch = data.batch.to(device)
            out = model(data)
        elif isinstance(model, GTN):
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.batch = data.batch.to(device)
            out = model(data)
        else:
            # data_x = torch.cat([data.x, torch.cat([torch.from_numpy(i) for i in data.pos], dim=0)], dim=-1)
            out = model(data.x.to(device).float(), data.edge_index.to(device), data.batch.to(device))  # Perform a single forward pass.
        if data_weight is None:
            loss = criterion(out, data.y.to(device)).mean()  # Compute the loss.
            loss.backward()
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
        else:
            # breakpoint()
            weight_trainer.step(sample_train=data, sample_val=data2, eta=scheduler.get_last_lr()[0], optimizer=optimizer)
            # breakpoint()
            # out = model()
            # out2 = model(data2.x.to(device).float(), data2.edge_index.to(device), data2.batch.to(device))
            addsum = torch.zeros_like(data.graph_index)
            addsum[1:] = torch.cumsum(torch.bincount(data.batch), dim=0)[:-1]
            weight_index = data.graph_index - addsum
            # data_weight_bs = data_weight((data2.graph_index - addsum).to(device))
            # data_weight_bs = data_weight_bs.exp() / data_weight.weight.exp().sum()
            # loss1 = criterion(out, data.y.to(device)).mean()
            # loss2 = (criterion(out2, data2.y.to(device)) * data_weight_bs).mean()
            # loss = loss1 + loss2 * lam

            score = model(data)

            weights = model.get_image_weights()[0][weight_index]
            loss, _ = weighted_cross_entropy(score, data.y.to(device), weights)
            # loss = torch.nn.functional.cross_entropy(score, gt_classes_train)

            optimizer.zero_grad()
            loss.backward()

            # clip_grad_norm_(model.parameters(), args.max_gradient)
            optimizer.step()
        # breakpoint()
          # Derive gradients.
        # breakpoint()
        # if data_weight is not None:
        #     data_weight_optimizer.step()
        #     data_weight_optimizer.zero_grad()

def test(model,loader, device):
    model.eval()
    correct = 0
    scores = []
    correct_label = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        if isinstance(model, my_spherenet) or isinstance(model, my_dimenet) or isinstance(model, my_transformer):
            data.x = data.x.to(device)
            # data.edge_index = data.edge_index.to(device)
            data.pos = torch.cat([torch.from_numpy(i) for i in data.pos], dim=0).to(device)
            data.batch = data.batch.to(device)
            out = model(data)
        elif isinstance(model, GTN):
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.batch = data.batch.to(device)
            out = model(data)
        else:
            # data_x = torch.cat([data.x, torch.cat([torch.from_numpy(i) for i in data.pos], dim=0)], dim=-1)
            out = model(data.x.to(device).float(), data.edge_index.to(device), data.batch.to(device))
        score = torch.softmax(out, dim=1)
        scores.append(score[:, 1])
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y.to(device)).sum())  # Check against ground-truth labels.
        correct_label.append(data.y)
    # breakpoint()
    correct_label = torch.cat(correct_label)
    scores = torch.cat(scores)
    fpr, tpr, thresholds = metrics.roc_curve(correct_label.detach().cpu().numpy(), scores.detach().cpu().numpy(), pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return correct / len(loader.dataset), auc, scores  # Derive ratio of correct predictions.

def train_GNN(model, train_loader,val_loader,test_loader,optimizer,criterion,n_epochs,patience,
              best_model_criterion, mu_sigma_criterion,prelin_update_criterion,device,early_stop,
              drop_ratio=None, data_weight=None, train_loader2=None, data_weight_optimizer=None,
              lam=None, scheduler=None, weight_trainer=None):

    best_val_acc = -0.1
    best_val_auc = -0.1
    best_val_epoch = 0
    best_model=None

    for epoch in range(0, int(n_epochs)):
        train(model=model,train_loader=train_loader,optimizer=optimizer,criterion=criterion,
              device=device,drop_ratio=drop_ratio,data_weight=data_weight, train_loader2=train_loader2,
              data_weight_optimizer=data_weight_optimizer, lam=lam, scheduler=scheduler, weight_trainer=weight_trainer)
        #train_acc = GNN_core.test(model=model,loader=train_loader)
        with torch.no_grad():
            this_val_acc, this_val_auc, _ = test(model=model,loader=val_loader, device=device)

        #if epoch%5==0:
        #    print('epoch:',epoch,'train acc: ',train_acc,'val acc: ',this_val_acc,'best val acc:',best_val_acc)

        if early_stop != 'y':
            get_better_model = True
        elif best_model_criterion == 'acc':
            get_better_model = this_val_acc > best_val_acc
        elif best_model_criterion == 'auc':
            get_better_model = this_val_auc > best_val_auc

        if get_better_model: #validation wrapper
            best_val_epoch = epoch
            if best_model_criterion == 'acc':
                best_val_acc = this_val_acc
                best_val_auc = this_val_auc
            elif best_model_criterion == 'auc':
                best_val_auc = this_val_auc
                best_val_acc = this_val_acc
            best_model = copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter+=1
        if patience_counter == patience:
            #print("ran out of patience")
            break
    with torch.no_grad():
        train_acc_best_model, train_auc_best_model, _ = test(model=best_model,loader=train_loader, device=device)
        test_acc_best_model, test_auc_best_model, test_score_best_model = test(model=best_model,loader=test_loader, device=device)
    #print('best epoch:',best_val_epoch,'train acc: ',train_acc_best_model,'test acc',test_acc_best_model,'best val acc:',best_val_acc)

        if mu_sigma_criterion == 'ce':
            if prelin_update_criterion == 'train':
                train_loss = calc_loss(model=best_model,loader=train_loader,criterion=criterion,device=device)
            elif prelin_update_criterion == 'val':
                train_loss = calc_loss(model=best_model, loader=val_loader, criterion=criterion, device=device)
            score = -train_loss.item()
        else:
            score = train_auc_best_model
    # if score<0:
    #     score=0.0
    # breakpoint()
    return score,best_val_acc,best_val_auc,\
        train_acc_best_model,train_auc_best_model,\
        test_acc_best_model,test_auc_best_model,\
        test_score_best_model,best_model

def calc_loss(model, loader, criterion, device):
    model.eval()
    loss = 0
    for data in loader:  # Iterate in batches over the training dataset.
        #model.conv1.register_forward_hook(get_activation('conv3'))
        if isinstance(model, my_spherenet) or isinstance(model, my_dimenet) or isinstance(model, my_transformer):
            data.x = data.x.to(device)
            # data.edge_index = data.edge_index.to(device)
            data.pos = torch.cat([torch.from_numpy(i) for i in data.pos], dim=0).to(device)
            data.batch = data.batch.to(device)
            out = model(data)
        elif isinstance(model, GTN):
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.batch = data.batch.to(device)
            out = model(data)
        else:
            # data_x = torch.cat([data.x, torch.cat([torch.from_numpy(i) for i in data.pos], dim=0)], dim=-1)
            out = model(data.x.to(device).float(), data.edge_index.to(device), data.batch.to(device))  # Perform a single forward pass.
        loss += criterion(out, data.y.to(device)).mean()
    return loss



def calculate_AUC(model, loader):
    model.eval()
    scores = []
    correct_label = []
    for data in loader:  # Iterate in batches over the training dataset.
        # model.conv1.register_forward_hook(get_activation('conv3'))
        if isinstance(model, my_spherenet) or isinstance(model, my_dimenet) or isinstance(model, my_transformer):
            out = model(data)
        else:
            data_x = torch.cat([data.x, torch.cat([torch.from_numpy(i) for i in data.pos], dim=0)], dim=-1)
            out = model(data.x.float(), data.edge_index, data.batch)  # Perform a single forward pass.
        score = torch.softmax(out, dim=1)
        scores.append(score[:, 1])
        correct_label.append(data.y)
    correct_label = torch.cat(correct_label)
    scores = torch.cat(scores)
    fpr, tpr, thresholds = metrics.roc_curve(correct_label.detach().numpy(), scores.detach().numpy(), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def predict(model,loader):
    model.eval()
    pred=[]
    outs = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        if isinstance(model, my_spherenet) or isinstance(model, my_dimenet) or isinstance(model, my_transformer):
            out = model(data)
        else:
            # data_x = torch.cat([data.x, torch.cat([torch.from_numpy(i) for i in data.pos], dim=0)], dim=-1)
            out = model(data.x.float(), data.edge_index, data.batch)
        outs.append(torch.softmax(out, dim=1)[:, 1].tolist())
        pred.append(out.argmax(dim=1).tolist())  # Use the class with highest probability.
    return pred, outs


def loss(model,loader,criterion):
    model.eval()
    loss=0.
    for data in loader:
        if isinstance(model, my_spherenet) or isinstance(model, my_dimenet) or isinstance(model, my_transformer):
            out = model(data)
        else:
            # data_x = torch.cat([data.x, torch.cat([torch.from_numpy(i) for i in data.pos], dim=0)], dim=-1)
            out = model(data.x.float(), data.edge_index, data.batch)
        loss += criterion(out, data.y).sum()
    return loss/len(loader.dataset)

def alternate_dataset(dataset):
    alternate=[]
    zeros=[]
    ones=[]
    for i,data in enumerate(dataset):
        label = data.y.item()
        if label == 0:
            zeros.append(i)
        elif label ==1:
            ones.append(i)
    if len(zeros)>len(ones):
        zeros=zeros[0:len(ones)]
    elif len(zeros)<len(ones):
        ones=ones[0:len(zeros)]
    for i,j in zip(zeros,ones):
        alternate.append(dataset[i])
        alternate.append(dataset[j])
    return alternate

def alternate_g(dataset):
    ones, zeros = sort_dataset(dataset)
    return coallated_dataset(ones, zeros)

def sort_dataset(dataset, a_label=1):
    labeled_a = []
    labeled_b = []
    for data in dataset:
        label = data.y.item()
        if label == a_label:
            labeled_a.append(data)
        else:
            labeled_b.append(data)
    return (labeled_a, labeled_b)


def coallated_dataset(set1, set2):
    dataset = []
    if len(set1)<len(set2):
        length = len(set1)
    else:
        length = len(set2)
    for i in range(length):
        dataset.append(set1[i])
        dataset.append(set2[i])
    return dataset

def get_info_dataset(dataset, verbose=False):
    """Determines the number of inputs labeled one and zero in a dataset."""
    zeros = 0
    ones = 0
    for data in dataset:
        label = data.y.item()
        if label == 0:
            zeros+=1
        elif label ==1:
            ones+=1
    if verbose:
        print(f'In this dataset, there are {zeros} inputs labeled "0" and {ones} inputs labeled "1". ')
    return (ones, zeros)

def balance_dataset(dataset):
    ones, zeros = get_info_dataset(dataset)
    if zeros==ones:
        return dataset
    if zeros>ones:
        major=zeros
        minor=ones
        the_major_one=0
    else:
        major=ones
        minor=zeros
        the_major_one=1
    major_index=0
    balanced = []
    for item in dataset:
        label = item.y.item()
        if label == the_major_one:
            if major_index<minor:    
                balanced.append(item)
                major_index=major_index+1
        else:
            balanced.append(item)
    return balanced

def calc_degree(graph):
    edge_first=graph['edge_index'].detach().numpy()[0]
    edge_second=graph['edge_index'].detach().numpy()[1]
    graph_dict=AffinityClustering.graph_dictionary(edge_first,edge_second,len(graph['x']))
    degree_matrix=AffinityClustering.calc_degree_matrix(graph_dict)
    return degree_matrix

def calc_distance(graph):
    # edge_first=graph['edge_index'].detach().numpy()[0]
    # edge_second=graph['edge_index'].detach().numpy()[1]
    # graph_dict=AffinityClustering.graph_dictionary(edge_first,edge_second,len(graph['x']))
    # distance_matrix=AffinityClustering.calc_distance_matrix(graph_dict)
    distance_matrix = AffinityClustering.calc_euclidean_distance_matrix(graph.pos)
    return distance_matrix

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args, device):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.get_image_weights(),
            lr=args.weight_learning_rate, betas=(0.5, 0.999), weight_decay=0)
        self.device = device
        self.args = args

    def _compute_unrolled_model(self, input, target, index, eta, network_optimizer):
        loss, _ = self.model.loss(input, target, weight_idx=index, partition='train')
        theta = _concat(self.model.parameters()).data
        # breakpoint()
        try:
          moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
          moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
        unrolled_model = self._construct_model_from_theta(theta.sub(moment+dtheta, alpha=eta))
        return unrolled_model

    def step(self, sample_train, sample_val, eta, optimizer):
        self.optimizer.zero_grad()
        val_loss = self._backward_step_unrolled(sample_train, sample_val, eta, optimizer)
        self.optimizer.step()
        self.model.get_image_weights()[0].data[self.model.get_image_weights()[0].data < 0] = 0
        if self.args.clip_1:
            self.model.get_image_weights()[0].data[self.model.get_image_weights()[0].data > 1] = 1
        return val_loss

    def _backward_step_unrolled(self, sample_train, sample_val, eta, optimizer):
        # input_train = sample_train['image'].to(self.device)
        # target_train = sample_train['false_class'].to(self.device)
        # train_index = sample_train['index'].to(self.device)
        # input_valid = sample_val['image'].to(self.device)
        # target_valid = sample_val['class'].to(self.device)
        # breakpoint()
        input_train = sample_train
        input_valid = sample_val
        input_train.x = input_train.x.to(self.device)
        input_train.edge_index = input_train.edge_index.to(self.device)
        input_train.batch = input_train.batch.to(self.device)
        input_valid.x = input_valid.x.to(self.device)
        input_valid.edge_index = input_valid.edge_index.to(self.device)
        input_valid.batch = input_valid.batch.to(self.device)
        target_train = sample_train.y.to(self.device)
        target_valid = sample_val.y.to(self.device)
        addsum = torch.zeros_like(input_train.graph_index)
        addsum[1:] = torch.cumsum(torch.bincount(input_train.batch), dim=0)[:-1]
        train_index = (input_train.graph_index - addsum).to(self.device)
        unrolled_model = self._compute_unrolled_model(input_train, target_train, train_index, eta, optimizer)
        unrolled_loss, _ = unrolled_model.loss(input_valid, target_valid, 'val')

        unrolled_loss.backward()
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train, train_index)

        for v, g in zip(self.model.get_image_weights(), implicit_grads):
          if v.grad is None:
            v.grad = Variable(- g.data)
          else:
            v.grad.data.copy_(- g.data)
        return unrolled_loss.item()

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
          v_length = np.prod(v.size())
          params[k] = theta[offset: offset+v_length].view(v.size())
          offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, index, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
          p.data.add_(v, alpha=R)
        loss, grads_p = self.model.loss(input, target, 'train', index, calculate_weight_grad=True)

        for p, v in zip(self.model.parameters(), vector):
          p.data.sub_(v, alpha=2*R)
        loss, grads_n = self.model.loss(input, target, 'train', index, calculate_weight_grad=True)

        for p, v in zip(self.model.parameters(), vector):
          p.data.add_(v, alpha=R)

        return [(x-y).div_(2*R) for x, y in zip([grads_p], [grads_n])]