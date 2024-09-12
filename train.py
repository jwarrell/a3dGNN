from __future__ import print_function
import argparse
import collections
import os
import time

import torch
import data
import GNN_core
import AffinityClustering
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from sklearn.cluster import DBSCAN
from Bio import SeqIO

parser = argparse.ArgumentParser(description="Simulate a Affinity training GNN with the appropriate hyperparameters.")
parser.add_argument('-d','--train-dataset', required=True, help='the protein dataset for training')
parser.add_argument('-s','--test-dataset', required=True, help='the protein dataset for testing')
parser.add_argument('--graph_path', required=True, help='path to the graph files')
parser.add_argument('-r','--partition_ratio', required=False, type=str, help="governs the ration of partition sizes in the training, validation, and test sets. a list of the form [train, val, test]", default="0.4:0.3:0.3")
parser.add_argument('--partition_size', required=False, help='sets partition size for the total size of dataset', default='max')
parser.add_argument('-e','--epochs', required=False, help='number of training epochs', default='201')
parser.add_argument('-nc','--num_layers', required=False, help='number of layers', default='3')
parser.add_argument('-p','--patience', required=False, type=int, help='upper limit for the patience counter used in validation', default=60)
parser.add_argument('-b','--batch_size', required=False, type=int, help='batch size for training, testing and validation', default=30)
parser.add_argument('-l','--learning_rate', required=False, type=float, help='initial learning rate', default=0.008)
parser.add_argument('--data_weight_learning_rate', required=False, type=float, help='initial learning rate', default=0.1)
parser.add_argument('-m','--model_type', required=False, type=str, help='the underlying model of the neural network', default='GCN')
parser.add_argument('-c','--hidden_channel', required=False, type=int, help='width of hidden layers', default=25)
parser.add_argument('--meta_feature_dim', required=False, help='the output size of the first linear layer or the input dimension of the clustering algorithm', default='same')
parser.add_argument('--cluster_params', required=False, help='p_factor, C_sp and C_degree', default="4,1.4,0.71")
parser.add_argument('--initial_mu', required=False, help='the starting point of the gaussian distribution mu for smoothing based optimization', default="identity")
parser.add_argument('--initial_sigma', required=False, type=float, help='the starting point of the gaussian distribution sigma for smoothing based optimization', default=3.)
parser.add_argument('-o','--params_storage_path',required=False,help='path to store the model params',default=None)
parser.add_argument('--constrained-names', required=False, type=str, default=None, help='protein names for training')
parser.add_argument('--constrained-names2', required=False, type=str, default=None, help='protein names for training')
parser.add_argument('--constrained-names-val', required=False, type=str, default=None, help='protein names for testing')
parser.add_argument('--tmp-dir', required=False, type=str, default='/tmp/ray', help='protein names for training')
parser.add_argument('--best_model_criterion', required=False, type=str, default='acc',
                    help='criterion to select the best model, acc or auc')
parser.add_argument('--mu_sigma_criterion', required=False, type=str, default='ce',
                    help='criterion to select the best model, ce (cross entropy) or auc')
parser.add_argument('--n_sample', required=False, type=int, default=15,
                    help='number of sample to draw for each training data')
parser.add_argument('--summary_dir', required=False, type=str, default='run',
                    help='criterion to select the best model, acc or auc')
parser.add_argument('--prelin_update_cri', required=False, type=str, default='val',
                    help='prelin layer update based on training score or validation loss')
parser.add_argument('--first_phase_ratio', required=False, default=None, type=float,
                    help='this is the ratio for training the prelinear layer')
parser.add_argument('--clustering', required=False, type=str, default='y',
                    help='doing clustering or not, y or n')
parser.add_argument('--step', required=False, type=int, default=30,
                    help='number of meta-epochs')
parser.add_argument('--early_stop', required=False, type=str, default='y',
                    help='number of meta-epochs')
parser.add_argument('--drop_ratio', required=False, type=float, help='percentage of edges to drop randomly', default=None)
parser.add_argument('--euc_thr', required=False, default=200, type=float, help='this is the distance threshold for removing edges between MHC an peptide')
parser.add_argument('--modify_way', required=False, type=str, default=None,
                    help='add edges or removing edges')
parser.add_argument('--prelin_type', required=False, type=str, default='linear',
                    help='linear or nonlinear for prelinear layer')
parser.add_argument('--weight-data', required=False, type=str, default='n',
                    help='whether to weight the data')
parser.add_argument('--lam', type=float, required=False, default=1)
parser.add_argument('--lam2', type=float, required=False, default=1)
parser.add_argument('--momentum', type=float, required=False, default=0.9)
parser.add_argument('--update-weight', action='store_true')
parser.add_argument('--corrupt', action='store_true')
parser.add_argument('--cluster-corrupt', action='store_true')
parser.add_argument('--kernel-weight', action='store_true')
parser.add_argument('--seq-file', required=False)
parser.add_argument('--eps', type=float, required=False, default=0.5)

args = parser.parse_args()

writer = SummaryWriter(log_dir=args.summary_dir)

output_path = args.params_storage_path
os.makedirs(args.params_storage_path, exist_ok=True)
print(f'output files will be stored in {args.params_storage_path}')
# if not os.path.isfile(f"{args.outdir}/config.txt") or (not os.path.samefile(args.my_config, f"{args.outdir}/config.txt")):
#     shutil.copy(args.my_config, f"{args.outdir}/config.txt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.cluster_corrupt:
    args.corrupt = False

data_maker = data.dataset_maker()
train_data, val_data = data_maker.make_data(protein_names_file=args.train_dataset,
                                            pdb_file_dir=args.graph_path,
                                            constrain_label_file=args.constrained_names,
                                            ratio=float(args.partition_ratio.split(":")[0]),
                                            make_balanced=True)
all_train_data, all_val_data = data_maker.make_data(protein_names_file=args.train_dataset,
                                                    pdb_file_dir=args.graph_path,
                                                    constrain_label_file=args.constrained_names2,
                                                    ratio=0.999,
                                                    make_balanced=True,
                                                    corrupt=args.corrupt)

if args.first_phase_ratio is None or args.first_phase_ratio == float(args.partition_ratio.split(":")[0]):
    first_phase_train_data, first_phase_val_data = train_data, val_data
    all_first_phase_train_data, all_first_phase_val_data = all_train_data, all_val_data
else:
    first_phase_train_data, first_phase_val_data = data_maker.make_data(protein_names_file=args.train_dataset,
                                            pdb_file_dir=args.graph_path,
                                            constrain_label_file=args.constrained_names,
                                            ratio=args.first_phase_ratio,
                                            make_balanced=True)
    all_first_phase_train_data, all_first_phase_val_data = data_maker.make_data(protein_names_file=args.train_dataset,
                                                                                pdb_file_dir=args.graph_path,
                                                                                constrain_label_file=args.constrained_names2,
                                                                                ratio=args.first_phase_ratio,
                                                                                make_balanced=True,
                                                                                corrupt=args.corrupt)

test_data = data_maker.make_data(protein_names_file=args.test_dataset,
                                 pdb_file_dir=args.graph_path,
                                 constrain_label_file=args.constrained_names_val,
                                 ratio=None,
                                 make_balanced=False,
                                 output_path=output_path)

if args.seq_file is not None:
    all_train_data_protein_name = [G.protein_name for G in all_train_data.graphs]
    seq = SeqIO.to_dict(SeqIO.parse(args.seq_file, "fasta"))
    seq = {i: seq[i] for i in all_train_data_protein_name}
    if args.kernel_weight:
        # breakpoint()
        seq_len = len(list(seq.values())[0])
        # gt_kernel = np.random.choice([0, 1], size=seq_len)
        gt_kernel = np.zeros(seq_len)
        gt_kernel[-84:] = 1
        hamm_dist = GNN_core.kernel_hamm_dist(seq, gt_kernel)
        torch.save(torch.from_numpy(gt_kernel), f'{output_path}/gt_kernel.pt')
        kernel = GNN_core.seq_kernel(seq_len=seq_len, sigma=1, device=device)
    else:
        hamm_dist = GNN_core.calc_dist_mat(seq)
    clustering = DBSCAN(eps=args.eps, min_samples=2).fit(hamm_dist.values).labels_
    torch.save(torch.from_numpy(clustering), f'{output_path}/gt_clustering.pt')
    # breakpoint()
    singleton_num = (clustering == -1).sum()
    max_cluster = clustering.max()
    singleton_num_cluster = np.arange(singleton_num) + clustering.max() + 1
    clustering[clustering == -1] = singleton_num_cluster
    if args.cluster_corrupt:
        selected_corrupted_cluster = set(np.random.choice(max_cluster, size=max_cluster//3, replace=False))
        for i, g in all_train_data.graphs:
            if clustering[i] in selected_corrupted_cluster:
                g.y = 1 - g.y
                g.corrupt = True
            else:
                g.corrupt = False
    # breakpoint()
else:
    # breakpoint()
    clustering = np.arange(len(all_train_data.graphs))
# if args.clustering == 'y':
#     train_degree_matrices, train_distance_matrices = train_data.get_degree_distance()
#     val_degree_matrices, val_distance_matrices = val_data.get_degree_distance()
#     test_degree_matrices, test_distance_matrices = test_data.get_degree_distance()

for current_data in [train_data.graphs, val_data.graphs, test_data.graphs, first_phase_train_data.graphs, first_phase_val_data.graphs]:
    for g in current_data:
        g['edge_index_old'] = torch.clone(g['edge_index'])

num_classes = 2

try:
    num_node_features = len(train_data.graphs[0].x[0])
except:
    num_node_features = 1

if args.initial_mu=='identity':
    initial_mu = 'identity'
else:
    initial_mu = torch.from_numpy(np.loadtxt(args.initial_mu))

if str(args.meta_feature_dim) == 'same':
    # meta_feature_dim = len(train_data.graphs[0]['x'][0])
    meta_feature_dim = num_node_features
else:
    meta_feature_dim = int(args.meta_feature_dim)

if args.prelin_type == 'linear':
    prelinear_mhc = GNN_core.preLin(input_dim=num_node_features+2, output_dim=meta_feature_dim+2,
                                    init_sigma=args.initial_sigma, init_mu=initial_mu, device=device)
    prelinear_peptide = GNN_core.preLin(input_dim=num_node_features+2, output_dim=meta_feature_dim+2,
                                        init_sigma=args.initial_sigma, init_mu=initial_mu, device=device)
else:
    prelinear_mhc = GNN_core.prelinear_twolayers(input_dim=num_node_features + 2, output_dim=meta_feature_dim + 2,
                                                 init_sigma=args.initial_sigma, init_mu=initial_mu, device=device)
    prelinear_peptide = GNN_core.prelinear_twolayers(input_dim=num_node_features + 2, output_dim=meta_feature_dim + 2,
                                                     init_sigma=args.initial_sigma, init_mu=initial_mu, device=device)

# if args.weight_data == 'y':
    # data_weight = GNN_core.data_weight(data_num=len(train_data.graphs), device=device, sigma=0.2)
    # data_weight = GNN_core.data_weight(data_num=len(train_data.graphs)).to(device)

# trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
# testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

# trainloader = DataLoader(train_data.graphs, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
# testloader = DataLoader(test_data.graphs, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
# valloader = DataLoader(val_data.graphs, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

cluster_params=args.cluster_params.split(",")
cluster_params = [float(entry) for entry in cluster_params]

def train_onePoint(current_train_data, current_val_data, current_train_data2, data_weight,
                   data_weight_optimizer, update_weight, pretrain_data_weight=None, predefined_clustering=None):
    # breakpoint()
    if args.model_type == 'GCN':
        model = GNN_core.GCN(hidden_channels=args.hidden_channel,
                             input_dim=num_node_features,
                             num_classes=num_classes,
                             num_layers=args.num_layers).to(device)
    if args.model_type == 'GTN':
        model = GNN_core.GTN(hidden_channels=args.hidden_channel,
                             input_dim=num_node_features,
                             num_classes=num_classes,
                             num_layers=args.num_layers).to(device)
    if args.model_type == 'GNN':
        model = GNN_core.GNN(hidden_channels=args.hidden_channel,
                             input_dim=num_node_features,
                             num_classes=num_classes,
                             num_layers=args.num_layers).to(device)
    if args.model_type == 'shperenet':
        model = GNN_core.my_spherenet(energy_and_force=False, cutoff=6, num_layers=4,
                                      hidden_channels=128, out_channels=2, int_emb_size=64,
                                      basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8,
                                      out_emb_channels=256, num_spherical=3, num_radial=6, envelope_exponent=5,
                                      num_before_skip=1, num_after_skip=2, num_output_layers=3).to(device)
    if args.model_type == 'dimenet':
        model = GNN_core.my_dimenet(input_feature_dim=num_node_features,
                                    energy_and_force=False, cutoff=4, num_layers=4, hidden_channels=64,
                                    out_channels=2, int_emb_size=64, basis_emb_size=8, out_emb_channels=64,
                                    num_spherical=7, num_radial=6, envelope_exponent=5, num_before_skip=1,
                                    num_after_skip=2, num_output_layers=3).to(device)
    if args.model_type == 'transformer':
        model = GNN_core.my_transformer(
            input_feature_dim=num_node_features,
            hidden_dim=128
        ).to(device)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    # if args.weight_data == 'y':
    #     data_weight_optimizer = torch.optim.Adam(data_weight.parameters(), lr=args.data_weight_learning_rate)

    sample_W_mhc, sample_b_mhc = prelinear_mhc.sample()
    sample_W_peptide, sample_b_peptide = prelinear_peptide.sample()
    if args.kernel_weight and predefined_clustering is None:
        sample_kernel = kernel.sample()
        hamm_dist = GNN_core.kernel_hamm_dist(seq, sample_kernel.cpu().numpy())
        clustering = DBSCAN(eps=args.eps, min_samples=2).fit(hamm_dist.values).labels_
        torch.save(torch.from_numpy(clustering), f'{output_path}/clustering.pt')
        singleton_num = (clustering == -1).sum()
        max_cluster = clustering.max()
        singleton_num_cluster = np.arange(singleton_num) + clustering.max() + 1
        clustering[clustering == -1] = singleton_num_cluster
    elif args.kernel_weight:
        sample_kernel = None
        clustering = predefined_clustering
    elif data_weight:
        sample_kernel = None
        clustering = np.arange(len(all_train_data.graphs))
    else:
        sample_kernel = None
        clustering = None

    if data_weight and pretrain_data_weight is None:
        data_weight = GNN_core.data_weight(data_num=clustering.max() + 1).to(device)
        data_weight_optimizer = torch.optim.Adam(data_weight.parameters(), lr=args.data_weight_learning_rate)
    elif data_weight and pretrain_data_weight is not None:
        data_weight = pretrain_data_weight
        data_weight_optimizer = None
    else:
        data_weight = None
        data_weight_optimizer = None

    t = time.time()
    if args.clustering == 'y':
        print(f'Doing clustering...')
        # for current_data, current_degree_matrices, current_distance_matrices in zip([train_data.graphs, val_data.graphs, test_data.graphs],
        #                                                                             [train_degree_matrices, val_degree_matrices, test_degree_matrices],
        #                                                                             [train_distance_matrices, val_distance_matrices, test_distance_matrices]):
        #     for index, g in enumerate(current_data):
        #         X = g['x'].to(device)
        #         X_m = prelinear.forward(X)
        #         new_edges=AffinityClustering.AffinityClustering_oneGraph(X_m.detach().cpu(),
        #                                                                  g['edge_index_old'],
        #                                                                  current_degree_matrices[index],
        #                                                                  current_distance_matrices[index],
        #                                                                  cluster_params)
        #         g['edge_index']=new_edges
        for current_data in [current_train_data.graphs, current_val_data.graphs, test_data.graphs]:
            for index, g in enumerate(current_data):
                if args.modify_way == 'add':
                    new_edges = AffinityClustering.add_edges_by_distance_and_type(x=g.x.to(device).float(),
                                                                                     pos=torch.stack([torch.from_numpy(i) for i in g.pos], dim=0).to(device).float(),
                                                                                     old_edges=g['edge_index_old'].to(device),
                                                                                     net_mhc=prelinear_mhc,
                                                                                     net_peptide=prelinear_peptide,
                                                                                     threshold=args.euc_thr)
                elif args.modify_way == 'remove':
                    new_edges = AffinityClustering.remove_edges_by_distance_and_type(x=g.x.to(device).float(),
                                                                                  pos=torch.stack(
                                                                                      [torch.from_numpy(i) for i in
                                                                                       g.pos], dim=0).to(
                                                                                      device).float(),
                                                                                  old_edges=g['edge_index_old'].to(
                                                                                      device),
                                                                                  net_mhc=prelinear_mhc,
                                                                                  net_peptide=prelinear_peptide,
                                                                                  threshold=args.euc_thr)
                elif args.modify_way == 'remove_mix':
                    new_edges = AffinityClustering.remove_edges_by_distance(x=g.x.to(device).float(),
                                                                            pos=torch.stack(
                                                                                [torch.from_numpy(i) for i in
                                                                                 g.pos], dim=0).to(device).float(),
                                                                            old_edges=g['edge_index_old'].to(device),
                                                                            net_mhc=prelinear_mhc,
                                                                            net_peptide=prelinear_peptide,
                                                                            threshold=args.euc_thr)

                g['edge_index'] = new_edges
        print(f'finish! take {time.time() - t} s')

    # if args.weight_data == 'y':
    #     current_data_weight = data_weight.sample()
    # else:
    #     current_data_weight = None

    degree_list=[]
    for current_data in [train_data.graphs, val_data.graphs, test_data.graphs]:
        for g in current_data:
            node_degree=len(g['edge_index'][0])/len(g['x'])
            degree_list.append(node_degree)
    print(f'train_graph_dataset: {np.mean(np.array(degree_list)),len(degree_list)}')

    train_loader = DataLoader(current_train_data.graphs, batch_size=args.batch_size, shuffle=True, num_workers=0,
                             drop_last=False)
    # train_loader2 = DataLoader(current_train_data.graphs, batch_size=args.batch_size, shuffle=True, num_workers=0,
    #                          drop_last=False)
    train_loader2 = DataLoader(current_train_data2.graphs, batch_size=args.batch_size, shuffle=True, num_workers=0,
                               drop_last=False)
    test_loader = DataLoader(test_data.graphs, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    val_loader = DataLoader(current_val_data.graphs, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    lr_lambda = lambda epoch: 0.9 ** (epoch / 100)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    # breakpoint()
    score,best_val_acc,best_val_auc,train_acc_best_model,train_auc_best_model,\
        test_acc_best_model,test_auc_best_model,test_score_best_model,best_model=\
        GNN_core.train_GNN(model=model,
                           train_loader=train_loader,
                           val_loader=val_loader,
                           test_loader=test_loader,
                           optimizer=optimizer,
                           criterion=criterion,
                           n_epochs=args.epochs,
                           patience=args.patience,
                           best_model_criterion=args.best_model_criterion,
                           mu_sigma_criterion=args.mu_sigma_criterion,
                           prelin_update_criterion=args.prelin_update_cri,
                           device=device,
                           early_stop=args.early_stop,
                           drop_ratio=args.drop_ratio,
                           # data_weight=current_data_weight,
                           data_weight=data_weight,
                           train_loader2=train_loader2,
                           data_weight_optimizer=data_weight_optimizer,
                           lam=args.lam,
                           lam2=args.lam2,
                           update_weight=update_weight,
                           clustering=clustering)
    return [(sample_W_mhc, sample_b_mhc), (sample_W_peptide, sample_b_peptide), data_weight, sample_kernel,
            score,best_val_acc,best_val_auc, train_acc_best_model,train_auc_best_model,
            test_acc_best_model,test_auc_best_model,test_score_best_model,best_model, clustering]
def train():
    results = {}
    for step in range(args.step):
        t = time.time()
        loss_score = []
        prelin_params_mhc = []
        prelin_params_peptide = []
        sample_kernel_list = []
        data_weight_list = []
        train_accs = []
        val_accs = []
        test_accs = []
        train_aucs = []
        val_aucs = []
        test_aucs = []
        test_score_best_model = []
        for sample_idx in range(args.n_sample):
            # if args.weight_data == 'y':
            #      data_weight = GNN_core.data_weight(data_num=len(train_data.graphs), device=device, sigma=0.2)
            if args.weight_data == 'y':
                sample_param_mhc, sample_param_peptide, sample_data_weight, sample_kernel, score, best_val_acc, best_val_auc,\
                    train_acc_best_model, train_auc_best_model,\
                    test_acc_best_model, test_auc_best_model, test_score_best_model, best_model, sample_cluster = \
                    train_onePoint(current_train_data=first_phase_train_data,
                                   current_val_data=first_phase_val_data,
                                   current_train_data2=all_first_phase_train_data,
                                   data_weight_optimizer=None,
                                   data_weight=True, update_weight=args.update_weight)
                # if args.weight_data == 'y':
                torch.save(sample_data_weight.weight, f'{output_path}/weight{sample_idx}.pt')
                torch.save([g.protein_name for g in all_train_data.graphs], f'{output_path}/weight{sample_idx}_protein_names.pt')
                if args.corrupt:
                    torch.save(torch.tensor([G.corrupt for G in all_train_data.graphs]),
                               f'{output_path}/corrupt.pt')
            else:
                sample_data_weight = None
                sample_cluster = None
                sample_kernel = None
                sample_param_mhc, sample_param_peptide, sample_data_weight, _, score, best_val_acc, best_val_auc, \
                    train_acc_best_model, train_auc_best_model, \
                    test_acc_best_model, test_auc_best_model, test_score_best_model, best_model, _ = \
                    train_onePoint(current_train_data=first_phase_train_data,
                                   current_val_data=first_phase_val_data,
                                   current_train_data2=all_first_phase_train_data,
                                   data_weight_optimizer=None,
                                   data_weight=args.weight_data == 'y', update_weight=False,
                                   pretrain_data_weight=sample_data_weight, predefined_clustering=sample_cluster)
            # breakpoint()
            # sample_param_mhc, sample_param_peptide, sample_data_weight, _, score, best_val_acc, best_val_auc, \
            #     train_acc_best_model, train_auc_best_model, \
            #     test_acc_best_model, test_auc_best_model, test_score_best_model, best_model, _ = \
            #     train_onePoint(current_train_data=first_phase_train_data,
            #                    current_val_data=first_phase_val_data,
            #                    current_train_data2=all_first_phase_train_data,
            #                    data_weight_optimizer=None,
            #                    data_weight=args.weight_data == 'y', update_weight=False,
            #                    pretrain_data_weight=sample_data_weight, predefined_clustering=sample_cluster)
            # breakpoint()
            loss_score.append(score)
            prelin_params_mhc.append(sample_param_mhc)
            prelin_params_peptide.append(sample_param_peptide)
            train_accs.append(train_acc_best_model)
            val_accs.append(best_val_acc)
            test_accs.append(test_acc_best_model)
            train_aucs.append(train_auc_best_model)
            val_aucs.append(best_val_auc)
            test_aucs.append(test_auc_best_model)
            data_weight_list.append(sample_data_weight)
            sample_kernel_list.append(sample_kernel)

            torch.save(test_score_best_model, f'{output_path}/predict_test_{sample_idx}.pt')

        # sample_W = torch.stack([i[0] for i in sample_param], dim=0)
        # sample_b = torch.stack([i[1] for i in sample_param], dim=0)
        train_accs = torch.tensor(train_accs)
        val_accs = torch.tensor(val_accs)
        test_accs = torch.tensor(test_accs)
        train_aucs = torch.tensor(train_aucs)
        val_aucs = torch.tensor(val_aucs)
        test_aucs = torch.tensor(test_aucs)

        # breakpoint()

        # writer.add_histogram(f'params_b_mu', prelinear.b_mu, global_step=step)
        # writer.add_scalar(f'params_sigma', prelinear.sigma, global_step=step)
        # writer.add_histogram(f'params_W_mu', prelinear.W_mu, global_step=step)
        #
        # writer.add_histogram(f'acc_train', train_accs, global_step=step)
        # writer.add_histogram(f'acc_val', val_accs, global_step=step)
        # writer.add_histogram(f'acc_test', test_accs, global_step=step)
        # writer.add_histogram(f'auc_train', train_aucs, global_step=step)
        # writer.add_histogram(f'auc_val', val_aucs, global_step=step)
        # writer.add_histogram(f'auc_test', test_aucs, global_step=step)
        #
        # writer.add_scalar(f'acc_train', train_accs.mean().item(), global_step=step)
        # writer.add_scalar(f'acc_val', val_accs.mean().item(), global_step=step)
        # writer.add_scalar(f'acc_test', test_accs.mean().item(), global_step=step)
        # writer.add_scalar(f'auc_train', train_aucs.mean().item(), global_step=step)
        # writer.add_scalar(f'auc_val', val_aucs.mean().item(), global_step=step)
        # writer.add_scalar(f'auc_test', test_aucs.mean().item(), global_step=step)

        # torch.save(test_score_best_model, f'{output_path}/predict_test.pt')
        if args.clustering == 'y':
            loss_score = torch.tensor(loss_score)
            if step < 10:
                offset = - loss_score.min()
                # offset = - torch.median(loss_score)
                # offset = 9
            objective_func = loss_score + offset
            objective_func[objective_func < 0] = 0.0

            prelinear_mhc.update(samples=prelin_params_mhc, objective_func=objective_func)
            prelinear_peptide.update(samples=prelin_params_peptide, objective_func=objective_func)

            if step < 10:
                prelinear_mhc.set_sigma(max(prelinear_mhc.sigma, torch.ones_like(prelinear_mhc.sigma)))
                prelinear_peptide.set_sigma(max(prelinear_peptide.sigma, torch.ones_like(prelinear_peptide.sigma)))

            if prelinear_mhc.sigma < 0.1 and prelinear_peptide.sigma < 0.1:
                break

        if args.kernel_weight:
            loss_score = torch.tensor(loss_score)
            if step < 10:
                offset = - loss_score.min()
                # offset = - torch.median(loss_score)
                # offset = 9
            objective_func = loss_score + offset
            objective_func[objective_func < 0] = 0.0
            kernel.update(sample_kernel_list, objective_func=objective_func)
            # breakpoint()
            torch.save(kernel.weight_mean, f'{output_path}/kernel{step}.pt')

        if args.weight_data == 'y':
            loss_score = torch.tensor(loss_score)
            if step < 10:
                offset = - loss_score.min()
                # offset = - torch.median(loss_score)
                # offset = 9
            objective_func = loss_score + offset
            objective_func[objective_func < 0] = 0.0
            if args.weight_data == 'y':
                torch.save(sample_data_weight.weight, f'{output_path}/weight.pt')
                if args.corrupt:
                    torch.save(torch.tensor([G.corrupt for G in all_train_data.graphs]), f'{output_path}/corrupt.pt')
            # data_weight.update(sample=torch.stack(data_weight_list, dim=0),objective_func=objective_func)

        print(f'finish step {step}, take time {time.time()-t} s')
        if args.clustering == 'y':
            print(f'MHC sigma: {prelinear_mhc.sigma.item()}')
            print(f'peptide sigma: {prelinear_peptide.sigma.item()}')
            print(f'loss score:', loss_score + offset, 'offset:', offset)
        else:
            print(f'loss score:', loss_score)
        print(f'acc train: {train_accs.mean().item()}, auc train: {train_aucs.mean().item()}')
        print(f'acc val: {val_accs.mean().item()}, auc val: {val_aucs.mean().item()}')
        print(f'acc test: {test_accs.mean().item()}, auc test: {test_aucs.mean().item()}')
        s = ""
        s += f'finish step {step}, take time {time.time()-t} s\n'
        if args.clustering == 'y':
            s += f'MHC sigma: {prelinear_mhc.sigma.item()}\n'
            s += f'peptide sigma: {prelinear_peptide.sigma.item()}\n'
            s += f'loss score: {loss_score + offset}, offset: {offset}\n'
        else:
            s += f'loss score: {loss_score}\n'
        s += f'acc train: {train_accs.mean().item()}, auc train: {train_aucs.mean().item()}\n'
        s += f'acc val: {val_accs.mean().item()}, auc val: {val_aucs.mean().item()}\n'
        s += f'acc test: {test_accs.mean().item()}, auc test: {test_aucs.mean().item()}\n'
        results[step] = {'acc_train': train_accs, 'auc_train': train_aucs,
                         'acc_val': val_accs, 'auc_val': val_aucs, 'acc_test': test_accs, 'auc_test': test_aucs}
        with open(f'{output_path}/output.log', 'a') as f:
            f.write(s)
        torch.save(results, f'{output_path}/output.pt')


if __name__ == '__main__':
    seed = 12345

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train()
