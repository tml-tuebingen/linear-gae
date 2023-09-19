import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import os

matplotlib.rcParams.update({'text.usetex': True,
                            'font.family': 'serif',
                            'font.size': 8,
                            'axes.titlesize': 9,
                            'xtick.labelsize': 7,
                            'ytick.labelsize': 7,
                            'legend.fontsize': 8,
                            'figure.titlesize': 9,
                            'text.latex.preamble': r'\usepackage{amsmath}'
                                                   r'\newcommand{\cora}{\texttt{Cora} }'
                                                   r'\newcommand{\cseer}{\texttt{Citeseer} }'
                                                   r'\newcommand{\pubmed}{\texttt{Pubmed} }'})

WIDTH = 6.2
HEIGHT = 1.3

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
'#f781bf', '#a65628', '#984ea3',
'#999999', '#e41a1c', '#dede00']

color_one = CB_color_cycle[0]
color_two = CB_color_cycle[1]

def gaussian_edge(folder):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), sharey=True)

    ax1_top = ax1.twiny()
    ax2_top = ax2.twiny()

    ################################################################
    # collect all data csvs model 2
    files = sorted([folder + '/' + f for f in os.listdir(folder) if
                    (os.path.isfile(folder + '/' + f)) & ('edge' in f) & ('MODEL2' in f)])
    tests = [pd.read_csv(f) for f in files if 'test' in f]
    trains = [pd.read_csv(f) for f in files if 'train' in f]
    params = [pd.read_csv(f) for f in files if 'params' in f]
    dists = np.array([eval(p['alignment'][0].replace('nan', 'None'))[1] for p in params])
    idx = np.argsort(dists)

    ################################################################
    # FEATURELESS
    files_fl = sorted([folder + '_featureless' + '/' + f for f in os.listdir(folder + '_featureless') if
                       (os.path.isfile(folder + '_featureless' + '/' + f)) & ('edge' in f) & ('MODEL2' in f)])
    tests_fl = [pd.read_csv(f) for f in files_fl if 'test' in f]
    trains_fl = [pd.read_csv(f) for f in files_fl if 'train' in f]

    ################################################################
    # PLOT FOR LABEL
    position = 1
    i = idx[0]
    te = tests[i]['test auc'].values
    tr = trains[i]['train auc'].values
    ax1.scatter([position - 0.15] * len(te), te, color=color_one, alpha=0.5,
                label='test performance', s=20)
    ax1.scatter([position + 0.15] * len(tr), tr, color=color_two, alpha=0.5,
                label='train performance', s=20)
    position += 1

    ################################################################
    # ITERATE
    for i in idx[1:]:
        te = tests[i]['test auc'].values
        tr = trains[i]['train auc'].values
        ax1.scatter([position - 0.15] * len(te), te, color=color_one, alpha=0.5, s=20)
        ax1.scatter([position + 0.15] * len(tr), tr, color=color_two, alpha=0.5, s=20)
        position += 1

    ################################################################
    # PLOT FEATURELESS
    position += 1
    te = tests_fl[0]['test auc'].values
    tr = trains_fl[0]['train auc'].values
    ax1.scatter([position - 0.15] * len(te), te, color=color_one, alpha=0.5, s=20)
    ax1.scatter([position + 0.15] * len(tr), tr, color=color_two, alpha=0.5, s=20)

    ################################################################
    # AXES AND LEGENDS
    ax1.set_xlim(0.5, 9.5)
    ax1_top.set_xlim(0.5, 9.5)
    ax1.set_ylim(0.5, 1)
    ax1.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax1.set_title(r'\texttt{GCN}')
    ax1.set_ylabel('AUC')
    ax1.set_xticks(np.concatenate([np.arange(7) + 1, [9]]), list(np.round(dists[idx]/64, 2)) + ['feature\nless'],
                   rotation=70)
    ax1_top.set_xticks(np.arange(7) + 1, [64, 32, 16, 8, 4, 2, 0], rotation=70)
    ax1_top.set_xlabel('overlapping dimensions $d$')
    ax1.set_xlabel("normalized misalignment", labelpad=-3)

    ################################################################
    # collect all data csvs MODEL1
    files = sorted([folder + '/' + f for f in os.listdir(folder) if
                    (os.path.isfile(folder + '/' + f)) & ('edge' in f) & ('MODEL1' in f)])
    tests = [pd.read_csv(f) for f in files if 'test' in f]
    trains = [pd.read_csv(f) for f in files if 'train' in f]
    params = [pd.read_csv(f) for f in files if 'params' in f]
    dists = np.array([eval(p['alignment'][0].replace('nan', 'None'))[1] for p in params])
    idx = np.argsort(dists)

    ################################################################
    # FEATURELSS
    files_fl = sorted([folder + '_featureless' + '/' + f for f in os.listdir(folder + '_featureless') if
                       (os.path.isfile(folder + '_featureless' + '/' + f)) & ('edge' in f) & ('MODEL1' in f)])
    tests_fl = [pd.read_csv(f) for f in files_fl if 'test' in f]
    trains_fl = [pd.read_csv(f) for f in files_fl if 'train' in f]

    ################################################################
    position = 1
    for i in idx:
        te = tests[i]['test auc'].values
        tr = trains[i]['train auc'].values
        ax2.scatter([position - 0.15] * len(te), te, color=color_one, alpha=0.5, s=20)
        ax2.scatter([position + 0.15] * len(tr), tr, color=color_two, alpha=0.5, s=20)
        position += 1

    ################################################################
    # FEATURELESS
    position += 1
    te = tests_fl[0]['test auc'].values
    tr = trains_fl[0]['train auc'].values
    ax2.scatter([position - 0.15] * len(te), te, color=color_one, alpha=0.5, s=20)
    ax2.scatter([position + 0.15] * len(tr), tr, color=color_two, alpha=0.5, s=20)

    ################################################################
    # AXES AND LEGENDS
    ax2.set_xlim(0.5, 9.5)
    ax2_top.set_xlim(0.5, 9.5)
    ax2.set_title(r'\texttt{linear}')
    ax2.set_xticks(np.concatenate([np.arange(7) + 1, [9]]), list(np.round(dists[idx]/64, 2)) + ['feature\nless'],
                   rotation=70)
    ax2_top.set_xticks(np.arange(7) + 1, [64, 32, 16, 8, 4, 2, 0], rotation=70)
    ax2_top.set_xlabel('overlapping dimensions $d$')
    ax2.set_xlabel("normalized misalignment", labelpad=-3)
    # fig.legend(bbox_to_anchor=[0.5, 1.19], loc='upper center', ncol=2)

    ax1.legend(handletextpad=0, borderpad=0.1, frameon=False, loc='lower left')

    plt.savefig('figures/gaussian_edge.tex', format='pgf', bbox_inches='tight')


def gaussian_node(folder):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), sharey=True)

    ax1_top = ax1.twiny()
    ax2_top = ax2.twiny()

    ################################################################
    # collect all data csvs Model2
    files = sorted([folder + '/' + f for f in os.listdir(folder) if
                    (os.path.isfile(folder + '/' + f)) & ('node' in f) & ('MODEL2' in f)])
    tests = [pd.read_csv(f) for f in files if 'test' in f]
    trains = [pd.read_csv(f) for f in files if 'train' in f]
    params = [pd.read_csv(f) for f in files if 'params' in f]
    dists = np.array([eval(p['alignment'][0].replace('nan', 'None'))[1] for p in params])
    idx = np.argsort(dists)

    ################################################################
    # FEATURELESS
    files_fl = sorted([folder + '_featureless' + '/' + f for f in os.listdir(folder + '_featureless') if
                       (os.path.isfile(folder + '_featureless' + '/' + f)) & ('node' in f) & ('MODEL2' in f)])
    tests_fl = [pd.read_csv(f) for f in files_fl if 'test' in f]
    trains_fl = [pd.read_csv(f) for f in files_fl if 'train' in f]

    ################################################################
    # PLOT FOR LEGEND
    position = 1
    i = idx[0]
    te = tests[i]['test auc'].values
    tr = trains[i]['train auc'].values
    ax1.scatter([position - 0.15] * len(te), te, color=color_one, alpha=0.5,
                label='test performance', s=20)
    ax1.scatter([position + 0.15] * len(tr), tr, color=color_two, alpha=0.5,
                label='train performance', s=20)

    ################################################################
    # ITERATE
    position += 1
    for i in idx[1:]:
        te = tests[i]['test auc'].values
        tr = trains[i]['train auc'].values
        ax1.scatter([position - 0.15] * len(te), te, color=color_one, alpha=0.5, s=20)
        ax1.scatter([position + 0.15] * len(tr), tr, color=color_two, alpha=0.5, s=20)
        position += 1

    ################################################################
    # PLOT FEATURELESS
    position += 1
    te = tests_fl[0]['test auc'].values
    tr = trains_fl[0]['train auc'].values
    ax1.scatter([position - 0.15] * len(te), te, color=color_one, alpha=0.5, s=20)
    ax1.scatter([position + 0.15] * len(tr), tr, color=color_two, alpha=0.5, s=20)

    ################################################################
    # AXES AND LEGEND
    ax1.set_ylim(0.5, 1)
    ax1.set_xlim(0.5, 9.5)
    ax1_top.set_xlim(0.5, 9.5)
    ax1.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax1.set_title(r'\texttt{GCN}')
    ax1.set_ylabel('AUC')
    ax1.set_xticks(np.concatenate([np.arange(7) + 1, [9]]), list(np.round(dists[idx]/64, 2)) + ['feature\nless'], rotation=70)
    ax1_top.set_xticks(np.arange(7) + 1, [64, 32, 16, 8, 4, 2, 0], rotation=70)
    ax1_top.set_xlabel('overlapping dimensions $d$')
    ax1.set_xlabel("normalized misalignment", labelpad=-3)

    ################################################################
    # collect all data csvs MODEL1
    files = sorted([folder + '/' + f for f in os.listdir(folder) if
                    (os.path.isfile(folder + '/' + f)) & ('node' in f) & ('MODEL1' in f)])
    tests = [pd.read_csv(f) for f in files if 'test' in f]
    trains = [pd.read_csv(f) for f in files if 'train' in f]
    params = [pd.read_csv(f) for f in files if 'params' in f]
    dists = np.array([eval(p['alignment'][0].replace('nan', 'None'))[1] for p in params])
    idx = np.argsort(dists)

    ################################################################
    # FEATURELESS
    files_fl = sorted([folder + '_featureless' + '/' + f for f in os.listdir(folder + '_featureless') if
                       (os.path.isfile(folder + '_featureless' + '/' + f)) & ('node' in f) & ('MODEL1' in f)])
    tests_fl = [pd.read_csv(f) for f in files_fl if 'test' in f]
    trains_fl = [pd.read_csv(f) for f in files_fl if 'train' in f]

    ################################################################
    # ITERATE
    position = 1
    for i in idx:
        te = tests[i]['test auc'].values
        tr = trains[i]['train auc'].values
        ax2.scatter([position - 0.15] * len(te), te, color=color_one, alpha=0.5, s=20)
        ax2.scatter([position + 0.15] * len(tr), tr, color=color_two, alpha=0.5, s=20)
        position += 1

    ################################################################
    # FEATURELESS
    position += 1
    te = tests_fl[0]['test auc'].values
    tr = trains_fl[0]['train auc'].values
    ax2.scatter([position - 0.15] * len(te), te, color=color_one, alpha=0.5, s=20)
    ax2.scatter([position + 0.15] * len(tr), tr, color=color_two, alpha=0.5, s=20)

    ################################################################
    # AXES AND LEGENDS
    ax2.set_title(r'\texttt{linear}')
    ax2.set_xlim(0.5, 9.5)
    ax2_top.set_xlim(0.5, 9.5)
    ax2.set_xticks(np.concatenate([np.arange(7) + 1, [9]]), list(np.round(dists[idx]/64, 2)) + ['feature\nless'], rotation=70)
    ax2_top.set_xticks(np.arange(7) + 1, [64, 32, 16, 8, 4, 2, 0], rotation=70)
    ax2_top.set_xlabel('overlapping dimensions $d$')
    ax2.set_xlabel("normalized misalignment", labelpad=-3)
    # fig.legend(bbox_to_anchor=[0.5, 1.4], loc='upper center', ncol=2)

    ax1.legend(handletextpad=0, borderpad=0.1, frameon=False, loc='lower left')
    plt.subplots_adjust(0.1)
    plt.savefig('figures/gaussian_node.tex', format='pgf', bbox_inches='tight')


def real_world(folder):
    dff = pd.read_csv(folder + '/summary.csv')
    df = dff[dff.task == 'edge']

#    fig, ([[ax11, ax12, ax13], [ax21, ax22, ax23]]) = plt.subplots(2, 3, figsize=(WIDTH, 2), sharey="row", sharex="col")
    fig, ([ax11, ax12, ax13]) = plt.subplots(1, 3, figsize=(2.5, 1.5), sharey="row")

    cora = df[df.dataset == 'cora']
    citeseer = df[df.dataset == 'citeseer']
    pubmed = df[df.dataset == 'pubmed']

    cmp_plot(ax11, cora, 1)
    ax11.set_title('cora')
    cmp_plot(ax12, citeseer)
    ax12.set_title('citeseer')
    cmp_plot(ax13, pubmed)
    ax13.set_title('pubmed')

    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    fig.legend(bbox_to_anchor=[0.5, 1], loc='lower center', frameon=False)
    ax11.set_ylim(0.75, 1)
    ax11.set_yticks(np.arange(0.75, 1.01, 0.05))
    ax11.set_ylabel('AUC link prediction')

    plt.savefig('figures/real_world1.tex', format='pgf', bbox_inches='tight')

    fig, ([ax11, ax12, ax13]) = plt.subplots(1, 3, figsize=(2.5, 1.5), sharey="row")

    df = dff[dff.task == 'node']

    cora = df[df.dataset == 'cora']
    citeseer = df[df.dataset == 'citeseer']
    pubmed = df[(df.dataset == 'pubmed') & ((df.out_dim == 4) | (df.featureless == True))]

    cmp_plot(ax11, cora, 2)
    ax11.set_title('cora')
    cmp_plot(ax12, citeseer)
    ax12.set_title('citeseer')
    cmp_plot(ax13, pubmed)
    ax13.set_title('pubmed')

    ax11.set_yticks(np.arange(0.95, 1.01, 0.01))
    ax11.set_ylabel('AUC node prediction')
    fig.legend(bbox_to_anchor=[0.5, 1], loc='lower center', frameon=False)


    fig.subplots_adjust(wspace=0.1, hspace=0.3)

    plt.savefig('figures/real_world2.tex', format='pgf', bbox_inches='tight')


def real_world_node(folder):
    df = pd.read_csv(folder + '/summary.csv')
    df = df[df.task == 'node']

    cora = df[df.dataset == 'cora']
    citeseer = df[df.dataset == 'citeseer']
    pubmed = df[(df.dataset == 'pubmed') & ((df.out_dim == 4) | (df.featureless == True))]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(WIDTH, HEIGHT), sharey='row')

    cmp_plot(ax1, cora, True)
    ax1.set_title('\cora')
    cmp_plot(ax2, citeseer)
    ax2.set_title('\cseer')
    cmp_plot(ax3, pubmed)
    ax3.set_title('\pubmed')

    fig.legend(bbox_to_anchor=[0.5, 1.05], loc='lower center', ncol=2)
    ax1.set_ylim(0.95, 1)
    ax1.set_yticks(np.arange(0.95, 1.01, 0.01))
    ax1.set_ylabel('AUC')

    plt.savefig('figures/real_world_node', format='pgf', bbox_inches='tight')


def cmp_plot(ax, data, label=False):
    value = data[(data.featureless == False) & (data.model == 2)]['test_auc_mean']
    error = data[(data.featureless == False) & (data.model == 2)]['test_auc_std']
    pos = 1
    color = color_one
    ax.scatter([pos] * len(value), value, color=color, marker='_', s=100)

    value = data[(data.featureless == False) & (data.model == 1)]['test_auc_mean']
    error = data[(data.featureless == False) & (data.model == 1)]['test_auc_std']
    pos = 2
    color = color_one
    if label == 1:
        ax.scatter([pos] * len(value), value, color=color, label='with features', marker='_', s=100)
    else:
        ax.scatter([pos] * len(value), value, color=color, marker='_', s=100)
    ax.errorbar(pos, value, error, ecolor=color)

    value = data[(data.featureless == True) & (data.model == 2)]['test_auc_mean']
    error = data[(data.featureless == True) & (data.model == 2)]['test_auc_std']
    pos = 1
    color = color_two
    ax.scatter([pos] * len(value), value, color=color, marker='_', s=100)
    ax.errorbar(pos, value, error, ecolor=color)

    value = data[(data.featureless == True) & (data.model == 1)]['test_auc_mean']
    error = data[(data.featureless == True) & (data.model == 2)]['test_auc_std']
    pos = 2
    color = color_two
    if label == 2:
        ax.scatter([pos] * len(value), value, color=color, label='featureless', marker='_', s=100)
    else:
        ax.scatter([pos] * len(value), value, color=color, marker='_', s=100)
    ax.errorbar(pos, value, error, ecolor=color)

    ax.set_xticks([1, 2], ['relu', 'linear'], rotation=45)
    ax.set_xlim(0.5, 2.5)


if __name__ == '__main__':
    gaussian_node('../results/cluster/gaussian_dot_innerproduct_indim64')
    gaussian_edge('../results/cluster/gaussian_dot_innerproduct_indim64')
    real_world('../results/cluster_new')
