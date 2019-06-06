import os, sys, random, itertools, json
dir_path = os.path.dirname(os.path.realpath(__file__))
import scipy.stats
from scipy.stats import ks_2samp
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
import seaborn as sns
import networkx as nx
from matplotlib.ticker import ScalarFormatter

import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 4.0)

DEL_THRESHOLD = 0

sns.set_context(
    "talk",
    font_scale=1,
    rc={
        "lines.linewidth": 2.5,
        "text.usetex": False,
        "font.family": 'serif',
        "font.serif": ['Palatino'],
        "font.size": 16
    })

sns.set_style('white')

################
# MISC
################

get_full_path = lambda p: os.path.join(os.path.dirname(os.path.abspath(__file__)), p)

def ig_to_nx(ig_graph, directed=False, nodes=None):
    g = nx.DiGraph() if directed else nx.Graph()
    nodes = nodes if nodes else ig_graph.vs
    edges = ig_graph.induced_subgraph(nodes).es if nodes else ig_graph.es
    for node in nodes: g.add_node(node.index, **node.attributes())
    for edge in edges: g.add_edge(edge.source, edge.target)
    return g

def _get_deg_dist(degree, del_threshold=DEL_THRESHOLD):
    counter = Counter(degree)
    for k, v in counter.items():
        if v <= del_threshold: del counter[k]
    deg, freq = map(np.array, zip(*sorted(counter.items())))
    prob = freq/float(np.sum(freq))
    return deg, prob

def renormalize_deg_dist(xk, pk, xk2):
    xk_, pk_ = [], []
    xk2_set = set(xk2)
    for x,p in zip(xk,pk):
        if x in xk2_set:
            xk_.append(x);pk_.append(p)
    xk_, pk_ = map(np.array, [xk_,pk_])
    pk_ /= np.sum(pk_)
    return xk_, pk_

def get_avg_nbor_indeg(graph, out_to_in=True):
    """Avg nbor indegree given node outdegree"""
    d = defaultdict(list)
    for node in graph.vs:
        deg = node.outdegree() if out_to_in else node.indegree()
        for nbor in node.neighbors(mode='OUT'):
            d[deg].append(nbor.indegree())
    d = {k:np.mean(v) for k,v in dict(d).items()}
    return map(np.array, zip(*sorted(d.items())))

def compute_indeg_map(graph, time_attr):
    src_tgt_indeg_map = defaultdict(dict)
    for node in graph.vs:
        nid = node.index
        nbors = [n for n in node.neighbors(mode='IN') if n[time_attr] == n[time_attr]]
        nbors = sorted(nbors, key=lambda n: n[time_attr])
        for indeg, nbor in enumerate(nbors): src_tgt_indeg_map[nbor.index][nid] = indeg
    return dict(src_tgt_indeg_map)

def get_chunk_degree_sequence(graph, time_attr, return_keys=True, use_median=False, get_int=True):
    outdegs, chunks = defaultdict(list), defaultdict(int)
    for node in graph.vs:
        time, outdeg = node[time_attr], node.outdegree()
        outdegs[time].append(outdeg)
        chunks[time] += 1
    f = np.median if use_median else np.mean
    outdegs = {k: int(round(f(v))) if get_int else f(v) for k,v in outdegs.items() if k in chunks}
    skeys = sorted(chunks.keys())
    chunks = [chunks[k] for k in skeys]
    outdegs = [outdegs[k] for k in skeys]
    if return_keys: return skeys, chunks, outdegs
    return chunks, outdegs

def get_config_model(g):
    config = ig.Graph.Degree_Sequence(g.vs.outdegree(), g.vs.indegree())
    for attr in g.vs.attributes(): config.vs[attr] = g.vs[attr]
    return config

def get_time_config_model(g, time_attr, debug=False):
    """
    configuration model controlled for time
    node u cannot link to nodes that join network in year  > year_u
    """
    indeg_map = dict(zip(g.vs.indices, g.vs.indegree()))
    outdeg_map = dict(zip(g.vs.indices, g.vs.outdegree()))
    year_nodes_map = defaultdict(list)
    for node in g.vs: year_nodes_map[node[time_attr]].append(node.index)
    years = sorted(year_nodes_map.keys())
    instubs, new_edges = [], []

    if debug: print ("{} time_attr vals".format(len(years)))
    for year in years:
        if debug: print year,
        new_nodes = year_nodes_map[year]
        for nid in new_nodes: instubs.extend([nid]*indeg_map[nid])
        random.shuffle(instubs)
        for nid in new_nodes:
            for _ in xrange(outdeg_map[nid]):
                new_edges.append((nid, instubs.pop()))

    config = g.copy()
    config.delete_edges(None)
    config.add_edges(new_edges)
    return config

def bin_clustering(cc, step=0.005):
    cc = pd.Series(cc)
    bins = np.arange(0, 1+step, step)
    cc = pd.cut(cc, bins=bins, right=False).value_counts()
    cc = cc.astype(float)/np.sum(cc)
    cc.index = np.arange(0, 1, step) #+ step/2.
    return cc

def clustering(graph, undirected_max=False, get_indeg=False, del_threshold=DEL_THRESHOLD):
    N = len(graph.vs)
    E_counter = [0.]*N

    # "parents" map
    target_map = defaultdict(set)
    for nhood in graph.neighborhood(mode='OUT'):
        target_map[nhood[0]] = set(nhood[1:])

    # increment edge counter of common parents
    for edge in graph.es:
        n1, n2 = edge.source, edge.target
        common_targets = target_map[n1].intersection(target_map[n2])
        for tgt in common_targets: E_counter[tgt] += 1

    cc, nids, degs = [], [], []
    for e, indeg, nid in zip(E_counter, graph.indegree(), graph.vs.indices):
        if indeg <= 1: continue
        degs.append(indeg)
        cc.append(e/(indeg*(indeg-1)))
        nids.append(nid)

    cc, nids, degs = map(np.array, [cc, nids, degs])
    cc, nids, degs = cc[degs>del_threshold], nids[degs>del_threshold], degs[degs>del_threshold]
    if undirected_max: cc *= 2
    if get_indeg: return cc, nids, degs
    return cc, nids

def ego_clustering(graph, attr_name, same_attr=True, undirected_max=False, get_indeg=False, del_threshold=DEL_THRESHOLD):
    N = len(graph.vs)
    E_counter = [0.]*N

    nid_attr_map = dict(zip(graph.vs.indices, graph.vs[attr_name]))

    # "parents" map
    target_map = defaultdict(set)
    for nhood in graph.neighborhood(mode='OUT'):
        target_map[nhood[0]] = set(nhood[1:])

    # increment edge counter of common parents
    for edge in graph.es:
        n1, n2 = edge.source, edge.target
        same_attr_val = nid_attr_map[n1] == nid_attr_map[n2]
        if same_attr != same_attr_val: continue
        common_targets = target_map[n1].intersection(target_map[n2])
        for tgt in common_targets: E_counter[tgt] += 1

    cc, nids, degs = [], [], []
    for e, indeg, nid in zip(E_counter, graph.indegree(), graph.vs.indices):
        if indeg <= 1: continue
        degs.append(indeg)
        cc.append(e/(indeg*(indeg-1)))
        nids.append(nid)

    cc, nids, degs = map(np.array, [cc, nids, degs])
    cc, nids, degs = cc[degs>del_threshold], nids[degs>del_threshold], degs[degs>del_threshold]
    if undirected_max: cc *= 2
    if get_indeg: return cc, nids, degs
    return cc, nids

def avg_clustering_degree(graph, undirected_max=False, del_threshold=DEL_THRESHOLD, data=None):
    if data: cc, nids, indeg = data
    else: cc, nids, indeg = clustering(graph, undirected_max=undirected_max, get_indeg=True)
    indeg_map = defaultdict(list)
    for cc_, indeg_ in zip(cc, indeg): indeg_map[indeg_].append(cc_)
    indeg_map = {k: np.mean(v) for k,v in indeg_map.items() if k > del_threshold}
    return zip(*sorted(indeg_map.items()))

def clustering_distribution(graph, undirected_max=False, del_threshold=DEL_THRESHOLD, step=0.015):
    cc, nids = clustering(graph, undirected_max=undirected_max, del_threshold=del_threshold)
    cc = bin_clustering(cc, step=step)
    if not undirected_max:
        cc = cc[cc.index <= 0.5]
        cc /= np.sum(cc)
    return cc

def ego_clustering_distribution(graph, attr_name, same_attr=True, undirected_max=False, del_threshold=DEL_THRESHOLD, step=0.015):
    cc, nids = ego_clustering(graph, attr_name, same_attr=same_attr, undirected_max=undirected_max, del_threshold=del_threshold)
    cc = bin_clustering(cc, step=step)
    if not undirected_max:
        cc = cc[cc.index <= 0.5]
        cc /= np.sum(cc)
    return cc

def get_nborhood_diversity_dist(g, attr, deg_mode='IN',ignore_zero_deg=True):
    nid_attr_map = dict(zip(g.vs.indices, g.vs[attr]))
    data = [len(set(nid_attr_map[nbor_nid] for nbor_nid in nhood[1:])) for nhood in g.neighborhood(mode=deg_mode)]
    if ignore_zero_deg: data = [x for x in data if x > 0]
    xk, pk = map(np.array, zip(*sorted(Counter(data).items())))
    pk = pk.astype(float)/np.sum(pk)
    return xk, pk

def get_nbor_degree_df(graph, deg_mode='IN', nbor_mode='OUT'):
    data = []
    for node in graph.vs:
        node_deg = node.degree(mode=deg_mode)
        for nbor in node.neighbors(mode=nbor_mode):
            nbor_deg = nbor.degree(mode=deg_mode)
            data.append((node_deg, nbor_deg))
    return pd.DataFrame(data=data, columns=['citing', 'cited'])

def get_avg_nbor_degree(graph, deg_mode='IN', nbor_mode='OUT'):
    df = get_nbor_degree_df(graph, deg_mode, nbor_mode)
    df2 = df.groupby('citing')['cited'].apply(np.mean)
    s =pd.Series(df2)
    return map(list, [s.index, s])

def _shortest_path_length(graph, N, deg_mode='ALL'):
    nids = np.random.choice(graph.vs.indices, size=N)
    lengths = graph.shortest_paths(source=nids, mode=deg_mode)
    all_lengths = []
    for l in lengths: all_lengths.extend(l)
    return all_lengths

def shortest_path_length_pmf(graph, N, deg_mode='ALL', cdf=False):
    lengths = _shortest_path_length(graph, N, deg_mode)
    length_pmf = Counter(lengths)
    del length_pmf[0]; del length_pmf[np.inf]
    total = float(sum(length_pmf.values()))
    hops, pmf = map(np.array, zip(*sorted({k:v/total for k,v in length_pmf.items()}.items())))
    if cdf: pmf = np.cumsum(pmf)
    return hops, pmf

def get_component_sizes(graph, mode='WEAK'):
    return Counter(graph.components(mode=mode).sizes())

################
# ATTRIBUTES
################

def get_discrete_rv(x, p):
    sum_p = sum(p)
    return scipy.stats.rv_discrete(values=(x,[px/float(sum_p) for px in p]))

def compute_ps(graph, attrs=None):
    attr_count = defaultdict(int)
    if not attrs: attrs = graph.vs.attributes()
    mode = 'OUT' if graph.is_directed() else 'ALL'
    total = 0.
    for node in graph.vs:
        for nbor in node.neighbors(mode='OUT'):
            total += 1
            for attr in attrs:
                if node[attr] == nbor[attr]:
                    attr_count[attr] += 1
    return {k:v/total for k,v in attr_count.items()}

def add_attr_to_graph(graph, df, attr):
    nid_attr = dict(zip(df.index, df[attr]))
    graph.vs[attr] = [nid_attr.get(nid, np.nan) for nid in graph.vs['name']]

def get_attr_dist(g, attr):
    x = np.array(g.vs[attr])
    x = x[~np.isnan(x)]
    xk, pk = map(np.array, zip(*sorted(Counter(x).items())))
    pk = pk.astype(float)/np.sum(pk)
    return xk, pk

def num_matches_distribution(g, attr='attrs'):
    num_matches = []
    for node in g.vs:
        attrs = node[attr]
        for nbor in node.neighbors(mode='OUT'):
            count = 0
            for a in nbor[attr]:
                if a in attrs: count += 1
            num_matches.append(count)
    xk, fk = map(np.array, zip(*sorted(Counter(num_matches).items())))
    pk = fk.astype(float)/fk.sum()
    return np.array(num_matches), xk, pk

def get_edge_attr_counter(g, attr):
    c = defaultdict(lambda: defaultdict(int))
    for node in g.vs:
        for nbor in node.neighbors(mode='OUT'):
            c[node[attr]][nbor[attr]] += 1
    return c

def get_edge_attr_matrix(g, attr, normalize=True, conditional=True, get_map=False):
    unicount = Counter(g.vs[attr])
    n = len(unicount)
    val_idx_map = dict(zip(unicount.keys(), xrange(n)))
    idx_val_map = dict(zip( xrange(n), unicount.keys()))
    bicount = get_edge_attr_counter(g, attr)
    mat = [[0.]*n for _ in xrange(n)]
    total = 0.
    for idx1 in xrange(n):
        for idx2 in xrange(n):
            val1, val2 = idx_val_map[idx1], idx_val_map[idx2]
            mat[idx1][idx2] = bicount[val1][val2]
            total += bicount[val1][val2]

    if not normalize: return mat

    for row in xrange(n):
        if conditional:
            s = float(np.sum(mat[row]))
            if s == 0: continue
            mat[row] = [x/s for x in mat[row]]
        else:
            mat[row] = [x/total for x in mat[row]]

    if get_map: return mat, idx_val_map
    return mat

def get_edge_attr_heatmap(g, attr, normalize=True, conditional=True, ax=None, title=None, **kw):
    ax = ax if ax else plt.subplot()
    mat = get_edge_attr_matrix(g, attr, normalize=normalize, conditional=conditional)
    kw['xticklabels'] = kw.get('xticklabels', False)
    kw['yticklabels'] = kw.get('yticklabels', False)
    sns.heatmap(mat, ax=ax, **kw)
    if title: ax.set_title(title)
    return np.matrix(mat), ax

def get_dpl_exponent(g, time_attr='time', debug=False):
    node_counter, edge_counter = defaultdict(int), defaultdict(int)
    for node in g.vs:
        t = node[time_attr]
        if t != t: continue
        node_counter[t] += 1
        edge_counter[t] += node.outdegree()
    _, nodes = zip(*sorted(node_counter.items()))
    _, edges = zip(*sorted(edge_counter.items()))
    nodes, edges = map(np.cumsum, map(np.array, [nodes, edges]))
    nodes, edges = nodes[edges>0], edges[edges>0]
    exp, b = np.polyfit(np.log10(nodes), np.log10(edges), 1)
    if debug: return exp, nodes, edges
    return exp

def get_source_and_target_dist(g, attr):
    """e_{i.}, e_{.i} distribution"""
    xkpk = lambda c: map(np.array, zip(*sorted(c.items())))

    attr_vals = np.array(g.vs[attr])
    outdeg = np.array(g.vs.outdegree())
    indeg = np.array(g.vs.indegree())

    outdeg = outdeg[~np.isnan(attr_vals)]
    indeg = indeg[~np.isnan(attr_vals)]
    attr_vals = attr_vals[~np.isnan(attr_vals)]

    src_counter = defaultdict(int)
    tgt_counter = defaultdict(int)

    for node_outdeg, node_indeg, node_attr in zip(outdeg, indeg, attr_vals):
        src_counter[node_attr] += node_outdeg
        tgt_counter[node_attr] += node_indeg

    src_xk, src_pk = xkpk(src_counter)
    tgt_xk, tgt_pk = xkpk(tgt_counter)

    src_pk = src_pk.astype(float)/np.sum(src_pk)
    tgt_pk = tgt_pk.astype(float)/np.sum(tgt_pk)

    exp_eii = np.dot(src_pk, tgt_pk)
    return dict(src=(src_xk, src_pk), tgt=(tgt_xk, tgt_pk), c=exp_eii)

def get_xk_pk(g, attr, sort=False):
    c = Counter(g.vs[attr])
    s = float(sum(c.values()))
    c = {k:v/s for k,v in c.items()}
    if sort: return zip(*sorted(c.items(), key=lambda t: t[-1], reverse=True))
    return zip(*c.items())

def get_binned_values(z, num=100):
    bins = np.linspace(.999*np.min(z), np.max(z)*1.001, num=num)
    binned_vals = list(bins[np.searchsorted(bins, z)])
    return _get_deg_dist(binned_vals)

################
# PLOT
################

graph_kw = dict(bbox=(300,300), vertex_size=3, edge_arrow_size=.3, edge_width=.1)

def plot_deg2(graphs, labels=None, cutoff=0, last_graph_baseline=True,
              ls='.-', ax=None, pdf_fname=None, loc='best', deg_mode='IN',
              lw=5, alpha=0.6, cmap='magma', inverse_cdf=True, del_threshold=DEL_THRESHOLD,
              add_mean=True):

    fig = None
    if ax is None: fig, ax = plt.subplots(1,1,figsize=(11,5))
    ax.set_title('In-degree Inverse CDF' if inverse_cdf else 'In-degree Distribution', fontsize=14)
    if not labels: labels = ['graph{}'.format(idx) for idx in range(1, len(graphs)+1)]
    cmap = plt.get_cmap(cmap, len(graphs))
    graphs = [g if type(g) is ig.Graph else g.g for g in graphs]

    if last_graph_baseline:
        graphs, baseline_graph = graphs[:-1], graphs[-1]
        labels, baseline_label = labels[:-1], labels[-1]
        baseline_indeg = baseline_graph.degree(mode=deg_mode)
        obxk, obpk = bxk, bpk = _get_deg_dist(baseline_indeg, del_threshold=del_threshold)

    RADIUS = 0

    for idx, (graph, label) in enumerate(zip(graphs, labels)):
        color = None
        indegs = graph.degree(mode=deg_mode)
        xk, pk = _get_deg_dist(indegs, del_threshold=del_threshold)
        if cutoff > 0: xk, pk = xk[cutoff:], pk[cutoff:]/np.sum(pk[cutoff:])
        if inverse_cdf: pk = 1 - pk.cumsum() + pk

        if last_graph_baseline:
            ks_stat = get_ks_statistic(indegs, baseline_indeg)
            label = "{} (KS: {})".format(label, round(ks_stat, 3))

        ax.plot(xk, pk, ls, label=label, alpha=alpha, color=color, mec='white', lw=lw)

    if last_graph_baseline:
        color = None
        if inverse_cdf: bpk = 1 - bpk.cumsum() + bpk
        ax.plot(bxk, bpk, ls, label=baseline_label, alpha=alpha+0.1, color=color, lw=lw)

    ax.set_xscale('log'); ax.set_yscale('log');
    xlabel = '{}degree $k$'.format(deg_mode.lower())
    ylabel = '$\Pr(K \geq k)$' if inverse_cdf else 'probability'
    title = 'In-degree Distribution'

    if add_mean:
        avg_cc = np.dot(obxk, obpk)
        ax.axvline(avg_cc, ls='--', alpha=0.4, label='average {}degree'.format(deg_mode.lower()))

    set_labels(ax, xlabel, ylabel, title)

    if pdf_fname and fig: plt.savefig('./figs/{}'.format(pdf_fname))
    return ax

def plot_matches(graphs, labels=None,  last_graph_baseline=True,
                ls='o-', ax=None, pdf_fname=None, loc='best', lw=4, alpha=0.6,
                cmap='magma', attr_name='attrs', del_threshold=DEL_THRESHOLD, inverse_cdf=False):

    fig = None
    if ax is None: fig, ax = plt.subplots(1,1,figsize=(11,5))
    ax.set_title('Matches CDF' if inverse_cdf else 'Matches PMF', fontsize=14)
    if not labels: labels = ['graph{}'.format(idx) for idx in range(1, len(graphs)+1)]
    cmap = plt.get_cmap(cmap, len(graphs))
    graphs = [g if type(g) is ig.Graph else g.g for g in graphs]

    if last_graph_baseline:
        graphs, baseline_graph = graphs[:-1], graphs[-1]
        labels, baseline_label = labels[:-1], labels[-1]
        bm, baseline_xk, baseline_pk = num_matches_distribution(baseline_graph, attr=attr_name)

    for idx, (graph, label) in enumerate(zip(graphs, labels)):
        color = None
        m, xk, pk = num_matches_distribution(graph, attr=attr_name)
        if inverse_cdf: pk = 1 - pk.cumsum() + pk

        if last_graph_baseline:
            ks_stat = get_ks_statistic(bm, m)
            label = "{} (KS: {})".format(label, round(ks_stat, 3))

        ax.plot(xk, pk, ls, label=label, alpha=alpha, color=color, lw=lw)

    if last_graph_baseline:
        color = None
        if inverse_cdf: baseline_pk = 1 - baseline_pk.cumsum() + baseline_pk
        ax.plot(baseline_xk, baseline_pk, ls, label=baseline_label, alpha=alpha+0.1, color=color, lw=lw)

    xlabel = r'number of matches $m$'
    ylabel = r'$\Pr(M \geq m)$' if inverse_cdf else r'$\Pr(M = m)$'
    set_labels(ax, xlabel, ylabel, r'Matches Distribution')
    if pdf_fname and fig: plt.savefig('./figs/{}'.format(pdf_fname))
    return ax

def plot_cc_deg2(graphs, labels=None, last_graph_baseline=True,
                 ls='.-', lw=5, ax=None, pdf_fname=None, loc='best', deg_mode='IN',
                 plot_dd=True, logy=False, cdf_cutoff=0.95, alpha=0.6, cmap='magma',
                 del_threshold=DEL_THRESHOLD):

    fig = None
    if ax is None: fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.set_title('Average Local Clustering and {}degree'.format(deg_mode.title()), fontsize=14)
    if not labels: labels = ['graph{}'.format(idx) for idx in range(1, len(graphs)+1)]
    cmap = plt.get_cmap(cmap, len(graphs))
    graphs = [g if type(g) is ig.Graph else g.g for g in graphs]
    idx = 0

    if last_graph_baseline:
        graphs, baseline_graph = graphs[:-1], graphs[-1]
        labels, baseline_label = labels[:-1], labels[-1]

        baseline_cc_data = clustering(baseline_graph, get_indeg=True)
        baseline_cc_xk, baseline_cc = avg_clustering_degree(baseline_graph, data=baseline_cc_data, del_threshold=del_threshold)

        baseline_deg = baseline_graph.degree(mode=deg_mode)
        bxk, bpk = _get_deg_dist(baseline_deg, del_threshold=del_threshold)
        idx = list(bxk).index(baseline_cc_xk[0])
        bxk, bpk = bxk[idx:], bpk[idx:]/bpk[idx:].sum()

    for idx, (graph, label) in enumerate(zip(graphs, labels)):
        color = None #cmap(idx)
        xk, pk = avg_clustering_degree(graph, del_threshold=del_threshold)

        if last_graph_baseline:
            cc_dist = get_cc_distance(xk, pk, baseline_cc_xk, baseline_cc, bxk, bpk, relative=True)
            label = "{} (WRE: {})".format(label, round(cc_dist, 3))

        ax.plot(xk, pk, ls, label=label, alpha=alpha, color=color, lw=lw)

    if last_graph_baseline:
        ax.plot(baseline_cc_xk, baseline_cc, ls, label=baseline_label, alpha=alpha, lw=lw)

    ax.set_xscale('log')
    if logy: ax.set_yscale('log')
    set_labels(ax, "{}degree".format(deg_mode.lower()), 'avg clustering')

    if plot_dd:
       cdf = np.cumsum(bpk)
       cutoff_idx = len(cdf[cdf < cdf_cutoff])+1
       cutoff_xval = bxk[cutoff_idx]
       ax.axvline(x=cutoff_xval, linestyle='--', lw=3, alpha=0.3, label='cdf={}'.format(cdf_cutoff))
       # twin_ax = ax.twinx()
       # twin_ax.plot(bxk, cdf, '.', ms=12, alpha=.3, color='black', label='degree dist')
       # twin_ax.set_ylabel('p(degree)')
       # tickify(twin_ax)
       # twin_ax.legend(loc=1)

    if pdf_fname and fig: plt.savefig('./figs/{}'.format(pdf_fname))
    return ax

def plot_deg_and_deg_cc(graphs, labels=None, last_graph_baseline=True,
                        ls='.-', dc_ls='o', lw=5, axs=None, pdf_fname=None, cc_loc='best', deg_mode='IN',
                        plot_dd=True, logy=False, cutoff=0, deg_loc='best', cc_mode='IN',
                        cmap='magma', del_threshold=DEL_THRESHOLD, dd_inverse_cdf=False, add_mean=True):
    fig = None

    if not axs: fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20, 6))
    else: (ax1, ax2) = axs

    cc_ax = plot_cc_deg2(graphs, labels=labels, last_graph_baseline=last_graph_baseline, ls=dc_ls, lw=lw,
                         ax=ax2, pdf_fname=None, loc=cc_loc, deg_mode=cc_mode, plot_dd=plot_dd,
                         logy=logy, cmap=cmap, del_threshold=del_threshold)

    dd_ax = plot_deg2(graphs, labels=labels, cutoff=cutoff, last_graph_baseline=last_graph_baseline,
                      ls=ls, ax=ax1, pdf_fname=None, loc=deg_loc, deg_mode=deg_mode, lw=lw,
                      cmap=cmap, del_threshold=del_threshold, inverse_cdf=dd_inverse_cdf, add_mean=add_mean)

    if pdf_fname: fig.savefig(pdf_fname)

    return fig, (ax1, ax2)

def plot_deg_and_cc_and_deg_cc(graphs, labels=None, last_graph_baseline=True, get_atty=True, attr_name='single_attr',
                              ls='.-', dc_ls='o', lw=5, axs=None, pdf_fname=None, cc_loc='best', deg_mode='IN',
                              plot_dd=False, logy=False, cutoff=0, deg_loc='best', cc_mode='IN',
                              undefined='nan', undirected_max=False, step=0.025, cc_plot_cdf=False,
                              cc_loglog=True, cmap='magma', del_threshold=DEL_THRESHOLD, dd_inverse_cdf=True,
                              figcaption=None, add_cc_mean=True, add_dd_mean=True):

    if get_atty:
        for g,l in zip(graphs, labels):
            print "{}: {:.3f}".format(l, g.assortativity_nominal(attr_name))

    fig = None
    if not axs: fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(25,6))
    else: (ax1, ax2, ax3) = axs

    ax2,ax3=ax3,ax2

    plot_deg_and_deg_cc(graphs, labels, last_graph_baseline=last_graph_baseline, ls=ls, dc_ls=dc_ls,
                        lw=lw, axs=(ax1, ax2), pdf_fname=None, cc_loc=cc_loc, deg_mode=deg_mode, cc_mode=cc_mode,
                        plot_dd=plot_dd, logy=logy, cutoff=cutoff, deg_loc=deg_loc, cmap=cmap,
                        del_threshold=del_threshold, dd_inverse_cdf=dd_inverse_cdf, add_mean=add_dd_mean)

    plot_binned_clustering_dist_graphs(graphs, labels=labels, ax=ax3, last_graph_baseline=last_graph_baseline,
                                       undefined=undefined, mode=cc_mode, undirected_max=undirected_max, step=step,
                                       plot_cdf=cc_plot_cdf, loglog=cc_loglog, cmap=cmap, add_mean=add_cc_mean)

    if pdf_fname: fig.savefig(pdf_fname)
    if figcaption: plt.gcf().suptitle(figcaption, y=1)
    return fig, (ax1,ax3,ax2)

def plot_all(graphs, labels=None, last_graph_baseline=True,
              ls='.-', dc_ls='o', lw=5, axs=None, pdf_fname=None, cc_loc='best', deg_mode='IN',
              plot_dd=False, logy=False, cutoff=0, deg_loc='best', cc_mode='IN',
              undefined='nan', undirected_max=False, step=0.025, cc_plot_cdf=False,
              cc_loglog=True, cmap='magma', del_threshold=DEL_THRESHOLD, dd_inverse_cdf=True,
              figcaption=None, add_cc_mean=True, add_dd_mean=True, attr_name='attrs', attr_inverse_cdf=False):

    fig = None
    if not axs: fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4,figsize=(30,6))
    else: (ax1, ax2, ax3, ax4) = axs


    plot_deg_and_cc_and_deg_cc(graphs, labels, last_graph_baseline=last_graph_baseline,
                                  ls=ls, dc_ls=dc_ls, lw=lw, axs=(ax1, ax2, ax3), pdf_fname=pdf_fname, cc_loc=cc_loc,
                                  deg_mode=deg_mode,
                                  plot_dd=plot_dd, logy=logy, cutoff=cutoff, deg_loc=deg_loc, cc_mode=cc_mode,
                                  undefined=undefined, undirected_max=undirected_max, step=step, cc_plot_cdf=cc_plot_cdf,
                                  cc_loglog=cc_loglog, cmap=cmap, del_threshold=del_threshold, dd_inverse_cdf=dd_inverse_cdf,
                                  figcaption=figcaption, add_cc_mean=add_cc_mean, add_dd_mean=add_dd_mean)

    plot_matches(graphs, labels, last_graph_baseline=last_graph_baseline, ax=ax4, attr_name=attr_name, inverse_cdf=attr_inverse_cdf)

    if pdf_fname: fig.savefig(pdf_fname)
    if figcaption: plt.gcf().suptitle(figcaption, y=1)
    return fig, (ax1,ax3,ax2,ax4)

def plot_binned_clustering_dist(binned_ccs, labels=None, ax=None, last_graph_baseline=True,
                                plot_cdf=False, loglog=True, cmap='magma', add_mean=True):
    if not ax: ax = plt.subplot()
    if not labels: labels = list(map(str, range(len(binned_ccs))))
    cmap = plt.get_cmap(cmap, len(labels)/2)
    baseline_cc = binned_ccs[-1]
    binned_baseline_cc = bin_clustering(baseline_cc)
    # baseline_bcc_cdf = np.cumsum(binned_ccs[-1])

    for idx, (bcc, label) in enumerate(zip(binned_ccs, labels),1):
        color = None #cmap(idx)
        if last_graph_baseline and idx != len(binned_ccs):
            binned_cc = bin_clustering(bcc)
            ks = get_ks_statistic(baseline_cc, bcc)
            label = "{} (KS: {})".format(label, round(ks, 3))
        bcc = bin_clustering(bcc)
        x = bcc.index
        if plot_cdf:
            ax.plot(x, np.cumsum(bcc), '--', lw=4, alpha=0.6, label=label, color=color)
        else:
            y = list(1-bcc.cumsum()+bcc)
            if loglog: x, y = x[1:], y[1:]
            ax.plot(x, y, '.-', label=label, alpha=0.6, lw=4, color=color)

    title = 'Local Clustering CDF' if plot_cdf else 'Local Clustering Inverse CDF'
    if add_mean: ax.axvline(np.mean(binned_ccs[-1]), ls='--', alpha=0.4, label='average clustering')
    set_labels(ax, xlabel='Local Clustering Coefficient c', ylabel='$\Pr(C \geq c)$', title=title)
    if loglog: ax.set_xscale('log')
    return ax

def plot_binned_clustering_dist_graphs(graphs, labels=None, ax=None, last_graph_baseline=True,
                                       undefined='nan', mode='IN', undirected_max=False, step=0.025,
                                       plot_cdf=False, loglog=False, cmap='magma', add_mean=True):

    graphs = [g if type(g) is ig.Graph else g.g for g in graphs]
    ccs = [clustering(g, undirected_max=undirected_max)[0] for g in graphs]
    # ccs = [clustering_distribution(g, undirected_max=undirected_max) for g in graphs]
    return plot_binned_clustering_dist(ccs, labels=labels, ax=ax, add_mean=add_mean, last_graph_baseline=last_graph_baseline, plot_cdf=plot_cdf, loglog=loglog, cmap=cmap)

def tickify(ax):
    ax.tick_params(direction='in', length=6, width=2, colors='k', which='major')
    ax.tick_params(direction='in', length=4, width=1, colors='k', which='minor')
    ax.minorticks_on()

def plot_avg_nbor_degree(xk, pk, deg_mode='IN', nbor_mode='OUT', ax=None, ls='--', basey=2, loc=2, **ax_kwargs):
    if not ax: ax = plt.subplot()
    ax_kwargs['lw'] = ax_kwargs.get('lw', 4)
    ax_kwargs['alpha'] = ax_kwargs.get('alpha', 0.6)
    ax.plot(xk, pk, ls, **ax_kwargs)
    ax.set_xscale('log'); ax.set_yscale('log', basey=basey)
    for axi in [ax.xaxis, ax.yaxis]: axi.set_major_formatter(ScalarFormatter())
    set_labels(ax, xlabel='node indeg', ylabel='average nbor indeg', title='E[nbor deg | node deg]', loc=loc)
    return ax

def set_labels(ax, xlabel=None, ylabel=None, title=None, use_legend=True, loc='best', ticks=True, despine=True, size=16):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if use_legend: ax.legend(loc=loc, prop={'size': size})
    if ticks: tickify(ax)
    if despine: sns.despine(ax=ax)


################
# EVAL
################

def get_ks_statistic(indeg1, indeg2):
    return ks_2samp(indeg1, indeg2)[0]

def get_cc_distance(xk, cc, xk_data, cc_data, xk_deg, pk_deg, relative=False):
    """(xk,cc) fitted graph data, (xk_data, cc_data) observed/ground truth data"""
    # filter degree in DD
    xk_data_set = set(xk_data)
    xk_deg2, pk_deg2 = map(np.array, zip(*[(x,p) for x,p in zip(xk_deg,pk_deg) if x in xk_data_set]))
    pk_deg2 = pk_deg/np.sum(pk_deg2)

    # filter zero to avoid zero div
    xk_data, cc_data, xk, cc = map(np.array, [xk_data, cc_data, xk, cc])
    xk_data, cc_data = xk_data[cc_data>0], cc_data[cc_data>0]
    xk, cc = xk[cc>0], cc[cc>0]

    # [deg] -> avg cc map
    obs_cc_map = dict(zip(xk_data, cc_data))
    fit_cc_map = dict(zip(xk, cc))

    wavg = 0
    for k,pk in zip(xk_deg2, pk_deg2):
        if k not in obs_cc_map: continue
        t = pk*abs(obs_cc_map[k]-fit_cc_map.get(k,0.))
        wavg += t/obs_cc_map[k] if relative else t
    return wavg


################
# AGGREGATE & IO
################

# main
def _get_graph_params(graph, time_attr, discard_nids, init_nids, post_nids, use_mean_outdeg=False, debug=True, attrs=None):
    get_summary = lambda g: g.summary().rsplit('--', 1)[0]

    discard_nids = set(discard_nids)
    all_nodes = graph.vs.select(list(post_nids))
    chunk_outdeg_map = defaultdict(list)
    chunk_attr_list = defaultdict(list)
    if debug: print ("next chunk + outdeg")

    # compute "actual" chunks (use main graph) and mean outdegree
    for node in all_nodes:
        nbors = [nbor for nbor in node.neighbors(mode='OUT') if nbor.index not in discard_nids]
        ta = node[time_attr]
        if ta != ta: continue
        chunk_outdeg_map[node[time_attr]].append(len(nbors))
        if attrs: chunk_attr_list[node[time_attr]].append(node[attrs])

    chunk_attr_seq = []
    for k in sorted(chunk_attr_list):
        chunk_attr_seq.append(chunk_attr_list[k])

    chunk_deg_seq = [(k, len(v), np.mean(v)) for k,v in sorted(chunk_outdeg_map.items())]
    time_keys, chunks, outdegs = zip(*chunk_deg_seq)
    if use_mean_outdeg: outdegs = np.array([np.mean(graph.outdegree())]*len(outdegs))

    # subgraphs
    if debug: print ("next subgraphs")

    gpre = graph.subgraph(list(init_nids))
    gpost = graph.subgraph(list(post_nids))
    geval = graph.subgraph(list(init_nids.union(post_nids)))

    if debug:
        for n,g in zip(['gpre', 'gpost', 'geval'], [gpre, gpost, geval]):
            print ("{}:{}".format(n,get_summary(g)))

        print ("Chunks: ", chunks[:5])
        print ("M: ", [str(np.round(x,2)) for x in outdegs][:5])
        print ("TBAdded: {}".format(np.dot(chunks, outdegs)))

    return {
        'graph': geval,
        'gpre': gpre,
        'gpost': gpost,
        'time_keys': time_keys,
        'chunk_sizes': chunks,
        'mean_outdegs': outdegs,
        'mean_chunk_size': np.mean(chunks),
        'mean_outdeg': np.mean(outdegs),
        'N': len(geval.vs),
        'nids_order': None,
        'chunk_sampler': chunk_attr_seq
    }

# discard + BFS
def extract_arw_input_data(graph, time_attr, discard_pct, init_pct, debug=True, use_mean_outdeg=False, attrs=None, min_bfs_size=10, eval_pct=1.):
    # sort nids by time
    nids, _ = zip(*sorted(zip(graph.vs.indices, graph.vs[time_attr]), key=lambda t: t[-1]))
    # discard nids setup
    discard_idx = int(round(discard_pct*len(nids)))
    use_nids = nids[discard_idx:]
    use_nids_iter = iter(use_nids)

    if debug: print ("nids sorted, next bfs")

    # construct initial subgraph using BFS
    N_gpre = int(round(init_pct*len(nids)))
    if debug: print ("initial graph size: {}".format(N_gpre))
    init_nids =  set()

    while len(init_nids) <= N_gpre:
        nid = random.choice(use_nids)
        if nid in init_nids: continue

        visited_nids = []

        for node in graph.bfsiter(nid, mode='OUT'):
            visited_nids.append(node.index)
            if len(visited_nids) > N_gpre: break

        if len(visited_nids) < min(min_bfs_size, N_gpre):
            continue # ignore if BFS too small

        num_add = min(N_gpre-len(init_nids)+1, len(visited_nids))
        init_nids.update(visited_nids[:num_add])

    if debug: print ("bfs, next graph_params")
    discard_nids = set(nids[:discard_idx])-init_nids

    discard_and_init = discard_nids.union(init_nids)
    post_nids = [n for n in nids if n not in discard_and_init]
    post_idx = int(round(eval_pct*len(post_nids)))
    post_nids = set(post_nids[:post_idx+1])

    return _get_graph_params(graph, time_attr, discard_nids, init_nids, post_nids, debug=debug, use_mean_outdeg=use_mean_outdeg, attrs=attrs)
