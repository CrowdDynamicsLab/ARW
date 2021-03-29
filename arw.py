import igraph as ig
import numpy as np
import random
import time
from collections import defaultdict, Counter, deque
import itertools

MAX_LENGTH = 1000 # maximum random walk length (hyperparameter)

class RandomWalkSingleAttribute(object):
    def __init__(self, p_diff, p_same, jump, out, gpre, attr_name='single_attr', debug=True):
        self.p_diff = p_diff
        self.p_same = p_same
        self.jump = jump
        self.out = out

        self.gpre = gpre
        self.attr_name = attr_name

        self.directed = True
        self.g = ig.Graph(directed=self.directed)
        self.attributed = attr_name in gpre.vs.attributes()

        self.debug = debug
        self.setup()

    def summary(self): return self.g.summary()
    def flip(self, p): return random.random() < p

    def setup(self):
        self.seed_same = self.p_same/(self.p_same+self.p_diff)
        self.seed_diff = 1.-self.seed_same
        self.n0 = len(self.gpre.vs)
        self.total_edges = len(self.gpre.es)
        self.next_nid = self.n0
        self.chunk_nid = self.next_nid-1
        self.chunk_size = 1

        self.nbors = defaultdict(list)
        self.out_nbors = defaultdict(list)
        self.in_nbors = defaultdict(list)
        self.nid_chunk_map = {}
        self.nid_attr_map = {}
        self.attr_nid_map = defaultdict(list)

        for nid, nbor_nids in enumerate(self.gpre.get_adjlist(mode='ALL')): self.nbors[nid] = nbor_nids
        for nid, nbor_nids in enumerate(self.gpre.get_adjlist(mode='OUT')): self.out_nbors[nid] = nbor_nids
        for nid, nbor_nids in enumerate(self.gpre.get_adjlist(mode='IN')): self.in_nbors[nid] = nbor_nids

        for node in self.gpre.vs: self.nid_attr_map[node.index] = node[self.attr_name] if self.attributed else None
        for nid, attr in self.nid_attr_map.items(): self.attr_nid_map[attr].append(nid)

        for nid in self.gpre.vs.indices: self.nid_chunk_map[nid] = 0

    def add_nodes(self, chunk_seq, mean_seq, chunk_attr_sampler=None):
        if self.attributed: assert chunk_attr_sampler
        num_chunks = len(chunk_seq)
        chunk_debug = num_chunks//10
        if (self.debug): print ("Total chunks: {}".format(num_chunks))
            
        for idx, (chunk_size, m) in enumerate(zip(chunk_seq, mean_seq)):
            if self.debug and (idx + 1) % chunk_debug == 0: print (idx, end=' ')
            self.chunk_size = chunk_size
            self.m = m
            self.add_chunk(idx, attr_sampler=chunk_attr_sampler[idx][:] if self.attributed else None)
            self.chunk_nid = self.next_nid-1

        self.build_graph()

    def add_chunk(self, chunk_id, attr_sampler=None):
        if self.attributed: assert attr_sampler
        marked = defaultdict(frozenset)

        for _ in range(self.chunk_size):
            new_nid = self.next_nid; self.next_nid += 1
            self.nid_chunk_map[new_nid] = chunk_id
            attrs = attr_sampler.pop() if self.attributed else None
            marked[new_nid] = self.add_node(new_nid, attrs=attrs)
            self.update_node(new_nid, marked[new_nid])

    def update_node(self, nid, marked):
        for nbor_nid in marked:
            self.out_nbors[nid].append(nbor_nid)
            self.in_nbors[nbor_nid].append(nid)

    def build_graph(self):
        self.edges = edges = set()
        all_nbors = self.out_nbors

        for node, nbors in all_nbors.items():
            for nbor in nbors: edges.add((node, nbor))

        self.g.add_vertices(self.next_nid)
        self.g.add_edges(list(edges))
        self.g.simplify()
        self.g.vs['chunk_id'] = [self.nid_chunk_map[n] for n in self.g.vs.indices]
        if self.attributed: self.g.vs[self.attr_name] = [self.nid_attr_map[n] for n in self.g.vs.indices]
        if self.debug: print ("\n{}".format(self.g.summary()))
            
    def link(self, cur_nid, attrs=None):
        if not self.attributed:
            return random.random() < self.p_diff
        else:
            cur_attrs =  self.nid_attr_map[cur_nid]
            p = self.p_same if cur_attrs == attrs else self.p_diff
            return random.random() < p

    def get_seed_nid(self, new_nid, attrs=None):
        if not self.attributed: return random.randint(0, new_nid-1)
        if random.random() < self.seed_diff: return random.randint(0, new_nid-1)
        same_nids = self.attr_nid_map[attrs]
        if same_nids: return random.choice(same_nids)
        return np.random.randint(0, new_nid-1)

    def add_node(self, new_nid, attrs=None):
        marked = set()
        m = int(round(self.m if self.flip(0.5) else self.m+0.5))
        cur_nid = seed_nid = self.get_seed_nid(new_nid, attrs=attrs)
        num_marked, length, max_length = 0, 0, MAX_LENGTH/max(self.p_same, self.p_diff)

        while num_marked < m:
            length += 1

            if length > max_length:
                break

            if cur_nid not in marked and self.link(cur_nid, attrs):
                num_marked += 1
                marked.add(cur_nid)

            if random.random() < self.jump:
                cur_nid = seed_nid
            else:
                use_out = random.random() < self.out
                nbors = self.out_nbors[cur_nid] if use_out else self.in_nbors[cur_nid]
                if not nbors: nbors = self.in_nbors[cur_nid] if use_out else self.out_nbors[cur_nid]

                if nbors: cur_nid = random.choice(nbors)
                else: cur_nid = seed_nid = self.get_seed_nid(new_nid, attrs=attrs)

        if self.attributed:
            self.nid_attr_map[new_nid] = attrs
            self.attr_nid_map[attrs].append(new_nid)

        return marked
