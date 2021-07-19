# from spn.io.Graphics import _get_networkx_obj
from spn.structure.Base import Sum, Product, Leaf, get_topological_order_layers

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

from algorithms.nodes import CollapsedNode


def _get_networkx_obj(spn, label_depth=None, label_tree_node=False, label_class_var=False, label_color_class=None, scope_letters=None):
    G = nx.Graph()

    # Add all nodes
    for layer in reversed(get_topological_order_layers(spn)):
        for n in layer:
            node_kwargs = {}
            node_kwargs['collapsed'] = False
            if label_depth != None:
                node_kwargs['depth'] = n._depth
            if label_tree_node:
                node_kwargs['tree_node'] = n._tree_node
            if label_color_class != None:
                node_kwargs['label_color'] = 'red' if label_color_class in n.scope and isinstance(n, Leaf) else 'gray'
            if isinstance(n, CollapsedNode):
                node_kwargs['collapsed'] = True
                G.add_node(n.id, color='lightgray', label='', **node_kwargs)
            elif isinstance(n, Leaf):
                subscript = n.scope[0]
                superscript  = ''
                if scope_letters is not None and scope_letters[n.scope[0]] is not None:
                    subscript = scope_letters[n.scope[0]]
                if hasattr(n, '_superscript'):
                    superscript = n._superscript
                G.add_node(n.id, color='#bdffc5', label=f'$X_{{{subscript}}}^{{{superscript}}}$', **node_kwargs)
            elif isinstance(n, Sum):
                G.add_node(n.id, color='#ffb8b3', label='+', **node_kwargs)
            elif isinstance(n, Product):
                G.add_node(n.id, color='#c2bdff', label='X', **node_kwargs)
            else:
                G.add_node(n.id, color='gray', label='?', **node_kwargs)

    # Add all edges
    for layer in reversed(get_topological_order_layers(spn)):
        for n in layer:
            if isinstance(n, Leaf):
                continue
            for c in n.children:
                G.add_edge(n.id, c.id)

    return G

def plot_labeled_spn(spn, fname='plot.png', label_depth=None, label_tree_node=False, label_class_var=False, label_color_class=None, large=False, scope_letters=None, save=True, margins=None):
    font_size=12
    node_size = 400
    linewidhts = 1
    if large:
        font_size = 20
        node_size = 1500
        linewidhts = 2
    
    G = _get_networkx_obj(spn, label_depth=label_depth, label_tree_node=label_tree_node, label_class_var=label_class_var, label_color_class=label_color_class, scope_letters=scope_letters)

    color_map = []
    node_sizes = []
    labels = {}
    for node in G:
        labels[node] = G.nodes[node]['label']
        if label_depth != None:
            color_map.append('steelblue' if G.nodes[node]['depth'] > label_depth else 'red')
        elif label_tree_node:
            color_map.append('steelblue' if not G.nodes[node]['tree_node'] else 'red')
        elif label_color_class:
            color_map.append(G.nodes[node]['label_color'])
        else:
            color_map.append(G.nodes[node]['color'])
        node_sizes.append(node_size / 2 if G.nodes[node]['collapsed'] else node_size)

    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, labels=labels, font_size=font_size, arrows=False, node_color=color_map, linewidths=linewidhts, node_size=node_sizes)
    ax = plt.gca() # to get the current axis
    ax.collections[0].set_edgecolor("#000000")
    ax.margins(0.1 if margins is None else margins)
    if save:
        plt.savefig(fname)
