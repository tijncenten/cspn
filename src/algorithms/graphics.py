# from spn.io.Graphics import _get_networkx_obj
from spn.structure.Base import Sum, Product, Leaf, get_topological_order_layers

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


def _get_networkx_obj(spn, label_depth=None, label_tree_node=False, label_class_var=False, label_color_class=None):
    G = nx.Graph()

    # Add all nodes
    for layer in reversed(get_topological_order_layers(spn)):
        for n in layer:
            node_kwargs = {}
            if label_depth != None:
                node_kwargs['depth'] = n._depth
            if label_tree_node:
                node_kwargs['tree_node'] = n._tree_node
            if label_color_class != None:
                node_kwargs['label_color'] = 'red' if label_color_class in n.scope and isinstance(n, Leaf) else 'gray'
            if isinstance(n, Leaf):
                G.add_node(n.id, color='#bdffc5', label=f'$X_{n.scope[0]}$', **node_kwargs)
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

def plot_labeled_spn(spn, fname='plot.png', label_depth=None, label_tree_node=False, label_class_var=False, label_color_class=None):
    G = _get_networkx_obj(spn, label_depth=label_depth, label_tree_node=label_tree_node, label_class_var=label_class_var, label_color_class=label_color_class)

    color_map = []
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

    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, labels=labels, font_size=10, arrows=False, node_color=color_map, linewidths=1)
    ax = plt.gca() # to get the current axis
    ax.collections[0].set_edgecolor("#000000")
    plt.savefig(fname)
