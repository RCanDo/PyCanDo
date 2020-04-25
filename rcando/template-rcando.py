#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: Plotti graphs 
subtitle:
version: 1.0
type: code
keywords: [graph, plot, draw]   # there are always some keywords!
description: |
remarks:    
    - NetworkX
todo:
sources:
    - title:
      chapter:
      pages:
      link:
      date:
      authors:
          - nick:
            fullname:
            email:
      usage: |
          idea & inspiration
file:
    usage:
        interactive: True
        terminal: True
    name: _.py
    path: ~/Projects/Python/RCanDo/..
    date: 2020-04-19
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
              - arek@staart.pl
"""

#%%
"""
pwd
cd ~/Projects/Python/RCanDo/...
ls
"""

import networkx as nx

#%%

from matplotlib.axes import Axes


def plot_network(graph: nx.DiGraph, 
                 # ax: Axes, 
                 node_size=1000, 
                 node_label_size=12, node_color='white'):
    pos = nx.drawing.circular_layout(graph)

    nx.draw_networkx_nodes(graph, pos,
                           node_shape='o',
                           edgecolors='black',
                           node_size=node_size,
                           node_color=node_color,
                           #ax=ax
                           )

    nx.draw_networkx_labels(graph, pos,
                            font_color='black',
                            font_weight='bold',
                            font_size=node_label_size,
                            #ax=ax
                            )

    nx.draw_networkx_edges(graph, pos,
                           edge_color='gray',
                           node_size=node_size,
                           #ax=ax
                           )

    nx.draw_networkx_edge_labels(graph, pos,
                                 font_size=9,
                                 edge_labels={edge: graph.edges[edge]['weight'] for edge in graph.edges},
                                 #ax=ax
                            )

#%%