#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: Number of graphs utilities
subtitle:
version: 1.0
type: code
keywords: [graph, plot, draw]
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

#%%

import networkx as nx
from matplotlib.axes import Axes


def plot_graph(graph: nx.DiGraph, 
               layout='circular',
               # ax: Axes,                #! see  get_network_plot()  below
               node_size=100, 
               node_label_size=7, 
               node_color='white'
              ):
    """plotting graph
    
    layout   one of 'random', 'circular', 'kamada_kawai', 'planar',
                    'shell', 'spring', 'spectral', 'spiral'
             see:
             https://networkx.github.io/documentation/stable/reference/drawing.html#module-networkx.drawing.layout
    """

    LAYOUTS = {'random': nx.drawing.random_layout, 
               'circular': nx.drawing.circular_layout, 
               'kamada_kawai': nx.drawing.kamada_kawai_layout, 
               'planar': nx.drawing.planar_layout,
               'shell': nx.drawing.shell_layout, 
               'spring': nx.drawing.spring_layout, 
               'spectral': nx.drawing.spectral_layout#, 
               #'spiral': nx.drawing.spiral_layout
              }
    
    pos = LAYOUTS[layout](graph)

    nx.draw_networkx_nodes(graph, pos,
                           node_shape='o',
                           edgecolors='grey',
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
    
    try:
        nx.draw_networkx_edge_labels(graph, pos,
                             font_size=9,
                             edge_labels={edge: graph.edges[edge]['weight'] for edge in graph.edges},
                             #ax=ax
                                )
    except KeyError as e:
        print('Probably no edge labels: {}'.format(e))

#%% 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
    
def get_network_plot(graph: nx.DiGraph, figsize=(10, 6)) -> Figure:
    """check it !!!
    """
    fig, ax = plt.subplots(figsize=figsize)   # !!!

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    for spine in plt.gca().spines.values():   # ???
        spine.set_visible(False)

    plot_graph(graph, ax)

    return fig
    
#%%
    

#%%
    

#%%
    
    
    