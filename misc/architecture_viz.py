from graphviz import Digraph

def create_dgcnn_diagram():
    dot = Digraph(comment='Modified DGCNN Architecture')
    
    # Set graph attributes
    dot.attr(rankdir='TB')  # Top to bottom direction
    dot.attr('node', shape='box', style='rounded,filled')
    
    # Color scheme
    input_color = '#E6F3FF'      # Light blue
    spatial_color = '#FFE6E6'    # Light red
    geometric_color = '#E6FFE6'  # Light green
    fusion_color = '#FFF2E6'     # Light orange
    
    # Input processing
    dot.node('input', 'Raw Point Cloud\n(xyz coordinates)', fillcolor=input_color)
    dot.node('geom_feat', 'Geometric Feature Computation\n• Normals\n• Curvature\n• Local density\n• Heights', 
             fillcolor=input_color)
    dot.node('combined', 'Combined Features\n(xyz + geometric)', fillcolor=input_color)
    
    # Create spatial path nodes
    with dot.subgraph(name='cluster_0') as spatial:
        spatial.attr(label='Spatial Path', style='rounded', bgcolor=spatial_color)
        spatial.node('conv1', 'DynamicEdgeConv1\n(6→64)', fillcolor=spatial_color)
        spatial.node('conv2', 'DynamicEdgeConv2\n(128→128)', fillcolor=spatial_color)
        spatial.node('conv3', 'DynamicEdgeConv3\n(256→256)', fillcolor=spatial_color)
    
    # Create geometric path nodes
    with dot.subgraph(name='cluster_1') as geometric:
        geometric.attr(label='Geometric Path', style='rounded', bgcolor=geometric_color)
        geometric.node('geom_encoder', 'Geometric Encoder\n(6→32→64)', fillcolor=geometric_color)
        geometric.node('local_att', 'Local Attention\n(64→32→64)', fillcolor=geometric_color)
        geometric.node('global_att', 'Global Attention\n(64→32→64)', fillcolor=geometric_color)
    
    # Fusion and output nodes
    dot.node('concat', 'Feature Concatenation\n(512 features)', fillcolor=fusion_color)
    dot.node('fusion', 'Fusion Layers\n(512→256→128→num_classes)', fillcolor=fusion_color)
    dot.node('output', 'Classification Output', fillcolor=fusion_color)
    
    # Add edges
    # Input processing
    dot.edge('input', 'geom_feat')
    dot.edge('geom_feat', 'combined')
    
    # Spatial path
    dot.edge('combined', 'conv1')
    dot.edge('conv1', 'conv2')
    dot.edge('conv2', 'conv3')
    dot.edge('conv3', 'concat')
    
    # Geometric path
    dot.edge('combined', 'geom_encoder')
    dot.edge('geom_encoder', 'local_att')
    dot.edge('geom_encoder', 'global_att')
    dot.edge('local_att', 'concat')
    dot.edge('global_att', 'concat')
    
    # Output path
    dot.edge('concat', 'fusion')
    dot.edge('fusion', 'output')
    
    return dot

# Create and save the diagram
diagram = create_dgcnn_diagram()
diagram.render('dgcnn_architecture', format='pdf', cleanup=True)
