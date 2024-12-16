from graphviz import Digraph

def create_loss_diagram():
    dot = Digraph(comment='Ordinal Focal Loss Architecture')
    
    # Set graph attributes
    dot.attr(rankdir='TB')  # Top to bottom direction
    dot.attr('node', shape='box', style='rounded,filled')
    
    # Color scheme
    input_color = '#E6F3FF'      # Light blue
    encoding_color = '#FFE6E6'   # Light red
    weighting_color = '#E6FFE6'  # Light green
    output_color = '#FFF2E6'     # Light orange
    
    # Input nodes
    dot.node('pred', 'Model Predictions\n(Logits)', fillcolor=input_color)
    dot.node('target', 'Target Labels\n(0-4)', fillcolor=input_color)
    dot.node('dataset', 'Training Dataset', fillcolor=input_color)
    
    # Class weights calculation
    with dot.subgraph(name='cluster_2') as class_weights:
        class_weights.attr(label='Class Weights Calculation', style='rounded', bgcolor=input_color)
        class_weights.node('count', 'Count Class\nFrequencies', fillcolor=input_color)
        class_weights.node('inverse', 'Compute Inverse\nFrequencies', fillcolor=input_color)
        class_weights.node('normalize', 'Normalize & Cap\nat 200', fillcolor=input_color)
    
    # Encoding nodes
    with dot.subgraph(name='cluster_0') as encoding:
        encoding.attr(label='Ordinal Encoding', style='rounded', bgcolor=encoding_color)
        encoding.node('softmax', 'Softmax\nProbabilities', fillcolor=encoding_color)
        encoding.node('triangular', 'Triangular\nOrdinal Encoding', fillcolor=encoding_color)
    
    # Weighting nodes
    with dot.subgraph(name='cluster_1') as weighting:
        weighting.attr(label='Loss Weighting', style='rounded', bgcolor=weighting_color)
        weighting.node('focal', 'Focal Weights\n|prob - target|ᵧ', fillcolor=weighting_color)
        weighting.node('distance', 'Distance Weights\n1 + |class - target|', fillcolor=weighting_color)
        weighting.node('class_weights', 'Class Weights\n(α)', fillcolor=weighting_color)
    
    # Output nodes
    dot.node('weighted_bce', 'Weighted BCE Loss', fillcolor=output_color)
    dot.node('final_loss', 'Final Loss', fillcolor=output_color)
    
    # Add edges
    # Class weights calculation
    dot.edge('dataset', 'count')
    dot.edge('count', 'inverse')
    dot.edge('inverse', 'normalize')
    dot.edge('normalize', 'class_weights')
    
    # Main loss computation
    dot.edge('pred', 'softmax')
    dot.edge('target', 'triangular')
    dot.edge('target', 'distance')
    dot.edge('softmax', 'focal')
    dot.edge('triangular', 'focal')
    dot.edge('focal', 'weighted_bce')
    dot.edge('distance', 'weighted_bce')
    dot.edge('class_weights', 'weighted_bce')
    dot.edge('weighted_bce', 'final_loss')
    
    return dot

# Create and save the diagram
diagram = create_loss_diagram()
diagram.render('ordinal_focal_loss', format='pdf', cleanup=True)
