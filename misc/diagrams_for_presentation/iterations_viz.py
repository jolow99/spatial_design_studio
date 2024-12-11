from graphviz import Digraph

def create_iterations_diagram():
    dot = Digraph(comment='Model Architecture Iterations')
    
    # Set graph attributes
    dot.attr(rankdir='LR')  # Left to right direction
    # Increase spacing between nodes
    dot.attr('graph', nodesep='1.0', ranksep='2.0')
    dot.attr('node', shape='box', style='rounded,filled')
    
    # Color scheme
    base_color = '#E6F3FF'      # Light blue
    
    # Define iterations in single row
    iterations = [
        ('model1', '1. Regression with\nMSE Loss'),
        ('model2', '2. 5-Class Classification\nCross-Entropy Loss'),
        ('model3', '3. Softmax Focal Loss'),
        ('model4', '4. Ordinal Weights +\nFocal Loss'),
        ('model5', '5. Geometric Features\n• Curvature\n• Density\n• Height\n• Normals'),
        ('model6', '6. Full Dataset Training\nNo Train-Test Split')
    ]
    
    # Create nodes and connections
    for i, (node_id, label) in enumerate(iterations):
        # Add width and height constraints to make boxes more uniform
        dot.node(node_id, label, fillcolor=base_color, style='filled,rounded', 
                width='2.5', height='1.5')
        if i < len(iterations) - 1:
            dot.edge(iterations[i][0], iterations[i+1][0])
    
    return dot

# Create and save the diagram
diagram = create_iterations_diagram()
diagram.render('model_iterations', format='pdf', cleanup=True)
