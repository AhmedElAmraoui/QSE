import networkx as nx
import matplotlib.pyplot as plt 

def xorsat_problem_for_plotting(xorsat_problem=tuple, num_variables=int):
    """
    Prepares variables and constraints for plotting a Max-3-XORSAT problem.

    :param xorsat_problem: A tuple representing the XOR-SAT problem. 
                           - The first element (xorsat_problem[0]) is a np.array, 
                             where each row contains indices of variables involved 
                             in a constraint.
                           - The second element (xorsat_problem[1]) is a list of binary 
                             values representing the XOR result for each constraint (0 or 1).
    :param num_variables: An integer representing the total number of variables 
                          in the XOR-SAT problem.

    :return: 
        - variables (list of str): List of variable names in the form ['x1', 'x2', ..., 'xn'].
        - constraints (list of tuples): List of tuples representing constraints. 
          Each tuple is of the form ([v1, v2, v3], b), where [v1, v2, v3] is a list 
          of variable indices involved in the constraint and b is the XOR result (0 or 1).
          
    Example usage:
    ---------------
    xorsat_problem = ([[0, 1, 2], [1, 2, 3]], [0, 1])
    num_variables = 4
    
    variables, constraints = xorsat_problem_for_plotting(xorsat_problem, num_variables)
    # variables -> ['x0', 'x1', 'x2', 'x3']
    # constraints -> [([0, 1, 2], 0), ([1, 2, 3], 1)]
    """
    # Create a list of variable names in the form ['x0', 'x1', ..., 'xn-1']
    variables = [f'x{str(i)}' for i in range(num_variables)]

    # Unpack the constraints and results from the xorsat_problem tuple
    # 'constr' contains lists of variable indices, and 'b' contains the XOR results
    constr, b = [list(row) for row in xorsat_problem[0]], list(xorsat_problem[1])

    # Combine each set of variables with its corresponding XOR result into a tuple
    constraints = [(i) for i in zip(constr, b)]

    # Return the list of variable names and the list of constraints
    return variables, constraints

def create_xorsat_graph(variables, constraints):
    """
    Visualize a Max-3-XORSAT problem using NetworkX, with constraint nodes colored 
    based on their XOR value (0 = red, 1 = green).
    
    :param variables: List of variable names (e.g., ['x1', 'x2', 'x3']).
    :param constraints: List of XOR constraints. Each constraint is a tuple of 
                        ([v1, v2, v3], b), where [v1, v2, v3] is a list of variables involved,
                        and b is the result of their XOR (0 or 1).
    :return: A tuple containing the graph object for Max 3-XORSAT and a list of node colors.
    """
    # Create a bipartite graph where one set is variables and the other is constraints
    graph = nx.Graph()
    # to add a specific colors for the variables and constraints 
    node_color = []
    # Add the variables as one set of nodes
    for var in variables:
        graph.add_node(var, bipartite=0)
        # node color is light blue for variables 
        node_color.append('lightblue')

    
    # Add the constraints as nodes and connect them to the variables involved
    for i, (vars_in_constraint, xor_result) in enumerate(constraints):
        # Add a node representing the constraint
        constraint_node = f"C{i}" # ={b}"
        graph.add_node(constraint_node, bipartite=1)
        # Color the constraint node based on the XOR result
        node_color.append('red' if xor_result == 0 else 'green')
                
        # Add edges between the variables and the constraint node
        for var_index in vars_in_constraint:
            variable_name = variables[var_index]  # access the variable from the list
            graph.add_edge(variable_name, constraint_node)

    return graph, node_color


def visualize_graph(Graph=None, variables =[], node_color= None):
    """
    Function to plot the graph based on a passed bipartite graph of vars and constraints
    :param Graph: Graph object.
    :param node_colors: List of colors for each node.

    """
    G = Graph
    node_color = node_color
    # Draw the graph
    pos = nx.bipartite_layout(G, nodes=variables)#nx.spring_layout(G)  # Position nodes using spring layout
    nx.draw(G, pos, with_labels=True, node_color=node_color, node_size=2000, font_size=10, font_weight="bold")
    plt.show()

def plot_graph(xorsat_problem = tuple, num_variables = int):
    """
    Main function to generate and visualize a Max-3-XORSAT problem.
    
    :param variables: List of variable names.
    :param constraints: List of XOR constraints. Each constraint is a tuple of 
                        ([v1, v2, v3], b), where [v1, v2, v3] is a list of variables involved,
                        and b is the result of their XOR (0 or 1).
    """
    variables, constraints = xorsat_problem_for_plotting(xorsat_problem, num_variables)
    graph, node_color = create_xorsat_graph(variables,constraints)
    visualize_graph(graph,variables, node_color)
    


