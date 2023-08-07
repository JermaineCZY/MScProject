import networkx as nx
import matplotlib.pyplot as plt

def plot_neural_network():
    G = nx.DiGraph()  # 创建一个有向图对象

    # Define nodes for different layers
    input_layer = ['x{}'.format(i+1) for i in range(2)]  # 创建输入层节点
    hidden_layer1 = ['h{}'.format(i+1) for i in range(8)]  # 创建第一层隐藏层节点
    dropout1 = ['d{}'.format(i+1) for i in range(8)]  # 创建第一层Dropout节点
    hidden_layer2 = ['h{}'.format(i+9) for i in range(4)]  # 创建第二层隐藏层节点
    dropout2 = ['d{}'.format(i+9) for i in range(4)]  # 创建第二层Dropout节点
    output_layer = ['y']  # 创建输出层节点

    # Add nodes to the graph
    G.add_nodes_from(input_layer, layer='Input')  # 将输入层节点添加到图中，并标记其层级为'Input'
    G.add_nodes_from(hidden_layer1, layer='Hidden1')  # 将第一层隐藏层节点添加到图中，并标记其层级为'Hidden1'
    G.add_nodes_from(dropout1, layer='Dropout1')  # 将第一层Dropout节点添加到图中，并标记其层级为'Dropout1'
    G.add_nodes_from(hidden_layer2, layer='Hidden2')  # 将第二层隐藏层节点添加到图中，并标记其层级为'Hidden2'
    G.add_nodes_from(dropout2, layer='Dropout2')  # 将第二层Dropout节点添加到图中，并标记其层级为'Dropout2'
    G.add_nodes_from(output_layer, layer='Output')  # 将输出层节点添加到图中，并标记其层级为'Output'

    # Add edges to the graph (connections between layers)
    for i in input_layer:  # 对于输入层的每一个节点
        for h in hidden_layer1:  # 对于第一层隐藏层的每一个节点
            G.add_edge(i, h, weight=1)  # 创建一个从输入层节点到隐藏层节点的边，并设置权重为1

    for h in hidden_layer1:  # 对于第一层隐藏层的每一个节点
        for d in dropout1:  # 对于第一层Dropout的每一个节点
            G.add_edge(h, d, weight=0.5 if int(h[1:]) != int(d[1:]) else 1)  # 创建一个从隐藏层节点到Dropout节点的边，如果隐藏层节点和Dropout节点编号相同，则权重为1，否则为0.5

    for d in dropout1:  # 对于第一层Dropout的每一个节点
        for h in hidden_layer2:  # 对于第二层隐藏层的每一个节点
            G.add_edge(d, h, weight=1)  # 创建一个从Dropout节点到隐藏层节点的边，并设置权重为1

    for h in hidden_layer2:  # 对于第二层隐藏层的每一个节点
        for d in dropout2:  # 对于第二层Dropout的每一个节点
            G.add_edge(h, d, weight=0.5 if int(h[1:]) != int(d[1:]) else 1)  # 创建一个从隐藏层节点到Dropout节点的边，如果隐藏层节点和Dropout节点编号相同，则权重为1，否则为0.5

    for d in dropout2:  # 对于第二层Dropout的每一个节点
        G.add_edge(d, 'y', weight=1)  # 创建一个从Dropout节点到输出层节点的边，并设置权重为1

    # Define node positions for different layers
    pos = {}  # 创建一个空字典来保存节点位置
    pos.update((node, (1, i - len(input_layer) / 2 + 0.5)) for i, node in enumerate(input_layer))  # 设置输入层节点的位置
    pos.update((node, (2, i - len(hidden_layer1) / 2 + 0.5)) for i, node in enumerate(hidden_layer1))  # 设置第一层隐藏层节点的位置
    pos.update((node, (3, i - len(dropout1) / 2 + 0.5)) for i, node in enumerate(dropout1))  # 设置第一层Dropout节点的位置
    pos.update((node, (4, i - len(hidden_layer2) / 2 + 0.5)) for i, node in enumerate(hidden_layer2))  # 设置第二层隐藏层节点的位置
    pos.update((node, (5, i - len(dropout2) / 2 + 0.5)) for i, node in enumerate(dropout2))  # 设置第二层Dropout节点的位置
    pos.update((node, (6, - len(output_layer) / 2 + 0.5)) for node in output_layer)  # 设置输出层节点的位置

    # Define node colors based on layers
    node_colors = {'Input': '#377eb8', 'Hidden1': '#984ea3', 'Dropout1': '#984ea3', 'Hidden2': '#ff7f00', 'Dropout2': '#ff7f00', 'Output': '#e41a1c'}  # 定义每一层节点的颜色
    colors = [node_colors[G.nodes[node]['layer']] for node in G.nodes]  # 获取所有节点的颜色
    edge_colors = ['#dedede' if G.edges[edge]['weight'] == 0.5 else 'black' for edge in G.edges]  # 根据边的权重获取边的颜色
    node_style = ['dotted' if 'd' in node else 'solid' for node in G.nodes]  # 根据节点是否在Dropout层确定节点边界的样式

    # Plot the neural network graph
    plt.figure(figsize=(16, 8), dpi=300)  # 创建一个新的图形，设置其大小和DPI
    nx.draw(G, pos, with_labels=False, node_color=colors, node_size=1000, font_size=16, font_weight='bold', arrows=True, edge_color=edge_colors, style=node_style)  # 画出神经网络的图形

    # Add labels to the input layer nodes on the left side
    input_labels = {'x1': 'Heroes', 'x2': 'Items'}
    input_pos = {node: (pos[node][0] - 0.3, pos[node][1]) for node in input_layer}
    nx.draw_networkx_labels(G, input_pos, labels=input_labels, font_size=14, font_weight='bold', verticalalignment='center')

    # Add layer titles
    plt.text(1, -len(input_layer) / 2 - 1.5, 'Input Layer', fontsize=20, ha='center')
    plt.text(2, -len(hidden_layer1) / 2 - 0.75, 'Hidden Layer 1\n(128 neurons)', fontsize=18, ha='center')
    plt.text(3, -len(dropout1) / 2 - 0.75, 'Dropout Layer 1', fontsize=18, ha='center')
    plt.text(4, -len(hidden_layer2) / 2 - 0.75, 'Hidden Layer 2\n(64 neurons)', fontsize=18, ha='center')
    plt.text(5, -len(dropout2) / 2 - 0.75, 'Dropout Layer 2', fontsize=18, ha='center')
    plt.text(6, -len(output_layer) / 2 - 3.0, 'Output Layer', fontsize=18, ha='center')
    plt.title('Neural Network', fontsize=20)  # 添加整个图形的标题
    #plt.savefig("neural_network.svg", format="svg")

    plt.show()# 保存图形
if __name__ == '__main__':
    plot_neural_network()  # 调用函数画出神经网络的图形
