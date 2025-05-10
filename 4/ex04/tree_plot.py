import matplotlib.pyplot as plt

dx_dict = {
    0: 1,
    1: 0.5,
    2: 0.25,
    3: 0.5,
    4: 0.3,
    5: 0.2,
    6: 0.5,
}

def draw(node, x=0.5, y=1.0, depth=0, ax=None, max_depth=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis("off")

    if max_depth is None:
        max_depth = get_max_depth(node)          # helper below

    dx = dx_dict.get(depth, 0.5)

    ax.text(x, y, node.label(), ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#e8e8ff"))

    if node.is_leaf():
        return

    # vertical step
    y_child = y - 0.18

    # left child
    ax.plot([x, x-dx], [y-0.02, y_child+0.02], "k-")
    draw(node.left,  x-dx, y_child, depth+1, ax, max_depth)

    # right child
    ax.plot([x, x+dx], [y-0.02, y_child+0.02], "k-")
    draw(node.right, x+dx, y_child, depth+1, ax, max_depth)

    if depth == 0:
        plt.show()

def get_max_depth(node):
    if node.is_leaf():
        return 0
    return 1 + max(get_max_depth(node.left), get_max_depth(node.right))