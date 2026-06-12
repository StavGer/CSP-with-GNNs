import torch


def S(a, b, n):
    return (a, (a + b) % n)
 
def S_inv(a, b, n):
    return (a, (b - a) % n)
 
def T(a, b, n):
    return ((a + b) % n, b)
 
def T_inv(a, b, n):
    return ((a - b) % n, b)
 
def flatten_index(x, y, z, n):
    """Flatten 3D coordinates (x, y, z) into a 1D index."""
    return x * n * n + y * n + z

def unflatten_index(index, n):
    x = index // (n * n)
    y = (index % (n * n)) // n
    z = index % n
    return x, y, z
 
def add_edges_for_plane(plane, n, edges):
    """Add edges for a specific 2D plane (xy, xz, yz) to the edge list."""
   
    if plane == 'xy':
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    node = flatten_index(x, y, z, n)  # 3D to 1D index
                    neighbors = [
                        (flatten_index((x + 1) % n, y, z, n)),
                        (flatten_index((x - 1) % n, y, z, n)),
                        (flatten_index(x, (y + 1) % n, z, n)),
                        (flatten_index(x, (y - 1) % n, z, n)),
                        flatten_index(*S(x, y, n), z, n),
                        flatten_index(*S_inv(x, y, n), z, n),
                        flatten_index(*T(x, y, n), z, n),
                        flatten_index(*T_inv(x, y, n), z, n)
                    ]
                    for neighbor in neighbors:
                        if [node,neighbor] not in edges and node!=neighbor:
                            edges.append([node, neighbor])
 
    elif plane == 'xz':
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    node = flatten_index(x, y, z, n)  # 3D to 1D index
                    neighbors = [
                        (flatten_index((x + 1) % n, y, z, n)),
                        (flatten_index((x - 1) % n, y, z, n)),
                        (flatten_index(x, y, (z + 1) % n, n)),
                        (flatten_index(x, y, (z - 1) % n, n)),
                        flatten_index(S(x, z, n)[0], y, S(x, z, n)[1], n),
                        flatten_index(S_inv(x, z, n)[0], y, S_inv(x, z, n)[1], n),
                        flatten_index(T(x, z, n)[0], y, T(x, z, n)[1], n),
                        flatten_index(T_inv(x, z, n)[0], y,T_inv(x, z, n)[1], n)
                    ]
                    for neighbor in neighbors :
                        if [node,neighbor] not in edges and node!=neighbor:
                            edges.append([node, neighbor])
 
    elif plane == 'yz':
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    node = flatten_index(x, y, z, n)  # 3D to 1D index
                    neighbors = [
                        (flatten_index(x, (y + 1) % n, z, n)),
                        (flatten_index(x, (y - 1) % n, z, n)),
                        (flatten_index(x, y, (z + 1) % n, n)),
                        (flatten_index(x, y, (z - 1) % n, n)),
                        flatten_index(x, *S(y, z, n), n),
                        flatten_index(x, *S_inv(y, z, n), n),
                        flatten_index(x, *T(y, z, n), n),
                        flatten_index(x, *T_inv(y, z, n), n)
                    ]
                    for neighbor in neighbors:
                        if [node,neighbor] not in edges and node!=neighbor:
                            edges.append([node, neighbor])
 
def stacked_planes_edges(n):
    """Construct the 3D Margulis-Gabber-Galil expander graph by stacking xy, xz, and yz planes."""
    edges = []
   
    # Add edges for each plane
    add_edges_for_plane('xy', n, edges)
    add_edges_for_plane('xz', n, edges)
    add_edges_for_plane('yz', n, edges)
   
    # Convert edge list to PyTorch tensor and transpose for PyG
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
   
    return edge_index
def modn(val, n):
    return val % n

def add_affine_edges_for_plane(plane, n, edges, edge_set):
    for x in range(n):
        for y in range(n):
            for z in range(n):
                node = flatten_index(x, y, z, n)

                neighbors = []

                if plane == 'xy':
                    # Affine maps on (x, y)
                    neighbors += [
                        (modn(x + 2*y, n), y, z),
                        (modn(x - 2*y, n), y, z),
                        (modn(x + 2*y + 1, n), y, z),
                        (modn(x - (2*y + 1), n), y, z)
                    ]

                elif plane == 'xz':
                    # Affine maps on (x, z)
                    neighbors += [
                        (x, y, modn(z + 2*x, n)),
                        (x, y, modn(z - 2*x, n)),
                        (x, y, modn(z + 2*x + 1, n)),
                        (x, y, modn(z - (2*x + 1), n))
                    ]

                elif plane == 'yz':
                    # Affine maps on (y, z)
                    neighbors += [
                        (x, modn(y + 2*z, n), z),
                        (x, modn(y - 2*z, n), z),
                        (x, modn(y + 2*z + 1, n), z),
                        (x, modn(y - (2*z + 1), n), z)
                    ]

                for (nx, ny, nz) in neighbors:
                    neighbor = flatten_index(nx, ny, nz, n)
                    if node != neighbor:
                        # edge = tuple(sorted([node, neighbor]))
                        # if edge not in edge_set:
                        #     edge_set.add(edge)
                        edges.append([node, neighbor])

def stacked_affine_edges(n):
    edges = []
    edge_set = set()

    add_affine_edges_for_plane('xy', n, edges, edge_set)
    add_affine_edges_for_plane('xz', n, edges, edge_set)
    add_affine_edges_for_plane('yz', n, edges, edge_set)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def add_affine_edges_for_all_planes(n):
    edges = []
    edge_set = set()

    for x in range(n):
        for y in range(n):
            for z in range(n):
                node = flatten_index(x, y, z, n)

                # --- XY Plane: vary x & y; fix z
                xy_neighbors = [
                    (modn(x + 2*y, n), y, z),
                    (modn(x - 2*y, n), y, z),
                    (modn(x + 2*y + 1, n), y, z),
                    (modn(x - (2*y + 1), n), y, z),
                    (x, modn(y + 2*x, n), z),
                    (x, modn(y - 2*x, n), z),
                    (x, modn(y + 2*x + 1, n), z),
                    (x, modn(y - (2*x + 1), n), z)
                ]

                # --- XZ Plane: vary x & z; fix y
                xz_neighbors = [
                    (modn(x + 2*z, n), y, z),
                    (modn(x - 2*z, n), y, z),
                    (modn(x + 2*z + 1, n), y, z),
                    (modn(x - (2*z + 1), n), y, z),
                    (x, y, modn(z + 2*x, n)),
                    (x, y, modn(z - 2*x, n)),
                    (x, y, modn(z + 2*x + 1, n)),
                    (x, y, modn(z - (2*x + 1), n))
                ]

                # --- YZ Plane: vary y & z; fix x
                yz_neighbors = [
                    (x, modn(y + 2*z, n), z),
                    (x, modn(y - 2*z, n), z),
                    (x, modn(y + 2*z + 1, n), z),
                    (x, modn(y - (2*z + 1), n), z),
                    (x, y, modn(z + 2*y, n)),
                    (x, y, modn(z - 2*y, n)),
                    (x, y, modn(z + 2*y + 1, n)),
                    (x, y, modn(z - (2*y + 1), n))
                ]

                for (n_x, ny, nz) in xy_neighbors + xz_neighbors + yz_neighbors:
                    neighbor = flatten_index(n_x, ny, nz, n)
                    if node != neighbor:
                        edge = tuple([node, neighbor])
                        if edge not in edge_set:
                            edge_set.add(edge)
                            edges.append([node, neighbor])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
 
    return edge_index


