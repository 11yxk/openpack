"""
- Ref.1: https://colab.research.google.com/github/machine-perception-robotics-group/MPRGDeepLearningLectureNotebook/blob/master/15_gcn/03_action_recognition_ST_GCN.ipynb#scrollTo=Vk-AMCVb5jqM
- Ref.2: https://github.com/open-mmlab/mmskeleton/blob/master/mmskeleton/ops/st_gcn/graph.py
"""
from typing import Tuple

import numpy as np

NUM_NODES_NTU_RGBD = 25
NTU_RGBD_SKELETON_LAYOUT = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                            (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                            (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                            (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                            (22, 23), (23, 8), (24, 25), (25, 12))

# NUM_NODES_MSCOCO = 17
# MSCOCO_SKELETON_LAYOUT = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
#                           (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
#                           (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
#                           (3, 5), (4, 6))


NUM_NODES_MSCOCO = 15
MSCOCO_SKELETON_LAYOUT = ((13, 11), (14, 12), (11, 12),
                          (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
                          (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
                          (3, 5), (4, 6))
class Graph():
    """
    Attributes:
        A (np.ndarray): adjacency matrix with shape = (NUM_HOP+1, NUM_NODE, NUM_NODE)
        skeleton (Tuple[Tuple[int,int], ...]): list of edges.
        hop_size (int): make connection with noded that can be reached in ``hop_size`` hop.
    Todo:
        - Update graph construction method with mmskeleton's implementation.
    """
    A: np.ndarray = None

    def __init__(self, hop_size: int = 2, num_nodes: int = None,
                 skeleton: Tuple[Tuple[int, int], ...] = None):
        self.skeleton = skeleton
        self.num_node = num_nodes
        self.get_edge()

        self.hop_size = hop_size
        self.hop_dist = self.get_hop_distance(
            self.num_node, self.edge, hop_size=hop_size)

        # TODO: Check original paper for the adjacency matrix definition.
        self.get_adjacency()

    def __str__(self):
        return str(self.A)

    def get_edge(self) -> None:
        self_link = [(i, i) for i in range(self.num_node)]  # ループ
        neighbor_link = [(i - 1, j - 1) for (i, j) in self.skeleton]
        self.edge = self_link + neighbor_link

    def get_adjacency(self):
        valid_hop = range(0, self.hop_size + 1, 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dist == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dist == hop] = normalize_adjacency[self.hop_dist == hop]
        self.A = A

    def get_hop_distance(self, num_node, edge, hop_size):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [
            np.linalg.matrix_power(
                A,
                d) for d in range(
                hop_size +
                1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(hop_size, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        DAD = np.dot(A, Dn)
        return DAD


def get_adjacency_matrix(
        layout: str = "MSCOCO",
        hop_size: int = 2) -> np.ndarray:
    """Returns adjacency matrix.

    Args:
        layout (str, optional): skeleton layout. {MSCOCO, NTU-RGBD}. Defaults to "MSCOCO".
        hop_size (int, optional): maximum distance of connection. Defaults to 2.

    Raises:
        ValueError: _description_

    Returns:
        np.ndarray: adjacency matrix
    """
    if layout.upper() == "NTU-RGBD":
        graph = Graph(hop_size, NUM_NODES_NTU_RGBD, NTU_RGBD_SKELETON_LAYOUT)
    elif layout.upper() == "MSCOCO":
        graph = Graph(hop_size, NUM_NODES_MSCOCO, MSCOCO_SKELETON_LAYOUT)
    else:
        raise ValueError(f"unknown layout, [layout = {layout}]")
    return graph.A
