import numpy as np
from scipy.spatial import cKDTree as KDTree
from dataLoader import is_solid_edge
def vox_to_points(x,y):
    assert len(x) == len(y)
    return x[np.where(y==1)]


def compute_chamfer_distance(x,predict_y,ground_truth_y):
    predict_edge=np.full(len(predict_y), 0, dtype=int)
    ground_truth_edge=np.full(len(predict_y), 0, dtype=int)
    for i in range(len(predict_y)):
        if(predict_y[i]==1 and is_solid_edge(i,predict_y,[130,130,130])):
            predict_edge[i]=1

    for i in range(len(ground_truth_y)):
        if(ground_truth_y[i]==1 and is_solid_edge(i,ground_truth_y,[130,130,130])):
            ground_truth_edge[i]=1

    predict=vox_to_points(x,predict_edge)
    ground_truth=vox_to_points(x,ground_truth_edge)
    # one direction
    gen_points_kd_tree = KDTree(predict)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(ground_truth)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(ground_truth)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(predict)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer
