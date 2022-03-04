from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt


def BEV_project(points, cfg, max_range=False):
    pc = np.copy(points)
    if max_range:
        x_max, x_min = cfg['x_max'], cfg['x_min']
        y_max, y_min = cfg['y_max'], cfg['y_min']
    else:
        x_max, x_min = np.max(pc[:, 0]), np.min(pc[:, 0])
        y_max, y_min = np.max(pc[:, 1]), np.min(pc[:, 1])
    h = np.int_(np.ceil((x_max - x_min) / cfg['x_res']))
    w = np.int_(np.ceil((y_max - y_min) / cfg['y_res']))
    BEV = np.zeros(shape=(h, w))
    pc[:, 0] = np.floor((x_max - pc[:, 0]) / cfg['x_res'])
    pc[:, 1] = np.floor((y_max - pc[:, 1]) / cfg['y_res'])
    indices = np.lexsort((-pc[:, 3], pc[:, 1], pc[:, 0]))
    pc = pc[indices]
    _, uni_indices = np.unique(pc[:, :2], return_index=True, axis=0)
    uni_pc = pc[uni_indices]
    BEV[np.int_(uni_pc[:, 0]), np.int_(uni_pc[:, 1])] = uni_pc[:, 3]
    return BEV


if __name__ == '__main__':
    from easydict import EasyDict as edict

    config = edict({"x_min": -120., "x_max": 120.,
                    "y_min": -120., "y_max": 120.,
                    "x_res": 0.2, "y_res": 0.2})
    data = load_data('../data/data.p')
    points = data['velodyne']
    # image -> real scene : up -> forward, down -> backward, left -> left ,right -> right
    BEV = BEV_project(points, config)
    # rotate for visualization
    BEV_vis = np.transpose(BEV)
    BEV_vis[:,:] = BEV_vis[:,::-1] # up -> left, down -> right, left -> backward ,right -> forward
    plt.imshow(BEV_vis, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.show()
