from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def identify_channels(data,cfg,use_classifier = False,use_specification = False):
    # TODO: the results look a little bit different from the reference solution
    points = np.copy(data['velodyne'])[:, :3]
    tan_elevation = points[:, 2] / np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    elevation = np.arctan(tan_elevation) * 180. / np.pi
    if not use_classifier:
        # lasers are evenly spaced in terms of vertical angles
        if use_specification:
            angle_max,angle_min = cfg['elevation_max'],cfg['elevation_min']
            # It turns out that there are quite a lot of points out of the specified angle range
        else:
            angle_max,angle_min = np.max(elevation),np.min(elevation)
        interval = (angle_max - angle_min)/(cfg['channels']-1)
        channels = np.clip(np.int_(np.round((angle_max - elevation)/interval) + 1),1,cfg['channels'])
        return channels
    else:
        # use K means algorithm to classify point clouds
        # it turns out that there will be about 7-10% of the points being assigned to different channels
        # can't tell which result is better
        random_state = 0
        init_centers = np.linspace(start = np.max(elevation),stop= np.min(elevation),num = cfg['channels'])
        init_centers = np.expand_dims(init_centers,axis = -1)
        k_means = KMeans(n_clusters = cfg['channels'],init= init_centers,random_state = random_state)
        k_means.fit(np.expand_dims(elevation,axis = -1))
        channels = k_means.labels_ + 1
        return channels

# modified from task2
def velo_to_image(data, channels, camera_id=2):
    assert camera_id in range(0, 4), "Wrong camera id"
    points = np.copy(data['velodyne'])
    T_cam_velo = data[f'T_cam{camera_id}_velo']
    P_rect = data[f'P_rect_{camera_id}0']
    points[:, 3] = np.ones_like(points[:, 3])
    points = np.expand_dims(points, axis=-1)
    cam_points = np.einsum('ij,mjk->mik', T_cam_velo, points)

    # filter out points that behind the camera
    mask_f = np.array(cam_points[:, 2] >= 0).squeeze()
    cam_points = cam_points[mask_f]
    channels = channels[mask_f]
    image_points = np.einsum('ij,mjk->mik', P_rect, cam_points)
    image_points = np.squeeze(image_points)
    dehomo_points = np.int_(np.divide(image_points[:, :2], np.expand_dims(image_points[:, 2], axis=-1)))

    # filter out points out of range of image
    h, w, _ = data['image_2'].shape
    mask_h = np.bitwise_and(dehomo_points[:, 1] < h, dehomo_points[:, 1] >= 0)
    mask_w = np.bitwise_and(dehomo_points[:, 0] < w, dehomo_points[:, 0] >= 0)
    mask_i = np.bitwise_and(mask_h, mask_w)
    image_points = dehomo_points[mask_i]
    channels = channels[mask_i]
    return image_points, channels


def visualize_2d(image, points, channels, color_ls):
    plt.imshow(image)
    plt.axis('off')
    x = points[:, 0]
    y = points[:, 1]
    c_len = len(color_ls)
    colors = [color_ls[c%c_len] for c in channels]
    plt.scatter(x, y, c=colors, s=0.01)
    plt.show()


if __name__ == '__main__':
    from easydict import EasyDict as edict

    config = edict({"x_min": -120., "x_max": 120.,
                    "y_min": -120., "y_max": 120.,
                    "x_res": 0.2, "y_res": 0.2,
                    'elevation_max': 2., 'elevation_min': -24.9,
                    'elevation_res': 0.4, 'channels': 64}) # resolution of vertical angles is an approximated value
    data = load_data('../data/data.p')
    camera_id = 2
    channels = identify_channels(data, config,use_classifier=False)
    points_image,channels = velo_to_image(data, channels, camera_id)
    color_list = ['lime','red','magenta','blue']
    visualize_2d(data['image_2'], points_image, channels,color_list)
