from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt


def velo_to_image(data, camera_id=2):
    assert camera_id in range(0, 4), "Wrong camera id"
    points = np.copy(data['velodyne'])
    T_cam_velo = data[f'T_cam{camera_id}_velo']
    P_rect = data[f'P_rect_{camera_id}0']
    points[:, 3] = np.ones_like(points[:, 3])
    points = np.expand_dims(points, axis=-1)
    cam_points = np.einsum('ij,mjk->mik', T_cam_velo, points)

    # filter out points that behind the camera
    mask_f = cam_points[:, 2] >= 0
    cam_points = cam_points[mask_f[:, 0]]

    image_points = np.einsum('ij,mjk->mik', P_rect, cam_points)
    image_points = np.squeeze(image_points)
    sem_labels = data['sem_label'][mask_f[:, 0]]
    dehomo_points = np.int_(np.divide(image_points[:, :2], np.expand_dims(image_points[:, 2], axis=-1)))

    # filter out points out of range of image
    h, w, _ = data['image_2'].shape
    mask_h = np.bitwise_and(dehomo_points[:, 1] < h, dehomo_points[:, 1] >= 0)
    mask_w = np.bitwise_and(dehomo_points[:, 0] < w, dehomo_points[:, 0] >= 0)
    mask_i = np.bitwise_and(mask_h, mask_w)
    image_points = dehomo_points[mask_i]
    sem_labels = sem_labels[mask_i]
    return image_points, sem_labels


def visualize_2d(image, points, sem_labels, color_map, bounding_box=None):
    plt.imshow(image)
    plt.axis('off')
    x = points[:, 0]
    y = points[:, 1]
    sem_labels = sem_labels.flatten()
    color = np.array([color_map[i] for i in sem_labels]) / 255.
    # convert to RGB
    color = color[:, ::-1]
    plt.scatter(x, y, c=color, s=0.01)

    if bounding_box is not None:
        connect = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                            [0, 4], [1, 5], [2, 6], [3, 7],
                            [4, 5], [5, 6], [6, 7], [7, 4]])
        for box in bounding_box:
            for c in connect:
                x, y = box[c, 0], box[c, 1]
                plt.plot(x, y, c='lime', linewidth=1)
    plt.show()


def get_velo_box_corner(data):
    objects = data['objects']
    boxes = []
    for obj in objects:
        box = []
        # no need to store catagory since all objects in this exercise are cars
        h, l, w = obj[8:11]  # ambiguous here
        x, y, z = obj[11:14]
        yaw = obj[14]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        box.append([x - w / 2 * cos_yaw - l / 2 * sin_yaw, y, z - w / 2 * sin_yaw + l / 2 * cos_yaw])
        box.append([x - w / 2 * cos_yaw + l / 2 * sin_yaw, y, z - w / 2 * sin_yaw - l / 2 * cos_yaw])
        box.append([x + w / 2 * cos_yaw + l / 2 * sin_yaw, y, z + w / 2 * sin_yaw - l / 2 * cos_yaw])
        box.append([x + w / 2 * cos_yaw - l / 2 * sin_yaw, y, z + w / 2 * sin_yaw + l / 2 * cos_yaw])
        # y: down, therefore minus h
        box.append([x - w / 2 * cos_yaw - l / 2 * sin_yaw, y - h, z - w / 2 * sin_yaw + l / 2 * cos_yaw])
        box.append([x - w / 2 * cos_yaw + l / 2 * sin_yaw, y - h, z - w / 2 * sin_yaw - l / 2 * cos_yaw])
        box.append([x + w / 2 * cos_yaw + l / 2 * sin_yaw, y - h, z + w / 2 * sin_yaw - l / 2 * cos_yaw])
        box.append([x + w / 2 * cos_yaw - l / 2 * sin_yaw, y - h, z + w / 2 * sin_yaw + l / 2 * cos_yaw])
        boxes.append(box)
    boxes = np.array(boxes)
    boxes = np.concatenate((boxes, np.ones_like(boxes[:, :, [0]])), axis=-1)
    boxes = np.expand_dims(boxes, axis=-1)
    T_cam_velo_0 = data['T_cam0_velo']
    T_cam_velo_0_inverse = np.linalg.inv(T_cam_velo_0)
    boxes_velo = np.einsum('ij,mnjk->mnik', T_cam_velo_0_inverse, boxes)
    boxes_velo = np.squeeze(boxes_velo)
    return boxes_velo


def get_boxes_corner_2d(data, boxes_velo, camera_id):
    assert camera_id in range(0, 4), "Wrong camera id"
    T_cam_velo = data[f'T_cam{camera_id}_velo']
    P_rect = data[f'P_rect_{camera_id}0']

    boxes_velo = np.expand_dims(boxes_velo, axis=-1)
    boxes_cam = np.einsum('ij,mnjk->mnik', T_cam_velo, boxes_velo)
    # mask out potential objects behind the camera
    mask_f = np.all(np.squeeze(boxes_cam[:, :, 2] >= 0), axis=1)
    boxes_cam = boxes_cam[mask_f]
    filtered_boxes_velo = boxes_velo[mask_f]

    boxes_image = np.einsum('ij,mnjk->mnik', P_rect, boxes_cam)
    boxes_image = np.squeeze(boxes_image)
    boxes_image = np.int_(np.divide(boxes_image[:, :, :2], np.expand_dims(boxes_image[:, :, 2], axis=-1)))

    # filter out objects out of range of image
    h, w, _ = data['image_2'].shape
    mask_h = np.bitwise_and(boxes_image[:, :, 1] < h, boxes_image[:, :, 1] >= 0)
    mask_w = np.bitwise_and(boxes_image[:, :, 0] < w, boxes_image[:, :, 0] >= 0)
    mask_i = np.all(np.bitwise_and(mask_h, mask_w), axis=1)
    boxes_image = boxes_image[mask_i]
    filtered_boxes_velo = filtered_boxes_velo[mask_i]
    return boxes_image, np.squeeze(filtered_boxes_velo)


if __name__ == '__main__':
    from easydict import EasyDict as edict
    import vispy

    vis = __import__('3dvis')
    config = edict({"x_min": -120., "x_max": 120.,
                    "y_min": -120., "y_max": 120.,
                    "x_res": 0.2, "y_res": 0.2})
    data = load_data('../data/data.p')
    camera_id = 2

    # question 1
    points_image, sem_labels = velo_to_image(data, camera_id)
    visualize_2d(data['image_2'], points_image, sem_labels, data['color_map'])

    # question 2
    boxes_velo = get_velo_box_corner(data)
    boxes_image, filtered_boxes_velo = get_boxes_corner_2d(data, boxes_velo, camera_id)
    visualize_2d(data['image_2'], points_image, sem_labels, data['color_map'], boxes_image)

    # question 3
    visualizer = vis.Visualizer()
    visualizer.update(data['velodyne'][:, :3], data['sem_label'], data['color_map'])
    visualizer.update_boxes(filtered_boxes_velo[:, :, :3])
    vispy.app.run()
