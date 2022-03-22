from load_data import load_data
import numpy as np
import cv2
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
    filtered_velo_points = points[mask_f[:, 0]]

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
    filtered_velo_points = filtered_velo_points[mask_i]
    return image_points, sem_labels, filtered_velo_points.squeeze()[:,:3]


def visualize_2d(image, points, sem_labels, color_map, bounding_box=None,save_path = None):
    sem_labels = sem_labels.flatten()
    color = np.array([color_map[i] for i in sem_labels])
    image = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_RGB2BGR)
    for i in range(points.shape[0]):
        image = cv2.circle(image, (np.int32(points[i][0]), np.int32(points[i][1])), 1, color[i].tolist(), -1)

    if bounding_box is not None:
        connect = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                            [0, 4], [1, 5], [2, 6], [3, 7],
                            [4, 5], [5, 6], [6, 7], [7, 4]])
        for box in bounding_box:
            for c in connect:
                start = box[c[0]]
                end = box[c[1]]
                image = cv2.line(image,start,end,color = [0,255,0],thickness=2)
    cv2.imshow('res', image)
    cv2.waitKey(0)
    if save_path:
        cv2.imwrite(save_path,image)


def get_velo_box_corner(data):
    objects = data['objects']
    boxes = []
    for obj in objects:
        box = []
        # no need to store catagory since all objects in this exercise are cars
        h, w, l = obj[8:11]  # ambiguous here
        x, y, z = obj[11:14]
        yaw = obj[14]
        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)
        box.append([x - w / 2 * sin_yaw - l / 2 * cos_yaw, y, z - w / 2 * cos_yaw + l / 2 * sin_yaw])
        box.append([x - w / 2 * sin_yaw + l / 2 * cos_yaw, y, z - w / 2 * cos_yaw - l / 2 * sin_yaw])
        box.append([x + w / 2 * sin_yaw + l / 2 * cos_yaw, y, z + w / 2 * cos_yaw - l / 2 * sin_yaw])
        box.append([x + w / 2 * sin_yaw - l / 2 * cos_yaw, y, z + w / 2 * cos_yaw + l / 2 * sin_yaw])
        # y: down, therefore minus h
        box.append([x - w / 2 * sin_yaw - l / 2 * cos_yaw, y-h, z - w / 2 * cos_yaw + l / 2 * sin_yaw])
        box.append([x - w / 2 * sin_yaw + l / 2 * cos_yaw, y-h, z - w / 2 * cos_yaw - l / 2 * sin_yaw])
        box.append([x + w / 2 * sin_yaw + l / 2 * cos_yaw, y-h, z + w / 2 * cos_yaw - l / 2 * sin_yaw])
        box.append([x + w / 2 * sin_yaw - l / 2 * cos_yaw, y-h, z + w / 2 * cos_yaw + l / 2 * sin_yaw])
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
    points_image, sem_labels, filtered_velo_points = velo_to_image(data, camera_id)
    visualize_2d(data['image_2'], points_image, sem_labels, data['color_map'],save_path = '../results/figure2_1.png')

    # question 2
    boxes_velo = get_velo_box_corner(data)
    boxes_image, filtered_boxes_velo = get_boxes_corner_2d(data, boxes_velo, camera_id)
    visualize_2d(data['image_2'], points_image, sem_labels, data['color_map'], boxes_image,save_path = '../results/figure2_2.png')

    # question 3: entire scene
    visualizer = vis.Visualizer()
    visualizer.update(data['velodyne'][:, :3], data['sem_label'], data['color_map'])
    visualizer.update_boxes(filtered_boxes_velo[:, :, :3])
    vispy.app.run()

    # question 3: partly visualize for counting boxes
    visualizer_part = vis.Visualizer()
    visualizer_part.update(filtered_velo_points, sem_labels, data['color_map'])
    visualizer_part.update_boxes(filtered_boxes_velo[:, :, :3])
    vispy.app.run()
