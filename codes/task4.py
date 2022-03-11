from data_utils import load_from_bin, load_oxts_velocity, load_oxts_angular_rate, calib_cam2cam, calib_velo2cam, \
    depth_color, print_projection_plt
import numpy as np
import os
import datetime
import cv2


class MultimodalData:
    def __init__(self, base_path, velo_cfg):
        self.velo_cfg = velo_cfg
        self.cam_dir = os.path.join(base_path, 'image_02')
        self.imu_dir = os.path.join(base_path, 'oxts')
        self.velo_dir = os.path.join(base_path, 'velodyne_points')
        self.read_all_timestamps()
        self.read_calib_matrices()

    def read_timestamps(self, filename):
        ts = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # t = datetime.datetime.strptime(line,'%Y-%m-%d %H:%M:%S.%f')
                # ts.append(t)
                timestamps_ = line[11:]
                timestamps_ = np.double(timestamps_[:2]) * 3600 + np.double(timestamps_[3:5]) * 60 + np.double(
                    timestamps_[6:])
                ts.append(timestamps_)
        return ts

    def read_all_timestamps(self):
        self.cam_ts = self.read_timestamps(os.path.join(self.cam_dir, 'timestamps.txt'))
        self.imu_ts = self.read_timestamps(os.path.join(self.imu_dir, 'timestamps.txt'))
        self.velo_ts = self.read_timestamps(os.path.join(self.velo_dir, 'timestamps.txt'))
        self.velo_ts_start = self.read_timestamps(os.path.join(self.velo_dir, 'timestamps_start.txt'))
        self.velo_ts_end = self.read_timestamps(os.path.join(self.velo_dir, 'timestamps_end.txt'))
        assert len(self.cam_ts) == len(self.imu_ts)
        assert len(self.imu_ts) == len(self.velo_ts)
        assert len(self.velo_ts) == len(self.velo_ts_start)
        assert len(self.velo_ts_start) == len(self.velo_ts_end)
        self.num_frames = len(self.cam_ts)
        self.file_list = [str(i).rjust(10, '0') for i in range(self.num_frames)]
        return

    def read_calib_matrices(self):
        R1, T1 = calib_velo2cam(os.path.join(base_path, 'calib_velo_to_cam.txt'))
        self.mat_velo2cam = np.vstack((np.hstack((R1, T1)), np.array([0., 0., 0., 1.])))
        self.mat_cam2cam = calib_cam2cam(os.path.join(base_path, 'calib_cam_to_cam.txt'), '02')
        R2, T2 = calib_velo2cam(os.path.join(base_path, 'calib_imu_to_velo.txt'))
        self.mat_imu2velo = np.vstack((np.hstack((R2, T2)), np.array([0., 0., 0., 1.])))
        return

    def read_image(self, index: int):
        assert 0 <= index < self.num_frames, 'Index out of range'
        file_path = os.path.join(self.cam_dir, 'data', self.file_list[index] + '.png')
        img = cv2.imread(file_path)  # BGR
        self.h, self.w, _ = img.shape
        return img

    def read_velo(self, index: int):
        assert 0 <= index < self.num_frames, 'Index out of range'
        file_path = os.path.join(self.velo_dir, 'data', self.file_list[index] + '.bin')
        points = load_from_bin(file_path)
        return points

    def read_imu(self, index: int):
        assert 0 <= index < self.num_frames, 'Index out of range'
        file_path = os.path.join(self.imu_dir, 'data', self.file_list[index] + '.txt')
        speed = load_oxts_velocity(file_path)
        angular_speed = np.array(load_oxts_angular_rate(file_path))
        return speed, angular_speed

    def velo_to_img(self, pc):
        points = np.copy(pc)
        points = np.hstack((points, np.ones_like(points[:, [0]])))
        points = np.expand_dims(points, axis=-1)
        cam_points = np.einsum('ij,mjk->mik', self.mat_velo2cam, points)

        # filter out points that behind the camera
        mask_f = cam_points[:, 2] >= 0
        cam_points = cam_points[mask_f[:, 0]]
        velo_points = points[mask_f[:, 0]]
        velo_points = np.squeeze(velo_points)

        image_points = np.einsum('ij,mjk->mik', self.mat_cam2cam, cam_points)
        image_points = np.squeeze(image_points)
        dehomo_points = np.int_(np.divide(image_points[:, :2], np.expand_dims(image_points[:, 2], axis=-1)))

        # filter out points out of range of image
        mask_h = np.bitwise_and(dehomo_points[:, 1] < self.h, dehomo_points[:, 1] >= 0)
        mask_w = np.bitwise_and(dehomo_points[:, 0] < self.w, dehomo_points[:, 0] >= 0)
        mask_i = np.bitwise_and(mask_h, mask_w)
        image_points = dehomo_points[mask_i]
        velo_points = velo_points[mask_i]
        return image_points, velo_points[:, :3]

    def visualize_2d(self, index: int, corrected=False):
        assert 0 <= index < self.num_frames, 'Index out of range'
        image = self.read_image(index)
        velo_points = self.read_velo(index)
        if corrected:
            raise NotImplementedError
        img_points, velo_points = self.velo_to_img(velo_points)
        distances = np.sqrt(np.sum(np.square(velo_points), axis=1))
        # TODO: point colors are not exactly the same as shown in the handout
        colors = depth_color(distances, min_d=np.min(distances), max_d=np.max(distances))
        img = print_projection_plt(img_points.T, colors, image)
        cv2.imshow('result', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    from easydict import EasyDict as edict

    base_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'problem_4')
    config = edict({"x_min": -120., "x_max": 120.,
                    "y_min": -120., "y_max": 120.,
                    "x_res": 0.2, "y_res": 0.2,
                    'elevation_max': 2., 'elevation_min': -24.9,
                    'elevation_res': 0.4, 'channels': 64})  # resolution of vertical angles is an approximated value
    data = MultimodalData(base_path, config)
    data.visualize_2d(37)
