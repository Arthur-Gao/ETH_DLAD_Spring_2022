from data_utils import load_from_bin, load_oxts_velocity, load_oxts_angular_rate, calib_cam2cam, calib_velo2cam, \
    depth_color, print_projection_plt
import numpy as np
import os
import datetime
import cv2


def shift_angle(angles, angle_min):
    # shift angles in the range [angle_min,angle_min + pi]
    angle_max = angle_min + 2 * np.pi
    if not isinstance(angles, np.ndarray):
        angles = np.array([angles])
    while np.any(np.bitwise_or(angles > angle_max, angles < angle_min)):
        angles[angles < angle_min] += 2 * np.pi
        angles[angles > angle_max] -= 2 * np.pi
    return angles


def homo_rigid_matrix_4d(yaw, translation):
    if len(translation.shape) == 2:
        translation = np.expand_dims(translation, axis=-1)
    if len(yaw.shape) == 1:
        yaw = np.expand_dims(yaw, axis=-1)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    ones = np.ones_like(cos_yaw)
    zeros = np.zeros_like(cos_yaw)
    rot_matrix = np.hstack((cos_yaw, -sin_yaw, zeros, sin_yaw, cos_yaw, zeros,zeros,zeros,ones))
    rot_matrix = np.reshape(rot_matrix,(-1,3,3))
    last_row = np.expand_dims(np.hstack((zeros,zeros,zeros,ones)),axis = 1)
    trans_matrix = np.concatenate((np.concatenate((rot_matrix,translation),axis = 2),last_row),axis = 1)
    return trans_matrix


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

    def read_calib_matrices(self):
        R1, T1 = calib_velo2cam(os.path.join(base_path, 'calib_velo_to_cam.txt'))
        self.mat_velo2cam = np.vstack((np.hstack((R1, T1)), np.array([0., 0., 0., 1.])))
        self.mat_cam2img = calib_cam2cam(os.path.join(base_path, 'calib_cam_to_cam.txt'), '02')
        R2, T2 = calib_velo2cam(os.path.join(base_path, 'calib_imu_to_velo.txt'))
        self.mat_imu2velo = np.vstack((np.hstack((R2, T2)), np.array([0., 0., 0., 1.])))
        # self.mat_velo2imu = np.linalg.inv(self.mat_imu2velo)
        R2_inv = np.linalg.inv(R2)
        self.mat_velo2imu = np.vstack((np.hstack((R2_inv, -np.matmul(R2_inv, T2))), np.array([0., 0., 0., 1.])))

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

    def convert_coord(self, pc, calib_mat):
        points = np.copy(pc)
        if points.shape[1] == 3:
            points = np.hstack((points, np.ones_like(points[:, [0]])))
        if len(points.shape) == 2:
            points = np.expand_dims(points, axis=-1)
        new_points = np.einsum('ij,mjk->mik', calib_mat, points)
        return new_points

    def velo_to_img(self, pc):
        points = np.copy(pc)
        cam_points = self.convert_coord(points, self.mat_velo2cam)

        # filter out points that behind the camera
        mask_f = cam_points[:, 2] >= 0
        cam_points = cam_points[mask_f[:, 0]]
        velo_points = points[mask_f[:, 0]]
        velo_points = np.squeeze(velo_points)

        image_points = self.convert_coord(cam_points, self.mat_cam2img)
        image_points = np.squeeze(image_points)
        dehomo_points = np.int_(np.divide(image_points[:, :2], np.expand_dims(image_points[:, 2], axis=-1)))

        # filter out points out of range of image
        mask_h = np.bitwise_and(dehomo_points[:, 1] < self.h, dehomo_points[:, 1] >= 0)
        mask_w = np.bitwise_and(dehomo_points[:, 0] < self.w, dehomo_points[:, 0] >= 0)
        mask_i = np.bitwise_and(mask_h, mask_w)
        image_points = dehomo_points[mask_i]
        velo_points = velo_points[mask_i]
        return image_points, velo_points[:, :3]

    # def set_num_interval(self,num:int):
    #     self.num_intervals = num

    def correct_distortion(self, index: int, velo):
        assert 0 <= index < self.num_frames, 'Index out of range'
        start_time = self.velo_ts_start[index]
        end_time = self.velo_ts_end[index]
        forward_time = self.velo_ts[index]

        start_azim = - 2 * (forward_time - start_time) / (end_time - start_time) * np.pi
        start_azim = shift_angle(start_azim, -np.pi)[0]

        tan_azimuth = velo[:, 1] / (velo[:, 0] + 1e-10)
        mask_III = np.bitwise_and(velo[:, 0] < 0, velo[:, 1] > 0)  ## pi/2 ~ pi
        mask_IV = np.bitwise_and(velo[:, 0] < 0, velo[:, 1] < 0)  ## -pi/2 ~ -pi
        azimuth = np.arctan(tan_azimuth)
        azimuth[mask_III] = azimuth[mask_III] + np.pi
        azimuth[mask_IV] = azimuth[mask_IV] - np.pi  # make azimuth in range [-pi,pi]

        angle_diff = shift_angle(azimuth - start_azim, 0) # lidar rotating angle difference with starting position
        time_ls = (end_time - start_time) * angle_diff / (2 * np.pi) # elapse time of each points since the starting moment
        velocity, angle_velocity = self.read_imu(index)

        displacement_imu = self.calculate_displacement(time_ls, velocity, acceleration=[0, 0, 0]) # displacement of car from the starting position
        angle_displace_imu = self.calculate_displacement(time_ls, angle_velocity, acceleration=[0, 0, 0]) # angular displacement of car from the starting position
        yaw_imu = angle_displace_imu[:, [2]] # only consider rotation around z axis

        # to convert points to be in the imu coordinates at the camera triggering moment
        # TODO: a strange problem： when using cam_ts, not working
        cam_time = self.velo_ts[index]
        displacement_cam = self.calculate_displacement(cam_time - start_time, velocity, acceleration=[0, 0, 0])
        angle_displace_cam = self.calculate_displacement(cam_time - start_time, angle_velocity, acceleration=[0, 0, 0])
        yaw_cam = angle_displace_cam[:, [2]]

        transform_mat = homo_rigid_matrix_4d(-yaw_imu+yaw_cam,-displacement_imu+displacement_cam)

        imu_points = self.convert_coord(velo, self.mat_velo2imu)
        corrected_points = np.einsum('mij,mjk->mik',transform_mat,imu_points)
        velo_points = self.convert_coord(corrected_points, self.mat_imu2velo)
        return np.squeeze(velo_points)[:, :3]

    def calculate_displacement(self, times, velocity, acceleration):
        if not isinstance(times,np.ndarray):
            times = np.array([times])
        l = times.shape[0]
        displacement = np.zeros(shape=(l, 3))
        for i in range(3):
            displacement[:, i] = velocity[i] * times + acceleration[i] * np.square(times) / 2.
        return displacement

    def visualize_2d(self, indice: list, show_both = True):
        for index in indice:
            assert 0 <= index < self.num_frames, f'Index {index} out of range'
            image = self.read_image(index)
            velo_points = self.read_velo(index)
            velo_points_cor = self.correct_distortion(index, velo_points)
            img_points, velo_points = self.velo_to_img(velo_points)
            img_points_cor, velo_points_cor = self.velo_to_img(velo_points_cor)
            distances = np.sqrt(np.sum(np.square(velo_points), axis=1))
            distances_cor = np.sqrt(np.sum(np.square(velo_points_cor), axis=1))

            # TODO: point colors are not exactly the same as shown in the handout
            colors = depth_color(distances, min_d=np.min(distances), max_d=np.max(distances))
            colors_cor = depth_color(distances_cor, min_d=np.min(distances_cor), max_d=np.max(distances_cor))
            img = print_projection_plt(img_points.T, colors, image)
            img_cor = print_projection_plt(img_points_cor.T, colors_cor, image)
            if not show_both:
                cv2.imshow(f'before_{index}', img)
                cv2.waitKey(0)
                cv2.imshow(f'corrected_{index}', img_cor)
                cv2.waitKey(0)
            else:
                img_con = cv2.vconcat([img,img_cor])
                # resize for visualization
                h,w,_ = img_con.shape
                img_con = cv2.resize(img_con,(int(h),int(w*0.5)))
                cv2.imshow(f'both_{index}', img_con)
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
    # data.set_num_interval(100)
    frame_ids = range(100)
    data.visualize_2d(frame_ids, True)
