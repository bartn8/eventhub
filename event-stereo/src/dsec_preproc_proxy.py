# %%
import os
import argparse
from pathlib import Path

import numpy as np
import cv2
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as Rot

import tqdm
import glob
import shutil

# %%


class Transform:
    def __init__(self, translation: np.ndarray, rotation: Rot):
        if translation.ndim > 1:
            self._translation = translation.flatten()
        else:
            self._translation = translation
        assert self._translation.size == 3
        self._rotation = rotation

    @staticmethod
    def from_transform_matrix(transform_matrix: np.ndarray):
        translation = transform_matrix[:3, 3]
        rotation = Rot.from_matrix(transform_matrix[:3, :3])
        return Transform(translation, rotation)

    @staticmethod
    def from_rotation(rotation: Rot):
        return Transform(np.zeros(3), rotation)

    def R_matrix(self):
        return self._rotation.as_matrix()

    def R(self):
        return self._rotation

    def t(self):
        return self._translation

    def T_matrix(self) -> np.ndarray:
        return self._T_matrix_from_tR(self._translation, self._rotation.as_matrix())

    def q(self):
        # returns (x, y, z, w)
        return self._rotation.as_quat()

    def euler(self):
        return self._rotation.as_euler('xyz', degrees=True)

    def __matmul__(self, other):
        # a (self), b (other)
        # returns a @ b
        #
        # R_A | t_A   R_B | t_B   R_A @ R_B | R_A @ t_B + t_A
        # --------- @ --------- = ---------------------------
        # 0   | 1     0   | 1     0         | 1
        #
        rotation = self._rotation * other._rotation
        translation = self._rotation.apply(other._translation) + self._translation
        return Transform(translation, rotation)

    def inverse(self):
        #           R_AB  | A_t_AB
        # T_AB =    ------|-------
        #           0     | 1
        #
        # to be converted to
        #
        #           R_BA  | B_t_BA    R_AB.T | -R_AB.T @ A_t_AB
        # T_BA =    ------|------- =  -------|-----------------
        #           0     | 1         0      | 1
        #
        # This is numerically more stable than matrix inversion of T_AB
        rotation = self._rotation.inv()
        translation = - rotation.apply(self._translation)
        return Transform(translation, rotation)

# %%

def reproject_depth(W_start, H_start, K, depth, K_end, RT, W_end, H_end):
    xx, yy = np.meshgrid(np.arange(W_start), np.arange(H_start))
    points_grid = np.stack(((xx-K[0,2])/K[0,0], (yy-K[1,2])/K[1,1], np.ones_like(xx)), axis=0) * depth
    mask = np.ones((H_start, W_start), dtype=bool)
    mask[depth<=0] = False
    depth_pts = points_grid.transpose(1,2,0)[mask]

    camera_points = (RT @ np.vstack([depth_pts.T, np.ones(depth_pts.shape[0])])).T

    if camera_points.shape[0] == 0:
        return np.zeros([H_end, W_end])

    rvecs = np.zeros((3,1)) 
    tvecs = np.zeros((3,1))
    D_end = np.zeros((4,1))

    _camera_points = camera_points[:,:3]

    imgpts, _ = cv2.projectPoints(_camera_points, rvecs, tvecs, K_end, D_end)
    
    imgpts = imgpts[:,0,:]
    valid_points = (imgpts[:, 1] >= 0) & (imgpts[:, 1] < H_end) & \
                (imgpts[:, 0] >= 0) & (imgpts[:, 0] < W_end)
    imgpts = imgpts[valid_points,:]

    _end_depth = camera_points[valid_points,2]

    end_depth = np.zeros([H_end, W_end])
    end_depth[imgpts[:,1].astype(int), imgpts[:,0].astype(int)] = _end_depth

    return end_depth

def process(seqpath, min_depth=0.5, max_depth=100, overwrite=False):

    seqpath = Path(seqpath)
    assert seqpath.is_dir()
    print(f'start processing: {seqpath}')

    confpath = seqpath / 'calibration' / 'cam_to_cam.yaml'
    assert confpath.exists()
    conf = OmegaConf.load(confpath)

    # # Copy timestamps
    # src_timestamp_path = seqpath / "images" / "timestamps.txt"
    # dst_timestamp_path = seqpath / "disparity" / "proxy_timestamps.txt"
    # os.makedirs(dst_timestamp_path.parent, exist_ok=True)
    # shutil.copy(src_timestamp_path, dst_timestamp_path)

    disparity_left_dir = seqpath / "disparity" / "proxy_image" / "left"
    outdir = seqpath / "disparity" / "proxy_event" / "left"

    if not disparity_left_dir.exists() or not disparity_left_dir.is_dir():
        print(f"Skipping {seqpath}: {disparity_left_dir} does not exist")
        return

    os.makedirs(outdir, exist_ok=True)

    disp_files = [x for x in disparity_left_dir.iterdir() if x.suffix == '.png']

    # Get mapping for this sequence: From rectified event left (camRect0) to rectified frame left (camRect1) (backward mapping)

    K_r0 = np.eye(3)
    K_r0[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect0']['camera_matrix']
    K_r1 = np.eye(3)
    K_r1[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect1']['camera_matrix']

    R_r0_0 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect0']))
    R_r1_1 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect1']))

    T_r0_0 = Transform.from_rotation(R_r0_0)
    T_r1_1 = Transform.from_rotation(R_r1_1)
    T_1_0 = Transform.from_transform_matrix(np.array(conf['extrinsics']['T_10']))
    

    T_r1_r0 = T_r1_1 @ T_1_0 @ T_r0_0.inverse()
    
    _T_r0_r1 = T_r1_r0.inverse()
    T_r0_r1 = np.eye(4)
    T_r0_r1[:3, :3] = _T_r0_r1.R().as_matrix()
    T_r0_r1[:3, 3] = _T_r0_r1.t()

    Q_12 = np.array(conf['disparity_to_depth']['cams_12'])

    focal_12 = Q_12[2, 3]
    baseline_12 = np.abs(1/Q_12[3, 2])

    Q_03 = np.array(conf['disparity_to_depth']['cams_03'])

    focal_03 = Q_03[2, 3]
    baseline_03 = np.abs(1/Q_03[3, 2])  

    assert focal_03 > 0
    assert baseline_03 > 0
    assert np.isclose(focal_03, K_r0[0,0])

    try:
        for entry in tqdm.tqdm(disp_files):
            disp_out_file = outdir / entry.name
            if disp_out_file.exists() and not overwrite:
                continue

            # disp_in = cv2.imread(str(entry), cv2.IMREAD_ANYDEPTH) / 256.0
            disp_in = cv2.imread(str(entry), cv2.IMREAD_ANYDEPTH) / 128.0
            disp_in = disp_in.astype(np.float32)
            
            # depth_in = cv2.reprojectImageTo3D(disp_in, Q_12)[...,-1]
            depth_in = disp_in.copy()
            depth_in[depth_in > 0] = (focal_12 * baseline_12) / depth_in[depth_in > 0]

            depth_in[depth_in < min_depth] = 0
            depth_in[depth_in > max_depth] = 0
            
            H_start, W_start = depth_in.shape
            H_end, W_end = 480, 640
            depth_out = reproject_depth(W_start, H_start, K_r1, depth_in, K_r0, T_r0_r1, W_end, H_end)
            
            disp_out = depth_out.copy()
            disp_out[disp_out > 0] = (focal_03 * baseline_03) / disp_out[disp_out > 0]

            cv2.imwrite(str(disp_out_file), np.clip(disp_out * 256, 0, 65535).astype(np.uint16))

    except KeyboardInterrupt as e:
        print(f'Error processing {seqpath}: {e}')
        # rm last disp_out_file
        if disp_out_file:
            os.remove(disp_out_file)

    print(f'done processing: {seqpath}')

def transform_inv(T):
    T_inv = np.eye(4)
    T_inv[:3,:3] = T[:3,:3].T
    T_inv[:3,3] = -1.0 * ( T_inv[:3,:3] @ T[:3,3] )
    return T_inv

# def process_right(seqpath):
#     seqpath = Path(seqpath)
#     assert seqpath.is_dir()
#     print(f'start processing: {seqpath}')

#     confpath = seqpath / 'calibration' / 'cam_to_cam.yaml'
#     assert confpath.exists()
#     conf = OmegaConf.load(confpath)

#     images_right_dir = seqpath / 'images' / 'right'
#     outdir = images_right_dir / 'ev_inf'
#     os.makedirs(outdir, exist_ok=True)

#     image_in_dir = images_right_dir / 'rectified'
#     # Get mapping for this sequence: From rectified event right (camRect3) to rectified frame right (camRect2) (backward mapping)

#     K_r3 = np.eye(3)
#     K_r3[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect3']['camera_matrix']
#     K_r2 = np.eye(3)
#     K_r2[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect2']['camera_matrix']

#     R_r3_3 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect3']))
#     R_r2_2 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect2']))

#     T_r3_3 = Transform.from_rotation(R_r3_3)
#     T_r2_2 = Transform.from_rotation(R_r2_2)
#     # T_2_3 = Transform.from_transform_matrix(np.array(conf['extrinsics']['T_23'])) # Not present in the config
#     T_2_3 = Transform.from_transform_matrix(transform_inv(np.array(conf['extrinsics']['T_32'])))

#     T_r1_r0 = T_r2_2 @ T_2_3 @ T_r3_3.inverse()
#     R_r1_r0_matrix = T_r1_r0.R().as_matrix()
#     P_r1_r0 = K_r2 @ R_r1_r0_matrix @ np.linalg.inv(K_r3)

#     ht = 480
#     wd = 640
#     # coords: ht, wd, 2
#     coords = np.stack(np.meshgrid(np.arange(wd), np.arange(ht)), axis=-1)
#     # coords_hom: ht, wd, 3
#     coords_hom = np.concatenate((coords, np.ones((ht, wd, 1))), axis=-1)
#     # mapping: ht, wd, 3
#     mapping = (P_r1_r0 @ coords_hom[..., None]).squeeze()
#     # mapping: ht, wd, 2
#     mapping = (mapping/mapping[..., -1][..., None])[..., :2]
#     mapping = mapping.astype('float32')

#     image_files = [x for x in image_in_dir.iterdir() if x.suffix == '.png']

#     try:
#         for entry in tqdm.tqdm(image_files):
#             disp_out_file = outdir / entry.name
#             if disp_out_file.exists():
#                 continue

#             image_in = cv2.imread(str(entry))
#             image_out = cv2.remap(image_in, mapping, None, interpolation=cv2.INTER_LINEAR)
#             cv2.imwrite(str(disp_out_file), image_out)
#     except KeyboardInterrupt as e:
#         print(f'Error processing {seqpath}: {e}')
#         # rm last disp_out_file
#         if disp_out_file:
#             os.remove(disp_out_file)

#     print(f'done processing: {seqpath}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DSEC dataset sequences.")
    parser.add_argument("input_path", type=str, help="Input path to DSEC dataset (supports glob patterns)")
    parser.add_argument("--min_depth", type=float, default=0.5, help="Minimum depth to consider")
    parser.add_argument("--max_depth", type=float, default=100.0, help="Maximum depth to consider")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed files")
    args = parser.parse_args()

    input_path = Path(args.input_path) / "*"

    datapaths = glob.glob(str(input_path))
    for datapath in datapaths:
        process(datapath, min_depth=args.min_depth, max_depth=args.max_depth, overwrite=args.overwrite)
        # process_right(datapath)

    print("Processing complete.")

