import numpy as np
import h5py
import hdf5plugin
import cv2
import glob
import os
import tqdm
import yaml
import numba
import argparse
import errno
import re


def transform_inv(T):
    T_inv = np.eye(4)
    T_inv[:3,:3] = T[:3,:3].T
    T_inv[:3,3] = -1.0 * ( T_inv[:3,:3] @ T[:3,3] )
    return T_inv

def rectify(stereo_left_cam_matrix, stereo_left_dist, stereo_right_cam_matrix, stereo_right_dist, stereo_left_to_stereo_right, h, w):
    R1,R2,P1,P2,Q,_,_ = cv2.stereoRectify(stereo_left_cam_matrix, stereo_left_dist, stereo_right_cam_matrix, stereo_right_dist, (w,h), stereo_left_to_stereo_right[:3,:3], stereo_left_to_stereo_right[:3,3], flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)
    leftmapX, leftmapY = cv2.initUndistortRectifyMap(stereo_left_cam_matrix, stereo_left_dist, R1, P1, (w,h), cv2.CV_32FC1)
    rightmapX, rightmapY = cv2.initUndistortRectifyMap(stereo_right_cam_matrix, stereo_right_dist, R2, P2, (w,h), cv2.CV_32FC1)

    leftmap = np.concatenate([np.expand_dims(leftmapX, -1), np.expand_dims(leftmapY, -1)], -1)
    rightmap = np.concatenate([np.expand_dims(rightmapX, -1), np.expand_dims(rightmapY, -1)], -1)

    return R1,R2,P1,P2, Q, leftmap, rightmap

#https://github.com/daniilidis-group/m3ed/blob/main/build_system/semantics/internimage.py
#https://stackoverflow.com/questions/41703210/inverting-a-real-valued-index-grid/68706787#68706787
#https://stackoverflow.com/questions/72635492/what-are-the-inaccuracies-of-this-inverse-map-function-in-opencv/72649764#72649764
def invert_map(F):
    # shape is (h, w, 2), an "xymap"
    (h, w) = F.shape[:2]
    I = np.zeros_like(F)
    I[:,:,1], I[:,:,0] = np.indices((h, w)) # identity map
    P = np.copy(I)
    for i in range(10):
        correction = I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
        P += correction * 0.5
    return P

@numba.njit
def forward_remap(disp: np.ndarray, map: np.ndarray):
    """
    disp: np.ndarray HxW
    map: np.ndarray HxWx2
    """
    re_disp = np.zeros_like(disp)
    H,W = disp.shape[:2]

    for y in range(H):
        for x in range(W):
            if disp[y,x] > 0:
                rx,ry = map[y,x]
                rx,ry = round(rx), round(ry)
                if 0 <= rx < W and 0 <= ry < H:
                    re_disp[ry,rx] = disp[y,x]
    
    return re_disp


def main(args):
    input_paths = args.input_folder
    output_path = args.output_folder
    delete = args.delete
    # ignore_glob = args.ignore_glob
    ignore_h5 = args.ignore_h5

    #Search for valid pair of data+depth_gt
    # if ignore_glob:
    #     data_glob = input_paths
    # else:
    data_glob = []
    for input_path in input_paths:
        data_glob += sorted(glob.glob(os.path.join(input_path, "*/*_data.h5")))

    data_list = []
    gt_list = []

    print("Checking for valid pairs...")

    for data_path in tqdm.tqdm(data_glob):
        gt_path = data_path.replace('_data.h5', '_depth_gt.h5')
        if os.path.exists(gt_path):
            data_list.append(data_path)
            gt_list.append(gt_path)

    print("Converting to DSEC style...")

    pbar = tqdm.tqdm(zip(data_list, gt_list), total=len(data_list))

    for data_path, gt_path in pbar:

        pbar.set_description(os.path.basename(data_path).replace('_data.h5', ''))

        data_file = h5py.File(data_path, 'r+')
        depth_file = h5py.File(gt_path, 'r+')

        basename = os.path.basename(data_path).replace("_data.h5", "")

        for dir in ["calibration", "disparity/event", "events/left", "events/right"]:
            os.makedirs(os.path.join(output_path, basename, dir), exist_ok=True)

        #Create left and right event data h5
        for cam in ['left', 'right']:
            pbar.set_description(f"Creating {cam} event history...")

            try:
                event_file = h5py.File(os.path.join(output_path, basename, f"events/{cam}/events.h5"), 'x')
                events_group = event_file.create_group("events")

                _compression_type = hdf5plugin.Zstd(clevel=3) if hdf5plugin is not None else 'lzf'

                events_group.create_dataset("p", data=data_file[f'/prophesee/{cam}/p'], compression=_compression_type)
                pbar.set_description(f"Creating {cam} p event history... (20%)")

                events_group.create_dataset("t", data=data_file[f'/prophesee/{cam}/t'], compression=_compression_type)
                pbar.set_description(f"Creating {cam} t event history... (40%)")

                events_group.create_dataset("x", data=data_file[f'/prophesee/{cam}/x'], compression=_compression_type)
                pbar.set_description(f"Creating {cam} x event history... (60%)")

                events_group.create_dataset("y", data=data_file[f'/prophesee/{cam}/y'], compression=_compression_type)
                pbar.set_description(f"Creating {cam} y event history... (80%)")

                event_file.create_dataset("ms_to_idx", data=data_file[f'/prophesee/{cam}/ms_map_idx'], compression=_compression_type)
                event_file.create_dataset("t_offset", data=0)
                pbar.set_description(f"Creating {cam} event history... (100%)")

                event_file.close()
            except OSError as e:
                if not ignore_h5:
                    raise e
                if e.errno is not None:
                    myerrno = e.errno
                else:
                    mymsgs = str(e).split(",")
                    myerrno = -1
                    for mymsg in mymsgs:
                        mymsg = mymsg.replace(" ", "")
                        if re.match("^errno=[0-9]+$", mymsg):
                            myerrno = int(mymsg.split("=")[1])
                            break
                if myerrno != errno.EEXIST:
                    raise e

        #Get calibration files
        pbar.set_description(f"Creating calibration files...")

        """
        Intrinsics
        cam0: Event camera left
        cam1: Frame camera left
        cam2: Frame camera right
        cam3: Event camera right
        camRectX: Rectified version of camX. E.g. camRect0 is the rectified version of cam0.
        Extrinsics
        T_XY: Rigid transformation that transforms a point from the camY coordinate frame into the camX coordinate frame.
        R_rectX: Rotation that transforms a point from the camX coordinate frame into the camRectX coordinate frame.
        """

        K_event_left = np.eye(3)
        K_event_left[0, 0] = data_file["/prophesee/left/calib/intrinsics"][0]
        K_event_left[1, 1] = data_file["/prophesee/left/calib/intrinsics"][1]
        K_event_left[0, 2] = data_file["/prophesee/left/calib/intrinsics"][2]
        K_event_left[1, 2] = data_file["/prophesee/left/calib/intrinsics"][3]
        D_event_left = np.array(data_file['/prophesee/left/calib/distortion_coeffs'])

        K_event_right = np.eye(3)
        K_event_right[0, 0] = data_file["/prophesee/right/calib/intrinsics"][0]
        K_event_right[1, 1] = data_file["/prophesee/right/calib/intrinsics"][1]
        K_event_right[0, 2] = data_file["/prophesee/right/calib/intrinsics"][2]
        K_event_right[1, 2] = data_file["/prophesee/right/calib/intrinsics"][3]
        D_event_right = np.array(data_file['/prophesee/right/calib/distortion_coeffs'])

        event_right_to_event_left = np.array(data_file["/prophesee/right/calib/T_to_prophesee_left"])
        RT_event = transform_inv(event_right_to_event_left)

        R1_event, R2_event, P1_event, P2_event, Q_event, leftmap_event, rightmap_event = rectify(K_event_left, D_event_left, K_event_right, D_event_right, RT_event, 720, 1280)

        inv_leftmap_event = invert_map(leftmap_event)
        inv_rightmap_event = invert_map(rightmap_event)
        
        R1_event_4x4 = np.eye(4)
        R1_event_4x4[:3,:3] = R1_event

        R2_event_4x4 = np.eye(4)
        R2_event_4x4[:3,:3] = R2_event

        focal_event = abs(Q_event[2,3])
        baseline_event = abs(1/Q_event[3,2])

        K_gray_left = np.eye(3)
        K_gray_left[0, 0] = data_file["/ovc/left/calib/intrinsics"][0]
        K_gray_left[1, 1] = data_file["/ovc/left/calib/intrinsics"][1]
        K_gray_left[0, 2] = data_file["/ovc/left/calib/intrinsics"][2]
        K_gray_left[1, 2] = data_file["/ovc/left/calib/intrinsics"][3]
        D_gray_left = np.array(data_file['/ovc/left/calib/distortion_coeffs'])

        K_gray_right = np.eye(3)
        K_gray_right[0, 0] = data_file["/ovc/right/calib/intrinsics"][0]
        K_gray_right[1, 1] = data_file["/ovc/right/calib/intrinsics"][1]
        K_gray_right[0, 2] = data_file["/ovc/right/calib/intrinsics"][2]
        K_gray_right[1, 2] = data_file["/ovc/right/calib/intrinsics"][3]
        D_gray_right = np.array(data_file['/ovc/right/calib/distortion_coeffs'])

        gray_left_to_event_left = np.array(data_file["/ovc/left/calib/T_to_prophesee_left"])
        gray_right_to_event_left = np.array(data_file["/ovc/right/calib/T_to_prophesee_left"])
        RT_gray = transform_inv(gray_right_to_event_left) @ gray_left_to_event_left

        R1_gray, R2_gray, P1_gray, P2_gray, Q_gray, _, _ = rectify(K_gray_left, D_gray_left, K_gray_right, D_gray_right, RT_gray, 800, 1280)

        R1_gray_4x4 = np.eye(4)
        R1_gray_4x4[:3,:3] = R1_gray

        R2_gray_4x4 = np.eye(4)
        R2_gray_4x4[:3,:3] = R2_gray

        cam_to_cam = {}
        cam_to_cam['intrinsics'] = {}
        
        cam_to_cam['intrinsics']['cam0'] = {}
        cam_to_cam['intrinsics']['cam0']['camera_type'] = 'event'
        cam_to_cam['intrinsics']['cam0']['canera_location'] = 'left'
        cam_to_cam['intrinsics']['cam0']['is_rectified'] = False
        cam_to_cam['intrinsics']['cam0']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['cam0']['distortion_coeffs'] = D_event_left.tolist()
        cam_to_cam['intrinsics']['cam0']['distortion_model'] = 'radtan'
        cam_to_cam['intrinsics']['cam0']['resolution'] = [1280, 720]
        cam_to_cam['intrinsics']['cam0']['camera_matrix'] = np.array(data_file['/prophesee/left/calib/intrinsics']).tolist()

        cam_to_cam['intrinsics']['camRect0'] = {}
        cam_to_cam['intrinsics']['camRect0']['camera_type'] = 'event'
        cam_to_cam['intrinsics']['camRect0']['canera_location'] = 'left'
        cam_to_cam['intrinsics']['camRect0']['is_rectified'] = True
        cam_to_cam['intrinsics']['camRect0']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['camRect0']['resolution'] = [1280, 720]
        cam_to_cam['intrinsics']['camRect0']['camera_matrix'] = np.array([P1_event[0,0], P1_event[1,1], P1_event[0,2], P1_event[1,2]]).tolist()

        cam_to_cam['intrinsics']['cam1'] = {}
        cam_to_cam['intrinsics']['cam1']['camera_type'] = 'frame'
        cam_to_cam['intrinsics']['cam1']['canera_location'] = 'left'
        cam_to_cam['intrinsics']['cam1']['is_rectified'] = False
        cam_to_cam['intrinsics']['cam1']['T_cn_cnm1'] = transform_inv(gray_left_to_event_left).tolist()#cam0->cam1
        cam_to_cam['intrinsics']['cam1']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['cam1']['distortion_coeffs'] = D_gray_left.tolist()
        cam_to_cam['intrinsics']['cam1']['distortion_model'] = 'radtan'
        cam_to_cam['intrinsics']['cam1']['resolution'] = [1280, 800]
        cam_to_cam['intrinsics']['cam1']['camera_matrix'] = np.array(data_file['/ovc/left/calib/intrinsics']).tolist()

        cam_to_cam['intrinsics']['camRect1'] = {}
        cam_to_cam['intrinsics']['camRect1']['camera_type'] = 'frame'
        cam_to_cam['intrinsics']['camRect1']['canera_location'] = 'left'
        cam_to_cam['intrinsics']['camRect1']['is_rectified'] = True
        cam_to_cam['intrinsics']['camRect1']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['camRect1']['resolution'] = [1280, 800]
        cam_to_cam['intrinsics']['camRect1']['camera_matrix'] = np.array([P1_gray[0,0], P1_gray[1,1], P1_gray[0,2], P1_gray[1,2]]).tolist()

        cam_to_cam['intrinsics']['cam2'] = {}
        cam_to_cam['intrinsics']['cam2']['camera_type'] = 'frame'
        cam_to_cam['intrinsics']['cam2']['canera_location'] = 'right'
        cam_to_cam['intrinsics']['cam2']['is_rectified'] = False
        cam_to_cam['intrinsics']['cam2']['T_cn_cnm1'] = (transform_inv(gray_right_to_event_left) @ gray_left_to_event_left).tolist()#cam1 -> cam2
        cam_to_cam['intrinsics']['cam2']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['cam2']['distortion_coeffs'] = D_gray_right.tolist()
        cam_to_cam['intrinsics']['cam2']['distortion_model'] = 'radtan'
        cam_to_cam['intrinsics']['cam2']['resolution'] = [1280, 800]
        cam_to_cam['intrinsics']['cam2']['camera_matrix'] = np.array(data_file['/ovc/right/calib/intrinsics']).tolist()

        cam_to_cam['intrinsics']['camRect2'] = {}
        cam_to_cam['intrinsics']['camRect2']['camera_type'] = 'frame'
        cam_to_cam['intrinsics']['camRect2']['canera_location'] = 'right'
        cam_to_cam['intrinsics']['camRect2']['is_rectified'] = True
        cam_to_cam['intrinsics']['camRect2']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['camRect2']['resolution'] = [1280, 800]
        cam_to_cam['intrinsics']['camRect2']['camera_matrix'] = np.array([P2_gray[0,0], P2_gray[1,1], P2_gray[0,2], P2_gray[1,2]]).tolist()


        cam_to_cam['intrinsics']['cam3'] = {}
        cam_to_cam['intrinsics']['cam3']['camera_type'] = 'event'
        cam_to_cam['intrinsics']['cam3']['canera_location'] = 'right'
        cam_to_cam['intrinsics']['cam3']['is_rectified'] = False
        cam_to_cam['intrinsics']['cam3']['T_cn_cnm1'] = (transform_inv(event_right_to_event_left) @ gray_right_to_event_left).tolist()#cam2 -> cam3
        cam_to_cam['intrinsics']['cam3']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['cam3']['distortion_coeffs'] = D_event_right.tolist()
        cam_to_cam['intrinsics']['cam3']['distortion_model'] = 'radtan'
        cam_to_cam['intrinsics']['cam3']['resolution'] = [1280, 720]
        cam_to_cam['intrinsics']['cam3']['camera_matrix'] = np.array(data_file['/prophesee/right/calib/intrinsics']).tolist()

        cam_to_cam['intrinsics']['camRect3'] = {}
        cam_to_cam['intrinsics']['camRect3']['camera_type'] = 'event'
        cam_to_cam['intrinsics']['camRect3']['canera_location'] = 'right'
        cam_to_cam['intrinsics']['camRect3']['is_rectified'] = True
        cam_to_cam['intrinsics']['camRect3']['camera_model'] = 'pinhole'
        cam_to_cam['intrinsics']['camRect3']['resolution'] = [1280, 720]
        cam_to_cam['intrinsics']['camRect3']['camera_matrix'] = np.array([P2_event[0,0], P2_event[1,1], P2_event[0,2], P2_event[1,2]]).tolist()


        cam_to_cam['extrinsics'] = {}
        cam_to_cam['extrinsics']['T_10'] = transform_inv(gray_left_to_event_left).tolist()
        cam_to_cam['extrinsics']['T_21'] = (transform_inv(gray_right_to_event_left) @ gray_left_to_event_left).tolist()
        cam_to_cam['extrinsics']['T_32'] = (transform_inv(event_right_to_event_left) @ gray_right_to_event_left).tolist()

        cam_to_cam['extrinsics']['R_rect0'] = R1_event.tolist()
        cam_to_cam['extrinsics']['R_rect1'] = R1_gray.tolist()
        cam_to_cam['extrinsics']['R_rect2'] = R2_gray.tolist()
        cam_to_cam['extrinsics']['R_rect3'] = R2_event.tolist()

        cam_to_cam['disparity_to_depth'] = {}
        cam_to_cam['disparity_to_depth']['cams_03'] = Q_event.tolist()
        cam_to_cam['disparity_to_depth']['cams_12'] = Q_gray.tolist()

        #Save cam_to_cam.yaml
        with open(os.path.join(output_path, basename, "calibration/cam_to_cam.yaml"), "w") as f:
            yaml.dump(cam_to_cam, f)

        cam_to_lidar = {}

        cam_to_lidar['T_lidar_cam0'] = transform_inv(np.array(data_file["/ouster/calib/T_to_prophesee_left"])).tolist()
        cam_to_lidar['T_lidar_camRect0'] = transform_inv(R1_event_4x4 @ np.array(data_file["/ouster/calib/T_to_prophesee_left"])).tolist()
        
        cam_to_lidar['T_lidar_cam1'] = transform_inv(np.array(transform_inv(gray_left_to_event_left) @ data_file["/ouster/calib/T_to_prophesee_left"])).tolist()
        cam_to_lidar['T_lidar_camRect1'] = transform_inv(R1_gray_4x4 @ transform_inv(gray_left_to_event_left) @ np.array(data_file["/ouster/calib/T_to_prophesee_left"])).tolist()
        
        cam_to_lidar['T_lidar_cam2'] = transform_inv(transform_inv(gray_right_to_event_left) @ np.array(data_file["/ouster/calib/T_to_prophesee_left"])).tolist()
        cam_to_lidar['T_lidar_camRect2'] = transform_inv(R2_gray_4x4 @ transform_inv(gray_right_to_event_left) @ np.array(data_file["/ouster/calib/T_to_prophesee_left"])).tolist()
        
        cam_to_lidar['T_lidar_cam3'] = transform_inv(np.array(transform_inv(event_right_to_event_left) @ data_file["/ouster/calib/T_to_prophesee_left"])).tolist()
        cam_to_lidar['T_lidar_camRect3'] = transform_inv(R2_event_4x4 @ transform_inv(event_right_to_event_left) @ np.array(data_file["/ouster/calib/T_to_prophesee_left"])).tolist()

        #Fake data
        cam_to_lidar['calibration_details'] = {}
        cam_to_lidar['calibration_details']['image_indices'] = [0,0]
        cam_to_lidar['calibration_details']['n_after'] = 0
        cam_to_lidar['calibration_details']['n_before'] = 0
        cam_to_lidar['calibration_details']['sequence'] = '1970-01-01-00-00-00'
        cam_to_lidar['calibration_details']['temporal_offset_us'] = 0

        #Save cam_to_lidar.yaml
        with open(os.path.join(output_path, basename, "calibration/cam_to_lidar.yaml"), "w") as f:
            yaml.dump(cam_to_lidar, f)

        pbar.set_description(f"Creating event rectification maps files...")

        #Create left and right rectification maps
        for cam, map in zip(['left', 'right'], [inv_leftmap_event, inv_rightmap_event]):
            rect_file = h5py.File(os.path.join(output_path, basename, f"events/{cam}/rectify_map.h5"), 'w')
            rect_file.create_dataset("rectify_map", data=map.astype(np.float32), compression='lzf')
            rect_file.close()

        #Save groundtruth disparity values.
        ts_gt = np.array(depth_file['ts'])
        n_imgs = len(depth_file['depth']['prophesee']['left'])
        timestamps_event = []

        disp_factor_dict = {}
        
        for i in range(n_imgs):
            pbar.set_description(f"Creating event groundtruth disparity files ({i+1}/{n_imgs}) ...")

            depth_i = depth_file['depth']['prophesee']['left'][i]
            depth_i[np.isinf(depth_i)] = 0
            depth_i = forward_remap(depth_i, inv_leftmap_event)

            timestamps_event.append(int(ts_gt[i]))

            #Convert depth to disparity and save with 6 zeros digits
            disp_i = depth_i.copy()
            disp_i[disp_i>0] = (focal_event*baseline_event) / disp_i[disp_i>0]

            quantization = np.floor(65535 / np.max(disp_i))
            if 'event' not in disp_factor_dict:
                disp_factor_dict['event'] = {}
            disp_factor_dict['event'][f'{i:06d}.png'] = quantization.item()

            cv2.imwrite(os.path.join(output_path, basename, f"disparity/event/{i:06d}.png"), np.clip(disp_i*quantization,0,65535).astype(np.uint16))

        with open(os.path.join(output_path, basename, "disparity/disp_factor.yaml"), "w") as f:
            yaml.dump(disp_factor_dict, f)

        np.savetxt(os.path.join(output_path, basename, "disparity/timestamps.txt"), np.array(timestamps_event).astype(np.uint64), fmt='%i')

        data_file.close()
        depth_file.close()

        if delete:
            os.remove(data_path)
            os.remove(gt_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="M3ED Converter into DSEC format"
    )

    parser.add_argument(
        "-i",
        "--input_folder",
        required=True,
        # nargs='+',
        default="datasets/M3ED/raw",
        help="""Input folder containing a list of folder and for each folder a p\
                air of *_data.h5 and *_depth_gt.h5"""
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        required=True,
        default="datasets/M3ED/processed",
        help="""Output folder where modified files will be moved from original source folder"""
    )
    parser.add_argument(
        "-d",
        "--delete",
        action='store_true',
        help="""Delete original files"""
    )
    # parser.add_argument(
    #     "--ignore_glob",
    #     action='store_true',
    #     help="""Ignore glob"""
    # )
    parser.add_argument(
        "--ignore_h5",
        action='store_true',
        help="""Ignore h5 exceptions"""
    )

    args = parser.parse_args()

    main(args)