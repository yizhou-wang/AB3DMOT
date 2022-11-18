import json
import os
import shutil
import numpy as np
import torch

from tqdm import tqdm
from math import ceil
from turtle import pos
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
from tridet.structures.boxes3d import GenericBoxes3D
from tridet.structures.pose import Pose

CLASS_DICT = {
    'Pedestrian': 0,
    'Car': 1
}

# cls2id = {
#     'pedestrian': 0,
#     'car': 1,
#     'van': 2,
#     'bus': 3,
#     'truck': 4
# }


def radx_to_matrix(rotx, N):
    device = rotx.device

    cos, sin = rotx.cos(), rotx.sin()

    i_temp = torch.tensor([[1, 0, 0],
                           [0, 1, -1],
                           [0, 1, 1]]).to(dtype=torch.float32,
                                          device=device)
    rx = i_temp.repeat(N, 1).view(N, -1, 3)

    rx[:, 1, 1] *= cos
    rx[:, 1, 2] *= sin
    rx[:, 2, 1] *= sin
    rx[:, 2, 2] *= cos

    return rx


def rady_to_matrix(rotys, N):
    device = rotys.device

    cos, sin = rotys.cos(), rotys.sin()

    i_temp = torch.tensor([[1, 0, 1],
                           [0, 1, 0],
                           [-1, 0, 1]]).to(dtype=torch.float32,
                                           device=device)
    ry = i_temp.repeat(N, 1).view(N, -1, 3)

    ry[:, 0, 0] *= cos
    ry[:, 0, 2] *= sin
    ry[:, 2, 0] *= sin
    ry[:, 2, 2] *= cos

    return ry


def radz_to_matrix(rotz, N):
    device = rotz.device

    cos, sin = rotz.cos(), rotz.sin()

    i_temp = torch.tensor([[1, -1, 0],
                           [1, 1, 0],
                           [0, 0, 1]]).to(dtype=torch.float32,
                                          device=device)
    rz = i_temp.repeat(N, 1).view(N, -1, 3)

    rz[:, 0, 0] *= cos
    rz[:, 0, 1] *= sin
    rz[:, 1, 0] *= sin
    rz[:, 1, 1] *= cos

    return rz


def encode_box3d(eulers, dims, locs):
    '''
    construct 3d bounding box for each object.
    Args:
        rotys: rotation in shape N
        dims: dimensions of objects
        locs: locations of objects

    Returns:
    '''
    device = eulers.device
    N = eulers.shape[0]
    rx = radx_to_matrix(eulers[:, 0], N)  # (N, 3, 3)
    ry = rady_to_matrix(eulers[:, 1], N)  # (N, 3, 3)
    rz = radz_to_matrix(eulers[:, 2], N)  # (N, 3, 3)

    # [[eight w], [eight h], [eight l]]  # (N*3, 8)
    dims = dims.view(-1, 1).repeat(1, 8)
    dims[::3, :4], dims[2::3, :4] = 0.5 * dims[::3, :4], 0.5 * dims[2::3, :4]
    dims[::3, 4:], dims[2::3, 4:] = -0.5 * dims[::3, 4:], -0.5 * dims[2::3, 4:]
    dims[1::3, :4], dims[1::3, 4:] = 0.5 * \
        dims[1::3, 4:], -0.5 * dims[1::3, 4:]
    index = torch.tensor([[4, 0, 1, 2, 3, 5, 6, 7],
                          [4, 5, 0, 1, 6, 7, 2, 3],
                          [4, 5, 6, 0, 1, 2, 3, 7]]).repeat(N, 1).to(device=device)
    box_3d_object = torch.gather(dims, 1, index)
    box_3d = box_3d_object.view(N, 3, -1)
    box_3d = torch.matmul(torch.matmul(
        rz, torch.matmul(rx, ry)), box_3d_object.view(N, 3, -1))
    box_3d += locs.unsqueeze(-1)

    return box_3d


def convert_3d_box_to_kitti(box):
    """Convert a single 3D bounding box (GenericBoxes3D) to KITTI convention. i.e. for evaluation. We
    assume the box is in the reference frame of camera_2 (annotations are given in this frame).

    Usage:
        >>> box_camera_2 = pose_02.inverse() * pose_0V * box_velodyne
        >>> kitti_bbox_params = convert_3d_box_to_kitti(box_camera_2)

    Parameters
    ----------
    box: GenericBoxes3D
        Box in camera frame (X-right, Y-down, Z-forward)

    Returns
    -------
    W, L, H, x, y, z, rot_y, alpha: float
        KITTI format bounding box parameters.
    """
    assert len(box) == 1

    quat = Quaternion(*box.quat.cpu().tolist()[0])
    tvec = box.tvec.cpu().numpy()[0]
    sizes = box.size.cpu().numpy()[0]

    inversion = Quaternion(axis=[1, 0, 0], radians=np.pi / 2).inverse
    quat = inversion * quat

    # Construct final pose in KITTI frame (use negative of angle if about positive z)
    if quat.axis[2] > 0:
        kitti_pose = Pose(wxyz=Quaternion(
            axis=[0, 1, 0], radians=-quat.angle), tvec=tvec)
        rot_y = -quat.angle
    else:
        kitti_pose = Pose(wxyz=Quaternion(
            axis=[0, 1, 0], radians=quat.angle), tvec=tvec)
        rot_y = quat.angle

    # Construct unit vector pointing in z direction (i.e. [0, 0, 1] direction)
    # The transform this unit vector by pose of car, and drop y component, thus keeping heading direction in BEV (x-z grid)
    v_ = np.float64([[0, 0, 1], [0, 0, 0]])
    v_ = (kitti_pose * v_)[:, ::2]

    # Getting positive theta angle (we define theta as the positive angle between
    # a ray from the origin through the base of the transformed unit vector and the z-axis
    theta = np.arctan2(abs(v_[1, 0]), abs(v_[1, 1]))

    # Depending on whether the base of the transformed unit vector is in the first or
    # second quadrant we add or subtract `theta` from `rot_y` to get alpha, respectively
    alpha = rot_y + theta if v_[1, 0] < 0 else rot_y - theta
    # Bound from [-pi, pi]
    if alpha > np.pi:
        alpha -= 2.0 * np.pi
    elif alpha < -np.pi:
        alpha += 2.0 * np.pi
    alpha = np.around(alpha, decimals=2)  # KITTI precision

    # W, L, H, x, y, z, rot-y, alpha
    return sizes[0], sizes[1], sizes[2], tvec[0], tvec[1], tvec[2], rot_y, alpha


def quaternion_upper_hemispher(q):
    """
    The quaternion q and −q represent the same rotation be-
    cause a rotation of θ in the direction v is equivalent to a
    rotation of 2π − θ in the direction −v. One way to force
    uniqueness of rotations is to require staying in the “upper
    half” of S 3 . For example, require that a ≥ 0, as long as
    the boundary case of a = 0 is handled properly because of
    antipodal points at the equator of S 3 . If a = 0, then require
    that b ≥ 0. However, if a = b = 0, then require that c ≥ 0
    because points such as (0,0,−1,0) and (0,0,1,0) are the
    same rotation. Finally, if a = b = c = 0, then only d = 1 is
    allowed.
    :param q:
    :return:
    """
    a, b, c, d = q
    if a < 0:
        q = -q
    if a == 0:
        if b < 0:
            q = -q
        if b == 0:
            if c < 0:
                q = -q
            if c == 0:
                q[3] = 1

    return q


def convert_label(Tc_l, rot, t, scale):
    Tl_o = np.identity(4)
    Tl_o[:3, :3] = Rotation.from_euler(
        'zyx', [rot['z'], rot['y'], rot['x']]).as_matrix()
    Tl_o[:3, 3] = np.array([t['x'], t['y'], t['z']])
    Tc_o = np.matmul(Tc_l, Tl_o)
    R = Tc_o[:3, :3]
    T = Tc_o[:3, 3]
    r = Rotation.from_matrix(R)
    yaw, pitch, roll = r.as_euler('yxz')
    tx, ty, tz = T
    quat = quaternion_upper_hemispher(r.as_quat()).tolist()

    return [pitch, yaw, roll], [tx, ty, tz], [scale['x'], scale['y'], scale['z']], quat


def in_fov(euler, position, scale, intrinsics, img_w=1440., img_h=1080.):
    box3d = encode_box3d(torch.tensor(euler).view(-1, 3), torch.tensor(
        scale).view(-1, 3), torch.tensor(position).view(-1, 3)).numpy()[0]
    box3d_homo = np.ones((4, 8), dtype=box3d.dtype)
    box3d_homo[:3, :] = box3d
    img_cor_points = np.dot(intrinsics, box3d_homo)
    img_cor_points = img_cor_points.T
    img_cor_points[:, 0] /= img_cor_points[:, 2]
    img_cor_points[:, 1] /= img_cor_points[:, 2]
    l, t, r, b = img_cor_points[:, 0].min(), img_cor_points[:, 1].min(
    ), img_cor_points[:, 0].max(), img_cor_points[:, 1].max()
    return (img_cor_points.min() >= 0.) and (img_cor_points[:, 0].max() < img_w) and (img_cor_points[:, 1].max() < img_h), l.item(), t.item(), (r - l).item(), (b - t).item()


def to_kitti_format(file_path):
    save_name = os.path.basename(file_path)
    dir = os.path.split(file_path)[0]
    save_path = os.path.join(dir, f'{save_name}_kitti.json')

    with open(file_path, 'r') as in_file:
        detections = json.load(in_file)
    for seq_name, frames in detections.items():
        print(f'Now processing: {seq_name}')
        for frame_name, objs in tqdm(frames.items()):
            new_objs = []
            for obj in objs:
                new_obj = obj.copy()
                box3d = GenericBoxes3D.from_vectors([obj['bbox3d']])
                W, L, H, x, y, z, rot_y, alpha = convert_3d_box_to_kitti(box3d)
                bbox3d = [float(W), float(L), float(H), float(
                    x), float(y), float(z), float(rot_y), float(alpha)]
                new_obj['bbox3d'] = bbox3d
                new_objs.append(new_obj)
            detections[seq_name][frame_name] = new_objs

    with open(save_path, 'w') as out_file:
        json.dump(detections, out_file, indent=2)


def to_kitti_det_format(file_path, out_dir, cat, conf_thres):
    save_name = os.path.basename(file_path)
    dir = os.path.split(file_path)[0]
    save_path = os.path.join(dir, f'{save_name}_kitti.json')

    with open(file_path, 'r') as in_file:
        detections = json.load(in_file)
    for seq_name, frames in detections.items():
        print(f'Now processing: {seq_name}')
        out_txt_path = os.path.join(out_dir, seq_name + '.txt')
        with open(out_txt_path, 'w') as f:
            pass
        for frame_name, objs in tqdm(frames.items()):
            for obj in objs:
                frame_id = int(frame_name)
                if obj['category'] != cat:
                    continue
                class_id = CLASS_DICT[obj['category']]
                bbox2d_x1 = obj['bbox'][0]
                bbox2d_y1 = obj['bbox'][1]
                bbox2d_x2 = obj['bbox'][0] + obj['bbox'][2]
                bbox2d_y2 = obj['bbox'][1] + obj['bbox'][3]
                score = obj['score_3d']
                if score < conf_thres:
                    continue

                box3d = GenericBoxes3D.from_vectors([obj['bbox3d']])
                bbox3d_w, bbox3d_l, bbox3d_h, bbox3d_x, bbox3d_y, bbox3d_z, bbox3d_roty, bbox3d_alpha = convert_3d_box_to_kitti(
                    box3d)

                obj_str = "%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (
                    frame_id - train_last_index, class_id, bbox2d_x1, bbox2d_y1, bbox2d_x2, bbox2d_y2, score,
                    bbox3d_h, bbox3d_w, bbox3d_l, bbox3d_x, bbox3d_y, bbox3d_z,
                    bbox3d_roty, bbox3d_alpha)
                with open(out_txt_path, 'a+') as f:
                    f.write(obj_str)


if __name__ == '__main__':

    convert_type = 'result'
    train_last_index = 1260

    if convert_type == 'result':

        ##################################################
        # convert results

        detection_json_root = '/home/yzwang/Research/Tracking/AB3DMOT/data/CRUW2022/detection_json/'
        # detection_name = 'dd3d/dla34/day_night_enh'
        detection_name = 'smoke_rot_y'
        conf_thres = 0.2 # change here
        # detection_json_name = 'bbox3d_predictions_3dnms_0.0.json'
        detection_json_name = 'bbox3d_predictions_2dnms_0.75.json'
        detection_json_path = os.path.join(
            detection_json_root, detection_name, detection_json_name)

        detection_txt_root = '/home/yzwang/Research/Tracking/AB3DMOT/data/CRUW2022/detection'
        detection_txt_dir = os.path.join(detection_txt_root, detection_name)

        for cat in CLASS_DICT.keys():
            detection_txt_dir_cat = detection_txt_dir + '_' + cat + '_test'
            os.makedirs(detection_txt_dir_cat, exist_ok=True)
            to_kitti_det_format(detection_json_path,
                                detection_txt_dir_cat, cat, conf_thres)

    if convert_type == 'label':

        ##################################################
        # convert label
        # to MOT format: <frame> <track_id> <type> <truncated> <occluded> <alpha> <bb_left> <bb_top> <bb_right> <bb_bottom> <height> <weight> <length> <x> <y> <z> <rotation_y> <score>

        split_name = 'test'
        root_path = '/home/yzwang/Research/Tracking/AB3DMOT/scripts/CRUW2022/label'
        mode = 'split_test'  # split_test or others
        # split_test scheme: first 70% train, last 30% test
        train_ratio = 0.7

        cruw_root = '/mnt/nas_cruw/CRUW_2022'
        cruw_label_root = '/mnt/disk2/CRUW_2022/CRUW_2022_label'
        cruw_calib_root = '/mnt/nas_cruw/cruw_left_calibs'
        calib_realtive_path = 'calib/camera/left.json'
        image_realtive_path = 'camera/left'

        save_root_path = os.path.join(root_path)

        print(f'Start to preparing files in {save_root_path}')
        # if os.path.exists(save_root_path):
        #     shutil.rmtree(save_root_path)

        os.makedirs(save_root_path, exist_ok=True)

        seqs = ['2021_1120_1616', '2021_1120_1618', '2021_1120_1619', '2021_1120_1632', '2021_1120_1634', '2022_0203_1428', '2022_0203_1439', '2022_0203_1441',
                '2022_0203_1443', '2022_0203_1445', '2022_0203_1512', '2022_0217_1232', '2022_0217_1251', '2022_0217_1307', '2022_0217_1322']
        # type_to_train = ['Car', 'Pedestrian']

        labels_train, labels_test = [], []
        # filter out instances larger than distance (meter)
        distance_threshold = 50

        tid_max = 0

        for seq_name in seqs:
            labels_in_seq = []
            print(f'Now processing {seq_name}')
            img_names = sorted(os.listdir(os.path.join(
                cruw_root, seq_name, image_realtive_path)))
            # train_last_index = int(ceil(train_ratio * len(img_names))) - \
            #     1 if mode == 'split_test' else len(img_names)

            trk_id_seq_dict = {}
            # ann_save_path = os.path.join(save_root_path, seq_name + '_train.txt')
            ann_save_path_test = os.path.join(
                save_root_path, seq_name + '.txt')
            ann_str_list_test = []

            for frame_index, img_name in enumerate(tqdm(img_names)):
                # if frame_index != 1260 + 12:
                #     continue
                frame_name = img_name.split('.')[0]
                label_path = os.path.join(
                    cruw_label_root, seq_name, 'label', f'{frame_name}.json')
                if not os.path.exists(label_path):
                    print(f'{frame_name} has no label')
                    continue

                with open(os.path.join(cruw_calib_root, seq_name, calib_realtive_path), 'r') as calib_file:
                    cam_calib = json.load(calib_file)
                with open(label_path, 'r') as label_file:
                    label = json.load(label_file)
                P2 = cam_calib['intrinsic']
                P1 = cam_calib['extrinsic']
                P2 = np.array(P2, dtype=np.float32).reshape(3, 4)
                P1 = np.array(P1, dtype=np.float32).reshape(4, 4)

                objs = []
                for obj in label:
                    # if not obj['obj_type'] in type_to_train:
                    #     continue
                    try:
                        obj['obj_type'].lower()
                    except:
                        continue
                    if obj['obj_type'] not in CLASS_DICT:
                        continue

                    psr = obj['psr']
                    euler, position, scale, quat = convert_label(
                        P1, psr['rotation'], psr['position'], psr['scale'])
                    is_in_fov, l, t, w, h = in_fov(euler, position, scale, P2)
                    if (np.linalg.norm(np.array(position)) > distance_threshold) or not is_in_fov:
                        continue
                    # if (np.linalg.norm(np.array(position)) > distance_threshold):
                    #     continue

                    if obj['obj_id'] not in trk_id_seq_dict:
                        tid_max += 1
                        tid_cur = tid_max
                        trk_id_seq_dict[obj['obj_id']] = tid_cur
                    else:
                        tid_cur = trk_id_seq_dict[obj['obj_id']]

                    if frame_index >= train_last_index:
                        box3d_vec = [quat[3], *quat[0:3], *
                                     position, scale[1], scale[0], scale[2]]
                        box3d = GenericBoxes3D.from_vectors([box3d_vec])
                        bbox3d_w, bbox3d_l, bbox3d_h, bbox3d_x, bbox3d_y, bbox3d_z, bbox3d_roty, bbox3d_alpha = convert_3d_box_to_kitti(
                            box3d)

                        # obj_str = "%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (
                        #     frame_id, class_id, bbox2d_x1, bbox2d_y1, bbox2d_x2, bbox2d_y2, score,
                        #     bbox3d_h, bbox3d_w, bbox3d_l, bbox3d_x, bbox3d_y, bbox3d_z,
                        #     bbox3d_roty, bbox3d_alpha)
                        # with open(out_txt_path, 'a+') as f:
                        #     f.write(obj_str)

                        ann_str = "%d %d %s -1 -1 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % (
                            frame_index -
                            train_last_index, tid_cur, obj['obj_type'], bbox3d_alpha, l, t, l+w, t+h,
                            bbox3d_h, bbox3d_w, bbox3d_l, bbox3d_x, bbox3d_y, bbox3d_z, bbox3d_roty)
                        ann_str_list_test.append(ann_str)

            print('last track ID for', seq_name, ':', tid_max)

            with open(ann_save_path_test, 'w') as f:
                for s in ann_str_list_test:
                    f.write(s)
