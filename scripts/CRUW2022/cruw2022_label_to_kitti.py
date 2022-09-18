import os
import shutil
from turtle import pos
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import json
from math import ceil
import torch


cls2id = {
    'pedestrian': 0,
    'car': 1,
    'van': 2,
    'bus': 3,
    'truck': 4
}

# 3D util


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


def conver_label(Tc_l, rot, t, scale):
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

# return bool indicating in fov and left, right,  width, height of projected 2d bbox


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


if __name__ == '__main__':

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

    save_root_path = os.path.join(root_path, split_name)

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
        if seq_name != '2021_1120_1619':
            continue

        labels_in_seq = []
        print(f'Now processing {seq_name}')
        img_names = sorted(os.listdir(os.path.join(
            cruw_root, seq_name, image_realtive_path)))
        # train_last_index = int(ceil(train_ratio * len(img_names))) - \
        #     1 if mode == 'split_test' else len(img_names)
        train_last_index = 1260

        trk_id_seq_dict = {}
        # ann_save_path = os.path.join(save_root_path, seq_name + '_train.txt')
        ann_save_path_test = os.path.join(
            save_root_path, seq_name + '.txt')
        ann_str_list = []
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
                if obj['obj_type'].lower() in cls2id:
                    cls_id = cls2id[obj['obj_type'].lower()]
                else:
                    cls_id = -1

                psr = obj['psr']
                euler, position, scale, quat = conver_label(
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
                

                if cls_id >= 0:
                    if frame_index <= train_last_index:
                        ann_str = "%d %d %s -1 -1 %.2f %.2f %.2f %.2f %.2f %.4f %.4f %.4f %d\n" % (
                            frame_index, tid_cur, obj['obj_type'], l, t, l+w, t+h, 1, position[0], position[1], position[2], cls_id)
                        ann_str_list.append(ann_str)
                    else:
                        ann_str = "%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%d\n" % (
                            frame_index - (train_last_index + 1), tid_cur, l, t, w, h, 1, position[0], position[1], position[2], cls_id)
                        ann_str_list_test.append(ann_str)

        print('last track ID for', seq_name, ':', tid_max)

        # with open(ann_save_path, 'w') as f:
        #     for s in ann_str_list:
        #         f.write(s)
        with open(ann_save_path_test, 'w') as f:
            for s in ann_str_list_test:
                f.write(s)
