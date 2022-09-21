import open3d as o3d
import numpy as np
import os
import argparse


path = "../data/object/training"
parse = argparse.ArgumentParser()
parse.add_argument('--index', type=str, default=None)
args = parse.parse_args()

def rot_y(rotation_y):
    cos = np.cos(rotation_y)
    sin = np.sin(rotation_y)
    R = np.array([[cos,0,sin],[0,1,0],[-sin,0,cos]])
    return R

class Object3d:
    def __init__(self, content):
        super(Object3d, self).__init__()
        lines = content.split()

        lines = list(filter(lambda x: len(x), lines))
        self.name, self.truncated, self.occluded, self.alpha = lines[0], float(lines[1]), float(lines[2]), float(lines[3])
        self.dimensions = [lines[8], lines[9], lines[10]]
        self.dimensions = np.array([float(x) for x in self.dimensions])
        self.location = [lines[11], lines[12], lines[13]]
        self.location = np.array([float(x) for x in self.location])
        self.rotation_y = float(lines[14])

class Calib:
    def __init__(self, dict_calib):
        super(Calib, self).__init__()
        self.P0 = dict_calib['P0'].reshape(3, 4)
        self.P1 = dict_calib['P1'].reshape(3, 4)
        self.P2 = dict_calib['P2'].reshape(3, 4)
        self.P3 = dict_calib['P3'].reshape(3, 4)
        self.R0_rect = dict_calib['R0_rect'].reshape(3, 3)
        self.Tr_velo_to_cam = dict_calib['Tr_velo_to_cam'].reshape(3, 4)
        self.Tr_imu_to_velo = dict_calib['Tr_imu_to_velo'].reshape(3, 4)
def get_calib(index):
    calib_path = os.path.join(path, "calib", "{:06d}.txt".format(index))
    with open(calib_path) as f:
        lines = f.readlines()
    lines = list(filter(lambda x: len(x) and x!='\n', lines))
    dict_calib = {}
    for line in lines:
        key,value = line.split(":")
        dict_calib[key] = np.array([float(x) for x in value.split()])
    return Calib(dict_calib)

def get_objects(vis, index):
    calib1 = get_calib(index)
    box_path = os.path.join(path, "label_2", "{:06d}.txt".format(index))
    with open(box_path) as f:
        lines = f.readlines()
    lines = list(filter(lambda x: len(x) > 0 and x !='\n', lines))
    obj = [Object3d(x) for x in lines]
    for obj_index in range(len(obj)):
        if obj[obj_index].name == "Car" or obj[obj_index].name == "Pedestrian" or obj[obj_index].name == "Cyclist":
            R = rot_y(obj[obj_index].rotation_y)
            h, w, l = obj[obj_index].dimensions[0], obj[obj_index].dimensions[1], obj[obj_index].dimensions[2]
            x = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
            #y = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
            y = [0, 0, 0, 0, -h, -h, -h, -h]
            z = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
            corner_3d = np.vstack([x,y,z])
            corner_3d = np.dot(R, corner_3d)

            corner_3d[0, :] += obj[obj_index].location[0]
            corner_3d[1, :] += obj[obj_index].location[1]
            corner_3d[2, :] += obj[obj_index].location[2]
            corner_3d = np.vstack((corner_3d, np.zeros((1, corner_3d.shape[-1]))))
            corner_3d[-1][-1] = 1

            inv_Tr = np.zeros_like(calib1.Tr_velo_to_cam)
            inv_Tr[0:3, 0:3] = np.transpose(calib1.Tr_velo_to_cam[0:3, 0:3])
            inv_Tr[0:3, 3] = np.dot(-np.transpose(calib1.Tr_velo_to_cam[0:3,0:3]), calib1.Tr_velo_to_cam[0:3, 3])

            Y = np.dot(inv_Tr, corner_3d)

            draw_box(vis,Y)

def draw_box(vis,Y):
    points_d3box = Y
    points_box = np.transpose(points_d3box)
    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6],
                          [6, 7], [0, 4], [1, 5], [2, 6], [3, 7], [0, 5], [1, 4], [3, 6], [2, 7]])
    print(lines_box.shape)
    colors = np.array([[0, 1, 0] for j in range(len(lines_box))])
    line_set = o3d.geometry.LineSet()
    line_set.lines = o3d.utility.Vector2iVector(lines_box)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.points = o3d.utility.Vector3dVector(points_box)


    #vis.update_geometry(line_set)
    vis.add_geometry(line_set)
    #vis.update_renderer()





def main(index):

    pc_path = os.path.join(path, "velodyne", "{:06d}.bin".format(index))
    raw_points = np.fromfile(pc_path, dtype=np.float32, count=-1, ).reshape([-1, 4])[:, :3]
    pcd = o3d.open3d.geometry.PointCloud()
    pcd.points = o3d.open3d.utility.Vector3dVector(raw_points)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="kitti")
    vis.get_render_option().point_size = 1
    vis.get_render_option().line_width = 5.0
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    vis.add_geometry(pcd)

    get_objects(vis,index)
    vis.run()








if __name__ == '__main__':
    index = int(args.index)
    main(index)