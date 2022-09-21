import open3d as o3d
import numpy as np




def main(pc_path):
    raw_points = np.fromfile(pc_path, dtype=np.float32, count=-1,).reshape([-1,4])[:,:3]
    print(raw_points.shape)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="kitti")
    vis.get_render_option().point_size = 1
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    pcd = o3d.open3d.geometry.PointCloud()
    pcd.points = o3d.open3d.utility.Vector3dVector(raw_points)
    vis.add_geometry(pcd)
    vis.run()















if __name__ == '__main__':
    pc_path = "../data/object/training/velodyne/000000.bin"
    main(pc_path)