# IGICP
This package includes the implementation of [1]. We propose a new point pair similarity method by combing the normal vector, the smallest eigenvalue of the spatial covariance matrix, and the KL divergence of local intensity values. In the pose optimization step, we use both the proposed point pair similarity and planarity as the weight.

We build our IGICP system based on fast_gicp and introduce several changes as introduced in [1]. We appreciate the efforts made by fast_gicp providers.

[1] Li He, Wen Li, Yisheng Guan, and Hong Zhang. **IGICP: Intensity and Geometry Enhanced LiDAR Odometry**. *IEEE Transactions on Intelligent Vehicles*, to appear.

hel@sustech.edu.cn

2112101119@mail2.gdut.edu.cn

Nov. 22, 2023


## 1. Install fast_gicp 
First, you need to install fast_gicp, please refer to https://github.com/SMRT-AIST/fast_gicp.

## 2. Copy IGICP files to fast_gicp folder
### Download these files and copy to your project folder
- Copy fast_igicp.cpp to fast_gicp/src/fast_gicp/gicp/.
- Copy fast_igicp.hpp to fast_gicp/include/fast_gicp/gicp/.
- Copy fast_igicp_impl.hpp to fast_gicp/include/fast_gicp/gicp/impl/. This file includes the main body of IGICP.
- Copy kitti_igicp.cpp to fast_gicp/src/.
- Copy the folder 'test' to src/fast_gicp/data/. The folder 'test' includes 0~50 scans of KITTI 07 for testing. If you want to load your data, please change the two paths in lines 194 and 195 of 'kitti_igicp.cpp', where 'SourceDataPath' is the path of your data and 'TrajSavePath' is the path where the results are saved.

### 3. Run IGICP
- Replace fast_gicp/CMakeLists.txt with the new CMakeLists.txt file attached.  
- Run the following command in your fast_gicp workspace.
```bash
catkin_make
source devel/setup.bash
rosrun fast_gicp igicp_kitti
```
