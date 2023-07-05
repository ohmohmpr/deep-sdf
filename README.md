How to run

pip3 uninstall kiss-icp -y && pip3 install --verbose .

kiss_icp_pipeline --visualize ../data/00/velodyne ../sequences/sequence00_01.npy
kiss_icp_pipeline --visualize ../data/04/velodyne ../sequences/sequence04_01.npy
kiss_icp_pipeline --visualize ../data/04/velodyne ../sequences/sequence04_02.npy
kiss_icp_pipeline --visualize ../data/07/velodyne ../sequences/sequence07_01.npy
kiss_icp_pipeline --visualize ../data/20/velodyne ../sequences/sequence20_01.npy
kiss_icp_pipeline --visualize ../data/20/velodyne ../sequences/sequence20_02.npy # highway 
kiss_icp_pipeline --visualize ../data/21/velodyne ../sequences/sequence21_01.npy # highway



#IPB
kiss_icp_pipeline --visualize ~/MSR-SEM1/DSP-SLAM/data/kitti/00/velodyne/ ../sequences/sequence00_01.npy
kiss_icp_pipeline --visualize ~/MSR-SEM1/DSP-SLAM/data/kitti/07/velodyne/ ../sequences/sequence07_01.npy
kiss_icp_pipeline --visualize ../data/04/velodyne ../sequences/sequence04_01.npy
kiss_icp_pipeline --visualize ../data/04/velodyne ../sequences/sequence04_02.npy
kiss_icp_pipeline --visualize ~/MSR-SEM1/DSP-SLAM/data_semantic_kitti/velodyne/dataset/sequences/20/velodyne/ ../sequences/sequence20_01.npy
kiss_icp_pipeline --visualize ~/MSR-SEM1/DSP-SLAM/data_semantic_kitti/velodyne/dataset/sequences/20/velodyne/ ../sequences/sequence20_02.npy
kiss_icp_pipeline --visualize ~/MSR-SEM1/DSP-SLAM/data_semantic_kitti/velodyne/dataset/sequences/21/velodyne/ ../sequences/sequence21_01.npy

