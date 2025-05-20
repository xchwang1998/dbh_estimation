#include "math.h"

#ifndef _HLP_H_Included_
#define _HLP_H_Included_
#include "../include/utils/Hlp.h"
#endif

#ifndef _FEC_H_Included_
#define _FEC_H_Included_
#include "../include/utils/FEC.h"
#endif

int main(int argc, char **argv) 
{
    // check the num of parameters
    if (argc < 0) {
        std::cout << RED << "Error, at least 2 parameter" << RESET << std::endl;
        std::cout << "USAGE: ./DBH_Estimation [Target Station's File Name] [Source Station's File Name]" << std::endl;
        return 1;
    }

    std::string data_path = PROJECT_PATH;
    std::cout << BOLDGREEN << "----------------DATA PROCESSING----------------" << RESET << std::endl;
    
    std::cout << BOLDGREEN << data_path+"/config/para.yaml" << RESET << std::endl;
    // read the setting parameters
    ConfigSetting config_setting;
    ReadParas(data_path+"/config/para.yaml", config_setting);
    
    // read pcd files
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    ReadPCD(config_setting.pcd_data, input_cloud);
    
    // downsample input point cloud
    down_sampling_voxel(*input_cloud, 0.05);

    // sor filter
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    sor_filter_noise(input_cloud, input_cloud_filtered);
    std::cout << "input_cloud SOR filtered: " << input_cloud_filtered->size() << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tree_points(new pcl::PointCloud<pcl::PointXYZ>);
    clothSimulationFilter(input_cloud_filtered, ground_points, tree_points, config_setting);

    std::cout << "ground_points: " << ground_points->size() << std::endl;
    std::cout << "tree_points: " << tree_points->size() << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr tree_points_croped(new pcl::PointCloud<pcl::PointXYZ>);
    stem_above_ground(tree_points, ground_points, tree_points_croped);
    pcl::io::savePCDFileBinary("/media/xiaochen/xch_disk/TreeScope_dataset/icra_challenge/development_phase/tree_croped.pcd", *tree_points_croped);

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> obj_cluster_points;
    // fec_cluster(tree_points, obj_cluster_points, config_setting);    
    // pcl_cluster(tree_points, obj_cluster_points, config_setting); 
    pcl_cluster(tree_points_croped, obj_cluster_points, config_setting);
    std::cout << "obj_cluster_points: " << obj_cluster_points.size() << std::endl;

    std::vector<Cluster> obj_clusters;
    std::vector<Cluster> discarded_clusters;
    cluster_attributes(obj_clusters, discarded_clusters, obj_cluster_points, config_setting);
    std::cout << "obj_clusters size: " << obj_clusters.size() << std::endl;
    std::cout << "discarded_clusters size: " << discarded_clusters.size() << std::endl;
    
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr obj_points(new pcl::PointCloud<pcl::PointXYZRGBL>());
    clusterTopoints(obj_clusters, obj_points);
    
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr discard_points(new pcl::PointCloud<pcl::PointXYZRGBL>());
    clusterTopoints(discarded_clusters, discard_points);
    
    pcl::io::savePCDFileBinary("/media/xiaochen/xch_disk/TreeScope_dataset/icra_challenge/development_phase/ground.pcd", *ground_points);
    pcl::io::savePCDFileBinary("/media/xiaochen/xch_disk/TreeScope_dataset/icra_challenge/development_phase/tree.pcd", *tree_points);
    pcl::io::savePCDFileBinary("/media/xiaochen/xch_disk/TreeScope_dataset/icra_challenge/development_phase/obj.pcd", *obj_points);
    pcl::io::savePCDFileBinary("/media/xiaochen/xch_disk/TreeScope_dataset/icra_challenge/development_phase/discard.pcd", *discard_points);
    return 0;
}