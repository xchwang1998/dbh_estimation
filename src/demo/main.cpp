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
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    ReadPCD(config_setting.pcd_data, input_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tree_points(new pcl::PointCloud<pcl::PointXYZ>);
    clothSimulationFilter(input_cloud, ground_points, tree_points, config_setting);

    std::cout << "ground_points: " << ground_points->size() << std::endl;
    std::cout << "tree_points: " << tree_points->size() << std::endl;
    
    return 0;
}