#include "stdio.h"
#include <iostream>
#include<fstream>
#include <stack>
#include <unordered_set>
#include <tuple>

#include <ctime>
#include <string>
#include <cstdlib>
#include <chrono>

#include "liblas/liblas.hpp"
#include "liblas/point.hpp"
#include "boost/regex.hpp"

// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>

// Eigen
#include <Eigen/Dense>

// CSF
#include "CSF.h"

// OpenCV
#include "opencv2/core.hpp"

#ifndef _DST_H_Included_
#define _DST_H_Included_
#include "../include/dst/DST.h"
#endif

#ifndef _FEC_H_Included_
#define _FEC_H_Included_
#include "../include/utils/FEC.h"
#endif

#define INSIGNIFICANCE -99999
#define INFINITE		99999
#define INFINITESIMAL 0.00000001


#define MAX(A,B) ((A) >= (B) ? (A) : (B))
#define MIN(A,B) ((A) <= (B) ? (A) : (B))


// Associated functions
bool New_Int(int **pnArray, long lEleNum);
bool New_Char(char **pcArray, long lEleNum);
bool New_Long(long **plArray, long lEleNum);
bool New_Double(double **pdArray, long lEleNum);
bool New_Bool(bool **pbArray, long lEleNum);

bool Del_Int(int **pnArray);
bool Del_Char(char **pcArray);
bool Del_Long(long **plArray);
bool Del_Double(double **pdArray);
bool Del_Bool(bool **pbArray);

void SetVal_Int(int* pnArray, long lEleNum, int nVal);
void SetVal_Long(long *plArray, long lEleNum, long lVal);
void SetVal_Double(double *pdArray, long lEleNum, double dVal);
void SetVal_Bool(bool *pbArray, long lEleNum, bool bVal);

bool Val_Equ(const int bVal_1, const int bVal_2);
bool Val_Equ(const double bVal_1, const double bVal_2);
bool Val_Equ(const long lVal_1, const long lVal_2);
bool Val_Equ(const char& cVal_1, const char& cVal_2);
// int round(double x);

long ArrayMax(const double *pdIn, const long lEleNum);
long ArrayMax(const long *plIn, const long lEleNum);
long ArrayMax(const int *pnIn, const long lEle_Num);

// read the parameters from yaml file 
void ReadParas(const std::string& file_path, ConfigSetting &config_setting);

void ReadPCD(const std::string& file_path, pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_data);

// spereate the ground points, selected the points by index
void addPointCloud(const std::vector<int>& index_vec, 
                   const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered);

// CSF, set the parameters, and get the indices of ground and object
void clothSimulationFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
						   pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_ground,
						   pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_obj,
                           ConfigSetting &config_setting);


void sor_filter_noise(pcl::PointCloud<pcl::PointXYZ>::Ptr &source, 
                        pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered);


// FEC cluster, get the cluster points
void fec_cluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                 std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cluster_points_,
                 ConfigSetting &config_setting);

void pcl_cluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                 std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cluster_points_,
                 ConfigSetting &config_setting);

// calculate the attributes of each cluster, then select the trunk
void cluster_attributes(std::vector<Cluster> &clusters,
                        std::vector<Cluster> &disgard_clusters,
                        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cluster_points_,
                        ConfigSetting &config_setting);

// split the line into string
std::vector<std::string> split(std::string str,std::string s);

void matrix_to_pair(Eigen::Matrix4f &trans_matrix,
                    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &trans_pair);
void pair_to_matrix(Eigen::Matrix4f &trans_matrix,
                    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &trans_pair);

void point_to_vector(pcl::PointCloud<pcl::PointXYZ>::Ptr &pclPoints, 
                        std::vector<Eigen::Vector3d> &vecPoints);

void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZ> &pl_feat,
                         double voxel_size);


//============the following are UBUNTU/LINUX ONLY terminal color codes.==========
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */
//============the following are UBUNTU/LINUX ONLY terminal color codes.==========

class TicToc
{
public:
    TicToc()
    {
        tic();
    }

    TicToc( bool _disp )
    {
        disp_ = _disp;
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    void toc( std::string _about_task )
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        double elapsed_ms = elapsed_seconds.count() * 1000;

        if( disp_ )
        {
          std::cout.precision(3); // 10 for sec, 3 for ms 
          std::cout << _about_task << ": " << elapsed_ms << " msec." << std::endl;
        }
    }

private:  
    std::chrono::time_point<std::chrono::system_clock> start, end;
    bool disp_ = false;
};
