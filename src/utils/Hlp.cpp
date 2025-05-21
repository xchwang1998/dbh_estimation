#include "math.h"
#include "float.h"
#include <unordered_map>

#ifndef _HLP_H_Included_
#define _HLP_H_Included_
#include "../include/utils/Hlp.h"
#endif

// Associated functions
// int round(double x)
// {
//    return int(x > 0.0 ? x + 0.5 : x - 0.5);
// }

// 分配数组内存
bool New_Double(double** pdArray, long lEleNum)
{
	*pdArray = new double[lEleNum];
	if( *pdArray )	{
		return true;
	}
	else	{
		return false;
	}
}
bool New_Int(int** pnArray, long lEleNum)
{
	*pnArray = new int[lEleNum];
	if( *pnArray )	{
		return true;
	}
	else	{
		return false;
	}
}
bool New_Long(long** plArray, long lEleNum)
{
	*plArray = new long[lEleNum];
	if( *plArray )	{
		return true;
	}
	else	{
		return false;
	}
}
bool New_Char(char** pcArray, long lEleNum)
{
	*pcArray = new char[lEleNum];
	if( *pcArray )	{
		return true;
	}
	else	{
		return false;
	}
}
bool New_Bool(bool** pbArray, long lEleNum)
{
	*pbArray = new bool[lEleNum];
	if( *pbArray )	{
		return true;
	}
	else	{
		return false;
	}
}

// 删除内存空间
bool Del_Double(double** pdArray)
{
	if ( *pdArray )
	{
		delete[] *pdArray;
		*pdArray = NULL;
		return true;
	}
	else
	{
		return false;
	}
}
bool Del_Int(int** pnArray)
{
	if ( *pnArray )
	{
		delete[] *pnArray;
		*pnArray = NULL;
		return true;
	}
	else
	{
		return false;
	}
}
bool Del_Long(long** plArray)
{
	if ( *plArray )
	{
		delete[] *plArray;
		*plArray = NULL;
		return true;
	}
	else
	{
		return false;
	}
}
bool Del_Char(char** pcArray)
{
	if ( *pcArray )
	{
		delete[] *pcArray;
		*pcArray = NULL;
		return true;
	}
	else
	{
		return false;
	}
}
bool Del_Bool(bool** pbArray)
{
	if ( *pbArray )
	{
		delete[] *pbArray;
		*pbArray = NULL;
		return true;
	}
	else
	{
		return false;
	}
}

// 统一设置数组的值
void SetVal_Int(int* pnArray, long lEleNum, int nVal)
{
	for(long i = 0; i < lEleNum; i++)
		*(pnArray + i) = nVal; 	
}
void SetVal_Long(long *plArray, long lEleNum, long lVal)
{
	for(long i = 0; i < lEleNum; i++)
		*(plArray + i) = lVal; 
}
void SetVal_Double(double *pdArray, long lEleNum, double dVal)
{
	for(long i = 0; i < lEleNum; i++)
		*(pdArray + i) = dVal; 
}
void SetVal_Bool(bool *pbArray, long lEleNum, bool bVal)
{
	for(long i = 0; i < lEleNum; i++)
		*(pbArray + i) = bVal; 
}

// 判断两个值是否相等
bool Val_Equ(const int nVal_1, const int nVal_2)
{
    double dDif = (double)( nVal_1 - nVal_2 );
	dDif = fabs( dDif );
    if ( DBL_EPSILON > dDif )
    {
        return true;
    }
    else
    {
        return false;
    }
}
bool Val_Equ(const double bVal_1, const double bVal_2)
{
    double dDif = fabs( bVal_1 - bVal_2 );
    if ( DBL_EPSILON > dDif )
    {
        return true;
    }
    else
    {
        return false;
    }
}
bool Val_Equ(const long lVal_1, const long lVal_2)
{
    long lDiff = abs( lVal_1 - lVal_2 );
    if ( 1 > lDiff )
    {
        return true;
    }
    else
    {
        return false;
    }
}
bool Val_Equ(const char& cVal_1, const char& cVal_2)
{
    if ( cVal_1 == cVal_2 )
    {
        return true;
    }
    else
    {
        return false;
    }
}


long ArrayMax(const double *pdIn, const long lEle_Num)
{
	long lMax_ID = 0;
	double dMax  = *pdIn;
	for(long lEle = 0; lEle < lEle_Num; lEle++)
	{
		if ( *(pdIn + lEle) > dMax )
		{
			dMax    = *(pdIn + lEle);
			lMax_ID = lEle;
		}
	}
	return lMax_ID;
}
long ArrayMax(const long *plIn, const long lEle_Num)
{
	long lMax_ID = 0;
	long lMax    = *plIn;
	for(long lEle = 0; lEle < lEle_Num; lEle++)
	{
		if ( *(plIn + lEle) > lMax )
		{
			lMax    = *(plIn + lEle);
			lMax_ID = lEle;
		}
	}
	return lMax_ID;
}
long ArrayMax(const int *pnIn, const long lEle_Num)
{
	long lMax_ID = 0;
	int nMax     = *pnIn;
	for(long lEle = 0; lEle < lEle_Num; lEle++)
	{
		if ( *(pnIn + lEle) > nMax )
		{
			nMax    = *(pnIn + lEle);
			lMax_ID = lEle;
		}
	}
	return lMax_ID;
}

// generate the rgb color randomly
int* rand_rgb() 
{
    int* rgb = new int[3];
    rgb[0] = rand() % 255;
    rgb[1] = rand() % 255;
    rgb[2] = rand() % 255;
    return rgb;
}

// read the parameters from yaml file 
void ReadParas(const std::string& file_path, 
				ConfigSetting &config_setting, 
				cylinderConfig &cy_setting,
				circleConfig &circle_setting)
{
	cv::FileStorage fs(file_path, cv::FileStorage::READ);

	config_setting.pcd_data = (std::string)fs["pcd_data"];
	config_setting.json_data = (std::string)fs["json_data"];
	
	std::cout << BOLDBLUE << "-----------Read Data Path-----------" << RESET << std::endl;
	std::cout << "pcd data path: " << config_setting.pcd_data << std::endl;
	std::cout << "JSON data path: " << config_setting.json_data << std::endl;

	// read the parameters for FEC cluster
	config_setting.min_component_size = (int)fs["min_component_size"];
	config_setting.tolorance = (double)fs["tolorance"];
	config_setting.max_n = (int)fs["max_n"];
	config_setting.merge_dist = (double)fs["merge_dist"];	
	std::cout << BOLDBLUE << "-----------Read FEC cluster Parameters-----------" << RESET << std::endl;
	std::cout << "min_component_size: " << config_setting.min_component_size << std::endl;
	std::cout << "tolorance: " << config_setting.tolorance << std::endl;
	std::cout << "max_n: " << config_setting.max_n << std::endl;
	std::cout << "merge_dist: " << config_setting.merge_dist << std::endl;

	// read the parameters to distinguish the type of cluster
	config_setting.clusterPointsNum = (int)fs["clusterPointsNum"];
	config_setting.linearThres = (double)fs["linearThres"];
	config_setting.planarThres = (double)fs["planarThres"];
	config_setting.scaterThres = (double)fs["scaterThres"];
	config_setting.directThres = (double)fs["directThres"];
	config_setting.heightThres = (double)fs["heightThres"];
	config_setting.horizonThres = (double)fs["horizonThres"];
	
	std::cout << BOLDBLUE << "-----------Read Parameters to distinguish the type of cluster-----------" << RESET << std::endl;
	std::cout << "clusterPointsNum: " << config_setting.clusterPointsNum << std::endl;
	std::cout << "linearThres: " << config_setting.linearThres << std::endl;
	std::cout << "planarThres: " << config_setting.planarThres << std::endl;
	std::cout << "scaterThres: " << config_setting.scaterThres << std::endl;
	std::cout << "directThres: " << config_setting.directThres << std::endl;
	std::cout << "heightThres: " << config_setting.heightThres << std::endl;
	std::cout << "horizonThres: " << config_setting.horizonThres << std::endl;

	// config_setting.bSloopSmooth = (bool)fs["bSloopSmooth"];
	fs["bSloopSmooth"] >> config_setting.bSloopSmooth;
	config_setting.cloth_resolution = (double)fs["cloth_resolution"];
	config_setting.rigidness = (int)fs["rigidness"];
	config_setting.time_step = (double)fs["time_step"];
	config_setting.class_threshold = (double)fs["class_threshold"];
	config_setting.iterations = (int)fs["iterations"];
	
	std::cout << BOLDBLUE << "-----------Read Parameters for CSF-----------" << RESET << std::endl;
	std::cout << "bSloopSmooth: " << config_setting.bSloopSmooth << std::endl;
	std::cout << "cloth_resolution: " << config_setting.cloth_resolution << std::endl;
	std::cout << "rigidness: " << config_setting.rigidness << std::endl;
	std::cout << "time_step: " << config_setting.time_step << std::endl;
	std::cout << "class_threshold: " << config_setting.class_threshold << std::endl;
	std::cout << "iterations: " << config_setting.iterations << std::endl;

	config_setting.min_height = (double)fs["min_height"];
	config_setting.max_height = (double)fs["max_height"];
	std::cout << BOLDBLUE << "-----------Read Parameters for Crop-----------" << RESET << std::endl;
	std::cout << "min_Tree_height: " << config_setting.min_height << std::endl;
	std::cout << "max_Tree_height: " << config_setting.max_height << std::endl;

	cy_setting.cyliner_point_num = (int)fs["cyliner_point_num"];
	cy_setting.model_dist_thres = (double)fs["model_dist_thres"];
	cy_setting.iteration_num = (int)fs["iteration_num"];
	cy_setting.min_radius = (double)fs["min_radius"];
	cy_setting.max_radius = (double)fs["max_radius"];
	fs["is_opti_cylinder_coeff"] >> cy_setting.is_opti_cylinder_coeff;
	std::cout << BOLDBLUE << "-----------Read Parameters for cylinder optimiziation-----------" << RESET << std::endl;
	std::cout << "cyliner_point_num: " << cy_setting.cyliner_point_num << std::endl;
	std::cout << "model_dist_thres: " << cy_setting.model_dist_thres << std::endl;
	std::cout << "iteration_num: " << cy_setting.iteration_num << std::endl;
	std::cout << "min_radius: " << cy_setting.min_radius << std::endl;
	std::cout << "max_radius: " << cy_setting.max_radius << std::endl;
	std::cout << "is_opti_cylinder_coeff: " << cy_setting.is_opti_cylinder_coeff << std::endl;

	circle_setting.circle_point_num = (int)fs["circle_point_num"];
	circle_setting.circle_model_dist_thres = (double)fs["circle_model_dist_thres"];
	circle_setting.circle_iteration_num = (int)fs["circle_iteration_num"];
	circle_setting.circle_min_radius = (double)fs["circle_min_radius"];
	circle_setting.circle_max_radius = (double)fs["circle_max_radius"];
	fs["is_opti_coeff"] >> circle_setting.is_opti_coeff;
	std::cout << BOLDBLUE << "-----------Read Parameters for circle optimiziation-----------" << RESET << std::endl;
	std::cout << "circle_point_num: " << circle_setting.circle_point_num << std::endl;
	std::cout << "circle_model_dist_thres: " << circle_setting.circle_model_dist_thres << std::endl;
	std::cout << "circle_iteration_num: " << circle_setting.circle_iteration_num << std::endl;
	std::cout << "circle_min_radius: " << circle_setting.circle_min_radius << std::endl;
	std::cout << "circle_min_radius: " << circle_setting.circle_min_radius << std::endl;
	std::cout << "circle_is_opti_coeff: " << circle_setting.is_opti_coeff << std::endl;

}

void ReadPCD(const std::string& file_path, pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_data)
{
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_path, *pcd_data) == -1) 
	{
		std::cerr << "Error: Could not read PCD file: " << file_path << std::endl;
		return;
    }
	std::cout << "Loaded PCD file: " << file_path << std::endl;
	std::cout << pcd_data->size() << " points in this file!" << std::endl;
}

void ReadJson(const std::string& file_path,
			   std::vector<std::pair<std::string, std::pair<double, double>>>& points)
{
    std::ifstream input(file_path);
    if (!input.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return;
    }

    nlohmann::json j;
    input >> j;

    for (auto& [id, value] : j.items()) {
        double x = value["x"];
        double y = value["y"];
        points.emplace_back(id, std::make_pair(x, y));
    }
}

// calculate the angle difference between two vectors
double angleBetweenVectors(Eigen::Vector3d v1, Eigen::Vector3d v2)
{
    double dot = v1.dot(v2);
    double norms = v1.norm() * v2.norm();
    double cos_theta = dot / norms;
    // limit the value in [-1, 1]
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
    // [0, M_PI]
    double angle_rad = std::acos(cos_theta);
    // [0, 180]
    double angle_deg = angle_rad * 180.0 / M_PI;

    if(angle_deg > 90)
    {
        // angle_deg = angle_deg-90;
        angle_deg = 180 - angle_deg;
    }
    return angle_deg;
}

// spereate the ground points, selected the points by index
void addPointCloud(const std::vector<int>& index_vec, 
                   const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                   pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered)
{
	auto& points = cloud_filtered->points;
	const auto& pointclouds = cloud->points;

	for_each(index_vec.begin(), index_vec.end(), [&](const auto& index) {
		pcl::PointXYZ pc;
		pc.x = pointclouds[index].x;
		pc.y = pointclouds[index].y;
		pc.z = pointclouds[index].z;
		// pc.intensity = pointclouds[index].intensity;

		points.push_back(pc);
	});

	cloud_filtered->height = 1;
	cloud_filtered->width = cloud_filtered->points.size();
}

// CSF, set the parameters, and get the indices of ground and object
void clothSimulationFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
						   pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_ground,
						   pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_obj,
                           ConfigSetting &config_setting)
{

	// new csf
	CSF csf;
	std::vector<int> groundIndexes;
	std::vector<int> offGroundIndexes;

	//step 1 read point cloud
	std::vector<csf::Point> pc;
	const auto& pointclouds = input_cloud->points;
    pc.resize(input_cloud->size());
    transform(pointclouds.begin(), pointclouds.end(), pc.begin(), [&](const auto& p)->csf::Point {
        csf::Point pp;
        pp.x = p.x;
        pp.y = p.y;
        pp.z = p.z;
        return pp;
    });	
	csf.setPointCloud(pc);// or csf.readPointsFromFile(pointClouds_filepath); 
	
    //step 2 parameter settings
	csf.params.bSloopSmooth = config_setting.bSloopSmooth;
	csf.params.cloth_resolution = config_setting.cloth_resolution;
	csf.params.rigidness = config_setting.rigidness;

	csf.params.time_step = config_setting.time_step;
	csf.params.class_threshold = config_setting.class_threshold;
	csf.params.interations = config_setting.iterations;

	//step 3 do filtering
	csf.do_filtering(groundIndexes, offGroundIndexes);

	addPointCloud(groundIndexes, input_cloud, cloud_ground);
    addPointCloud(offGroundIndexes, input_cloud, cloud_obj);
}

void sor_filter_noise(pcl::PointCloud<pcl::PointXYZ>::Ptr &source, pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered)
{

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(source);
    sor.setMeanK(25);
    sor.setStddevMulThresh(1.0);
    std::vector<int> filterd_indices;
    sor.filter(filterd_indices);
    
    addPointCloud(filterd_indices, source, filtered);
}


// FEC cluster, get the cluster points
void fec_cluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                 std::vector<pcl::PointCloud<pcl::PointXYZ>> &cluster_points_,
                 ConfigSetting &config_setting)
{
    if(cloud->size() <= 0)
        return;
    int min_component_size = config_setting.min_component_size;
    double tolorance = config_setting.tolorance;
    int max_n = config_setting.max_n;
    std::vector<pcl::PointIndices> cluster_indices;

    // fec, get the indices of clusters
    cluster_indices = FEC(cloud, min_component_size, tolorance, max_n);
    
    // loop all clusters
    pcl::PointCloud<pcl::PointXYZ>::Ptr centers_(new pcl::PointCloud<pcl::PointXYZ>);
    // std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_points_;
    for (int i = 0; i < cluster_indices.size(); i++) 
    {
        pcl::PointXYZ center_tmp_;
        pcl::PointCloud<pcl::PointXYZ> points_tmp_;
        for (int j = 0; j < cluster_indices[i].indices.size(); j++)
        {
            center_tmp_.x += cloud->points[cluster_indices[i].indices[j]].x;
            center_tmp_.y += cloud->points[cluster_indices[i].indices[j]].y;
            center_tmp_.z += cloud->points[cluster_indices[i].indices[j]].z;
            
            points_tmp_.push_back(cloud->points[cluster_indices[i].indices[j]]);
        }
        center_tmp_.x /= cluster_indices[i].indices.size();
        center_tmp_.y /= cluster_indices[i].indices.size();
        center_tmp_.z /= cluster_indices[i].indices.size();

        centers_->push_back(center_tmp_);
        cluster_points_.push_back(points_tmp_);
    }
    std::cout << "cluster_indices size: " << cluster_indices.size() 
              << ", cluster_points_: " << cluster_points_.size() << std::endl;
}

void pcl_cluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                 std::vector<pcl::PointCloud<pcl::PointXYZ>> &cluster_points_,
                 ConfigSetting &config_setting)
{
	if(cloud->size() <= 0)
        return;
    int min_component_size = config_setting.min_component_size;
    double tolorance = config_setting.tolorance;
    int max_n = config_setting.max_n;
	
	// build a kd-tree
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

	// store the cluster indices
    std::vector<pcl::PointIndices> cluster_indices;

    // clusters
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(tolorance);  // 设置聚类距离阈值
    ec.setMinClusterSize(min_component_size);     // 设置最小聚类点数
    // ec.setMaxClusterSize(max_cluster_size);     // 设置最大聚类点数
    ec.setSearchMethod(tree);                   // 设置搜索方法
    ec.setInputCloud(cloud);                    // 设置输入点云
    ec.extract(cluster_indices);                // 执行聚类

    // loop all clusters
    pcl::PointCloud<pcl::PointXYZ>::Ptr centers_(new pcl::PointCloud<pcl::PointXYZ>);
    // std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cluster_points_;
    for (int i = 0; i < cluster_indices.size(); i++) 
    {
        pcl::PointXYZ center_tmp_;
        pcl::PointCloud<pcl::PointXYZ> points_tmp_;
        for (int j = 0; j < cluster_indices[i].indices.size(); j++)
        {
            center_tmp_.x += cloud->points[cluster_indices[i].indices[j]].x;
            center_tmp_.y += cloud->points[cluster_indices[i].indices[j]].y;
            center_tmp_.z += cloud->points[cluster_indices[i].indices[j]].z;
            
            points_tmp_.push_back(cloud->points[cluster_indices[i].indices[j]]);
        }
        center_tmp_.x /= cluster_indices[i].indices.size();
        center_tmp_.y /= cluster_indices[i].indices.size();
        center_tmp_.z /= cluster_indices[i].indices.size();

        centers_->push_back(center_tmp_);
        cluster_points_.push_back(points_tmp_);
    }
    
    std::cout << "cluster_indices size: " << cluster_indices.size() 
              << ", cluster_points_: " << cluster_points_.size() << std::endl;
}

// calculate the cluster geometry
void getClusterGeometry(std::vector<pcl::PointCloud<pcl::PointXYZ>> clusterPoints, std::vector<Cluster>& clusters)
{ 
    for(int i=0; i<clusters.size(); i++)
    {
        if (clusters[i].points_.size() < 10)
            continue;
        
        calculateGeometry(clusters[i].points_, clusters[i]);
    }
}

// calculate single cluster geometry
void calculateGeometry(pcl::PointCloud<pcl::PointXYZ> clusterPoints, Cluster& cluster)
{
    if(clusterPoints.size() < 10)
        return;

    cluster.center_ = Eigen::Vector3d::Zero();
    cluster.covariance_ = Eigen::Matrix3d::Zero();
    cluster.normal_ = Eigen::Vector3d::Zero();
    cluster.eig_value_ = Eigen::Vector3d::Zero();
    cluster.points_.clear();

    double min_x = clusterPoints.points[0].x;
    double max_x = clusterPoints.points[0].x;
    double min_y = clusterPoints.points[0].y;
    double max_y = clusterPoints.points[0].y;
    double min_z = clusterPoints.points[0].z;
    double max_z = clusterPoints.points[0].z;
    
    for (int j=0; j<clusterPoints.size(); j++)
    {
        Eigen::Vector3d pi;
        pi[0] = clusterPoints.points[j].x;
        pi[1] = clusterPoints.points[j].y;
        pi[2] = clusterPoints.points[j].z;
        
        // statistic the data
        cluster.center_ += pi;
        cluster.covariance_ += pi * pi.transpose();

        // find the min coordinate
        min_x = MIN(min_x, clusterPoints.points[j].x);
        min_y = MIN(min_y, clusterPoints.points[j].y);
        min_z = MIN(min_z, clusterPoints.points[j].z);
        
        // find the max coordinate
        max_x = MAX(max_x, clusterPoints.points[j].x);
        max_y = MAX(max_y, clusterPoints.points[j].y);
        max_z = MAX(max_z, clusterPoints.points[j].z);
    }
    
    cluster.points_ = clusterPoints;

    cluster.center_ = cluster.center_ / clusterPoints.size();
    cluster.covariance_ = cluster.covariance_/clusterPoints.size() -
                            cluster.center_ * cluster.center_.transpose();

    // find the eigen values of matrix, and then sort    
    Eigen::EigenSolver<Eigen::Matrix3d> es(cluster.covariance_);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3d::Index evalsMin, evalsMax; 
    evalsReal.minCoeff(&evalsMin);
    evalsReal.maxCoeff(&evalsMax); 
    int evalsMid = 3 - evalsMin - evalsMax;

    // the attributes in cluster
    cluster.min_point << min_x, min_y, min_z;
    cluster.max_point << max_x, max_y, max_z;

    cluster.linearity_ = (evalsReal(evalsMax) - evalsReal(evalsMid)) / evalsReal(evalsMax);
    cluster.planarity_ = (evalsReal(evalsMid) - evalsReal(evalsMin)) / evalsReal(evalsMax);
    cluster.scatering_ = evalsReal(evalsMin) / evalsReal(evalsMax);
    cluster.eig_value_ << evalsReal(evalsMax), evalsReal(evalsMid), evalsReal(evalsMin);
    // normal vector
    cluster.normal_ << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
                                evecs.real()(2, evalsMax);
    // center point with normal
    cluster.p_center_.x = cluster.center_[0];
    cluster.p_center_.y = cluster.center_[1];
    cluster.p_center_.z = cluster.center_[2];
    cluster.p_center_.normal_x = cluster.normal_[0];
    cluster.p_center_.normal_y = cluster.normal_[1];
    cluster.p_center_.normal_z = cluster.normal_[2];
}

// get the center points of all clusters (in 2D plane)
void getClusterCenters2D(std::vector<Cluster>& clusters, pcl::PointCloud<pcl::PointXY>::Ptr& centers)
{
    int cluster_num = clusters.size();
    centers->clear();
    
    // the num of clusters
    if(cluster_num == 0)
    {
        return;
    }

    // get the center points (in 2d format)
    for(int i=0; i<cluster_num; i++)
    {
        pcl::PointXYZINormal pi = clusters[i].p_center_;
        
        pcl::PointXY pi_2d;
        pi_2d.x = pi.x;
        pi_2d.y = pi.y;
        
        centers->push_back(pi_2d);
    }
}

// direct add cluster
void addClusters(Cluster& cluster1, Cluster& cluster2)
{
    // update the cluster points
    for(int i=0; i<cluster2.points_.size(); i++)
    {
        cluster1.points_.push_back(cluster2.points_[i]);
    }

    cluster1.gnd_height = MIN(cluster1.gnd_height, cluster2.gnd_height);
    
    // calculate the geometry
    // calculateGeometry(cluster1.points_, cluster1);
    
    // only update the min and max points
    cluster1.min_point[0] = MIN(cluster1.min_point[0], cluster2.min_point[0]);
    cluster1.min_point[1] = MIN(cluster1.min_point[1], cluster2.min_point[1]);
    cluster1.min_point[2] = MIN(cluster1.min_point[2], cluster2.min_point[2]);
    
    cluster1.max_point[0] = MAX(cluster1.max_point[0], cluster2.max_point[0]);
    cluster1.max_point[1] = MAX(cluster1.max_point[1], cluster2.max_point[1]);
    cluster1.max_point[2] = MAX(cluster1.max_point[2], cluster2.max_point[2]);
}

// calculate the volume of box
double boxVolume(Eigen::Vector3d& min_point, Eigen::Vector3d& max_point)
{
    double dx = max_point[0] - min_point[0];
    double dy = max_point[1] - min_point[1];
    double dz = max_point[2] - min_point[2];
    dx = MAX(dx,0.0);
    dy = MAX(dy,0.0);
    dz = MAX(dz,0.0);

    double vol = dx * dy * dz;

    return vol;
}

// calculate the intersect volume of two box
double intersectVolume(Eigen::Vector3d& min_point1, Eigen::Vector3d& max_point1,
                        Eigen::Vector3d& min_point2, Eigen::Vector3d& max_point2)
{
    Eigen::Vector3d min_inter, max_inter;
    min_inter[0] = MAX(min_point1[0], min_point2[0]);
    min_inter[1] = MAX(min_point1[1], min_point2[1]);
    min_inter[2] = MAX(min_point1[2], min_point2[2]);
    
    max_inter[0] = MIN(max_point1[0], max_point2[0]);
    max_inter[1] = MIN(max_point1[1], max_point2[1]);
    max_inter[2] = MIN(max_point1[2], max_point2[2]);

    // calculate the length of intersect box
    double dx = max_inter[0] - min_inter[0];
    double dy = max_inter[1] - min_inter[1];
    double dz = max_inter[2] - min_inter[2];
    // std::cout << "dx: " << dx << ", dy: " << dy << ", dz: " << dz << std::endl;
    // make sure the length is great than 0
    dx = MAX(dx,0.0);
    dy = MAX(dy,0.0);
    dz = MAX(dz,0.0);

    double vol = dx * dy * dz;
    return vol;
}


// box intersection
double boxIntersect(Cluster& cluster1, Cluster& cluster2)
{
	// // calculate the distance between centers
	// double d = horiDist(cluster1.p_center_, cluster2.p_center_);

	// calculate the volume
	Eigen::Vector3d min_point1, max_point1;
	min_point1 = cluster1.min_point;
	max_point1 = cluster1.max_point;
	Eigen::Vector3d min_point2, max_point2;
	min_point2 = cluster2.min_point;
	max_point2 = cluster2.max_point;
	
	double v1 = boxVolume(min_point1, max_point1);
	double v2 = boxVolume(min_point2, max_point2);
	double inter_v = intersectVolume(min_point1, max_point1, min_point2, max_point2);
	
	double min_v = MIN(v1,v2);
	double ratio = inter_v/min_v;
	
	// std::cout << BOLDBLUE << "V1: " << v1 << ", V2: " << v2 << ", inter_v: " << inter_v << ", ratio: " << ratio << RESET << std::endl;
	
	// if(d > merg_dist)
	//     std::cout << BOLDBLUE << "V1: " << v1 << ", V2: " << v2 << ", inter_v: " << inter_v << ", ratio: " << ratio << ", Dist: " << d << RESET << std::endl;
	
	return ratio;
}

// is valid cluster or not
bool isValidCluster(Cluster& cluster_in, ConfigSetting &config_setting)
{
	bool isValid = false;
	Eigen::Vector3d orientation_vector(0,0,1);
	// the direction difference between cluster's normal vector and the z aixs
	// double direcDiff = abs(cluster_in.normal_[2]);
	double direcDiff = pointNormalVectorDiff(cluster_in.normal_, orientation_vector);

	// the difference along with three aixs
	double xaxisDiff = abs(cluster_in.max_point[0] - cluster_in.min_point[0]);
	double yaxisDiff = abs(cluster_in.max_point[1] - cluster_in.min_point[1]);
	double heightDiff = abs(cluster_in.max_point[2] - cluster_in.min_point[2]);
	
	// calculate point density in bounding box
	double dx = cluster_in.max_point[0] - cluster_in.min_point[0];
	double dy = cluster_in.max_point[1] - cluster_in.min_point[1];
	double dz = cluster_in.max_point[2] - cluster_in.min_point[2];
	double point_density = cluster_in.points_.size() / (dx * dy * dz);
	
	// cluster's threshold
	if(cluster_in.points_.size() > config_setting.clusterPointsNum &&
		cluster_in.linearity_ > config_setting.linearThres && 
		cluster_in.planarity_ < config_setting.planarThres &&
		cluster_in.scatering_ < config_setting.scaterThres &&
		direcDiff  < config_setting.directThres && 
		heightDiff > config_setting.heightThres && 
		heightDiff > xaxisDiff && heightDiff > yaxisDiff &&
		(xaxisDiff+yaxisDiff) < config_setting.horizonThres) 
	{ // && point_density > 20 (use this condition, wall surface is hard to remove)
		isValid = true;
	}

	return isValid;
}

// optimize the cluster vector
void optiClusters(std::vector<Cluster>& totalClusters, ConfigSetting &config_setting)
{
	// std::cout << BOLDGREEN << "1-The totalClusters size: " << totalClusters.size() << RESET << std::endl;
	if(totalClusters.size() == 0)
	{
		return;
	}
	std::vector<bool> merged(totalClusters.size(), false); 
	
	// build kdtree on the global cluster's centers (in 2D format)
	pcl::PointCloud<pcl::PointXY>::Ptr globalCenter2D(new pcl::PointCloud<pcl::PointXY>());
	getClusterCenters2D(totalClusters, globalCenter2D);
	if(globalCenter2D->size() == 0)
		return;
	pcl::KdTreeFLANN<pcl::PointXY> kdtree2d;
	kdtree2d.setInputCloud(globalCenter2D);
	// find the closed clusters and then merge
	for(int i=0; i<totalClusters.size(); i++)
	{
		// whether this cluster is merged
		if (merged[i])
			continue;

		// get the querry point
		pcl::PointXY query_center;
		query_center.x = totalClusters[i].p_center_.x;
		query_center.y = totalClusters[i].p_center_.y;
		
		// find the candidate clusters
		double radius = 0.5;
		std::vector<int> point_indices;
		std::vector<float> distances;
		if(kdtree2d.radiusSearch(query_center, radius, point_indices, distances) > 0)
		{
			for (size_t j = 0; j < point_indices.size(); ++j)
			{
				int index = point_indices[j]+1;
				if(index != i && !merged[index])
				{
					pcl::PointXY p_index;
					p_index.x = totalClusters[index].p_center_.x;
					p_index.y = totalClusters[index].p_center_.y;

					double d = horiDist(query_center, p_index);
					double ratio = boxIntersect(totalClusters[i], totalClusters[index]);
					double inter_ratio = 0.1;
					if(ratio > inter_ratio)
					{
						std::cout << BOLDYELLOW << "inter_ratio: " << inter_ratio << ", NUM: " << point_indices.size() << ", d: " << d 
											<< ", plane_d: " << d << ", ratio: " << ratio 
											<< ", Index: " << index << ", i: " << i << RESET << std::endl; 

						// add two clusters, then recalculate cluster's geometry
						addClusters(totalClusters[i], totalClusters[index]);
						calculateGeometry(totalClusters[i].points_, totalClusters[i]);

						merged[index] = true;
						break;
					}
				}
			}
		}
	}
	
	// filter the invalid cluster
	size_t write_idx = 1;
	for (size_t read_idx = 1; read_idx < totalClusters.size(); ++read_idx)
	{
		if (!merged[read_idx] && isValidCluster(totalClusters[read_idx], config_setting))
		{
			totalClusters[read_idx].label = write_idx + 1;
			if (read_idx != write_idx) {
				std::swap(totalClusters[write_idx], totalClusters[read_idx]);
			}
			++write_idx;
		}
	}
	// resize, remove the last part
	totalClusters.resize(write_idx);
}

// fit the cylinders of frame or total cloud
void clusterFit(std::vector<pcl::PointCloud<pcl::PointXYZ>> &cluster_points_, 
			std::vector<Cluster>& clustersData, cylinderConfig& fitting_config)
{
	std::vector<Cluster> tmpClusters;
	int count = 1;
	for(int i=0; i<cluster_points_.size(); i++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr tempPoints(new pcl::PointCloud<pcl::PointXYZ>());
		*tempPoints = cluster_points_[i];
		
		// fit the cylinder
		Cluster clusterI;
		// bool fit_flag = pcl_ransac_cylinder(cluster_points_[i], clusterI, fitting_config);
		bool fit_flag = fitCylinder(tempPoints, clusterI, fitting_config);
		if(fit_flag)
		{
			// std::cout << "success!" << std::endl;
			clusterI.is_cylinder_ = true;
			clusterI.label = count;
			count++;
			tmpClusters.push_back(clusterI);
		}
		else
		{
			clusterI.is_cylinder_ = false;
		}
	}
	clustersData = tmpClusters;
}

// sum the cluster points into a total point cloud
void clusterTopoints(std::vector<Cluster>& clusters, pcl::PointCloud<pcl::PointXYZRGBL>::Ptr& points)
{
    for(int i=0; i<clusters.size(); i++)
    {
        int label = clusters[i].label;
        int* rgb = rand_rgb();
        // in this study, the label-1 respresents ground point and is colored in brown
        if(label == 1)
        {
            rgb[0] = 216;
            rgb[1] = 144;
            rgb[2] = 5;
        }
          
        pcl::PointXYZRGBL thisPoint;
        for (int j=0; j<clusters[i].points_.size(); j++)
        {
            thisPoint.x = clusters[i].points_[j].x;
            thisPoint.y = clusters[i].points_[j].y;
            thisPoint.z = clusters[i].points_[j].z;
            thisPoint.r = rgb[0];
            thisPoint.g = rgb[1];
            thisPoint.b = rgb[2];
            thisPoint.label = j;

            // push the points with color and label
            points->push_back(thisPoint);
        }
    }
}

// down sample the point cloud, by voxel
void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZ> &pl_feat, double voxel_size) 
{
	int pointsNum = pl_feat.size();
	if (voxel_size < 0.01) 
		return;

	// a container, include key and value
	std::unordered_map<UNI_VOXEL_LOC, M_POINT> voxel_map;
	voxel_map.reserve(200000000);
	// point cloud size
	uint plsize = pl_feat.size();
	// loop for all points, and sum the information in each grid
	for (uint i = 0; i < plsize; i++) {
	// current points
	pcl::PointXYZ &p_c = pl_feat[i];
	float loc_xyz[3];
	for (int j = 0; j < 3; j++) {
		loc_xyz[j] = p_c.data[j] / voxel_size;
		// <0, then loc-1, floor
		if (loc_xyz[j] < 0) {
		loc_xyz[j] -= 1.0;
		}
	}
	// generate the voxel location
	UNI_VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
						(int64_t)loc_xyz[2]);
	// 
	auto iter = voxel_map.find(position);
	if (iter != voxel_map.end()) {
		iter->second.xyz[0] += p_c.x;
		iter->second.xyz[1] += p_c.y;
		iter->second.xyz[2] += p_c.z;
		iter->second.count++;
	} else {
		M_POINT anp;
		anp.xyz[0] = p_c.x;
		anp.xyz[1] = p_c.y;
		anp.xyz[2] = p_c.z;
		anp.count = 1;
		voxel_map[position] = anp;
	}
	}

	// reset the ori point cloud data
	plsize = voxel_map.size();
	pl_feat.clear();
	pl_feat.resize(plsize);

	// downSample the point cloud, the centeroid points are selected 
	uint i = 0;
	for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter) 
	{
		pl_feat[i].x = iter->second.xyz[0] / iter->second.count;
		pl_feat[i].y = iter->second.xyz[1] / iter->second.count;
		pl_feat[i].z = iter->second.xyz[2] / iter->second.count;
		i++;
	}
	std::cout << "Input points num: " << pointsNum << ", downsample voxel size: " << voxel_size 
			  << ", downsampled points num: " << plsize << std::endl;
}

void stem_above_ground(pcl::PointCloud<pcl::PointXYZ>::Ptr &tree_points, 
						pcl::PointCloud<pcl::PointXYZ>::Ptr &ground_points,
						pcl::PointCloud<pcl::PointXYZ>::Ptr &croped_tree,
						ConfigSetting &config_setting)
{
	if(tree_points->size() == 0 || ground_points->size() == 0)
	{
		std::cerr << "Error: tree or ground no points" << std::endl;
		return;
	}

	pcl::PointCloud<pcl::PointXY>::Ptr ground_2d(new pcl::PointCloud<pcl::PointXY>());
	point_to_XY<pcl::PointXYZ>(ground_points, ground_2d);

	// pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
    // tree->setInputCloud(ground_points);
	pcl::KdTreeFLANN<pcl::PointXY>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXY>);
	tree->setInputCloud(ground_2d);

	for(int i=0; i<tree_points->size(); i++)
	{
		// pcl::PointXYZ query_center = tree_points->points[i];
		pcl::PointXY query_center;
		query_center.x = tree_points->points[i].x;
		query_center.y = tree_points->points[i].y;

		std::vector<int> point_indices;
		std::vector<float> distances;
		if(tree->nearestKSearch(query_center, 1, point_indices, distances) > 0)
		{
			int index = point_indices[0];
			pcl::PointXYZ near_ground_point = ground_points->points[index];
			if(std::abs(tree_points->points[i].z - near_ground_point.z) < config_setting.max_height &&
				std::abs(tree_points->points[i].z - near_ground_point.z) > config_setting.min_height)
			{
				croped_tree->push_back(tree_points->points[i]);
			}
		}
	}
}

bool fitCylinder(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                 Cluster& source,
                 cylinderConfig& fitting_config)
{
    if (cloud->empty()) {
        std::cerr << "[fitCylinder] Input cloud is empty!" << std::endl;
        return false;
    }
    source.points_ = *cloud;
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(20); // 邻域大小可调
    ne.compute(*normals);

    // 创建分割对象
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(0.1);
    seg.setMaxIterations(fitting_config.iteration_num);
    seg.setDistanceThreshold(fitting_config.model_dist_thres);
    seg.setRadiusLimits(fitting_config.min_radius, fitting_config.max_radius);
    seg.setInputCloud(cloud);
    seg.setInputNormals(normals);

	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        std::cerr << "[fitCylinder] Could not estimate a cylindrical model for the given dataset." << std::endl;
        return false;
    }
    // no inlier points have been found
    if (inliers->indices.empty() && inliers->indices.size() < 0.5*fitting_config.cyliner_point_num) 
    {
        // std::cerr << "Not found the cylinder！" << std::endl;
        source.is_cylinder_ = false;
        return false;
    }

    // extract the cylinder points (inlier)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cylinder_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for(int i=0; i<inliers->indices.size(); i++)
    {
        pcl::PointXYZ pi;
        pi = source.points_.points[inliers->indices[i]];
        cylinder_cloud->push_back(pi);
    }

    // std::cout << "1-source.points_: " << source.points_.size() << std::endl;
    // update the information of cluster
    source.points_ = *cylinder_cloud;
    source.cylinder_coff = *coefficients;
    source.is_cylinder_ = true;
    // std::cout << "2-source.points_: " << source.points_.size() << std::endl;
    return true;
}

// fit the trunk by PCL ransac (circle)
bool pcl_ransac_circle(pcl::PointCloud<pcl::PointXYZ>& source, 
                       pcl::ModelCoefficients& coff, 
                       circleConfig& fitting_config)
{
    if(source.size() < fitting_config.circle_point_num)
    {
        return false;
    }
    
    // pcl::PointXYZ convert to XYZ format
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_points(new pcl::PointCloud<pcl::PointXYZ>());
    for(int i=0; i<source.size(); i++)
    {
        pcl::PointXYZ pi;
        pi.x = source.points[i].x;
        pi.y = source.points[i].y;
        pi.z = source.points[i].z;
        
        tmp_points->push_back(pi);
    }

    // pcl::SampleConsensusModelCircle2D<pcl::PointXYZ>::Ptr model(
    //     new pcl::SampleConsensusModelCircle2D<pcl::PointXYZ>(tmp_points));
    
    // find a circle model (in XY plane)
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setModelType(pcl::SACMODEL_CIRCLE2D);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(fitting_config.circle_iteration_num);
    seg.setDistanceThreshold(fitting_config.circle_model_dist_thres);
    seg.setRadiusLimits(fitting_config.circle_min_radius, fitting_config.circle_max_radius);
    seg.setInputCloud(tmp_points);
    seg.setOptimizeCoefficients(fitting_config.is_opti_coeff);
    // perform the segment
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() < 0.5*fitting_config.circle_point_num) 
    {
        // std::cerr << "Not found the cylinder！" << std::endl;
        return false;
    }
    
    // get the circle coefficients
    coff = *coefficients;
    return true;
}


// calculate the DBH of stem
void calculateDBH(std::vector<Cluster>& clusters, circleConfig& fitting_config)
{
    int count = 0;

    // loop all clusters
    for(int i=0; i<clusters.size(); i++)
    {
		// get the basic attributes (height)
		double min_z, max_z;
		min_z = clusters[i].min_point[2];
		max_z = clusters[i].max_point[2];
		double stem_height = max_z - min_z;
		// double gnd_height = clusters[i].gnd_height;
		double gnd_height = min_z;

		// calculate the profile circle 
		if(stem_height > 0) // clusters[i].is_cylinder_ && stem_height > 2
		{
			// get the location of stem
			pcl::PointXY cluster_i_loc;
			cluster_i_loc.x = clusters[i].center_[0];
			cluster_i_loc.y = clusters[i].center_[1];
			
			// variables for the following process, three circles for different heights
			std::vector<double> inter_h;
			inter_h.resize(3);
			inter_h[0] = gnd_height + 1.3; inter_h[1] = gnd_height + 1.4; inter_h[2] = gnd_height + 1.5; // set the height of interval
			std::vector<pcl::PointCloud<pcl::PointXYZ>> profiles;
			profiles.resize(3);
			std::vector<bool> flags;
			flags.resize(3);
			std::vector<pcl::ModelCoefficients> coffes;
			coffes.resize(3);
			
			// split the cluster point cloud
			pcl::PointCloud<pcl::PointXYZ> cluster_points = clusters[i].points_;
			for(int i=0; i<cluster_points.size(); i++)
			{
				for (int j = 0; j < 3; ++j)
				{
					if (std::abs(cluster_points.points[i].z - inter_h[j]) <= 0.1) 
					{
						profiles[j].push_back(cluster_points.points[i]);
					}
				}
			}

			// fit the circles of profile
			for (int k = 0; k < 3; ++k)
			{
				pcl::ModelCoefficients circle_coff;
				bool circle_flag = pcl_ransac_circle(profiles[k], circle_coff, fitting_config);
				flags[k] = circle_flag;
				coffes[k] = circle_coff;
				// std::cout << BOLDGREEN << "profiles " << k << ": "<< profiles[k].size() << RESET << std::endl;
			}

			// whether the flag is true
			if(flags[0] && flags[1] && flags[2])
			{
				clusters[i].dbh_flag = true;
				clusters[i].dbh_coff.values.resize(3);
				clusters[i].dbh_coff.values[0] = (coffes[0].values[0] + coffes[1].values[0] + coffes[2].values[0])/3;
				clusters[i].dbh_coff.values[1] = (coffes[0].values[1] + coffes[1].values[1] + coffes[2].values[1])/3;
				clusters[i].dbh_coff.values[2] = (coffes[0].values[2] + coffes[1].values[2] + coffes[2].values[2])/3;
				clusters[i].gnd_height = gnd_height;
				// std::cout << BOLDGREEN << "dbh_coff:( " << clusters[i].dbh_coff.values[0] << ", " 
				//                 << clusters[i].dbh_coff.values[1] << ", " 
				//                 << clusters[i].dbh_coff.values[2] << ")" << RESET << std::endl;
			}
			else if(flags[0] && flags[1])
			{
				clusters[i].dbh_flag = true;
				clusters[i].dbh_coff.values.resize(3);
				clusters[i].dbh_coff.values[0] = (coffes[0].values[0] + coffes[1].values[0])/2;
				clusters[i].dbh_coff.values[1] = (coffes[0].values[1] + coffes[1].values[1])/2;
				clusters[i].dbh_coff.values[2] = (coffes[0].values[2] + coffes[1].values[2])/2;
				clusters[i].gnd_height = gnd_height;
			}
			else if(flags[0] && flags[2])
			{
				clusters[i].dbh_flag = true;
				clusters[i].dbh_coff.values.resize(3);
				clusters[i].dbh_coff.values[0] = (coffes[0].values[0] + coffes[2].values[0])/2;
				clusters[i].dbh_coff.values[1] = (coffes[0].values[1] + coffes[2].values[1])/2;
				clusters[i].dbh_coff.values[2] = (coffes[0].values[2] + coffes[2].values[2])/2;
				clusters[i].gnd_height = gnd_height;
			}
			else if(flags[1] && flags[2])
			{
				clusters[i].dbh_flag = true;
				clusters[i].dbh_coff.values.resize(3);
				clusters[i].dbh_coff.values[0] = (coffes[1].values[0] + coffes[2].values[0])/2;
				clusters[i].dbh_coff.values[1] = (coffes[1].values[1] + coffes[2].values[1])/2;
				clusters[i].dbh_coff.values[2] = (coffes[1].values[2] + coffes[2].values[2])/2;
				clusters[i].gnd_height = gnd_height;
			}
			else if(flags[0])
			{
				clusters[i].dbh_flag = true;
				clusters[i].dbh_coff.values.resize(3);
				clusters[i].dbh_coff.values[0] = coffes[0].values[0];
				clusters[i].dbh_coff.values[1] = coffes[0].values[1];
				clusters[i].dbh_coff.values[2] = coffes[0].values[2];
				clusters[i].gnd_height = gnd_height;
			}
			else if(flags[1])
			{
				clusters[i].dbh_flag = true;
				clusters[i].dbh_coff.values.resize(3);
				clusters[i].dbh_coff.values[0] = coffes[1].values[0];
				clusters[i].dbh_coff.values[1] = coffes[1].values[1];
				clusters[i].dbh_coff.values[2] = coffes[1].values[2];
				clusters[i].gnd_height = gnd_height;
			}
			else if(flags[2])
			{
				clusters[i].dbh_flag = true;
				clusters[i].dbh_coff.values.resize(3);
				clusters[i].dbh_coff.values[0] = coffes[2].values[0];
				clusters[i].dbh_coff.values[1] = coffes[2].values[1];
				clusters[i].dbh_coff.values[2] = coffes[2].values[2];
				clusters[i].gnd_height = gnd_height;
			}
			else
			{
				count++;

				// std::cout << BOLDGREEN << flags[0] << flags[1] << flags[2]
				//           << ", profiles: " << profiles[0].size() << ", " << profiles[1].size()
				//           << ", " << profiles[2].size() << ", gnd_h: " << gnd_height 
				//           << ", min_z: " << min_z << ", max_z: " << max_z << ", stem_h: " << stem_height << RESET << std::endl;
			}
		}
    }
    std::cout << "Total num: " << clusters.size() << ", valid count: " << (clusters.size() - count) << std::endl;
}


// transfromation type
void matrix_to_pair(Eigen::Matrix4f &trans_matrix,
                    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &trans_pair)
{
    trans_pair.first = trans_matrix.block<3,1>(0,3).cast<double>();
    trans_pair.second = trans_matrix.block<3,3>(0,0).cast<double>();
}

void pair_to_matrix(Eigen::Matrix4f &trans_matrix,
                    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &trans_pair)
{
    trans_matrix.block<3,1>(0,3) = trans_pair.first.cast<float>();
    trans_matrix.block<3,3>(0,0) = trans_pair.second.cast<float>();
}

void point_to_vector(pcl::PointCloud<pcl::PointXYZ>::Ptr &pclPoints, 
                        std::vector<Eigen::Vector3d> &vecPoints)
{
	int Num = pclPoints->size();
	
	for(int i=0; i<Num; i++)
	{
		Eigen::Vector3d p;
		p[0] = pclPoints->points[i].x;
		p[1] = pclPoints->points[i].y;
		p[2] = pclPoints->points[i].z;

		vecPoints.push_back(p);
	}
}

// get ground height of current clusters
void clusterGndHeight(std::vector<Cluster>& clustersData, pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_gnd_data)
{
	// get the ground points and convert into the xy format
	pcl::PointCloud<pcl::PointXY>::Ptr gnd_data2d(new pcl::PointCloud<pcl::PointXY>());
	point_to_XY<pcl::PointXYZ>(tmp_gnd_data, gnd_data2d);
	// build the kdtree on 2d points cloud
	pcl::KdTreeFLANN<pcl::PointXY> kdtree2d;
	kdtree2d.setInputCloud(gnd_data2d);

	for(int i=0; i<clustersData.size(); i++)
	{
		// get the location of stem
		pcl::PointXY cluster_i_loc;
		cluster_i_loc.x = clustersData[i].center_[0];
		cluster_i_loc.y = clustersData[i].center_[1];
		
		double radius = 1.0;
		std::vector<int> point_indices;
		std::vector<float> distances;  
		if(kdtree2d.radiusSearch(cluster_i_loc, radius, point_indices, distances) > 0)
		{
			int index = point_indices[0];
			clustersData[i].gnd_height = tmp_gnd_data->points[index].z;
			
			// make sure the ground height is smaller than stem base
			if(clustersData[i].gnd_height > clustersData[i].min_point[2])
				clustersData[i].gnd_height = clustersData[i].min_point[2];
		}
		else
		{
			clustersData[i].gnd_height = clustersData[i].min_point[2];
		}
	}
}


void getDBHInfo(std::vector<Cluster>& clustersData, 
				std::vector<std::pair<std::string, std::pair<double, double>>> search_points)
{
	// build kdtree on the global cluster's centers (in 2D format)
	pcl::PointCloud<pcl::PointXY>::Ptr globalCenter2D(new pcl::PointCloud<pcl::PointXY>());
	getClusterCenters2D(clustersData, globalCenter2D);
	if(globalCenter2D->size() == 0)
		return;
	pcl::KdTreeFLANN<pcl::PointXY> kdtree2d;
	kdtree2d.setInputCloud(globalCenter2D);
	
	std::cout << "globalCenter2D: " << globalCenter2D->size() << std::endl;

	pcl::PointCloud<pcl::PointXY> querry;
	for(int i=0; i<search_points.size(); i++)
	{
		std::cout << "search_points: " << search_points.size() << std::endl;

		pcl::PointXY p_i;
		p_i.x = search_points[i].second.first;
		p_i.y = search_points[i].second.second;

		// find the candidate clusters
		std::vector<int> point_indices;
		std::vector<float> distances;
		if(kdtree2d.nearestKSearch(p_i, 1, point_indices, distances) > 0)
		{
			int index = point_indices[0];
			if(clustersData[index].dbh_flag)
			{
				double dbh = clustersData[index].dbh_coff.values[2]*2;
				std::cout << BOLDGREEN << search_points[i].first << " X: " << p_i.x << " Y: " << p_i.y
					 << ", distances: " << distances[0] << ", dbh: " << dbh << RESET << std::endl;
				
				std::string data_name = "/media/xiaochen/xch_disk/TreeScope_dataset/icra_challenge/testing_phase/"
										+ search_points[i].first + ".pcd";
				pcl::io::savePCDFileBinary(data_name, clustersData[index].points_);
			}
			else
			{
				std::cout << BOLDRED << search_points[i].first << " X: " << p_i.x << " Y: " << p_i.y
					 << ", distances: " << distances[0] << RESET << std::endl;
				
				std::string data_name = "/media/xiaochen/xch_disk/TreeScope_dataset/icra_challenge/testing_phase/"
						+ search_points[i].first + "-false.pcd";
				pcl::io::savePCDFileBinary(data_name, clustersData[index].points_);
			}
			


		}
	}
}