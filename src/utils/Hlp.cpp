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
void ReadParas(const std::string& file_path, ConfigSetting &config_setting)
{
	cv::FileStorage fs(file_path, cv::FileStorage::READ);

	config_setting.pcd_data = (std::string)fs["pcd_data"];
	std::cout << BOLDBLUE << "-----------Read Data Path-----------" << RESET << std::endl;
	std::cout << "pcd data path: " << config_setting.pcd_data << std::endl;

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
	config_setting.linearityThres = (double)fs["linearityThres"];
	config_setting.scateringThres = (double)fs["scateringThres"];
	config_setting.upThres = (double)fs["upThres"];
	config_setting.clusterHeight = (double)fs["clusterHeight"];
	config_setting.centerSelection = (int)fs["centerSelection"];
	std::cout << BOLDBLUE << "-----------Read Parameters to distinguish the type of cluster-----------" << RESET << std::endl;
	std::cout << "linearityThres: " << config_setting.linearityThres << std::endl;
	std::cout << "scateringThres: " << config_setting.scateringThres << std::endl;
	std::cout << "upThres: " << config_setting.upThres << std::endl;
	std::cout << "clusterHeight: " << config_setting.clusterHeight << std::endl;
	std::cout << "centerSelection: " << config_setting.centerSelection << std::endl;


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
                 std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cluster_points_,
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
        pcl::PointCloud<pcl::PointXYZ>::Ptr points_tmp_(new pcl::PointCloud<pcl::PointXYZ>);
        for (int j = 0; j < cluster_indices[i].indices.size(); j++)
        {
            center_tmp_.x += cloud->points[cluster_indices[i].indices[j]].x;
            center_tmp_.y += cloud->points[cluster_indices[i].indices[j]].y;
            center_tmp_.z += cloud->points[cluster_indices[i].indices[j]].z;
            
            points_tmp_->push_back(cloud->points[cluster_indices[i].indices[j]]);
        }
        center_tmp_.x /= cluster_indices[i].indices.size();
        center_tmp_.y /= cluster_indices[i].indices.size();
        center_tmp_.z /= cluster_indices[i].indices.size();

        centers_->push_back(center_tmp_);
        cluster_points_.push_back(points_tmp_);
    }
    
    for (size_t i = 0; i < centers_->size(); i++)
    {
        for (size_t j = i+1; j < centers_->size();)
        {
            double dx = centers_->points[i].x - centers_->points[j].x;
            double dy = centers_->points[i].y - centers_->points[j].y;
            double dz = centers_->points[i].z - centers_->points[j].z;
            double d2 = sqrt(dx*dx + dy*dy);
            double d3 = sqrt(dx*dx + dy*dy + dz*dz);
            
            if(d2 < config_setting.merge_dist)
            {
                *cluster_points_[i] = *cluster_points_[i] + *cluster_points_[j];
                
                centers_->points[i].x = (centers_->points[i].x + centers_->points[j].x) / 2;
                centers_->points[i].y = (centers_->points[i].y + centers_->points[j].y) / 2;
                centers_->points[i].z = (centers_->points[i].z + centers_->points[j].z) / 2;

                centers_->erase(centers_->begin() + j);
                cluster_points_.erase(cluster_points_.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
    
    std::cout << "cluster_indices size: " << cluster_indices.size() 
              << ", cluster_points_: " << cluster_points_.size() << std::endl;
}

void pcl_cluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                 std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cluster_points_,
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
        pcl::PointCloud<pcl::PointXYZ>::Ptr points_tmp_(new pcl::PointCloud<pcl::PointXYZ>);
        for (int j = 0; j < cluster_indices[i].indices.size(); j++)
        {
            center_tmp_.x += cloud->points[cluster_indices[i].indices[j]].x;
            center_tmp_.y += cloud->points[cluster_indices[i].indices[j]].y;
            center_tmp_.z += cloud->points[cluster_indices[i].indices[j]].z;
            
            points_tmp_->push_back(cloud->points[cluster_indices[i].indices[j]]);
        }
        center_tmp_.x /= cluster_indices[i].indices.size();
        center_tmp_.y /= cluster_indices[i].indices.size();
        center_tmp_.z /= cluster_indices[i].indices.size();

        centers_->push_back(center_tmp_);
        cluster_points_.push_back(points_tmp_);
    }
    
    for (size_t i = 0; i < centers_->size(); i++)
    {
        for (size_t j = i+1; j < centers_->size();)
        {
            double dx = centers_->points[i].x - centers_->points[j].x;
            double dy = centers_->points[i].y - centers_->points[j].y;
            double dz = centers_->points[i].z - centers_->points[j].z;
            double d2 = sqrt(dx*dx + dy*dy);
            double d3 = sqrt(dx*dx + dy*dy + dz*dz);
            
            if(d2 < config_setting.merge_dist)
            {
                *cluster_points_[i] = *cluster_points_[i] + *cluster_points_[j];
                
                centers_->points[i].x = (centers_->points[i].x + centers_->points[j].x) / 2;
                centers_->points[i].y = (centers_->points[i].y + centers_->points[j].y) / 2;
                centers_->points[i].z = (centers_->points[i].z + centers_->points[j].z) / 2;

                centers_->erase(centers_->begin() + j);
                cluster_points_.erase(cluster_points_.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
    
    std::cout << "cluster_indices size: " << cluster_indices.size() 
              << ", cluster_points_: " << cluster_points_.size() << std::endl;
}
// calculate the attributes of each cluster, then select the trunk
void cluster_attributes(std::vector<Cluster> &clusters,
                        std::vector<Cluster> &disgard_clusters,
                        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cluster_points_,
                        ConfigSetting &config_setting)
{
    if(cluster_points_.size() <= 0)
        return;

    // center of obj cloud
    Cluster cluster_tmp;
    
    // int line=0, plane=0;
    
    // loop all clusters
    // #pragma omp parallel for num_threads(8)
    for (int i = 0; i < cluster_points_.size(); i++) 
    {
        // init value
        cluster_tmp.center_ = Eigen::Vector3d::Zero();
        cluster_tmp.covariance_ = Eigen::Matrix3d::Zero();
        cluster_tmp.normal_ = Eigen::Vector3d::Zero();
        cluster_tmp.eig_value_ = Eigen::Vector3d::Zero();
        cluster_tmp.points_.clear();

        double minZ = MAX_INF;
        double maxZ = MIN_INF;
        // loop the points in each cluster
        for (int j = 0; j < cluster_points_[i]->points.size(); j++)
        {
            Eigen::Vector3d pi;
            pi[0] = cluster_points_[i]->points[j].x;
            pi[1] = cluster_points_[i]->points[j].y;
            pi[2] = cluster_points_[i]->points[j].z;
            if (minZ > pi[2])
                minZ = pi[2];
            if (maxZ < pi[2])
                maxZ = pi[2];
            cluster_tmp.center_ += pi;
            cluster_tmp.covariance_ += pi * pi.transpose();
        }
        cluster_tmp.points_ = *cluster_points_[i];

        // calculate the center and covariance
        cluster_tmp.center_ = cluster_tmp.center_ / cluster_points_[i]->points.size();
        cluster_tmp.covariance_ = cluster_tmp.covariance_/cluster_points_[i]->points.size() -
                                cluster_tmp.center_ * cluster_tmp.center_.transpose();
        cluster_tmp.center_[2] = minZ; // minZ; 1.5
        cluster_tmp.minZ = minZ;
        cluster_tmp.maxZ = maxZ;

        Eigen::EigenSolver<Eigen::Matrix3d> es(cluster_tmp.covariance_);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real();
        Eigen::Matrix3d::Index evalsMin, evalsMax; 
        evalsReal.minCoeff(&evalsMin);
        evalsReal.maxCoeff(&evalsMax); 
        int evalsMid = 3 - evalsMin - evalsMax;
        // the attributes in cluster
        cluster_tmp.linearity_ = (evalsReal(evalsMax) - evalsReal(evalsMid)) / evalsReal(evalsMax);
        cluster_tmp.planarity_ = (evalsReal(evalsMid) - evalsReal(evalsMin)) / evalsReal(evalsMax);
        cluster_tmp.scatering_ = evalsReal(evalsMin) / evalsReal(evalsMax);
        cluster_tmp.eig_value_ << evalsReal(evalsMax), evalsReal(evalsMid), evalsReal(evalsMin);
        cluster_tmp.normal_ << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
                        evecs.real()(2, evalsMax);
        Eigen::Vector3d Z = Eigen::Vector3d::UnitZ();
        Eigen::Vector3d normal_inc = cluster_tmp.normal_ - Z;
        Eigen::Vector3d normal_add = cluster_tmp.normal_ + Z; 
        // std::cout << "normal_inc: " << normal_inc.norm() << ", normal_add: " << normal_add.norm() << std::endl;
        // put it in the cluster
        if(cluster_tmp.scatering_ > config_setting.scateringThres)
        {
            // center point and the direction of line (trunk)
            cluster_tmp.p_center_.x = cluster_tmp.center_[0];
            cluster_tmp.p_center_.y = cluster_tmp.center_[1];
            cluster_tmp.p_center_.z = cluster_tmp.center_[2];
            cluster_tmp.p_center_.normal_x = cluster_tmp.normal_[0];
            cluster_tmp.p_center_.normal_y = cluster_tmp.normal_[1];
            cluster_tmp.p_center_.normal_z = cluster_tmp.normal_[2];
            cluster_tmp.p_center_.intensity = 1;
                        
            cluster_tmp.is_line_ = false;
            disgard_clusters.push_back(cluster_tmp);
            continue;
        }
        else if(cluster_tmp.linearity_ > config_setting.linearityThres && 
                (normal_inc.norm()< config_setting.upThres || normal_add.norm()<config_setting.upThres) &&
                (cluster_tmp.maxZ - cluster_tmp.minZ) > config_setting.clusterHeight)
        {
            // center point and the direction of line (trunk)
            cluster_tmp.p_center_.x = cluster_tmp.center_[0];
            cluster_tmp.p_center_.y = cluster_tmp.center_[1];
            cluster_tmp.p_center_.z = cluster_tmp.center_[2];
            cluster_tmp.p_center_.normal_x = cluster_tmp.normal_[0];
            cluster_tmp.p_center_.normal_y = cluster_tmp.normal_[1];
            cluster_tmp.p_center_.normal_z = cluster_tmp.normal_[2];
            cluster_tmp.p_center_.intensity = 1;
                        
            cluster_tmp.is_line_ = true;
            clusters.push_back(cluster_tmp);
            // line++;
        }
        else
        {
            // center point and the direction of line (trunk)
            cluster_tmp.p_center_.x = cluster_tmp.center_[0];
            cluster_tmp.p_center_.y = cluster_tmp.center_[1];
            cluster_tmp.p_center_.z = cluster_tmp.center_[2];
            cluster_tmp.p_center_.normal_x = cluster_tmp.normal_[0];
            cluster_tmp.p_center_.normal_y = cluster_tmp.normal_[1];
            cluster_tmp.p_center_.normal_z = cluster_tmp.normal_[2];
            cluster_tmp.p_center_.intensity = 1;
                        
            cluster_tmp.is_line_ = false;
            disgard_clusters.push_back(cluster_tmp);
        }
    }
    // std::cout << "Line: " << line << ", Plane: " << plane << std::endl;
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
						pcl::PointCloud<pcl::PointXYZ>::Ptr &croped_tree)
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
			if(std::abs(tree_points->points[i].z - near_ground_point.z) < 2 &&
				std::abs(tree_points->points[i].z - near_ground_point.z) > 1)
			{
				croped_tree->push_back(tree_points->points[i]);
			}
		}
	}
}

void estimateNormals(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    pcl::PointCloud<pcl::Normal>::Ptr& normals,
    double radius)
{
    // 创建法向量估计对象
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    // 创建KD树用于近邻搜索
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);

    // 设置搜索半径
    ne.setRadiusSearch(radius);

    // 计算法向量
    normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*normals);
}

// fit the trunk by PCL ransac (cylinder)
bool pcl_ransac_cylinder(Cluster& source, cylinderConfig& fitting_config)
{
    if(source.points_.size() < fitting_config.cyliner_point_num)
    {
        source.is_cylinder_ = false;
        return false;
    }

    // get the point and normal data
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_points(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr tmp_normal(new pcl::PointCloud<pcl::Normal>());
    for(int i=0; i<source.points_.size(); i++)
    {
        pcl::PointXYZ pi;
        pcl::Normal Ni;
        
        pi.x = source.points_[i].x;
        pi.y = source.points_[i].y;
        pi.z = source.points_[i].z;
        
        tmp_points->push_back(pi);
    }
	estimateNormals(tmp_points, tmp_normal, 0.3);

    // ransac segmentation
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(fitting_config.model_dist_thres);  // inlier point to model distance
    seg.setMaxIterations(fitting_config.iteration_num);
    seg.setRadiusLimits(fitting_config.min_radius, fitting_config.max_radius); // limit the cylinder radius
    seg.setOptimizeCoefficients(fitting_config.is_opti_cylinder_coeff);
    seg.setInputCloud(tmp_points);
    seg.setInputNormals(tmp_normal);
    
    // perform the segmentation
    // coefficients is consist of 7 parameters, the first three is the axis of cylinder (0-2), 
    // the second three is a point on the axis (3-5), and the last is the radius of cylineder (6) 
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    seg.segment(*inliers, *coefficients);
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

