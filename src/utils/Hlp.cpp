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

// down sample the point cloud, by voxel
void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZ> &pl_feat, double voxel_size) 
{
	int pointsNum = pl_feat.size();
	if (voxel_size < 0.01) 
		return;

	// a container, include key and value
	std::unordered_map<UNI_VOXEL_LOC, M_POINT> voxel_map;
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

// calculate the accuracy of pose
void accur_evaluation(std::pair<Eigen::Vector3d, Eigen::Matrix3d> esti, Eigen::Affine3d truth,
					 std::pair<Eigen::Vector3d, Eigen::Vector3d> &errors)
{
	Eigen::Vector3d trans_error;
	Eigen::Vector3d rotation_error;
	
	// // translation error
	// trans_error = esti.first - truth.translation();
	// // rotation error
	// Eigen::Vector3d estiEuler=esti.second.eulerAngles(2,1,0);
	// Eigen::Vector3d truthEuler=truth.rotation().eulerAngles(2,1,0);
	// rotation_error = estiEuler - truthEuler;


	// trans the pose type into the pair vector and matrix
	std::pair<Eigen::Vector3d, Eigen::Matrix3d> truth_pose;
	truth_pose.first = truth.translation();
	truth_pose.second = truth.rotation();
	// calculate the error
	std::pair<Eigen::Vector3d, Eigen::Matrix3d> esti_inv = matrixInv(esti);
	std::pair<Eigen::Vector3d, Eigen::Matrix3d> esti_err = matrixMultip(esti_inv, truth_pose);
	// translation and rotation error
	trans_error = esti_err.first;
	rotation_error = esti_err.second.eulerAngles(2,1,0);

	for(int i=0; i<3; i++)
	{
		if(abs(rotation_error[i] - 2*M_PI) < 0.3)
			rotation_error[i] -= 2*M_PI;

		if(abs(rotation_error[i] + 2*M_PI) < 0.3)
			rotation_error[i] += 2*M_PI;
		
		if(abs(rotation_error[i] - M_PI) < 0.3)
			rotation_error[i] -= M_PI;

		if(abs(rotation_error[i] + M_PI) < 0.3)
			rotation_error[i] += M_PI;
	}
	
	// use the degree to evoluate the rotation error
	// rotation_error = rotation_error * (180/M_PI);
	
	// use the mrad to evoluate the rotation error
	rotation_error = rotation_error * 1000;
	
	errors.first = trans_error;
	errors.second = rotation_error;
	// std::cout << "trans: \n" << trans_error << "\nrotation: \n" << rotation_error << std::endl;
}
// write the accuracy result
void write_error(std::string filePath, std::pair<Eigen::Vector3d, Eigen::Vector3d> &errors)
{
	Eigen::Vector3d trans_error = errors.first;
	Eigen::Vector3d rotation_error = errors.second;

	std::ofstream outfile(filePath);
	if (!outfile.is_open()) 
	{
		std::cerr << "failed open this file!" << std::endl;
		return;
	}
	outfile << trans_error[0] << ", " << trans_error[1] << ", " << trans_error[2] << ", ";
	outfile << rotation_error[0] << ", " << rotation_error[1] << ", " << rotation_error[2] << std::endl;
	outfile.close();
}

// calculate the accuracy vector
void accur_evaluation_vec(std::vector<TLSPos> esti, std::vector<Eigen::Affine3d> truh, std::vector<PosError> &errors)
{
	std::pair<Eigen::Vector3d, Eigen::Vector3d> err;
	int ID;
	PosError pe;

	for(int i=0; i<esti.size(); i++)
	{
		// get the station ID
		ID = esti[i].ID;
		
		// get the estimated pose
		std::pair<Eigen::Vector3d, Eigen::Matrix3d> esti_pose;
		esti_pose.first = esti[i].t;
		esti_pose.second = esti[i].R;
		
		// evaluaion
		accur_evaluation(esti_pose, truh[ID], err);
		
		// get the ID and evaluation result
		pe.ID = ID;
		pe.error = err;
		errors.push_back(pe);
	}
}

// write the result vector
void write_error_vec(std::string filePath, std::vector<PosError> &errors)
{
	std::ofstream outfile(filePath);
	if (!outfile.is_open()) 
	{
		std::cerr << "failed open this file!" << std::endl;
		return;
	}
	
	for(int i=0; i<errors.size(); i++)
	{
		int id = errors[i].ID;
		Eigen::Vector3d trans_error = errors[i].error.first;
		Eigen::Vector3d rotation_error = errors[i].error.second;

		outfile << "Station: " << id << std::endl;
		outfile << trans_error[0] << ", " << trans_error[1] << ", " << trans_error[2] << ", ";
		outfile << rotation_error[0] << ", " << rotation_error[1] << ", " << rotation_error[2] << std::endl;
	}
	outfile.close();
}

void write_pose(std::string filePath, std::vector<TLSPos> poses)
{
	std::ofstream outfile(filePath);
	if (!outfile.is_open()) 
	{
		std::cerr << "failed open this file: " << filePath << std::endl;
		return;
	}
	for(int i=0; i<poses.size(); i++)
	{
		int id = poses[i].ID;
		Eigen::Matrix3d R = poses[i].R;
		Eigen::Vector3d t = poses[i].t;
		outfile << "Station: " << id << std::endl;
		outfile << t[0] << ", " << t[1] << ", " << t[2] << std::endl;
		outfile << R << std::endl << std::endl;
	}
	outfile.close();
}

void write_relative_pose(std::string filePath, std::pair<Eigen::Vector3d, Eigen::Matrix3d> poses)
{
	std::ofstream outfile(filePath);
	if (!outfile.is_open()) 
	{
		std::cerr << "failed open this file: " << filePath << std::endl;
		return;
	}
	outfile << "Trans: \n" << poses.first << std::endl;
	outfile << "Rot: \n" << poses.second << std::endl;
	
	outfile.close();
}

// get the absolut pose by relative pose between nodes
void RelaToAbs(std::vector<CandidateInfo> &candidates_vec, std::vector<TLSPos> &tlsVec)
{
    for(int num=0; num<2; num++)
	{
		// calculated the initial pos of each TLS station
		for(int i=0; i<candidates_vec.size(); i++)
		{
			TLSPos currPos;
			int candtlsID, tlsID = candidates_vec[i].currFrameID;
			// set the first station
			if(tlsID == 0)
			{
				currPos.ID = tlsID;
				currPos.R = Eigen::Matrix3d::Identity();
				currPos.t = Eigen::Vector3d::Zero();
				currPos.isValued = true;
				tlsVec[tlsID] = currPos;
			}
			// loop the candiates of each station, and calculate the pose
			for(int j=0; j<candidates_vec[i].candidateIDScore.size(); j++)
			{
				candtlsID = candidates_vec[i].candidateIDScore[j].first;
				// from currt to candidiate pose
				if(tlsVec[tlsID].isValued && !tlsVec[candtlsID].isValued)
				{
					currPos.ID = candtlsID;
					currPos.R = tlsVec[tlsID].R * candidates_vec[i].relativePose[j].second.inverse();
					currPos.t = tlsVec[tlsID].t - 
								tlsVec[tlsID].R * candidates_vec[i].relativePose[j].second.inverse() * candidates_vec[i].relativePose[j].first;
					currPos.isValued = true;
					tlsVec[candtlsID] = currPos;
				}
				// from candidiate to currt
				if(!tlsVec[tlsID].isValued && tlsVec[candtlsID].isValued)
				{
					currPos.ID = tlsID;
					currPos.R = tlsVec[candtlsID].R * candidates_vec[i].relativePose[j].second;
					currPos.t = tlsVec[candtlsID].R * candidates_vec[i].relativePose[j].first + tlsVec[candtlsID].t;
					currPos.isValued = true;
					tlsVec[tlsID] = currPos;
				}
			}
		}
	}
}

// get the absolut pose (DFS)
void AbsByDFS(int current_node, TLSPos &current_pose, std::vector<CandidateInfo> &candidates_vec, 
				std::vector<TLSPos> &tlsVec, std::unordered_set<int>& visited)
{
	visited.insert(current_node);
	tlsVec[current_node] = current_pose;
	for (int i=0; i<candidates_vec[current_node].candidateIDScore.size(); i++)
	{
		if (visited.find(candidates_vec[current_node].candidateIDScore[i].first) == visited.end())
		{
			int targetID = candidates_vec[current_node].candidateIDScore[i].first;
			TLSPos newPose;
			newPose.ID = targetID;
			newPose.isValued = true;
			newPose.R = current_pose.R * candidates_vec[current_node].relativePose[i].second.inverse();
			newPose.t = current_pose.t - 
				current_pose.R * candidates_vec[current_node].relativePose[i].second.inverse() * candidates_vec[current_node].relativePose[i].first;

			AbsByDFS(targetID, newPose, candidates_vec, tlsVec, visited);
		}
	}
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> matrixInv(std::pair<Eigen::Vector3d, Eigen::Matrix3d> m1)
{
	std::pair<Eigen::Vector3d, Eigen::Matrix3d> res;
	
	res.second = m1.second.inverse();
	res.first = -m1.second.inverse() * m1.first;

	return res;
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> matrixMultip(std::pair<Eigen::Vector3d, Eigen::Matrix3d> m1, 
                                                        std::pair<Eigen::Vector3d, Eigen::Matrix3d> m2)
{
	std::pair<Eigen::Vector3d, Eigen::Matrix3d> res;

	res.second = m1.second * m2.second;
	res.first = m1.second*m2.first + m1.first;

	return res;
}

