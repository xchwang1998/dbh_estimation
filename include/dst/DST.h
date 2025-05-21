
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// for hash function
#define HASH_P 116101
#define MAX_N 10000000000
#define MAX_FRAME_N 20000

struct GrpStaic
{
	long lGrpID;    // Group ID
	long lEleNum;   // Group element number
	long lStaID;	// Start element ID
	long lEndID;    // End element ID
};

struct ProfLabel
{
	long lXPrior;       //X Prior
	long lXPstor;		//X Postorior
	long lYPrior;       //Y Prior
	long lYPstor;		//Y Postorior
	long lLabel;        //Group ID
};	

// parameters for preprocessing, std, and place recognition
typedef struct ConfigSetting {
	
  std::string pcd_data = "/media/xiaochen/xch_disk/TreeScope_dataset/";
  std::string json_data = "/media/xiaochen/xch_disk/TreeScope_dataset/";
  
	// for CSF, ground filter
	bool bSloopSmooth = false;
	double cloth_resolution = 0.5;
	int rigidness = 3;
	double time_step = 0.65;
	double class_threshold = 0.5;
	int iterations = 500;  

	// for FEC Cluster
	int min_component_size = 50;
	double tolorance = 0.2;
	int max_n = 50;
	double merge_dist = 0.1;
	
	// for ground Cluster
	double gnd_grid_size = 0.5;
	double gnd_points_num = 5;
	double gnd_points_dist = 0.5;
	
	// for cluster constraint
  int clusterPointsNum = 100;
  double linearThres = 0.7;
  double planarThres = 0.3;
  double scaterThres = 0.3;
  double directThres = 0.7;
  double heightThres = 2.0;
  double horizonThres = 5.0;

  // crop tree points
  double min_height = 0.1;
  double max_height = 2.0;
} ConfigSetting;

// for point to image
struct POINT_D {
  pcl::PointXYZ p;
  double d = -1;
};

// structure for Cluster
typedef struct Cluster {
  int label;
  pcl::PointXYZINormal p_center_;
  pcl::PointCloud<pcl::PointXYZ> points_;
  pcl::PointCloud<pcl::PointXYZINormal> point_normal_;
  Eigen::Vector3d center_;
  Eigen::Vector3d normal_;
  Eigen::Matrix3d covariance_;
  Eigen::Vector3d eig_value_;
  double minZ;
  double maxZ;
  double root;
  
  double linearity_ = 0;
  double planarity_ = 0;
  double scatering_ = 0;


  bool is_cylinder_ = false;
  pcl::ModelCoefficients cylinder_coff;

  bool dbh_flag = false;
  pcl::ModelCoefficients dbh_coff;

  Eigen::Vector3d min_point;
  Eigen::Vector3d max_point;
  double gnd_height = 0;
  std::vector<int> rgb;
} Cluster;

// fitting cylinder parameters
typedef struct cylinderConfig {
    int cyliner_point_num = 50;
    double model_dist_thres = 0.2;
    int iteration_num = 2000;
    double min_radius = 1;
    double max_radius = 0.025;
    bool is_opti_cylinder_coeff = true;
} cylinderConfig;

// fitting circle parameters
typedef struct circleConfig {
    int circle_point_num;
    double circle_model_dist_thres;
    int circle_iteration_num;
    double circle_min_radius;
    double circle_max_radius;
    bool is_opti_coeff;

    // split the profile, use the fix num or use the fix step
    bool use_num_or_step;
    int profile_num;
    double profile_step;
} circleConfig;


// location of STD, and a operator to define whether is equal or not
class TriDesc_LOC {
public:
  int64_t x, y, z, a, b, c;

  TriDesc_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0, int64_t va = 0,
             int64_t vb = 0, int64_t vc = 0)
      : x(vx), y(vy), z(vz), a(va), b(vb), c(vc) {}

  bool operator==(const TriDesc_LOC &other) const {
    // use three attributes
    return (x == other.x && y == other.y && z == other.z);
    // use six attributes
    // return (x == other.x && y == other.y && z == other.z && a == other.a &&
    //         b == other.b && c == other.c);
  }
};

// hash mapping for TriDesc_LOC, input TriDesc_LOC, output int64
template <> struct std::hash<TriDesc_LOC> {
  int64_t operator()(const TriDesc_LOC &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};

// location of uniform voxel
class UNI_VOXEL_LOC {
public:
    int64_t x, y, z;

    UNI_VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
        : x(vx), y(vy), z(vz) {}

    bool operator==(const UNI_VOXEL_LOC &other) const {
        return (x == other.x && y == other.y && z == other.z);
    }
};

// hash mapping for VOXEL_LOC, input VOXEL_LOC, output int64
template <> struct std::hash<UNI_VOXEL_LOC> {
  int64_t operator()(const UNI_VOXEL_LOC &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};

// for down sample function
struct M_POINT {
  float xyz[3];
  int count = 0;
};