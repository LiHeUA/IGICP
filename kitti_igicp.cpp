#include <chrono>
#include <iostream>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/circular_buffer.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <fast_gicp/gicp/fast_igicp.hpp>
#include <fast_gicp/gicp/impl/lsq_registration_impl.hpp>
#include<time.h> 
#include <dirent.h>

#ifdef USE_VGICP_CUDA
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif

using namespace std;

void GetFileNames(std::string path,std::vector<std::string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str())))
        return;
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
            filenames.push_back(path + "/" + ptr->d_name);
    }
    closedir(pDir);
}

bool cmp(string a, string b)
{
  std::vector<std::string> a_compare;
  std::vector<std::string> b_compare;
  std::string a_single="";
  std::string b_single="";
  for(int i=0;i<a.size();i++)
  {
    if(a[i]<='9'&&a[i]>='0')
    {
        a_single+=a[i];
    }
    else
    {
      if(a_single.size())
        {
          a_compare.push_back(a_single);
          a_single="";
        }
    }
	}
  for(int i=0;i<b.size();i++)
  {
    if(b[i]<='9'&&b[i]>='0')
    {
        b_single+=b[i];
    }
    else
    {
      if(b_single.size())
        {
          b_compare.push_back(b_single);
          b_single="";
        }
    }
	}
  int num_com;
  if(a_compare.size()<b_compare.size())
  {
    num_com=a_compare.size();
  }
  else
  {
    num_com=b_compare.size();
  }
  for(int i=0;i<num_com;i++)
  {
    int a_num = atoi(a_compare[i].c_str());
    int b_num  = atoi(b_compare[i].c_str());
    if(a_num==b_num)
    {
      continue;
    }
    else
    {
      return a_num<b_num;
    }
  }
  return a_compare.size()<b_compare.size();
}

int get_pcd_name(std::string path, std::vector<std::string>& filenames)
{   
  std::vector<std::string> filename_single;
	struct dirent **entry_list;
	int count;
	char* p = (char*)path.c_str();
	count = scandir(p, &entry_list, 0, alphasort);
	if (count < 0) 
	{
    perror("scandir");
    return EXIT_FAILURE;
	}
	for (int i = 0; i < count; i++) 
	{
    struct dirent *entry;
    entry = entry_list[i];
		if(strcmp(entry->d_name,".") != 0  &&  strcmp(entry->d_name,"..") != 0)
		{
      filename_single.push_back(entry->d_name);
		}
    free(entry);
	}
  free(entry_list);
  sort(filename_single.begin(), filename_single.end(), cmp);
  for (int i = 0; i < filename_single.size(); i++) 
	{
        filenames.push_back(path+"/" +filename_single[i]);
	}
	return filenames.size();
}

class KittiLoader {
public:
  KittiLoader(const std::string& dataset_path) : dataset_path(dataset_path) 
  {
    get_pcd_name(dataset_path,filenames);
    num_frames=filenames.size();
    if (num_frames == 0) 
    {
      std::cerr << "error: no files in " << dataset_path << std::endl;
    }
  }
  ~KittiLoader() {}

  size_t size() const { return num_frames; }

  pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud(size_t i) 
  {
    int begin_id=filenames[i].find_last_of('/');
    int end_id=filenames[i].find_last_of('.');
    time_each_pcd = filenames[i].substr(begin_id+1,end_id-begin_id-1);
    FILE* file = fopen(filenames[i].c_str(), "rb");
    if (!file) 
    {
      std::cerr << "error: failed to load " << filenames[i] << std::endl;
      return nullptr;
    }

    std::vector<float> buffer(1000000);
    size_t num_points = fread(reinterpret_cast<char*>(buffer.data()), sizeof(float), buffer.size(), file) / 4;
    fclose(file);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    
    for (int i = 0; i < num_points; i++) 
    {
      pcl::PointXYZI pt;
      pt.x = buffer[i * 4];
      pt.y = buffer[i * 4 + 1];
      pt.z = buffer[i * 4 + 2];
      pt.intensity = buffer[i * 4 + 3];
      if(isnan(pt.x)||isnan(pt.y)||isnan(pt.z))
      {
          continue;
      }
      if((pt.x==0)&&(pt.y==0)&&(pt.z==0))
      {
          continue;
      }
      cloud->push_back(pt); 
    }

    return cloud;
  }
public:
  string time_each_pcd;
private:
  int num_frames;
  std::vector<std::string> filenames;
  std::string dataset_path;
};

int main(int argc, char** argv) 
{
  // file path, you need to change
  std::string SourceDataPath="./src/fast_gicp/data/test/source_data/07";
  std::string TrajSavePath="./src/fast_gicp/data/test/result/traj.txt";
  std::cout << "SourceDataPath:"<<SourceDataPath << std::endl;
  std::vector<std::string> filename;

  KittiLoader kitti(SourceDataPath);

  // use downsample_resolution=1.0 for fast registration
  double downsample_resolution = 0.25; //0.25
  pcl::ApproximateVoxelGrid<pcl::PointXYZI> voxelgrid;
  voxelgrid.setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);

  // registration method
  // you should fine-tune hyper-parameters (e.g., voxel resolution, max correspondence distance) for the best result
  // fast_gicp::FastGICP<pcl::PointXYZI , pcl::PointXYZI> gicp;
  fast_gicp::FastIGICP<pcl::PointXYZI , pcl::PointXYZI> gicp;
  // fast_gicp::FastVGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
  // fast_gicp::FastVGICPCuda<pcl::PointXYZ, pcl::PointXYZ> gicp;
  // fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ> gicp;
  // pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;

  // gicp.setResolution(1.0);
  // gicp.setNearestNeighborSearchMethod(fast_gicp::NearestNeighborMethod::GPU_RBF_KERNEL);
  gicp.setMaxCorrespondenceDistance(4);//4
  gicp.setNumThreads(16);

  // set initial frame as target
  voxelgrid.setInputCloud(kitti.cloud(0));
  pcl::PointCloud<pcl::PointXYZI>::Ptr target(new pcl::PointCloud<pcl::PointXYZI>);
  voxelgrid.filter(*target);

  gicp.setInputTarget(target);

  // sensor pose sequence
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses(kitti.size());
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses_gt(kitti.size());
  poses[0].setIdentity();
  
  // trajectory for visualization
  pcl::PointCloud<pcl::PointXYZ>::Ptr trajectory(new pcl::PointCloud<pcl::PointXYZ>);
  trajectory->push_back(pcl::PointXYZ(0.0f, 0.0f, 0.0f));

  pcl::visualization::PCLVisualizer vis; 
  vis.addPointCloud<pcl::PointXYZ>(trajectory, "trajectory");

  // for calculating FPS
  boost::circular_buffer<std::chrono::high_resolution_clock::time_point> stamps(30);//30
  stamps.push_back(std::chrono::high_resolution_clock::now());

  double single_fps=0;
  double average_fps=0;
  double curcle_time=0;
  double curcle_total_time=0;
  for (int i = 1; i < kitti.size(); i++) 
  {
    // set the current frame as source
    voxelgrid.setInputCloud(kitti.cloud(i));
    pcl::PointCloud<pcl::PointXYZI>::Ptr source(new pcl::PointCloud<pcl::PointXYZI>);
    voxelgrid.filter(*source);
    gicp.setInputSource(source);

    // align and swap source and target cloud for next registration
    pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>);
    gicp.align(*aligned);
    gicp.swapSourceAndTarget();

    // accumulate pose
    poses[i] = poses[i - 1] * gicp.getFinalTransformation().cast<double>();

    // // FPS display
    // stamps.push_back(std::chrono::high_resolution_clock::now());
    // single_fps=stamps.size() / (std::chrono::duration_cast<std::chrono::nanoseconds>(stamps.back() - stamps.front()).count() / 1e9);
    // std::cout << single_fps << "fps" << std::endl;
    // average_fps+=single_fps;

    // visualization
    trajectory->push_back(pcl::PointXYZ(poses[i](0, 3), poses[i](1, 3), poses[i](2, 3)));
    vis.updatePointCloud<pcl::PointXYZ>(trajectory, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(trajectory, 255.0, 0.0, 0.0), "trajectory");
    vis.spinOnce();

    curcle_time=1/single_fps;//
    curcle_total_time+=curcle_time;
  }
  // average_fps=average_fps/kitti.size();
  // std::cout << "average_fps:"<<average_fps  << std::endl;
  // std::cout << "curcle_total_time:"<<curcle_total_time  << std::endl;
  
  // save the estimated poses
  std::ofstream ofs(TrajSavePath);
  for ( auto& pose : poses) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4; j++) {
        if (i || j) {
          ofs << " ";
        }

        ofs << pose(i, j);
      }
    }
    ofs << std::endl;
  }
  std::cout << "Trajectory results are saved at: "<<TrajSavePath  << std::endl;

  while (!vis.wasStopped()) 
  {
      vis.spinOnce();
  }
  
  return 0;
}