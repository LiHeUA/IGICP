#ifndef FAST_GICP_FAST_IGICP_IMPL_HPP
#define FAST_GICP_FAST_IGICP_IMPL_HPP

#include <fast_gicp/so3/so3.hpp>

/*
This file includes the main body of IGICP [1]. Line 22~26 set the parameters for IGICP. Function update_correspondences corresponds to Sec. III B~D, the 
correspondence selection part. As to Sec. III E Weighted Cost Function, we calculate the weight \omega_i in Eq. (16) of [1] in update_correspondences.
In function calculate_covariances, we first calibrate the intensity values and then calcualte the variance of intensity. 

We appreciate the fast_gicp source code (https://github.com/SMRT-AIST/fast_gicp) providers and build our system on fast_gicp framework. 

[1] Li He, Wen Li, Yisheng Guan, and Hong Zhang, IGICP: Intensity and Geometry Enhanced LiDAR Odometry, IEEE Transactions on Intelligent Vehicles, to appear

hel@sustech.edu.cn
2112101119@mail2.gdut.edu.cn

Nov. 22, 2023
*/

namespace fast_gicp {

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::FastIGICP() {
  /*parameters*/
  K_nest=5;                 //Nearest Neighbors for searching correspondence
  P_tao=60;                 //Gaussian scale parameter for normalizing KL distance, Eq. (10) of [1]
  P_mu=0;                   //Gaussian scale parameter for normalize KL distance
  near_num_intensity=5;     //Nearest Neighbors for Calculating Intensity Distributions
  alpha=5;                  //gain constant in the vector of local geometry features

#ifdef _OPENMP
  num_threads_ = omp_get_max_threads();
#else
  num_threads_ = 1;
#endif
  k_correspondences_ = 20;
  reg_name_ = "FastIGICP";
  corr_dist_threshold_ = std::numeric_limits<float>::max();

  regularization_method_ = RegularizationMethod::PLANE;
  search_source_.reset(new SearchMethodSource);
  search_target_.reset(new SearchMethodTarget);
  source_kdtree_.reset(new pcl::search::KdTree<PointSource>);
  target_kdtree_.reset(new pcl::search::KdTree<PointTarget>);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::~FastIGICP() {}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setNumThreads(int n) {
  num_threads_ = n;

#ifdef _OPENMP
  if (n == 0) {
    num_threads_ = omp_get_max_threads();
  }
#endif
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setCorrespondenceRandomness(int k) {
  k_correspondences_ = k;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  search_source_.swap(search_target_);
  source_covs_.swap(target_covs_);

  /*swap weight, vector and intensity information*/
  source_kdtree_.swap(target_kdtree_);
  source_covs_org_.swap(target_covs_org_);
  distance_source_weight_.swap(distance_target_weight_);
  singularValues_vector_source_.swap(singularValues_vector_target_);
  intensity_source_mean_.swap(intensity_target_mean_);
  intensity_source_cov_.swap(intensity_target_cov_);
  /***************************/

  correspondences_.clear();
  sq_distances_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::clearSource() {
  input_.reset();
  source_covs_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::clearTarget() {
  target_.reset();
  target_covs_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  search_source_->setInputCloud(cloud);
  source_covs_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    return;
  }
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  search_target_->setInputCloud(cloud);
  target_kdtree_->setInputCloud(cloud);
  target_covs_.clear();
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setSourceCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  source_covs_ = covs;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::setTargetCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  target_covs_ = covs;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {
  if (output.points.data() == input_->points.data() || output.points.data() == target_->points.data()) {
    throw std::invalid_argument("FastIGICP: destination cloud cannot be identical to source or target");
  }
  /*Calculate the covariance of intensity and points*/
  if (source_covs_.size() != input_->size()) {
    calculate_covariances(input_, *source_kdtree_, source_covs_,source_covs_org_,intensity_source_mean_,intensity_source_cov_,distance_source_weight_,singularValues_vector_source_);
  }
  if (target_covs_.size() != target_->size()) {
    calculate_covariances(target_, *target_kdtree_, target_covs_,target_covs_org_,intensity_target_mean_,intensity_target_cov_,distance_target_weight_,singularValues_vector_target_);
  }
  /*************************/

  LsqRegistration<PointSource, PointTarget>::computeTransformation(output, guess);
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
void FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  assert(source_covs_.size() == input_->size());
  assert(target_covs_.size() == target_->size());
  Eigen::Isometry3f trans_f = trans.cast<float>();
  correspondences_.resize(input_->size());
  mahalanobis_.resize(input_->size());
  Cost_weight.resize(input_->size());
  intensity_err_.resize(input_->size());
  intensity_weight_.resize(input_->size());
  std::vector<int> k_indices(K_nest);
  std::vector<float> k_sq_dists(K_nest);

#pragma omp parallel for num_threads(num_threads_) firstprivate(k_indices, k_sq_dists) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    PointTarget pt;
    pt.getVector4fMap() = trans_f * input_->at(i).getVector4fMap();
    
    target_kdtree_->nearestKSearch(pt, K_nest, k_indices, k_sq_dists);
    double intensity_big=100;
    double intensity_small=-1;
    bool correspondences_true=false;

    const auto& intensity_cov_A = intensity_source_cov_[i];
    const auto& intensity_mean_A = intensity_source_mean_[i];
    double weight_A=distance_source_weight_[i];
    Eigen::Vector4d source_sigval_A=trans.matrix()*singularValues_vector_source_[i];
    if(weight_A<0.2)
    {
      correspondences_[i] =-1;
      continue;
    }

    /*Intensity and Geometry Based Similarity Matching */
    for(int j = 0; j < K_nest; j++)
    {
      if(k_sq_dists[j] >corr_dist_threshold_ * corr_dist_threshold_)
      {
        continue;
      }
      int correspondences_small=  k_indices[j];
      double weight_B=distance_target_weight_[correspondences_small];
      if(weight_B<0.2)
      {
        continue;
      }
      const auto& intensity_cov_B = intensity_target_cov_[correspondences_small];
      const auto& intensity_mean_B = intensity_target_mean_[correspondences_small];
      Eigen::Vector4d source_sigval_B=singularValues_vector_target_[correspondences_small];

      /*geometry similarity*/ 
      double Sg = abs(source_sigval_A.dot(source_sigval_B) / (source_sigval_A.norm()*source_sigval_B.norm()));

      /*intensity similarity*/ 
        //KL distance, Eq. (9)
      double intensity_KL_dis = (intensity_cov_A*intensity_cov_A+(intensity_mean_A-intensity_mean_B)*(intensity_mean_A-intensity_mean_B))/(4*intensity_cov_B*intensity_cov_B)
                        +(intensity_cov_B*intensity_cov_B+(intensity_mean_A-intensity_mean_B)*(intensity_mean_A-intensity_mean_B))/(4*intensity_cov_A*intensity_cov_A)
                        -1/2;
        //intensity similarity, Eq. (10)
      double Si=exp(-(1/(2*P_tao*P_tao))*(intensity_KL_dis)*(intensity_KL_dis));
        // final similarity;
      double simularity_each=Si*Sg;

      /*weight of planarity*/ 
      double weight_p=2*weight_A*weight_B/(weight_A+weight_B);

      /*Find the most similar matching and calculate the weight \omega_i in Eq. (16) of [1]*/ 
      if(simularity_each>intensity_small)
      {
        intensity_small=simularity_each;
        correspondences_true=true;
        Cost_weight[i]=simularity_each*weight_p; // \omega_i, Eq. (16), Sec III E of [1]
        correspondences_[i] =correspondences_small;
      }
    }
    if(correspondences_true)
    {
      /*Weighted Cost Function
      Cost_weight contains the weights, or \omega_i, in Eq. (16), Sec III E of [1], and is pushed to mahalanobis_ for optimization
      */
      const auto& cov_A = source_covs_[i];
      const auto& cov_B = target_covs_[correspondences_[i]];
      Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();
      RCR(3, 3) = 1.0;
      mahalanobis_[i] = Cost_weight[i]*RCR.inverse();
      mahalanobis_[i](3, 3) = 0.0f;
    }
    else
    {
      Cost_weight[i]=0;
      correspondences_[i] =-1;
    }
  }
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
double FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {
  update_correspondences(trans);

  double sum_errors = 0.0;
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    Hs[i].setZero();
    bs[i].setZero();
  }

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[i];

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs_[target_index];

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error =mean_B - transed_mean_A ;

    sum_errors += error.transpose() * mahalanobis_[i] * error;

    if (H == nullptr || b == nullptr) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

    Eigen::Matrix<double, 6, 6> Hi = jlossexp.transpose() * mahalanobis_[i] * jlossexp;
    Eigen::Matrix<double, 6, 1> bi = jlossexp.transpose() * mahalanobis_[i] * error;

    Hs[omp_get_thread_num()] += Hi;
    bs[omp_get_thread_num()] += bi;
  }

  if (H && b) {
    H->setZero();
    b->setZero();
    for (int i = 0; i < num_threads_; i++) {
      (*H) += Hs[i];
      (*b) += bs[i];
    }
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
double FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::compute_error(const Eigen::Isometry3d& trans) {
  double sum_errors = 0.0;

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[i];

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs_[target_index];

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    sum_errors += error.transpose() * mahalanobis_[i] * error;
  }

  return sum_errors;
}

template <typename PointSource, typename PointTarget, typename SearchMethodSource, typename SearchMethodTarget>
template <typename PointT>
bool FastIGICP<PointSource, PointTarget, SearchMethodSource, SearchMethodTarget>::calculate_covariances(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::KdTree<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances_org,
  std::vector<double>& intensity_mean,
  std::vector<double>& intensity_cov,
  std::vector<double>& weight_distance,
  std::vector<Eigen::Vector4d>& singularValues_vector) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }
  covariances.resize(cloud->size());
  covariances_org.resize(cloud->size());
  intensity_mean.resize(cloud->size());
  intensity_cov.resize(cloud->size());
  weight_distance.resize(cloud->size());
  singularValues_vector.resize(cloud->size());

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_intensity_calib(new pcl::PointCloud<pcl::PointXYZI> );
  *cloud_intensity_calib=*cloud;

  double intensity_sum=0;
  double intensity_sigma=0;

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);
    Eigen::Matrix<double, 4, -1> neighbors(4, k_correspondences_);
    for (int j = 0; j < k_indices.size(); j++) {
      neighbors.col(j) = cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / k_correspondences_;
    covariances_org[i] = cov;

    if (regularization_method_ == RegularizationMethod::NONE) {
      covariances[i] = cov;
    } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
      double lambda = 1e-3;
      Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d C_inv = C.inverse();
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
    } else {
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Vector3d values;
      switch (regularization_method_) {
        default:
          std::cerr << "here must not be reached" << std::endl;
          abort();
        case RegularizationMethod::PLANE:
          values = Eigen::Vector3d(1, 1, 1e-3);
          break;
        case RegularizationMethod::MIN_EIG:
          values = svd.singularValues().array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_MIN_EIG:
          values = svd.singularValues() / svd.singularValues().maxCoeff();
          values = values.array().max(1e-3);
          break;
      }
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
      
      /*vector of local geometry features*/
      double planer=(svd.singularValues()(1)-svd.singularValues()(2))/svd.singularValues()(0);
      weight_distance[i]=planer;
      singularValues_vector[i]=Eigen::Vector4d(svd.matrixU()(0, 2), svd.matrixU()(1, 2), svd.matrixU()(2, 2),alpha*svd.singularValues()(2));

      // intensity calibration, Eq. (13) of [1]
      Eigen::Vector3d point_normal;
      point_normal=svd.matrixU().col(2);
      Eigen::Vector3d point_position(cloud->at(i).x,cloud->at(i).y,cloud->at(i).z);
      double intensiry_sigle=cloud_intensity_calib->points[i].intensity;
      if(cloud_intensity_calib->points[i].intensity==0)
      {
        cloud_intensity_calib->points[i].intensity=0.01;
      }
      //calibration for incident angle
      double intensity_angle=intensiry_sigle/(abs(point_normal.dot(point_position))/(point_normal.norm()*point_position.norm()));
      
      // //calibration for distance. Typically, the raw LiDAR data are distance-calibrated, such as KITTI. So, for KITTI, we do not calibrate for distance on KITTI.
      // cloud_intensity_calib->points[i].intensity=intensiry_sigle*point_position.norm()*point_position.norm();

      // //calibration for both angle and distance
      // double intensity_angle=intensiry_sigle/(abs(point_normal.dot(point_position))/(point_normal.norm()*point_position.norm()));
      // cloud_intensity_calib->points[i].intensity=intensity_angle*point_position.norm()*point_position.norm();

      // if(cloud_intensity_calib->points[i].intensity>1)
      // {
      //   cloud_intensity_calib->points[i].intensity=1;
      // }
    }
  }

  // Calculate the mean and variance of intensity used in Eq. (9) of [1]
  for (int j = 0; j < cloud_intensity_calib->size(); j++) 
  {
    intensity_sum=0;
    intensity_sigma=0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    kdtree.nearestKSearch(cloud_intensity_calib->points[j], near_num_intensity, k_indices, k_sq_distances);
    //Calculate the mean and variance
    for (int k = 0; k < k_indices.size(); k++) 
    {
      intensity_sum+=cloud_intensity_calib->points[k_indices[k]].intensity;
      intensity_sigma+=cloud_intensity_calib->points[k_indices[k]].intensity*cloud_intensity_calib->points[k_indices[k]].intensity;
    }
    intensity_mean[j]=intensity_sum/near_num_intensity;
    intensity_cov[j]=intensity_sigma/near_num_intensity - intensity_mean[j]*intensity_mean[j];
    if(intensity_cov[j]==0)
    {
      intensity_cov[j]=0.0001;
    }
  }
  return true;
}

} 
#endif
