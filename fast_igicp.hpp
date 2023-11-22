#ifndef FAST_GICP_FAST_IGICP_HPP
#define FAST_GICP_FAST_IGICP_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/search.h>
#include <pcl/registration/registration.h>
#include <fast_gicp/gicp/lsq_registration.hpp>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {

/**
 * @brief Fast GICP algorithm optimized for multi threading with OpenMP
 */
template<typename PointSource, typename PointTarget, typename SearchMethodSource = pcl::search::KdTree<PointSource>, typename SearchMethodTarget = pcl::search::KdTree<PointTarget>>
class FastIGICP : public LsqRegistration<PointSource, PointTarget> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
  using Ptr = pcl::shared_ptr<FastIGICP<PointSource, PointTarget>>;
  using ConstPtr = pcl::shared_ptr<const FastIGICP<PointSource, PointTarget>>;
#else
  using Ptr = boost::shared_ptr<FastIGICP<PointSource, PointTarget>>;
  using ConstPtr = boost::shared_ptr<const FastIGICP<PointSource, PointTarget>>;
#endif

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::reg_name_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;

public:
  FastIGICP();
  virtual ~FastIGICP() override;

  void setNumThreads(int n);
  void setCorrespondenceRandomness(int k);
  void setRegularizationMethod(RegularizationMethod method);

  virtual void swapSourceAndTarget() override;
  virtual void clearSource() override;
  virtual void clearTarget() override;

  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;
  virtual void setSourceCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs);
  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;
  virtual void setTargetCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs);

  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& getSourceCovariances() const {
    return source_covs_;
  }

  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& getTargetCovariances() const {
    return target_covs_;
  }

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

  virtual void update_correspondences(const Eigen::Isometry3d& trans);

  virtual double linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) override;

  virtual double compute_error(const Eigen::Isometry3d& trans) override;

  template<typename PointT>
  // bool calculate_covariances(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, pcl::search::Search<PointT>& kdtree, std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances);
  bool calculate_covariances(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, pcl::search::KdTree<PointT>& kdtree, std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances_org,std::vector<double>& intensity_mean,std::vector<double>& intensity_cov,std::vector<double>& weight_distance, std::vector<Eigen::Vector4d>& singularValues_vector);
  
  
protected:
  int K_nest;
  double P_tao;
  double P_mu;
  int near_num_intensity;
  int alpha;

  int num_threads_;
  int k_correspondences_;

  RegularizationMethod regularization_method_;

  std::shared_ptr<SearchMethodSource> search_source_;
  std::shared_ptr<SearchMethodTarget> search_target_;

  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> source_covs_;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> target_covs_;

  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mahalanobis_;

  std::vector<int> correspondences_;
  std::vector<float> sq_distances_;

  std::shared_ptr<pcl::search::KdTree<PointSource>> source_kdtree_;
  std::shared_ptr<pcl::search::KdTree<PointTarget>> target_kdtree_;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> source_covs_org_;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> target_covs_org_;
  std::vector<double> distance_source_weight_;
  std::vector<double> distance_target_weight_;
  std::vector<Eigen::Vector4d> singularValues_vector_source_;
  std::vector<Eigen::Vector4d> singularValues_vector_target_;
  std::vector<double> intensity_source_mean_;
  std::vector<double> intensity_source_cov_;
  std::vector<double> intensity_target_mean_;
  std::vector<double> intensity_target_cov_;
  std::vector<double> Cost_weight;
  std::vector<double> intensity_err_;
  std::vector<double> intensity_weight_;

};
}  // namespace fast_gicp

#endif
