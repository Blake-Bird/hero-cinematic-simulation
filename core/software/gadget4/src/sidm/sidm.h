/* Native elastic SIDM differential-table interface. Rates use proper cgs
 * quantities; interpolation is linear in cosine and logarithmic in speed.
 * A retained differential interpolant never silently changes convention.
 */
#ifndef RESEARCH_NATIVE_ELASTIC_SIDM_H
#define RESEARCH_NATIVE_ELASTIC_SIDM_H
#include "research_angular_transport.h"
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <memory>

namespace research_sidm {
class differential_table
{
 public:
  static differential_table read(const std::string &path,const std::string &model_hash,const std::string &table_sha256);
  double total_cross_section(double speed_kms) const;
  double differential_cross_section(double speed_kms,double cosine) const;
  double sample_cosine(double speed_kms,double uniform,double minimum_angle,bool folded) const;
  angular_transport_moments moments(double speed_kms,double split_angle,bool folded) const;
  long double maximum_sigma_speed(long double speed_upper_kms) const;
  const std::string &model_hash() const {return model_hash_;}
  const std::string &table_hash() const {return table_hash_;}
  double minimum_speed() const {return velocities_.front();}
  double maximum_speed() const {return velocities_.back();}
  std::uint64_t memory_bytes() const {return std::uint64_t(velocities_.capacity()+cosines_.capacity()+differential_.capacity())*sizeof(double);}
 private:
  std::vector<double> row(double speed) const;
  std::vector<double> velocities_,cosines_,differential_;
  int low_policy_=0,high_policy_=0;
  double absolute_tolerance_=0,relative_tolerance_=0;
  std::string model_hash_,table_hash_;
};
const differential_table *native_differential_model();
void prepare_native_differential_model(int angular_kind,const char *path,const char *model_hash,const char *table_hash);
double native_total_cross_section(int angular_kind,double normalization,double speed,double screening_velocity);
long double native_sigma_speed_bound(int angular_kind,long double normalization,long double speed_upper,long double screening_velocity);

struct dm_neighbor_point
{
  std::uint64_t id;
  std::size_t source_index;
  std::array<double,3> position_cm;
  double support_cm;
};
/* Dedicated DM tree uses GADGET4 basenode nextnode/sibling topology. Its point
 * population is fine/coarse DM support ghosts, independent of gas SPH lists.
 */
class dm_neighbor_tree
{
 public:
  dm_neighbor_tree(std::vector<dm_neighbor_point> points,double periodic_box_cm,std::uint64_t memory_bytes);
  ~dm_neighbor_tree();
  dm_neighbor_tree(const dm_neighbor_tree &)=delete;
  dm_neighbor_tree &operator=(const dm_neighbor_tree &)=delete;
  std::vector<std::size_t> overlapping(const dm_neighbor_point &target,std::uint64_t &visits,std::uint64_t maximum_visits) const;
 private:
  struct implementation;
  std::unique_ptr<implementation> data_;
};
}
#endif
