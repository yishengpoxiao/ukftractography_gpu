/**
 * \file gpu_tractography.h
 * \brief Host interface of the GPU tractography path. Included only by
 *        tractography.cc; all CUDA specifics live behind gpu_device_api.h.
 */
#ifndef UKF_GPU_TRACTOGRAPHY_H_
#define UKF_GPU_TRACTOGRAPHY_H_

#include <string>
#include <vector>

class ISignalData;
class FilterModel;
struct UKFSettings;
class UKFFiber;
class SeedPointInfo;
struct BranchingSeedAffiliation;

namespace ukf
{
namespace gpu
{

/** True if a usable CUDA device is present; info receives a description. */
bool available(std::string& info);

/**
 * Runs tensor, FreeWater and NODDI tractography for all seeds on the GPU.
 * fibers is resized to seeds.size() and filled in seed order, matching the
 * CPU Follow*T output. Returns false (with err set) when the given
 * configuration is not supported; caller should fall back to CPU.
 */
bool run_tractography(ISignalData* signal_data,
                      const UKFSettings& s,
                      int filter_model_type,
                      const FilterModel* filter_model,
                      const std::vector<SeedPointInfo>& seeds,
                      std::vector<UKFFiber>& fibers,
                      std::string& err,
                      std::vector<SeedPointInfo>* branch_seeds = NULL,
                      std::vector<BranchingSeedAffiliation>* affiliations = NULL);

}
}

#endif  /* UKF_GPU_TRACTOGRAPHY_H_ */
