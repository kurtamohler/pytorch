#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/EmbeddingBag.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/EmbeddingBag.h>
#include <ATen/native/Pool.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_embedding_bag_forward_only_native.h>
#include <ATen/ops/_embedding_bag_native.h>
#include <ATen/ops/empty.h>
#endif

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/EmbeddingBag_metallib.h>
#endif

namespace {

std::pair<Tensor, Tensor> promoteIndicesAndOffsets(const Tensor& indices, const Tensor& offsets) {
  const auto commonType = promoteTypes(offsets.scalar_type(), indices.scalar_type());
  return {indices.scalar_type() == commonType ? indices : indices.toType(commonType),
          offsets.scalar_type() == commonType ? offsets : offsets.toType(commonType)};
}

} // namespace

namespace mps {

static std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_mps_impl(
    const Tensor& weight,
    const Tensor& indices_,
    const Tensor& offsets_,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const std::optional<Tensor>& per_sample_weights_opt,
    bool include_last_offset,
    int64_t padding_idx) {
  TORCH_CHECK(indices_.dim() == 1,
              "input has to be a 1D Tensor, but got Tensor of dimension ",
              indices_.dim());
  if (indices_.dim() == 1) {
    TORCH_CHECK(offsets_.dim() == 1, "offsets has to be a 1D Tensor, but got Tensor of dimension ", offsets_.dim());
  }
  TORCH_CHECK(weight.dim() == 2, "weight has to be a 2D Tensor, but got Tensor of dimension ", weight.dim());
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  Tensor indices, offsets;
  std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_bag_mps", indices_arg, {kLong, kInt});
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarTypes("embedding_bag_mps", offsets_arg, {kLong, kInt});
  checkSameType("embedding_bag_mps", indices_arg, offsets_arg);
  auto weight_arg = TensorArg(weight, "weight", 1);

  int64_t numIndices = indices.size(0);
  int64_t numBags = offsets.size(0);
  if (include_last_offset) {
    // Check https://github.com/pytorch/pytorch/issues/29019
    // We plan to add one more element in offsets, which is equal to the size of
    // indices. Currently for cuda devices, we still use the legacy
    // implementation even this flag is enabled.
    TORCH_CHECK(numBags >= 1, "include_last_offset: numBags should be at least 1");
    numBags -= 1;
  }
  int64_t featureSize = weight.size(1);

  auto bag_size = at::empty(offsets.sizes(), indices.options());
  auto offset2bag = at::empty({indices.size(0)}, indices.options()); // offset2bag = [0 0 0 0 0]

  auto output = at::empty({numBags, featureSize}, weight.options());

  Tensor max_indices;

  if (mode == EmbeddingBagMode::MAX) {
    max_indices = at::empty({numBags, featureSize}, indices.options());
  } else {
    // No need to allocate if we aren't doing a backwards pass
    max_indices = at::empty({0}, indices.options());
  }

  EmbeddingBagParams<uint32_t> params;

  for (const auto dim : c10::irange(weight.dim())) {
    params.weight_strides[dim] = safe_downcast<uint32_t, int64_t>(weight.stride(dim));
    params.output_strides[dim] = safe_downcast<uint32_t, int64_t>(output.stride(dim));

    if (mode == EmbeddingBagMode::MAX) {
      params.max_indices_strides[dim] = safe_downcast<uint32_t, int64_t>(max_indices.stride(dim));
    }
  }

  params.num_indices = numIndices;
  params.num_bags = numBags;
  params.feature_size = featureSize;

  params.mode = static_cast<EmbeddingBagMode>(mode);
  params.padding_idx = padding_idx;

  auto num_threads = output.numel();
  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto pipeline_state = lib.getPipelineStateForFunc("embedding_bag_" + scalarToMetalTypeString(weight) + "_" +
                                                        scalarToMetalTypeString(indices));

      getMPSProfiler().beginProfileKernel(pipeline_state, "embedding_bag", {weight});
      [computeEncoder setComputePipelineState:pipeline_state];
      mtl_setArgs(computeEncoder, weight, indices, offsets, output, offset2bag, bag_size, max_indices, params);

      mtl_dispatch1DJob(computeEncoder, pipeline_state, num_threads);
      getMPSProfiler().endProfileKernel(pipeline_state);
    }
  });

  return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, offset2bag, bag_size, max_indices);
}

} // namespace mps

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_mps(const Tensor& weight,
                                                              const Tensor& indices,
                                                              const Tensor& offsets,
                                                              const bool scale_grad_by_freq,
                                                              const int64_t mode,
                                                              bool sparse,
                                                              const std::optional<Tensor>& per_sample_weights_opt,
                                                              bool include_last_offset,
                                                              int64_t padding_idx) {
  return mps::_embedding_bag_mps_impl(weight,
                                      indices,
                                      offsets,
                                      scale_grad_by_freq,
                                      mode,
                                      sparse,
                                      per_sample_weights_opt,
                                      include_last_offset,
                                      padding_idx);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_forward_only_mps(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const std::optional<Tensor>& per_sample_weights_opt,
    bool include_last_offset,
    int64_t padding_idx) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  return _embedding_bag_mps(
      weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
}

} // namespace at::native