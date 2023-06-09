#ifndef _SPARSE_QUERY_CPU
#define _SPARSE_QUERY_CPU

#include <torch/torch.h>

at::Tensor hash_query_cpu(const at::Tensor hash_query,
                          const at::Tensor hash_target,
                          const at::Tensor idx_target);

#endif
