
#ifndef COMPUTE_CAPABILITY_CONFIG_H
#define COMPUTE_CAPABILITY_CONFIG_H

#include <vector>

enum COMPUTE_CAPABILITY_CONFIG {
  WARP_SIZE = 0,
  MAX_BLOCK_SIZE,
  SMEM_ALLOCATION_SIZE,
  MAX_REGISTERS_PER_SM,
  MAX_REGISTERS_PER_BLOCK,
  MAX_REGISTERS_PER_THREAD,
  REGISTER_ALLOCATION_SIZE,
  MAX_ACTIVE_BLOCKS_PER_SM,
  MAX_ACTIVE_THREADS_PER_SM,

  MAX_CONCURRENT_KERNELS_NUM,
  NUM_COMPUTE_CAPABILITY_CONFIG,
};

struct cc_config_t {
  cc_config_t(int _num_sm, int _warp_size, int _max_block_size,
              int _smem_allocation_size, int _max_registers_per_SM,
              int _max_registers_per_block, int _max_registers_per_thread,
              int _register_allocation_size, int _max_active_blocks_per_SM,
              int _max_active_threads_per_SM, int _max_concurrent_kernels_num) {
    num_sm = _num_sm;
    warp_size = _warp_size;
    max_block_size = _max_block_size;
    smem_allocation_size = _smem_allocation_size;
    max_registers_per_SM = _max_registers_per_SM;
    max_registers_per_block = _max_registers_per_block;
    max_registers_per_thread = _max_registers_per_thread;
    register_allocation_size = _register_allocation_size;
    max_active_blocks_per_SM = _max_active_blocks_per_SM;
    max_active_threads_per_SM = _max_active_threads_per_SM;
    max_concurrent_kernels_num = _max_concurrent_kernels_num;
  }

  int num_sm;
  int warp_size;
  int max_block_size;
  int smem_allocation_size;
  int max_registers_per_SM;
  int max_registers_per_block;
  int max_registers_per_thread;
  int register_allocation_size;
  int max_active_blocks_per_SM;
  int max_active_threads_per_SM;
  int max_concurrent_kernels_num;
};

enum SM_COMPUTE_CAPACITY {
  SM50 = 0,
  SM52,
  SM53,
  SM60,
  SM61,
  SM70,
  SM75,
  NUM_SM_COMPUTE_CAPACITY,
};

enum GPU_ARCH {
  V100 = 0,
  NUM_GPU_ARCH,
};

std::vector<struct cc_config_t> gpu_config{
    cc_config_t(80, 32, 1024, 256, 65536, 65536, 255, 256, 32, 2048, 128),
};

#endif