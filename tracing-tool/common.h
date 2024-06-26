#include <stdint.h>

typedef struct {
  int pred_inst;
  int pred_off_threads;
  int pred_num;
  int sm_id;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  int warp_id;
  int opcode_id;
  int pc;
  int is_mem_inst;
  int mref_id;
  uint64_t mem_addrs1[32];
  uint64_t mem_addrs2[32];
  int dst_oprnd;
  int dst_oprnd_type;
  int src_oprnds[5];
  int src_oprnds_type[5];
  uint64_t curr_clk;
  int gwarp_id;
  bool isPredNeg;
  bool isPredUniform;
  uint32_t active_mask;
  uint32_t predicate_mask;
  bool stride_or_delta;
} inst_access_t;

#define cta_addresses_size_width 10000
#define cta_addresses_size_depth 10000