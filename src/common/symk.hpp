#pragma once

#include "sym_kernels.h"

#include <cstdint>
#include <cstring>

namespace nccl_deviceapi_test {

inline ncclSymkDevWorkArgs4K make_single_work_args(ncclDevComm_t dev_comm, ncclWindow_t input_win,
                                                   ncclWindow_t output_win, size_t n_elts,
                                                   int num_blocks) {
  ncclSymkDevWorkArgs4K args4k;
  std::memset(&args4k, 0, sizeof(args4k));
  args4k.args.kcomm.devComm = dev_comm;
  args4k.args.nMaxChannels = num_blocks;
  args4k.args.maxDynamicSmem = 0;

  ncclSymkChannelWorkRange* ranges = args4k.args.getWorkRange();
  for (int b = 0; b < num_blocks; b++) {
    uint32_t end = static_cast<uint32_t>((static_cast<uint64_t>(b + 1) * 0x10000ull) /
                                         static_cast<uint64_t>(num_blocks));
    ranges[b].workHi = 0;
    ranges[b].fracHi = static_cast<uint16_t>(end - 1);
  }

  ncclSymkDevWork* works = args4k.args.getWorks(num_blocks);
  works[0].redOpArg = 0;
  works[0].nElts = n_elts;
  works[0].inputWin = input_win;
  works[0].outputWin = output_win;
  works[0].inputOff = 0;
  works[0].outputOff = 0;
  works[0].rootRank = 0;
  works[0].sChannelId = 0;
  works[0].nChannels = num_blocks;
  return args4k;
}

}  // namespace nccl_deviceapi_test
