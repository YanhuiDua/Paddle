// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/backends/gpu/gpu_info.h"

__global__ void allreduce_start(int a) { a = a; }
__global__ void allreduce_stop(int a) { a = a; }
__global__ void allgather_start(int a) { a = a; }
__global__ void allgather_stop(int a) { a = a; }
__global__ void broadcast_start(int a) { a = a; }
__global__ void broadcast_stop(int a) { a = a; }
__global__ void recv_start(int a) { a = a; }
__global__ void recv_stop(int a) { a = a; }
__global__ void send_start(int a) { a = a; }
__global__ void send_stop(int a) { a = a; }

void AllReduceStart(hipStream_t stream) {
  allreduce_start<<<1, 1, 0, stream>>>(1);
}

void AllReduceStop(hipStream_t stream) {
  allreduce_stop<<<1, 1, 0, stream>>>(1);
}

void AllGatherStart(hipStream_t stream) {
  allgather_start<<<1, 1, 0, stream>>>(1);
}

void AllGatherStop(hipStream_t stream) {
  allgather_stop<<<1, 1, 0, stream>>>(1);
}

void BroadcastStart(hipStream_t stream) {
  broadcast_start<<<1, 1, 0, stream>>>(1);
}

void BroadcastStop(hipStream_t stream) {
  broadcast_stop<<<1, 1, 0, stream>>>(1);
}

void SendStart(hipStream_t stream) { send_start<<<1, 1, 0, stream>>>(1); }
void SendStop(hipStream_t stream) { send_stop<<<1, 1, 0, stream>>>(1); }
void RecvStart(hipStream_t stream) { recv_start<<<1, 1, 0, stream>>>(1); }
void RecvStop(hipStream_t stream) { recv_stop<<<1, 1, 0, stream>>>(1); }
