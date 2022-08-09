#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
from white_list import (
    op_accuracy_white_list,
    check_shape_white_list,
    compile_vs_runtime_white_list,
    no_check_set_white_list,
    op_threshold_white_list,
    no_grad_set_white_list,
)
paddle.enable_static()


def AffineGrid3D(theta, grid_shape):
    n = grid_shape[0]
    d = grid_shape[1]
    h = grid_shape[2]
    w = grid_shape[3]
    d_idx = np.repeat(np.repeat(np.linspace(-1, 1, d)[:, np.newaxis, np.newaxis], h,
                      axis=1), w, axis=2)[:, :, :, np.newaxis] 
    h_idx = np.repeat(np.repeat(np.linspace(-1, 1, h)[np.newaxis, :, np.newaxis], w,
                      axis=2), d, axis=0)[:, :, :, np.newaxis]
    w_idx = np.repeat(np.repeat(np.linspace(-1, 1, w)[np.newaxis, np.newaxis, :], h,
                      axis=1), d, axis=0)[:, :, :, np.newaxis]
    grid = np.concatenate([w_idx, h_idx, d_idx, np.ones([d, h, w, 1])],
                          axis=3)  # d * h * w * 4
    grid = np.repeat(grid[np.newaxis, :], n, axis=0)  # n * d * h * w *4
    ret = np.zeros([n, d * h * w, 3])
    theta = theta.transpose([0, 2, 1])
    for i in range(len(theta)):
        ret[i] = np.dot(grid[i].reshape([d * h * w, 4]), theta[i])

    return ret.reshape([n, d, h, w, 3]).astype("float64")

def getGridPointValue3D(data, x, y, z):
    data_shape = data.shape
    N = data_shape[0]
    C = data_shape[1]
    in_D = data_shape[2]
    in_H = data_shape[3]
    in_W = data_shape[4]
    out_D = x.shape[1]
    out_H = x.shape[2]
    out_W = x.shape[3]

    #out = np.zeros(data_shape, dtype='float64')
    out = np.zeros([N, C, out_D, out_H, out_W], dtype='float64')
    for i in range(N):
        for j in range(out_D):
            for k in range(out_H):
                for l in range(out_W):
                    if y[i, j, k, l] < 0 or y[i, j, k, l] > in_H - 1 or x[
                            i, j, k, l] < 0 or x[i, j, k, l] > in_W - 1 or z[
                                i, j, k, l] < 0 or z[i, j, k, l] > in_D - 1:
                        out[i, :, j, k, l] = 0
                    else:
                        out[i, :, j, k, l] = data[i, :, z[i, j, k, l], y[i, j, k, l], x[i, j, k, l]]

    return out

def clip(x, min_n, max_n):
    return np.maximum(np.minimum(x, max_n), min_n)

def unnormalizeAndClip(grid_slice, max_val, align_corners, padding_mode):
    if align_corners:
        grid_slice = 0.5 * ((grid_slice.astype('float64') + 1.0) * max_val)
    else:
        grid_slice = 0.5 * ((grid_slice.astype('float64') + 1.0) *
                            (max_val + 1)) - 0.5

    if padding_mode == "border":
        grid_slice = clip(grid_slice, 0, max_val)
    elif padding_mode == "reflection":
        double_range = 2 * max_val if align_corners else (max_val + 1) * 2
        grid_abs = np.abs(grid_slice) if align_corners else np.abs(grid_slice +
                                                                   0.5)
        extra = grid_abs - np.floor(grid_abs / double_range) * double_range
        grid_slice = np.minimum(extra, double_range - extra)
        grid_slice = grid_slice if align_corners else clip(
            grid_slice - 0.5, 0, max_val)
    return grid_slice

def GridSampler3D(data,
                  grid,
                  align_corners=True,
                  mode="bilinear",
                  padding_mode="zeros"):
    dims = data.shape
    N = dims[0]
    in_C = dims[1]
    in_D = dims[2]
    in_H = dims[3]
    in_W = dims[4]

    out_D = grid.shape[1]
    out_H = grid.shape[2]
    out_W = grid.shape[3]

    x = grid[:, :, :, :, 0]
    y = grid[:, :, :, :, 1]
    z = grid[:, :, :, :, 2]

    z_max = in_D - 1
    y_max = in_H - 1
    x_max = in_W - 1

    x = unnormalizeAndClip(x, x_max, align_corners, padding_mode)
    y = unnormalizeAndClip(y, y_max, align_corners, padding_mode)
    z = unnormalizeAndClip(z, z_max, align_corners, padding_mode)

    if mode == "bilinear":
        x0 = np.floor(x).astype('int32')
        x1 = x0 + 1
        y0 = np.floor(y).astype('int32')
        y1 = y0 + 1
        z0 = np.floor(z).astype('int32')
        z1 = z0 + 1 

        w_tnw = np.tile(((x1 - x) * (y1 - y) * (z1 - z)).reshape((N, 1, out_D, out_H, out_W)),
                     (1, in_C, 1, 1, 1))
        w_tne = np.tile(((x - x0) * (y1 - y) * (z1 - z)).reshape((N, 1, out_D, out_H, out_W)),
                     (1, in_C, 1, 1, 1))
        w_tsw = np.tile(((x1 - x) * (y - y0) * (z1 - z)).reshape((N, 1, out_D, out_H, out_W)),
                     (1, in_C, 1, 1, 1))
        w_tse = np.tile(((x - x0) * (y - y0) * (z1 - z)).reshape((N, 1, out_D, out_H, out_W)),
                     (1, in_C, 1, 1, 1))
        w_bnw = np.tile(((x1 - x) * (y1 - y) * (z - z0)).reshape((N, 1, out_D, out_H, out_W)),
                     (1, in_C, 1, 1, 1))
        w_bne = np.tile(((x - x0) * (y1 - y) * (z - z0)).reshape((N, 1, out_D, out_H, out_W)),
                     (1, in_C, 1, 1, 1))
        w_bsw = np.tile(((x1 - x) * (y - y0) * (z - z0)).reshape((N, 1, out_D, out_H, out_W)),
                     (1, in_C, 1, 1, 1))
        w_bse = np.tile(((x - x0) * (y - y0) * (z - z0)).reshape((N, 1, out_D, out_H, out_W)),
                     (1, in_C, 1, 1, 1))

        v_tnw = getGridPointValue3D(data, x0, y0, z0)
        v_tne = getGridPointValue3D(data, x1, y0, z0)
        v_tsw = getGridPointValue3D(data, x0, y1, z0)
        v_tse = getGridPointValue3D(data, x1, y1, z0)
        v_bnw = getGridPointValue3D(data, x0, y0, z1)
        v_bne = getGridPointValue3D(data, x1, y0, z1)
        v_bsw = getGridPointValue3D(data, x0, y1, z1)
        v_bse = getGridPointValue3D(data, x1, y1, z1)

        out = (w_tnw * v_tnw + w_tne * v_tne + w_tsw * v_tsw + w_tse * v_tse 
             + w_bnw * v_bnw + w_bne * v_bne + w_bsw * v_bsw + w_bse * v_bse).astype('float64')

    elif mode == "nearest":
        x = np.round(x).astype('int32')
        y = np.round(y).astype('int32')
        z = np.round(z).astype('int32')
        out = getGridPointValue3D(data, x, y, z)
    return out

class TestGridSamplerOp(OpTest):

    def setUp(self):
        self.use_cudnn = False
        self.numeric_grad_delta = 0.0001
        self.op_type = 'grid_sampler'
        self.align_corners = True
        self.padding_mode = "zeros"
        self.mode = "bilinear"
        self.initTestCase()
        x = np.random.randint(0, 255, self.x_shape).astype('float64')

        theta = np.zeros(self.theta_shape).astype('float64')
        for i in range(self.theta_shape[0]):
            for j in range(2):
                for k in range(3):
                    theta[i, j, k] = np.random.rand(1)[0]
        grid = AffineGrid3D(theta, self.grid_shape)

        self.inputs = {'X': x, 'Grid': grid}
        self.attrs = {
            'use_cudnn': self.use_cudnn,
            "align_corners": self.align_corners,
            "padding_mode": self.padding_mode,
            "mode": self.mode
        }
        self.outputs = {
            'Output':
            GridSampler3D(x, grid, self.align_corners, self.mode,
                        self.padding_mode)
        }

    def test_check_output(self):
        check_eager = False
        check_dygraph = True
        for place in self.get_places():
            res = self.check_output_with_place(place,
                                               atol=1e-5,
                                               no_check_set=None,
                                               equal_nan=False,
                                               check_dygraph=check_dygraph,
                                               inplace_atol=None,
                                               check_eager=check_eager)
            if check_eager:
                assert check_dygraph == True
                outs, dygraph_outs, eager_dygraph_outs, fetch_list = res
            elif check_dygraph:
                outs, dygraph_outs, fetch_list = res
            else:
                outs, fetch_list = res
            if self.op_type not in compile_vs_runtime_white_list.COMPILE_RUN_OP_WHITE_LIST:
                self.check_compile_vs_runtime(fetch_list, outs)

    def test_check_grad_normal(self):
        self._check_grad_helper()
        for place in self.get_places():
            self.check_grad_with_place(place,
                                       ['X'],
                                       'Output',
                                       no_grad_set=None,
                                       numeric_grad_delta=self.numeric_grad_delta,
                                       in_place=False,
                                       max_relative_error=0.01,
                                       user_defined_grads=None,
                                       user_defined_grad_outputs=None,
                                       check_dygraph=True,
                                       check_eager=False)

    def get_places(self):
        places = []
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def initTestCase(self):
        self.x_shape = (2, 3, 5, 6, 7)
        self.grid_shape = (2, 8, 9, 10, 3)
        self.theta_shape = (2, 3, 4)
        self.align_corners = True
        self.padding_mode = "zeros"
        self.mode = "bilinear"
        self.use_cudnn = False #if core.is_compiled_with_rocm() else True


class Case1(TestGridSamplerOp):
    def get_places(self):
        places = []
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def initTestCase(self):
        self.x_shape = (2, 3, 5, 6, 7)
        self.grid_shape = (2, 8, 9, 10, 3)
        self.theta_shape = (2, 3, 4)
        self.align_corners = False
        self.padding_mode = "zeros"
        self.mode = "bilinear"


class Case1_(TestGridSamplerOp):
    def get_places(self):
        places = []
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def initTestCase(self):
        self.x_shape = (2, 3, 5, 6, 7)
        self.grid_shape = (2, 8, 9, 10, 3)
        self.theta_shape = (2, 3, 4)
        self.align_corners = False
        self.padding_mode = "border"
        self.mode = "bilinear"


class Case2(TestGridSamplerOp):

    def initTestCase(self):
        self.x_shape = (2, 3, 5, 6, 7)
        self.grid_shape = (2, 8, 9, 10, 3)
        self.theta_shape = (2, 3, 4)
        self.align_corners = False
        self.padding_mode = "reflection"
        self.mode = "bilinear"


class Case3(TestGridSamplerOp):

    def initTestCase(self):
        self.x_shape = (2, 3, 5, 6, 7)
        self.grid_shape = (2, 8, 9, 10, 3)
        self.theta_shape = (2, 3, 4)
        self.align_corners = True
        self.padding_mode = "reflection"
        self.mode = "bilinear"


class Case4(TestGridSamplerOp):

    def initTestCase(self):
        self.x_shape = (2, 3, 5, 6, 7)
        self.grid_shape = (2, 8, 9, 10, 3)
        self.theta_shape = (2, 3, 4)
        self.align_corners = False
        self.padding_mode = "reflection"
        self.mode = "nearest"
        self.numeric_grad_delta = 0.0001


@skip_check_grad_ci(reason="'check_grad' on large inputs is too slow, " +
                    "however it is desirable to cover the forward pass")
class LargeInputCase(TestGridSamplerOp):

    def get_places(self):
        places = []
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def initTestCase(self):
        self.no_need_check_grad = True
        self.x_shape = (2, 3, 24, 24, 12)
        self.grid_shape = (2, 25, 25, 12, 3)
        self.theta_shape = (2, 3, 4)
        self.align_corners = False
        self.padding_mode = "reflection"
        self.mode = "bilinear"

    def test_check_grad_normal(self):
        pass


@skip_check_grad_ci(reason="'check_grad' on large inputs is too slow, " +
                    "however it is desirable to cover the forward pass")
class Case5(LargeInputCase):

    def initTestCase(self):
        self.no_need_check_grad = True
        self.x_shape = (2, 3, 24, 24, 12)
        self.grid_shape = (2, 25, 25, 12, 3)
        self.theta_shape = (2, 3, 4)
        self.align_corners = True
        self.padding_mode = "zeros"
        self.mode = "bilinear"
        self.use_cudnn = False #if core.is_compiled_with_rocm() else True


if __name__ == "__main__":
    unittest.main()
