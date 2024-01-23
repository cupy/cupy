
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime

from cupyx.scipy.spatial.delaunay_2d._schewchuk import init_predicate
from cupyx.scipy.spatial.delaunay_2d._kernels import (
    make_first_tri, init_point_location_fast, init_point_location_exact,
    vote_for_point, pick_winner_point, shift, shift_opp_tri,
    shift_tri_idx, split_points_fast, split_points_exact, split_tri,
    is_tri_active, mark_special_tris, check_delaunay_fast,
    check_delaunay_exact_fast, check_delaunay_exact_exact, mark_rejected_flips,
    flip, update_opp, update_flip_trace, relocate_points_fast,
    relocate_points_exact, mark_inf_tri, collect_free_slots, make_compact_map,
    compact_tris, update_vert_idx, find_closest_tri_to_point)


def _get_typename(dtype):
    typename = get_typename(dtype)
    if cupy.dtype(dtype).kind == 'c':
        typename = 'thrust::' + typename
    elif typename == 'float16':
        if runtime.is_hip:
            # 'half' in name_expressions weirdly raises
            # HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID in getLoweredName() on
            # ROCm
            typename = '__half'
        else:
            typename = 'half'
    return typename


def _get_module_func(module, func_name, *template_args):
    args_dtypes = [_get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel


PREAMBLE_MODULE = r'''

template<typename T>
__global__ void get_morton_number(
        const int n_points, const T* points, const T* min_val, const T* range,
        int* out) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const T* point = points + 2 * idx;

    const int Gap08 = 0x00FF00FF;   // Creates 16-bit gap between value bits
    const int Gap04 = 0x0F0F0F0F;   // ... and so on ...
    const int Gap02 = 0x33333333;   // ...
    const int Gap01 = 0x55555555;   // ...

    const int minInt = 0x0;
    const int maxInt = 0x7FFF;

    int mortonNum = 0;

    // Iterate coordinates of point
    for ( int vi = 0; vi < 2; ++vi )
    {
        // Read
        int v = int( ( point[ vi ] - min_val[0] ) / range[0] * 32768.0 );

        if ( v < minInt )
            v = minInt;

        if ( v > maxInt )
            v = maxInt;

        // Create 1-bit gaps between the 10 value bits
        // Ex: 1010101010101010101
        v = ( v | ( v <<  8 ) ) & Gap08;
        v = ( v | ( v <<  4 ) ) & Gap04;
        v = ( v | ( v <<  2 ) ) & Gap02;
        v = ( v | ( v <<  1 ) ) & Gap01;

        // Interleave bits of x-y coordinates
        mortonNum |= ( v << vi );
    }

    out[idx] = mortonNum;
}


template<typename T>
__global__ void compute_distance_2d(
        const int n_points, const T* points, const long long* a_idx,
        const long long* b_idx, int* out) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const T* p_c = points + 2 * idx;
    const T* p_a = points + 2 * a_idx[0];
    const T* p_b = points + 2 * b_idx[0];

    T abx = p_b[0] - p_a[0];
    T aby = p_b[1] - p_a[1];

    T acx = p_c[0] - p_a[0];
    T acy = p_c[1] - p_a[1];

    T dist = abx * acy - aby * acx;
    int int_dist = __float_as_int( fabs((float) dist) );
    out[idx] = int_dist;
}
'''

_preamble_module = cupy.RawModule(
    code=PREAMBLE_MODULE, options=('-std=c++11',),
    name_expressions=[
        f'get_morton_number<{x}>' for x in ['float', 'double']] + [
            f'compute_distance_2d<{x}>' for x in ['float', 'double']
    ])


def _check_if_coplanar_points(points, pa_idx, pb_idx, pc_idx):
    pa = points[pa_idx]
    pb = points[pb_idx]
    pc = points[pc_idx]

    detleft = (pa[0] - pc[0]) * (pb[1] - pc[1])
    detright = (pa[1] - pc[1]) * (pb[0] - pc[0])
    det = detleft - detright

    is_det_left_pos = cupy.where(detleft > 0, True, False)
    is_det_left_neg = cupy.where(detleft < 0, True, False)

    is_det_right_pos = cupy.where(detright >= 0, True, False)
    is_det_right_neg = cupy.where(detright <= 0, True, False)

    detsum = 0
    if is_det_left_pos:
        if is_det_right_neg:
            return det
        else:
            detsum = detleft + detright
    elif is_det_left_neg:
        if is_det_right_pos:
            return det
        else:
            detsum = -detleft - detright
    else:
        return det

    epsilon = cupy.finfo(points.dtype)
    ccwerrboundA = (3.0 + 16.0 * epsilon) * epsilon
    errbound = ccwerrboundA * detsum

    return cupy.where(
        cupy.logical_or(det >= errbound, -det >= errbound), det,
        cupy.asarray(0, dtype=points.dtype))


def _compute_triangle_orientation(det):
    return cupy.where(det > 0, 1, cupy.where(det < 0, -1, 0))


class CheckDelaunayMode:
    CircleFastOrientFast = 0
    CircleExactOrientSoS = 1


class ActTriMode:
    ActTriMarkCompact = 0
    ActTriCollectCompact = 1


class GDel2D:
    def __init__(self, points):
        self.n_points = points.shape[0] + 1
        self.max_triangles = 2 * self.n_points
        self.tri_num = 0

        # self.points = cupy.array(points, copy=True)
        self.points = points
        self.point_vec = cupy.empty((self.n_points, 2), dtype=points.dtype)
        self.point_vec[:-1] = points

        self.triangles = cupy.empty((4, 3), dtype=cupy.int32)
        self.triangle_opp = cupy.empty_like(self.triangles)
        self.triangle_info = cupy.zeros(4, dtype=cupy.int8)
        self._counters = cupy.zeros(8192, dtype=cupy.int32)
        self.counters_offset = 0
        self.counters_size = 2

        self.flip = cupy.empty((self.max_triangles, 2, 2), dtype=cupy.int32)
        self.tri_msg = cupy.empty((self.max_triangles, 2), dtype=cupy.int32)
        self.values = cupy.empty(self.n_points, dtype=cupy.int32)
        self.act_tri = cupy.empty(self.max_triangles, dtype=cupy.int32)
        self.vert_tri = cupy.empty(self.n_points, dtype=cupy.int32)

        self._org_flip_num = []

    @property
    def counters(self):
        return self._counters[self.counters_offset:]

    def _renew_counters(self):
        self.counters_offset += self.counters_size
        if self.counters_offset > 8192:
            self.counters_offset = 0
            self._counters[:] = 0

    def _construct_initial_triangles(self):
        # Find extreme points in the x-axis
        v0 = cupy.argmin(self.point_vec[:-1, 0])
        v1 = cupy.argmax(self.point_vec[:-1, 0])

        block_sz = 128
        n_blocks = (self.points.shape[0] + block_sz - 1) // block_sz

        _compute_distance_2d = _get_module_func(
            _preamble_module, 'compute_distance_2d', self.points)

        # Find furthest point from v0 and v1, a.k.a the biggest
        # triangle available
        _compute_distance_2d(
            (block_sz,), (n_blocks,),
            (self.n_points, self.point_vec, v0, v1, self.values))

        v2 = cupy.argmax(self.values[:-1])

        # Check if the three points are not coplanar
        ori = _check_if_coplanar_points(self.point_vec, v0, v1, v2)

        is_coplanar = cupy.where(ori == 0.0, True, False)
        if is_coplanar:
            raise ValueError(
                'The input is degenerate, the extreme points are close to '
                'coplanar')

        tri_ort = _compute_triangle_orientation(ori)
        tri = cupy.r_[v0, v1].astype(cupy.int32)
        tri = cupy.where(tri_ort == -1, tri[::-1], tri)

        # Create the initial triangulation
        # Compute the centroid of v0 v1 v2, to be used as the kernel point.
        tri = cupy.r_[tri, v2].astype(cupy.int32)
        self.point_vec[-1] = self.point_vec[tri].mean(0)

        self.pred_consts = cupy.empty(18, cupy.float64)
        init_predicate(self.pred_consts)

        # Put the initial triangles at the Inf list
        make_first_tri(
            self.triangles, self.triangle_opp, self.triangle_info,
            tri, self.n_points - 1)

        self._renew_counters()

        exact_check = cupy.empty(self.n_points, dtype=cupy.int32)
        init_point_location_fast(
            self.vert_tri, self.n_points, exact_check,
            self.counters, tri, self.n_points - 1, self.point_vec,
            self.points_idx, self.pred_consts)

        init_point_location_exact(
            self.vert_tri, self.n_points, exact_check,
            self.counters, tri,
            self.n_points - 1, self.point_vec, self.points_idx,
            self.pred_consts)

        self.available_points = self.n_points - 4
        self.tri_num = 4

    def _init_for_flip(self):
        min_val = self.points.min()
        max_val = self.points.max()
        range_val = max_val - min_val

        _get_morton_number = _get_module_func(
            _preamble_module, 'get_morton_number', self.points)

        block_sz = 128
        n_blocks = (self.points.shape[0] + block_sz - 1) // block_sz

        # Sort the points spatially according to their Morton numbers
        _get_morton_number(
            (block_sz,), (n_blocks,),
            (self.n_points - 1, self.points, min_val, range_val, self.values))

        self.values[-1] = 2 ** 31 - 1
        unique_values = cupy.unique(self.values)
        if unique_values.shape[0] != self.values.shape[0]:
            self.values = unique_values
        self.points_idx = cupy.argsort(self.values).astype(cupy.int32)
        self.point_vec = self.point_vec[self.points_idx]

        self._construct_initial_triangles()

    def _shift_replace(self, shift_vec, data, size, type_str, zeros=False):
        init = cupy.empty if not zeros else cupy.zeros
        shift_values = init(size, dtype=data.dtype)
        shift(shift_vec, data, shift_values, type_str)
        return shift_values

    def _shift_opp_tri(self, shift_vec, size):
        shift_values = cupy.empty((size, 3), dtype=cupy.int32)
        shift_opp_tri(shift_vec, self.triangle_opp, shift_values)
        return shift_values

    def _shift_tri(self, tri_to_vert, split_tri_vec):
        tri_num = self.tri_num + 2 * split_tri_vec.shape[0]
        shift_vec = cupy.cumsum(cupy.where(tri_to_vert < 0x7FFFFFFF - 1, 2, 0))
        shift_vec = shift_vec.astype(cupy.int32)
        shift_vec[1:] = shift_vec[:-1]
        shift_vec[0] = 0

        self.triangles = self._shift_replace(
            shift_vec, self.triangles, (tri_num, 3), 'Tri')
        self.triangle_info = self._shift_replace(
            shift_vec, self.triangle_info, tri_num, 'char', zeros=True)
        tri_to_vert = self._shift_replace(
            shift_vec, tri_to_vert, tri_num, 'int')

        self.triangle_opp = self._shift_opp_tri(shift_vec, tri_num)

        shift_tri_idx(self.vert_tri, shift_vec)
        shift_tri_idx(split_tri_vec, shift_vec)
        return tri_to_vert

    def _expand_copy(self, in_arr, new_size, zeros=False):
        init = cupy.empty if not zeros else cupy.zeros
        out = init(new_size, dtype=in_arr.dtype)
        out[:in_arr.shape[0]] = in_arr
        return out

    def _expand_tri(self, new_tri_num):
        if new_tri_num > self.tri_num:
            self.triangles = self._expand_copy(
                self.triangles, (new_tri_num, 3))
            self.triangle_opp = self._expand_copy(
                self.triangle_opp, (new_tri_num, 3))
            self.triangle_info = self._expand_copy(
                self.triangle_info, new_tri_num, zeros=True)

    def _split_tri(self):
        max_sample_per_tri = 100

        # Rank points
        tri_num = self.tri_num
        no_sample = self.n_points

        if no_sample / self.tri_num > max_sample_per_tri:
            no_sample = self.tri_num * max_sample_per_tri

        tri_circle = cupy.full(
            tri_num, cupy.iinfo(cupy.int32).min, dtype=cupy.int32)

        vert_circle = cupy.empty(no_sample, dtype=cupy.int32)
        vote_for_point(self.vert_tri, self.n_points, self.triangles,
                       vert_circle, tri_circle, no_sample,
                       self.n_points - 1, self.point_vec, self.pred_consts)

        tri_to_vert = cupy.full(
            tri_num, 0x7FFFFFFF, dtype=cupy.int32)
        pick_winner_point(self.vert_tri, self.n_points, vert_circle,
                          tri_circle, tri_to_vert, no_sample)

        del vert_circle
        del tri_circle

        split_tri_vec = cupy.where(tri_to_vert < 0x7FFFFFFF - 1)[0]
        split_tri_vec = split_tri_vec.astype(cupy.int32)

        self.ins_num = split_tri_vec.shape[0]
        extra_tri_num = 2 * self.ins_num
        split_tri_num = self.tri_num + extra_tri_num

        if (self.available_points - self.ins_num < self.ins_num and
                self.ins_num < 0.1 * self.n_points):
            self.do_flipping = False

        if self.do_flipping:
            tri_to_vert = self._shift_tri(tri_to_vert, split_tri_vec)
            tri_num = -1

        sz = split_tri_num if tri_num < 0 else tri_num

        ins_tri_map = cupy.full(sz, -1, dtype=cupy.int32)
        ins_tri_map[split_tri_vec] = cupy.arange(
            split_tri_vec.shape[0], dtype=cupy.int32)

        self._expand_tri(split_tri_num)

        # Update the location of the points
        exact_check = cupy.empty(self.n_points, dtype=cupy.int32)
        self._renew_counters()

        split_points_fast(
            self.vert_tri, tri_to_vert, self.triangles, ins_tri_map,
            exact_check, self.counters, tri_num,
            self.ins_num, self.n_points - 1, self.point_vec, self.points_idx,
            self.pred_consts)

        split_points_exact(
            self.vert_tri, tri_to_vert, self.triangles, ins_tri_map,
            exact_check, self.counters, tri_num,
            self.ins_num, self.n_points - 1, self.point_vec, self.points_idx,
            self.pred_consts)

        del exact_check

        # Split old into new triangle and copy them to new array
        split_tri(split_tri_vec, self.triangles, self.triangle_opp,
                  self.triangle_info, ins_tri_map, tri_to_vert,
                  tri_num, self.ins_num)

        self.available_points -= self.ins_num
        self.tri_num = self.triangles.shape[0]

    def _relocate_all(self):
        if self._flip_vec.shape[0] == 0:
            return

        if self.available_points > 0:
            tri_num = self.triangles.shape[0]
            tri_to_flip = cupy.empty(tri_num, dtype=cupy.int32)

            # Rebuild the pointers from back to forth
            next_flip_num = self._flip_vec.shape[0]
            for i in range(len(self._org_flip_num) - 1, -1, -1):
                prev_flip_num = self._org_flip_num[i]
                flip_num = next_flip_num - prev_flip_num
                update_flip_trace(
                    self._flip_vec, tri_to_flip, prev_flip_num, flip_num)
                next_flip_num = prev_flip_num

            # Relocate points
            exact_check = cupy.empty(self.n_points)
            self._renew_counters()

            relocate_points_fast(
                self.vert_tri, tri_to_flip, self._flip_vec,
                exact_check, self.counters, self.n_points - 1, self.point_vec,
                self.points_idx, self.pred_consts)

            relocate_points_exact(
                self.vert_tri, tri_to_flip, self._flip_vec,
                exact_check, self.counters, self.n_points - 1, self.point_vec,
                self.points_idx, self.pred_consts)

        self._flip_vec = cupy.empty((0, 4), dtype=cupy.int32)
        self._flip_vec_cap = self.max_triangles
        self._tri_msg = cupy.full(
            (self.max_triangles, 2), -1, dtype=cupy.int32)
        self._org_flip_num = []

    def _dispatch_check_delaunay(self, check_mode, org_act_num, tri_vote):
        if check_mode == CheckDelaunayMode.CircleFastOrientFast:
            check_delaunay_fast(
                self._act_tri, self.triangles, self.triangle_opp,
                self.triangle_info, tri_vote, org_act_num, self.n_points - 1,
                self.point_vec, self.pred_consts)
        elif check_mode == CheckDelaunayMode.CircleExactOrientSoS:
            exact_check_vi = self._tri_msg
            self._renew_counters()

            check_delaunay_exact_fast(
                self._act_tri, self.triangles, self.triangle_opp,
                self.triangle_info, tri_vote, exact_check_vi, org_act_num,
                self.counters, self.n_points - 1, self.point_vec,
                self.pred_consts)

            check_delaunay_exact_exact(
                self.triangles, self.triangle_opp, tri_vote, exact_check_vi,
                self.counters, self.n_points - 1, self.point_vec,
                self.points_idx, self.pred_consts)

    def _do_flipping(self, check_mode):
        tri_num = self.triangles.shape[0]
        if self._act_tri_mode == ActTriMode.ActTriMarkCompact:
            active_tri = is_tri_active(self.triangle_info)
            self._act_tri = cupy.where(active_tri)[0].astype(cupy.int32)
        elif self._act_tri_mode == ActTriMode.ActTriCollectCompact:
            self._act_tri = self._act_tri[self._act_tri >= 0]

        org_act_num = self._act_tri.shape[0]

        # Check actNum, switch mode or quit if necessary
        if org_act_num == 0:
            # No more work
            return False

        if (check_mode != CheckDelaunayMode.CircleExactOrientSoS and
                org_act_num < 64 * 32):
            # Little work, leave it for the Exact iterations
            return False

        # See if there's little work enough to switch to collect mode.
        if (org_act_num < 512 * 128 and
                org_act_num * 2 < self.max_triangles and
                org_act_num * 2 < tri_num):
            self._act_tri_mode = ActTriMode.ActTriCollectCompact
        else:
            self._act_tri_mode = ActTriMode.ActTriMarkCompact

        # Vote for flips
        tri_vote = cupy.full(tri_num, 0x7FFFFFFF, dtype=cupy.int32)
        self._dispatch_check_delaunay(check_mode, org_act_num, tri_vote)

        # Mark rejected flips
        flip_to_tri = cupy.empty(org_act_num, dtype=cupy.int32)
        mark_rejected_flips(
            self._act_tri, self.triangle_opp, tri_vote, self.triangle_info,
            flip_to_tri, org_act_num)

        del tri_vote

        # Compact flips
        flip_to_tri = flip_to_tri[flip_to_tri >= 0]
        flip_num = flip_to_tri.shape[0]

        if flip_num == 0:
            return False

        # Expand flip vector
        org_flip_num = self._flip_vec.shape[0]
        exp_flip_num = org_flip_num + flip_num

        if exp_flip_num > self._flip_vec_cap:
            self._relocate_all()
            org_flip_num = 0
            exp_flip_num = flip_num
            self._flip_vec_cap = exp_flip_num

        self._flip_vec = self._expand_copy(
            self._flip_vec, (exp_flip_num, 4), zeros=True)

        # self._tri_msg contains two components.
        # - .x is the encoded new neighbor information
        # - .y is the flipIdx as in the flipVec (i.e. globIdx)
        # As such, we do not need to initialize it to -1 to
        # know which tris are not flipped in the current round.
        # We can rely on the flipIdx being > or < than orgFlipIdx.
        # Note that we have to initialize everything to -1
        # when we clear the flipVec and reset the flip indexing.
        if self.triangles.shape[0] > self._tri_msg.shape[0]:
            self._tri_msg = cupy.empty(
                (self.triangles.shape[0], 2), dtype=cupy.int32)
        else:
            self._tri_msg = self._tri_msg[:self.triangles.shape[0]]

        # Expand active tri vector
        if self._act_tri_mode == ActTriMode.ActTriCollectCompact:
            self._act_tri = self._expand_copy(
                self._act_tri, org_act_num + flip_num)

        # Flipping
        flip(flip_to_tri, self.triangles, self.triangle_opp,
             self.triangle_info, self._tri_msg, self._act_tri, self._flip_vec,
             org_flip_num, org_act_num)

        self._org_flip_num.append(org_flip_num)

        # Update oppTri
        update_opp(self._flip_vec[org_flip_num:], self.triangle_opp,
                   self._tri_msg, flip_to_tri, org_flip_num, flip_num)

        return True

    def _do_flipping_loop(self, check_mode):
        self._flip_vec = cupy.empty((0, 4), dtype=cupy.int32)
        self._flip_vec_cap = self.max_triangles
        self._tri_msg = cupy.full(
            (self.max_triangles, 2), -1, dtype=cupy.int32)

        self._act_tri_mode = ActTriMode.ActTriMarkCompact

        flip_loop = 0
        while self._do_flipping(check_mode):
            flip_loop += 1

        self._relocate_all()

        self._flip_vec = None
        self._act_tri = None
        self._tri_msg = None

    def _split_and_flip(self):
        insert_loop = 0
        self.do_flipping = True

        while self.available_points > 0:
            self._split_tri()
            if self.do_flipping:
                self._do_flipping_loop(CheckDelaunayMode.CircleFastOrientFast)
            insert_loop += 1

        if not self.do_flipping:
            self._do_flipping_loop(CheckDelaunayMode.CircleFastOrientFast)

        mark_special_tris(self.triangle_info, self.triangle_opp)
        self._do_flipping_loop(CheckDelaunayMode.CircleExactOrientSoS)
        self._do_flipping_loop(CheckDelaunayMode.CircleFastOrientFast)

        mark_special_tris(self.triangle_info, self.triangle_opp)
        self._do_flipping_loop(CheckDelaunayMode.CircleExactOrientSoS)

    def _compact_tris(self):
        tri_num = self.triangles.shape[0]
        prefix = cupy.cumsum(self.triangle_info, dtype=cupy.int32)

        new_tri_num = prefix[tri_num - 1].item()
        free_num = tri_num - new_tri_num

        free_vec = cupy.empty(free_num, dtype=cupy.int32)
        collect_free_slots(self.triangle_info, prefix, free_vec, new_tri_num)

        # Make map
        make_compact_map(self.triangle_info, prefix, free_vec, new_tri_num)

        # Reorder the triangles
        compact_tris(
            self.triangle_info, prefix, self.triangles,
            self.triangle_opp, new_tri_num)

        self.triangles = self.triangles[:new_tri_num]
        self.triangle_opp = self.triangle_opp[:new_tri_num]
        self.triangle_info = self.triangle_info[:new_tri_num]

    def _output(self):
        mark_inf_tri(
            self.triangles, self.triangle_info, self.triangle_opp,
            self.n_points - 1)

        self._compact_tris()

        update_vert_idx(self.triangles, self.triangle_info, self.points_idx)

    def compute(self):
        self._init_for_flip()
        self._split_and_flip()
        self._output()
        return self.triangles, self.triangle_opp

    def find_point_in_triangulation(self, xi, eps, find_coords=False):
        out = cupy.empty((xi.shape[0],), dtype=cupy.int32)
        c = cupy.empty(0, dtype=cupy.float64)
        if find_coords:
            c = cupy.empty((xi.shape[0], xi.shape[-1] + 1), dtype=cupy.float64)

        find_closest_tri_to_point(xi, self.points, self.triangles,
                                  out, c, eps, find_coords)
        return out, c
