#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid >= size) return;
  size_t in_idx = offset;
  size_t t = gid;
  for (int d = (int)shape.size - 1; d >= 0; --d) {
    int32_t dim = shape.data[d];
    int32_t coord = (int32_t)(t % dim);
    t /= dim;
    in_idx += (size_t)coord * (size_t)strides.data[d];
  }
  out[gid] = a[in_idx];
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size,
                                   CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= size) return;

  size_t out_idx = offset;
  size_t t = gid;
  for (int d = (int)shape.size - 1; d >= 0; --d) {
    int32_t dim = shape.data[d];
    int32_t coord = (int32_t)(t % dim);
    t /= dim;
    out_idx += (size_t)coord * (size_t)strides.data[d];
  }
  out[out_idx] = a[gid];
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  size_t total = 1;
  for (auto s : shape) total *= (size_t)s;
  CudaDims dim = CudaOneDim(total);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(
      a.ptr, out->ptr, total, VecToCuda(shape), VecToCuda(strides), offset);
  /// END SOLUTION
}


__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t* out,
                                    CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= size) return;

  size_t out_idx = offset;
  size_t t = gid;
  for (int d = (int)shape.size - 1; d >= 0; --d) {
    int32_t dim = shape.data[d];
    int32_t coord = (int32_t)(t % dim);
    t /= dim;
    out_idx += (size_t)coord * (size_t)strides.data[d];
  }
  out[out_idx] = val;
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(
      size, val, out->ptr, VecToCuda(shape), VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * b[gid];
}
void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}
__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * val;
}
void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / b[gid];
}
void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}
__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / val;
}
void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = powf(a[gid], val);
}
void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = fmaxf(a[gid], b[gid]);
}
void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}
__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = fmaxf(a[gid], val);
}
void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] == b[gid]) ? 1.0f : 0.0f;
}
void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}
__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] == val) ? 1.0f : 0.0f;
}
void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] >= b[gid]) ? 1.0f : 0.0f;
}
void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}
__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] >= val) ? 1.0f : 0.0f;
}
void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = logf(a[gid]);
}
void EwiseLog(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = expf(a[gid]);
}
void EwiseExp(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = tanhf(a[gid]);
}
void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void MatmulKernel(const scalar_t* A, const scalar_t* B, scalar_t* C,
                             uint32_t M, uint32_t N, uint32_t P) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)M * (size_t)P;
  if (gid >= total) return;
  uint32_t i = gid / P;
  uint32_t j = gid % P;

  float acc = 0.0f;
  size_t arow = (size_t)i * N;
  for (uint32_t k = 0; k < N; ++k) {
    acc += A[arow + k] * B[(size_t)k * P + j];
  }
  C[(size_t)i * P + j] = acc;
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  size_t total = (size_t)M * (size_t)P;
  CudaDims dim = CudaOneDim(total);
  MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out,
                                size_t out_size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= out_size) return;

  size_t base = gid * reduce_size;
  float v = a[base];
  for (size_t k = 1; k < reduce_size; ++k) {
    float x = a[base + k];
    v = fmaxf(v, x);
  }
  out[gid] = v;
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  size_t out_size = out->size;
  CudaDims dim = CudaOneDim(out_size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out_size, reduce_size);
  /// END SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out,
                                size_t out_size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= out_size) return;

  size_t base = gid * reduce_size;
  float acc = 0.0f;
  for (size_t k = 0; k < reduce_size; ++k) acc += a[base + k];
  out[gid] = acc;
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you
   * can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  size_t out_size = out->size;
  CudaDims dim = CudaOneDim(out_size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out_size, reduce_size);
  /// END SOLUTION
}

// ============================================================================
// CUDA FFT Implementation (Cooley-Tukey Algorithm)
// ============================================================================

/**
 * CUDA Kernel: Bit-reversal permutation
 *
 * Each thread handles one element, swapping with its bit-reversed counterpart.
 */
__global__ void BitReversalKernel(scalar_t* real, scalar_t* imag, size_t n, size_t bits) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // Compute bit-reversed index
  size_t j = 0;
  for (size_t b = 0; b < bits; b++) {
    j = (j << 1) | ((i >> b) & 1);
  }

  // Swap only if j > i to avoid double-swapping
  if (j > i) {
    scalar_t temp_real = real[i];
    scalar_t temp_imag = imag[i];
    real[i] = real[j];
    imag[i] = imag[j];
    real[j] = temp_real;
    imag[j] = temp_imag;
  }
}

/**
 * CUDA Kernel: FFT Butterfly Operation for a single stage
 *
 * Each thread handles one butterfly computation.
 *
 * Butterfly:
 *   t = W * a[idx2]
 *   a[idx1] = u + t
 *   a[idx2] = u - t
 */
__global__ void FFTButterflyKernel(scalar_t* real, scalar_t* imag, size_t n,
                                    size_t stage) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  size_t m = 1 << stage;  // 2^stage
  size_t num_butterflies = n / 2;

  if (tid >= num_butterflies) return;

  // Determine which group and position within group
  size_t group_size = m / 2;
  size_t group_id = tid / group_size;
  size_t j = tid % group_size;

  size_t k = group_id * m;
  size_t idx1 = k + j;
  size_t idx2 = k + j + m / 2;

  // Compute twiddle factor: W = exp(-2πi * j / m)
  const double PI = 3.141592653589793;
  double angle = -2.0 * PI * j / m;
  double w_real = cos(angle);
  double w_imag = sin(angle);

  // Complex multiplication: t = W * a[idx2]
  double t_real = w_real * real[idx2] - w_imag * imag[idx2];
  double t_imag = w_real * imag[idx2] + w_imag * real[idx2];

  double u_real = real[idx1];
  double u_imag = imag[idx1];

  // Butterfly computation
  real[idx1] = u_real + t_real;
  imag[idx1] = u_imag + t_imag;
  real[idx2] = u_real - t_real;
  imag[idx2] = u_imag - t_imag;
}

/**
 * CUDA Cooley-Tukey FFT
 *
 * Parallel FFT implementation using CUDA.
 * Time complexity: O(log N) with O(N) parallelism per stage
 *
 * Args:
 *   a: Input array (real-valued)
 *   out_real: Output real part
 *   out_imag: Output imaginary part
 *   n: Array size (must be power of 2)
 */
void CooleyTukeyFFTCuda(const CudaArray& a, CudaArray* out_real,
                        CudaArray* out_imag, size_t n) {
  // Check that n is a power of 2
  if (n == 0 || (n & (n - 1)) != 0) {
    throw std::invalid_argument("FFT size must be a power of 2");
  }

  // Initialize output: copy input to real part, zero imaginary part
  cudaMemcpy(out_real->ptr, a.ptr, n * ELEM_SIZE, cudaMemcpyDeviceToDevice);
  cudaMemset(out_imag->ptr, 0, n * ELEM_SIZE);

  // Compute number of stages
  size_t num_stages = 0;
  size_t temp = n;
  while (temp > 1) {
    temp >>= 1;
    num_stages++;
  }

  // Step 1: Bit-reversal permutation
  CudaDims dim = CudaOneDim(n);
  BitReversalKernel<<<dim.grid, dim.block>>>(out_real->ptr, out_imag->ptr, n, num_stages);
  cudaDeviceSynchronize();  // Wait for bit-reversal to complete

  // Step 2: FFT butterfly stages
  for (size_t stage = 1; stage <= num_stages; stage++) {
    size_t num_butterflies = n / 2;
    CudaDims butterfly_dim = CudaOneDim(num_butterflies);
    FFTButterflyKernel<<<butterfly_dim.grid, butterfly_dim.block>>>(
        out_real->ptr, out_imag->ptr, n, stage);
    cudaDeviceSynchronize();  // Wait for each stage to complete
  }
}

/**
 * CUDA Kernel: Complex conjugate and normalize
 */
__global__ void ConjugateAndNormalizeKernel(scalar_t* real, scalar_t* imag,
                                             size_t n, scalar_t scale) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;

  imag[gid] = -imag[gid];  // Conjugate
  real[gid] *= scale;      // Normalize
  imag[gid] *= scale;
}

/**
 * CUDA Cooley-Tukey IFFT
 *
 * IFFT(X) = conj(FFT(conj(X))) / N
 *
 * Args:
 *   a_real: Input real part
 *   a_imag: Input imaginary part
 *   out: Output (real-valued)
 *   n: Array size (must be power of 2)
 */
void CooleyTukeyIFFTCuda(const CudaArray& a_real, const CudaArray& a_imag,
                         CudaArray* out, size_t n) {
  if (n == 0 || (n & (n - 1)) != 0) {
    throw std::invalid_argument("IFFT size must be a power of 2");
  }

  // Allocate temporary arrays on GPU
  CudaArray result_real(n);
  CudaArray result_imag(n);

  // Step 1: Copy input and negate imaginary part (conjugate)
  cudaMemcpy(result_real.ptr, a_real.ptr, n * ELEM_SIZE, cudaMemcpyDeviceToDevice);
  cudaMemcpy(result_imag.ptr, a_imag.ptr, n * ELEM_SIZE, cudaMemcpyDeviceToDevice);

  CudaDims dim = CudaOneDim(n);
  ConjugateAndNormalizeKernel<<<dim.grid, dim.block>>>(
      result_real.ptr, result_imag.ptr, n, 1.0f);
  cudaDeviceSynchronize();

  // Step 2: Bit-reversal permutation
  size_t num_stages = 0;
  size_t temp = n;
  while (temp > 1) {
    temp >>= 1;
    num_stages++;
  }

  BitReversalKernel<<<dim.grid, dim.block>>>(result_real.ptr, result_imag.ptr, n, num_stages);
  cudaDeviceSynchronize();

  // Step 3: FFT butterfly stages (same as FFT)
  for (size_t stage = 1; stage <= num_stages; stage++) {
    size_t num_butterflies = n / 2;
    CudaDims butterfly_dim = CudaOneDim(num_butterflies);

    FFTButterflyKernel<<<butterfly_dim.grid, butterfly_dim.block>>>(
        result_real.ptr, result_imag.ptr, n, stage);

    cudaDeviceSynchronize();
  }

  // Step 4: Conjugate and normalize by 1/N
  ConjugateAndNormalizeKernel<<<dim.grid, dim.block>>>(
      result_real.ptr, result_imag.ptr, n, 1.0f / n);
  cudaDeviceSynchronize();

  // Copy real part to output
  cudaMemcpy(out->ptr, result_real.ptr, n * ELEM_SIZE, cudaMemcpyDeviceToDevice);
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);

  // FFT functions
  m.def("cooley_tukey_fft_cuda", CooleyTukeyFFTCuda);
  m.def("cooley_tukey_ifft_cuda", CooleyTukeyIFFTCuda);
}
