#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  size_t ndim = shape.size();
  size_t total = 1;
  for (int32_t s : shape) total *= (size_t)s;

  std::vector<size_t> idx(ndim, 0);
  for (size_t cnt = 0; cnt < total; ++cnt) {
    size_t pos = offset;
    for (size_t d = 0; d < ndim; ++d) pos += (size_t)strides[d] * idx[d];
    out->ptr[cnt] = a.ptr[pos];

    for (ssize_t d = (ssize_t)ndim - 1; d >= 0; --d) {
      idx[d]++;
      if (idx[d] < (size_t)shape[d]) break;
      idx[d] = 0;
      if (d == 0) break;
    }
  }
  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  size_t ndim = shape.size();
  size_t total = 1;
  for (int32_t s : shape) total *= (size_t)s;

  std::vector<size_t> idx(ndim, 0);
  for (size_t cnt = 0; cnt < total; ++cnt) {
    size_t pos = offset;
    for (size_t d = 0; d < ndim; ++d) pos += (size_t)strides[d] * idx[d];
    out->ptr[pos] = a.ptr[cnt];

    for (ssize_t d = (ssize_t)ndim - 1; d >= 0; --d) {
      idx[d]++;
      if (idx[d] < (size_t)shape[d]) break;
      idx[d] = 0;
      if (d == 0) break;
    }
  }
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  size_t ndim = shape.size();
  std::vector<size_t> idx(ndim, 0);
  for (size_t cnt = 0; cnt < size; ++cnt) {
    size_t pos = offset;
    for (size_t d = 0; d < ndim; ++d) pos += (size_t)strides[d] * idx[d];
    out->ptr[pos] = val;

    for (ssize_t d = (ssize_t)ndim - 1; d >= 0; --d) {
      idx[d]++;
      if (idx[d] < (size_t)shape[d]) break;
      idx[d] = 0;
      if (d == 0) break;
    }
  }
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
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

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = a.ptr[i] * b.ptr[i];
}
void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = a.ptr[i] * val;
}
void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = a.ptr[i] / b.ptr[i];
}
void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = a.ptr[i] / val;
}
void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = std::pow(a.ptr[i], val);
}
void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
}
void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = std::max(a.ptr[i], val);
}
void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = (a.ptr[i] == b.ptr[i]) ? 1.0f : 0.0f;
}
void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = (a.ptr[i] == val) ? 1.0f : 0.0f;
}
void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = (a.ptr[i] >= b.ptr[i]) ? 1.0f : 0.0f;
}
void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = (a.ptr[i] >= val) ? 1.0f : 0.0f;
}
void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = std::log(a.ptr[i]);
}
void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = std::exp(a.ptr[i]);
}
void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  for (size_t i = 0; i < a.size; ++i) out->ptr[i] = std::tanh(a.ptr[i]);
}

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < p; ++j) {
      float acc = 0.0f;
      size_t a_row = (size_t)i * n;
      size_t b_col = (size_t)j;
      for (uint32_t k = 0; k < n; ++k) {
        acc += a.ptr[a_row + k] * b.ptr[(size_t)k * p + b_col];
      }
      out->ptr[(size_t)i * p + j] = acc;
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for (int i = 0; i < TILE; ++i) {
    for (int j = 0; j < TILE; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < TILE; ++k) {
        acc += a[i * TILE + k] * b[k * TILE + j];
      }
      out[i * TILE + j] += acc;
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  const uint32_t mt = m / TILE;
  const uint32_t nt = n / TILE;
  const uint32_t pt = p / TILE;
  const size_t tile_area = (size_t)TILE * TILE;

  for (int i = 0; i < m * p; i++) out->ptr[i] = 0;
  for (int i = 0; i < mt; i++) {
    for (int j = 0; j < pt; j++) {
      for (int k = 0; k < nt; k++) {
        AlignedDot(&a.ptr[i * n * TILE + k * tile_area], 
                   &b.ptr[k * p * TILE + j * tile_area], 
                   &out->ptr[i * p * TILE + j * tile_area]);
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  size_t groups = out->size;
  for (size_t g = 0; g < groups; ++g) {
    size_t base = g * reduce_size;
    float v = a.ptr[base];
    for (size_t k = 1; k < reduce_size; ++k) {
      float x = a.ptr[base + k];
      if (x > v) v = x;
    }
    out->ptr[g] = v;
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  size_t groups = out->size;
  for (size_t g = 0; g < groups; ++g) {
    size_t base = g * reduce_size;
    float acc = 0.0f;
    for (size_t k = 0; k < reduce_size; ++k) acc += a.ptr[base + k];
    out->ptr[g] = acc;
  }
  /// END SOLUTION
}

// ============================================================================
// FFT Implementation (Cooley-Tukey Algorithm)
// ============================================================================

const double PI = 3.141592653589793;

/**
 * Bit-reversal permutation for FFT
 *
 * Rearranges array elements according to bit-reversed indices.
 * For example, for n=8:
 *   0 (000) <-> 0 (000)
 *   1 (001) <-> 4 (100)
 *   2 (010) <-> 2 (010)
 *   3 (011) <-> 6 (110)
 *   etc.
 */
void BitReversalPermutation(scalar_t* real, scalar_t* imag, size_t n) {
  size_t bits = 0;
  size_t temp = n;
  while (temp > 1) {
    temp >>= 1;
    bits++;
  }

  for (size_t i = 0; i < n; i++) {
    // Compute bit-reversed index
    size_t j = 0;
    for (size_t b = 0; b < bits; b++) {
      j = (j << 1) | ((i >> b) & 1);
    }

    // Swap if j > i (to avoid swapping twice)
    if (j > i) {
      std::swap(real[i], real[j]);
      std::swap(imag[i], imag[j]);
    }
  }
}

/**
 * Cooley-Tukey FFT (iterative, in-place)
 *
 * Implements the radix-2 decimation-in-time FFT algorithm.
 * Time complexity: O(N log N)
 * Space complexity: O(1) (in-place)
 *
 * Args:
 *   a: Input array (real-valued, compact)
 *   out_real: Output array for real part
 *   out_imag: Output array for imaginary part
 *   n: Array size (must be power of 2)
 *
 * Algorithm:
 *   1. Bit-reversal permutation
 *   2. Iterative butterfly operations
 *   3. Uses twiddle factors: W_N^k = exp(-2πi k/N)
 */
void CooleyTukeyFFT(const AlignedArray& a, AlignedArray* out_real,
                    AlignedArray* out_imag, size_t n) {
  // Check that n is a power of 2
  if (n == 0 || (n & (n - 1)) != 0) {
    throw std::invalid_argument("FFT size must be a power of 2");
  }

  // Initialize output: copy input to real part, zero imaginary part
  for (size_t i = 0; i < n; i++) {
    out_real->ptr[i] = a.ptr[i];
    out_imag->ptr[i] = 0.0f;
  }

  // Step 1: Bit-reversal permutation
  BitReversalPermutation(out_real->ptr, out_imag->ptr, n);

  // Step 2: Iterative FFT (butterfly operations)
  // Iterate over stages (log2(n) stages total)
  for (size_t stage = 1; stage <= static_cast<size_t>(log2(n)); stage++) {
    size_t m = 1 << stage;  // Size of DFT for this stage (2^stage)

    // Twiddle factor: W_m = exp(-2πi/m)
    double wm_real = cos(-2.0 * PI / m);
    double wm_imag = sin(-2.0 * PI / m);

    // Process each group of size m
    for (size_t k = 0; k < n; k += m) {
      double w_real = 1.0;
      double w_imag = 0.0;

      // Butterfly operations within this group
      for (size_t j = 0; j < m / 2; j++) {
        size_t idx1 = k + j;
        size_t idx2 = k + j + m / 2;

        // Complex multiplication: t = W * a[idx2]
        double t_real = w_real * out_real->ptr[idx2] - w_imag * out_imag->ptr[idx2];
        double t_imag = w_real * out_imag->ptr[idx2] + w_imag * out_real->ptr[idx2];

        double u_real = out_real->ptr[idx1];
        double u_imag = out_imag->ptr[idx1];

        // Butterfly:
        //   a[idx1] = u + t
        //   a[idx2] = u - t
        out_real->ptr[idx1] = u_real + t_real;
        out_imag->ptr[idx1] = u_imag + t_imag;
        out_real->ptr[idx2] = u_real - t_real;
        out_imag->ptr[idx2] = u_imag - t_imag;

        // Update twiddle factor: W = W * W_m
        double w_temp = w_real;
        w_real = w_real * wm_real - w_imag * wm_imag;
        w_imag = w_temp * wm_imag + w_imag * wm_real;
      }
    }
  }
}

/**
 * Cooley-Tukey IFFT (using FFT with conjugate trick)
 *
 * IFFT(X) = conj(FFT(conj(X))) / N
 *
 * Args:
 *   a_real: Input real part
 *   a_imag: Input imaginary part
 *   out: Output array (real-valued, imaginary part should be ~0)
 *   n: Array size (must be power of 2)
 */
void CooleyTukeyIFFT(const AlignedArray& a_real, const AlignedArray& a_imag,
                     AlignedArray* out, size_t n) {
  if (n == 0 || (n & (n - 1)) != 0) {
    throw std::invalid_argument("IFFT size must be a power of 2");
  }

  // Create temporary arrays for result
  AlignedArray result_real(n);
  AlignedArray result_imag(n);

  // Step 1: Copy and negate imaginary part (conjugate)
  for (size_t i = 0; i < n; i++) {
    result_real.ptr[i] = a_real.ptr[i];
    result_imag.ptr[i] = -a_imag.ptr[i];
  }

  // Step 2: Bit-reversal permutation
  BitReversalPermutation(result_real.ptr, result_imag.ptr, n);

  // Step 3: Iterative FFT stages (same as FFT)
  size_t num_stages = 0;
  size_t temp = n;
  while (temp > 1) {
    temp >>= 1;
    num_stages++;
  }

  for (size_t stage = 1; stage <= num_stages; stage++) {
    size_t m = 1 << stage;  // 2^stage
    double wm_real = cos(-2.0 * PI / m);
    double wm_imag = sin(-2.0 * PI / m);

    for (size_t k = 0; k < n; k += m) {
      double w_real = 1.0;
      double w_imag = 0.0;

      for (size_t j = 0; j < m / 2; j++) {
        size_t idx1 = k + j;
        size_t idx2 = k + j + m / 2;

        // Complex multiplication: t = w * x[idx2]
        double t_real = w_real * result_real.ptr[idx2] - w_imag * result_imag.ptr[idx2];
        double t_imag = w_real * result_imag.ptr[idx2] + w_imag * result_real.ptr[idx2];

        double u_real = result_real.ptr[idx1];
        double u_imag = result_imag.ptr[idx1];

        // Butterfly operations
        result_real.ptr[idx1] = u_real + t_real;
        result_imag.ptr[idx1] = u_imag + t_imag;
        result_real.ptr[idx2] = u_real - t_real;
        result_imag.ptr[idx2] = u_imag - t_imag;

        // Update twiddle factor: w *= wm
        double w_temp = w_real;
        w_real = w_real * wm_real - w_imag * wm_imag;
        w_imag = w_temp * wm_imag + w_imag * wm_real;
      }
    }
  }

  // Step 4: Conjugate output and normalize
  for (size_t i = 0; i < n; i++) {
    out->ptr[i] = result_real.ptr[i] / static_cast<double>(n);
    // Note: Imaginary part should be ~0 for real input signals
    // We ignore it as the output is real-valued
  }
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
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
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);

  // FFT functions
  m.def("cooley_tukey_fft", CooleyTukeyFFT);
  m.def("cooley_tukey_ifft", CooleyTukeyIFFT);
}
