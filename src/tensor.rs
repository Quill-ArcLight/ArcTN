//! Contiguous row-major dense tensors backed by `Vec`.
//!
//! Permutations copy data; reshaping changes only shape metadata.
use num_complex::{Complex32, Complex64};
use rand::Rng;

/// Returns the element count while rejecting zero-length axes and overflow.
///
/// Scalars use shape `[]` and contain one element. A successful check makes
/// later row-major strides, offsets, and GEMM sizes non-wrapping.
pub fn checked_numel(shape: &[usize]) -> Result<usize, String> {
    let mut n = 1usize;
    for (axis, &dim) in shape.iter().enumerate() {
        if dim == 0 {
            return Err(format!("shape 第 {axis} 轴为 0；ArcTN 暂不支持零长度轴"));
        }
        n = n
            .checked_mul(dim)
            .ok_or_else(|| format!("shape {:?} 的元素数溢出 usize", shape))?;
    }
    Ok(n)
}

/// Scalar operations required by dense tensor kernels.
///
/// GEMM dispatches to the matching `matrixmultiply` kernel.
pub trait Scalar:
    Copy
    + Send
    + Sync
    + 'static
    + std::fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::AddAssign
    + num_traits::Zero
    + num_traits::One
{
    /// Computes `C = alpha * A * B + beta * C` with arbitrary strides.
    ///
    /// # Safety
    ///
    /// `a`, `b`, and `c` must address every element implied by their dimensions
    /// and strides. All reachable offsets must fit `isize`, logical outputs in
    /// `c` must not alias, and backend pointer/alignment requirements must hold.
    /// Only `checked_batch_gemm` may call this method after validating them.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm(
        m: usize,
        k: usize,
        n: usize,
        alpha: Self,
        a: *const Self,
        rsa: isize,
        csa: isize,
        b: *const Self,
        rsb: isize,
        csb: isize,
        beta: Self,
        c: *mut Self,
        rsc: isize,
        csc: isize,
    );

    fn random<R: Rng>(rng: &mut R) -> Self;

    /// Squared magnitude used by error norms.
    fn abs_sq(self) -> f64;

    /// Magnitude computed without first squaring, preserving tiny SVD phases.
    fn abs(self) -> f64;

    /// Converts an `f64` real scalar, with zero imaginary part when complex.
    fn from_real(x: f64) -> Self;

    /// Divides by an `f64` denominator without narrowing its reciprocal first.
    /// Custom scalar types should override this for very small denominators.
    fn div_real(self, denominator: f64) -> Self {
        self * Self::from_real(1.0 / denominator)
    }

    /// Encoded bytes per little-endian element.
    const NBYTES: usize;
    /// Decodes one element from exactly `NBYTES` little-endian bytes.
    fn from_le_bytes(bytes: &[u8]) -> Self;

    /// Complex conjugate; identity for real values.
    fn conj(self) -> Self;

    /// Relative Jacobi convergence threshold appropriate for this precision.
    const SVD_EPSILON: f64 = 1.0e-15;
}

impl Scalar for f32 {
    unsafe fn gemm(
        m: usize,
        k: usize,
        n: usize,
        alpha: Self,
        a: *const Self,
        rsa: isize,
        csa: isize,
        b: *const Self,
        rsb: isize,
        csb: isize,
        beta: Self,
        c: *mut Self,
        rsc: isize,
        csc: isize,
    ) {
        unsafe {
            matrixmultiply::sgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
        }
    }

    fn random<R: Rng>(rng: &mut R) -> Self {
        rng.gen_range(-1.0f32..1.0f32)
    }

    fn abs_sq(self) -> f64 {
        let x = f64::from(self);
        x * x
    }
    fn abs(self) -> f64 {
        f64::from(f32::abs(self))
    }
    fn from_real(x: f64) -> Self {
        x as f32
    }
    fn div_real(self, denominator: f64) -> Self {
        (f64::from(self) / denominator) as f32
    }
    const NBYTES: usize = 4;
    fn from_le_bytes(bytes: &[u8]) -> Self {
        f32::from_le_bytes(bytes.try_into().expect("f32 需 4 字节"))
    }
    fn conj(self) -> Self {
        self
    }
    const SVD_EPSILON: f64 = 1.0e-6;
}

impl Scalar for f64 {
    unsafe fn gemm(
        m: usize,
        k: usize,
        n: usize,
        alpha: Self,
        a: *const Self,
        rsa: isize,
        csa: isize,
        b: *const Self,
        rsb: isize,
        csb: isize,
        beta: Self,
        c: *mut Self,
        rsc: isize,
        csc: isize,
    ) {
        unsafe {
            matrixmultiply::dgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc)
        }
    }

    fn random<R: Rng>(rng: &mut R) -> Self {
        rng.gen_range(-1.0..1.0)
    }

    fn abs_sq(self) -> f64 {
        self * self
    }
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    fn from_real(x: f64) -> Self {
        x
    }
    fn div_real(self, denominator: f64) -> Self {
        self / denominator
    }
    const NBYTES: usize = 8;
    fn from_le_bytes(bytes: &[u8]) -> Self {
        f64::from_le_bytes(bytes.try_into().expect("f64 需 8 字节"))
    }
    fn conj(self) -> Self {
        self
    }
    const SVD_EPSILON: f64 = 1.0e-15;
}

impl Scalar for Complex32 {
    unsafe fn gemm(
        m: usize,
        k: usize,
        n: usize,
        alpha: Self,
        a: *const Self,
        rsa: isize,
        csa: isize,
        b: *const Self,
        rsb: isize,
        csb: isize,
        beta: Self,
        c: *mut Self,
        rsc: isize,
        csc: isize,
    ) {
        use matrixmultiply::CGemmOption;
        // Complex32 matches the C layout required by `matrixmultiply::cgemm`.
        unsafe {
            matrixmultiply::cgemm(
                CGemmOption::Standard,
                CGemmOption::Standard,
                m,
                k,
                n,
                [alpha.re, alpha.im],
                a as *const [f32; 2],
                rsa,
                csa,
                b as *const [f32; 2],
                rsb,
                csb,
                [beta.re, beta.im],
                c as *mut [f32; 2],
                rsc,
                csc,
            )
        }
    }

    fn random<R: Rng>(rng: &mut R) -> Self {
        Complex32::new(
            rng.gen_range(-1.0f32..1.0f32),
            rng.gen_range(-1.0f32..1.0f32),
        )
    }

    fn abs_sq(self) -> f64 {
        let re = f64::from(self.re);
        let im = f64::from(self.im);
        re * re + im * im
    }
    fn abs(self) -> f64 {
        f64::from(self.re).hypot(f64::from(self.im))
    }
    fn from_real(x: f64) -> Self {
        Complex32::new(x as f32, 0.0)
    }
    fn div_real(self, denominator: f64) -> Self {
        Complex32::new(
            (f64::from(self.re) / denominator) as f32,
            (f64::from(self.im) / denominator) as f32,
        )
    }
    const NBYTES: usize = 8;
    fn from_le_bytes(bytes: &[u8]) -> Self {
        let re = f32::from_le_bytes(bytes[0..4].try_into().expect("re 需 4 字节"));
        let im = f32::from_le_bytes(bytes[4..8].try_into().expect("im 需 4 字节"));
        Complex32::new(re, im)
    }
    fn conj(self) -> Self {
        Complex32::conj(&self)
    }
    const SVD_EPSILON: f64 = 1.0e-6;
}

impl Scalar for Complex64 {
    unsafe fn gemm(
        m: usize,
        k: usize,
        n: usize,
        alpha: Self,
        a: *const Self,
        rsa: isize,
        csa: isize,
        b: *const Self,
        rsb: isize,
        csb: isize,
        beta: Self,
        c: *mut Self,
        rsc: isize,
        csc: isize,
    ) {
        use matrixmultiply::CGemmOption;
        // Complex64 is C-compatible with two consecutive f64 values.
        unsafe {
            matrixmultiply::zgemm(
                CGemmOption::Standard,
                CGemmOption::Standard,
                m,
                k,
                n,
                [alpha.re, alpha.im],
                a as *const [f64; 2],
                rsa,
                csa,
                b as *const [f64; 2],
                rsb,
                csb,
                [beta.re, beta.im],
                c as *mut [f64; 2],
                rsc,
                csc,
            )
        }
    }

    fn random<R: Rng>(rng: &mut R) -> Self {
        Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
    }

    fn abs_sq(self) -> f64 {
        self.norm_sqr()
    }
    fn abs(self) -> f64 {
        self.norm()
    }
    fn from_real(x: f64) -> Self {
        Complex64::new(x, 0.0)
    }
    fn div_real(self, denominator: f64) -> Self {
        Complex64::new(self.re / denominator, self.im / denominator)
    }
    const NBYTES: usize = 16;
    fn from_le_bytes(bytes: &[u8]) -> Self {
        // Decode little-endian real and imaginary parts in NumPy complex128 order.
        let re = f64::from_le_bytes(bytes[0..8].try_into().expect("re 需 8 字节"));
        let im = f64::from_le_bytes(bytes[8..16].try_into().expect("im 需 8 字节"));
        Complex64::new(re, im)
    }
    fn conj(self) -> Self {
        Complex64::conj(&self)
    }
    const SVD_EPSILON: f64 = 1.0e-15;
}

#[derive(Clone, Debug)]
pub struct DenseTensor<T> {
    // Layout is private because unsafe GEMM relies on checked construction.
    pub(crate) shape: Vec<usize>,
    pub(crate) data: Vec<T>,
}

impl<T> DenseTensor<T> {
    /// Constructs a tensor after validating its row-major layout.
    pub fn try_from_data(shape: Vec<usize>, data: Vec<T>) -> Result<Self, String> {
        let n = checked_numel(&shape)?;
        if n != data.len() {
            return Err(format!(
                "张量 shape {:?} 需要 {n} 个元素，实际给了 {} 个",
                shape,
                data.len()
            ));
        }
        Ok(DenseTensor { shape, data })
    }

    /// Panicking counterpart to [`Self::try_from_data`].
    pub fn from_data(shape: Vec<usize>, data: Vec<T>) -> Self {
        Self::try_from_data(shape, data).unwrap_or_else(|e| panic!("DenseTensor::from_data: {e}"))
    }

    /// Returns the immutable shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the row-major elements.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns mutable elements without exposing layout metadata.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consumes the tensor and returns its elements.
    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    /// Consumes the tensor and returns `(shape, data)`.
    pub fn into_parts(self) -> (Vec<usize>, Vec<T>) {
        (self.shape, self.data)
    }

    /// Revalidates layout at executor and FFI boundaries.
    pub fn validate_layout(&self) -> Result<(), String> {
        let n = checked_numel(&self.shape)?;
        if n != self.data.len() {
            return Err(format!(
                "张量布局无效：shape {:?} 需要 {n} 个元素，data 长度为 {}",
                self.shape,
                self.data.len()
            ));
        }
        Ok(())
    }
}

impl<T: Scalar> DenseTensor<T> {
    /// Constructs zeros without wrapping an unrepresentable element count.
    pub fn try_zeros(shape: Vec<usize>) -> Result<Self, String> {
        let n = checked_numel(&shape)?;
        Ok(DenseTensor {
            shape,
            data: vec![T::zero(); n],
        })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::try_zeros(shape).unwrap_or_else(|e| panic!("DenseTensor::zeros: {e}"))
    }

    pub fn scalar(v: T) -> Self {
        DenseTensor {
            shape: vec![],
            data: vec![v],
        }
    }

    pub fn random<R: Rng>(shape: Vec<usize>, rng: &mut R) -> Self {
        let n = checked_numel(&shape).unwrap_or_else(|e| panic!("DenseTensor::random: {e}"));
        DenseTensor::try_from_data(shape, (0..n).map(|_| T::random(rng)).collect())
            .expect("checked numel 的随机张量布局应合法")
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns row-major strides.
    pub fn strides(&self) -> Vec<usize> {
        self.validate_layout()
            .expect("DenseTensor::strides 只能用于合法布局");
        let mut s = vec![1usize; self.shape.len()];
        for i in (0..self.shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1]
                .checked_mul(self.shape[i + 1])
                .expect("合法 DenseTensor 的 stride 不应溢出");
        }
        s
    }

    /// Copies axes in NumPy transpose order: output axis `d` is input `axes[d]`.
    pub fn permute(&self, axes: &[usize]) -> Self {
        assert_eq!(axes.len(), self.ndim());
        // Reject repeated and out-of-range axes before offset arithmetic.
        assert!(
            {
                let mut seen = vec![false; self.ndim()];
                axes.iter()
                    .all(|&a| a < self.ndim() && !std::mem::replace(&mut seen[a], true))
            },
            "permute: axes {axes:?} 不是 shape {:?} 的合法排列",
            self.shape
        );
        if axes.iter().enumerate().all(|(i, &a)| i == a) {
            return self.clone();
        }
        let in_strides = self.strides();
        let out_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let step: Vec<usize> = axes.iter().map(|&a| in_strides[a]).collect();
        let n = self.numel();
        let mut out = Vec::with_capacity(n);
        let ndim = out_shape.len();
        let mut idx = vec![0usize; ndim];
        let mut off = 0usize;
        for _ in 0..n {
            out.push(self.data[off]);
            for d in (0..ndim).rev() {
                idx[d] += 1;
                off += step[d];
                if idx[d] < out_shape[d] {
                    break;
                }
                off -= step[d] * out_shape[d];
                idx[d] = 0;
            }
        }
        DenseTensor {
            shape: out_shape,
            data: out,
        }
    }

    /// Reshapes contiguous data without copying.
    pub fn reshape(mut self, shape: Vec<usize>) -> Self {
        let n = checked_numel(&shape).unwrap_or_else(|e| panic!("reshape 的目标 shape 非法: {e}"));
        assert_eq!(n, self.data.len(), "reshape 元素数不匹配");
        self.shape = shape;
        self
    }

    /// Traces equal-sized axes `ax_i < ax_j`.
    pub fn trace_pair(&self, ax_i: usize, ax_j: usize) -> Self {
        assert!(ax_i < ax_j);
        let d = self.shape[ax_i];
        assert_eq!(d, self.shape[ax_j]);
        let strides = self.strides();
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != ax_i && *i != ax_j)
            .map(|(_, &s)| s)
            .collect();
        let out_axes: Vec<usize> = (0..self.ndim())
            .filter(|&i| i != ax_i && i != ax_j)
            .collect();
        let out_steps: Vec<usize> = out_axes.iter().map(|&a| strides[a]).collect();
        let diag_step = strides[ax_i] + strides[ax_j];
        let n_out: usize = out_shape.iter().product();
        let mut out = vec![T::zero(); n_out];
        let ndim_out = out_shape.len();
        let mut idx = vec![0usize; ndim_out];
        let mut base = 0usize;
        for slot in out.iter_mut() {
            let mut acc = T::zero();
            let mut off = base;
            for _ in 0..d {
                acc += self.data[off];
                off += diag_step;
            }
            *slot = acc;
            for dd in (0..ndim_out).rev() {
                idx[dd] += 1;
                base += out_steps[dd];
                if idx[dd] < out_shape[dd] {
                    break;
                }
                base -= out_steps[dd] * out_shape[dd];
                idx[dd] = 0;
            }
        }
        DenseTensor {
            shape: out_shape,
            data: out,
        }
    }

    /// Extracts the diagonal of equal-sized axes while retaining `ax_i`.
    pub fn diag_pair(&self, ax_i: usize, ax_j: usize) -> Self {
        assert!(ax_i < ax_j);
        let d = self.shape[ax_i];
        assert_eq!(d, self.shape[ax_j], "diag_pair: 两轴维度须相同");
        let strides = self.strides();
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != ax_j)
            .map(|(_, &s)| s)
            .collect();
        let ndim_out = out_shape.len();
        // Advance both source axes when incrementing the retained diagonal axis.
        let step: Vec<usize> = (0..ndim_out)
            .map(|o| {
                let in_ax = if o < ax_j { o } else { o + 1 };
                strides[in_ax] + if o == ax_i { strides[ax_j] } else { 0 }
            })
            .collect();
        let n_out: usize = out_shape.iter().product();
        let mut out = Vec::with_capacity(n_out);
        let mut idx = vec![0usize; ndim_out];
        let mut off = 0usize;
        for _ in 0..n_out {
            out.push(self.data[off]);
            for dd in (0..ndim_out).rev() {
                idx[dd] += 1;
                off += step[dd];
                if idx[dd] < out_shape[dd] {
                    break;
                }
                off -= step[dd] * out_shape[dd];
                idx[dd] = 0;
            }
        }
        DenseTensor {
            shape: out_shape,
            data: out,
        }
    }

    /// Sums over one axis.
    pub fn sum_axis(&self, ax: usize) -> Self {
        let d = self.shape[ax];
        let strides = self.strides();
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != ax)
            .map(|(_, &s)| s)
            .collect();
        let out_axes: Vec<usize> = (0..self.ndim()).filter(|&i| i != ax).collect();
        let out_steps: Vec<usize> = out_axes.iter().map(|&a| strides[a]).collect();
        let ax_step = strides[ax];
        let n_out: usize = out_shape.iter().product();
        let mut out = vec![T::zero(); n_out];
        let ndim_out = out_shape.len();
        let mut idx = vec![0usize; ndim_out];
        let mut base = 0usize;
        for slot in out.iter_mut() {
            let mut acc = T::zero();
            let mut off = base;
            for _ in 0..d {
                acc += self.data[off];
                off += ax_step;
            }
            *slot = acc;
            for dd in (0..ndim_out).rev() {
                idx[dd] += 1;
                base += out_steps[dd];
                if idx[dd] < out_shape[dd] {
                    break;
                }
                base -= out_steps[dd] * out_shape[dd];
                idx[dd] = 0;
            }
        }
        DenseTensor {
            shape: out_shape,
            data: out,
        }
    }

    /// Selects one index along an axis and removes that axis.
    pub fn select_axis(&self, ax: usize, i: usize) -> Self {
        assert!(i < self.shape[ax]);
        let strides = self.strides();
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|(d, _)| *d != ax)
            .map(|(_, &s)| s)
            .collect();
        let out_axes: Vec<usize> = (0..self.ndim()).filter(|&d| d != ax).collect();
        let out_steps: Vec<usize> = out_axes.iter().map(|&a| strides[a]).collect();
        let n_out: usize = out_shape.iter().product();
        let mut out = Vec::with_capacity(n_out);
        let ndim_out = out_shape.len();
        let mut idx = vec![0usize; ndim_out];
        let mut base = i * strides[ax];
        for _ in 0..n_out {
            out.push(self.data[base]);
            for dd in (0..ndim_out).rev() {
                idx[dd] += 1;
                base += out_steps[dd];
                if idx[dd] < out_shape[dd] {
                    break;
                }
                base -= out_steps[dd] * out_shape[dd];
                idx[dd] = 0;
            }
        }
        DenseTensor {
            shape: out_shape,
            data: out,
        }
    }

    /// Returns the maximum elementwise error, or infinity for non-finite differences.
    pub fn max_abs_diff(&self, other: &Self) -> f64 {
        assert_eq!(self.shape, other.shape);
        self.data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| (a - b).abs_sq().sqrt())
            .fold(0.0, |acc, d| {
                if d.is_finite() {
                    acc.max(d)
                } else {
                    f64::INFINITY
                }
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{checked_numel, DenseTensor};

    #[test]
    fn checked_layout_rejects_zero_overflow_and_short_data() {
        assert!(checked_numel(&[0]).is_err());
        assert!(checked_numel(&[usize::MAX, 2]).is_err());
        assert!(DenseTensor::<f64>::try_from_data(vec![2, 2], vec![1.0]).is_err());
    }

    #[test]
    fn public_accessors_preserve_layout_invariant() {
        let mut t = DenseTensor::try_from_data(vec![2], vec![1.0, 2.0]).unwrap();
        assert_eq!(t.shape(), &[2]);
        t.data_mut()[0] = 3.0;
        assert_eq!(t.data(), &[3.0, 2.0]);
        let (shape, data) = t.into_parts();
        assert_eq!(shape, vec![2]);
        assert_eq!(data, vec![3.0, 2.0]);
    }
}
