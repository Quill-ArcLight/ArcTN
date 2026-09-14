//! Lightweight linear algebra for dense tensor operations.
//!
//! The one-sided Jacobi SVD computes the full spectrum without LAPACK or
//! ndarray; callers may retain the leading χ components.
//!
//! One generic `T: Scalar` implementation supports real and complex values at
//! both precisions. The complex algorithm phase-aligns each Hermitian Gram
//! off-diagonal before applying a real Jacobi rotation and accumulating it in
//! `V`.
//!
//! Matrices are row-major. The result satisfies `A = U · diag(s) · Vᴴ`,
//! where `U` is `m×k`, `s` is real and descending, and `Vᴴ` is `k×n`.
//! `k = min(m, n)`. For real values, `Vᴴ = Vᵀ`.

use crate::tensor::Scalar;

/// Convergence information for one-sided Jacobi SVD.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SvdInfo {
    /// Whether the final sweep met the scalar type's relative Gram threshold.
    pub converged: bool,
    /// Number of completed Jacobi sweeps.
    pub sweeps: usize,
    /// Maximum relative Gram off-diagonal in the final sweep.
    pub max_relative_off_diagonal: f64,
}

/// Computes a row-major matrix length without integer wraparound.
#[inline]
fn checked_matrix_len(rows: usize, cols: usize, op: &str) -> usize {
    rows.checked_mul(cols)
        .unwrap_or_else(|| panic!("{op}: 矩阵 {rows}×{cols} 的元素数溢出 usize"))
}

/// Computes row-major one-sided Jacobi SVD as `(U, s, Vᴴ)`.
pub fn svd<T: Scalar>(a: &[T], m: usize, n: usize) -> (Vec<T>, Vec<f64>, Vec<T>) {
    let (u, s, vt, _) = svd_with_info(a, m, n);
    (u, s, vt)
}

/// Equivalent to [`svd`] with Jacobi convergence information.
///
/// Callers interpreting singular values as Schmidt coefficients or error data
/// should require [`SvdInfo::converged`].
pub fn svd_with_info<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
) -> (Vec<T>, Vec<f64>, Vec<T>, SvdInfo) {
    let a_len = checked_matrix_len(m, n, "svd");
    assert_eq!(a.len(), a_len, "svd: 尺寸不符");
    if m == 0 || n == 0 {
        return (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            SvdInfo {
                converged: true,
                sweeps: 0,
                max_relative_off_diagonal: 0.0,
            },
        );
    }
    // One-sided Jacobi rotates columns of a tall matrix. For `m < n`, solve
    // `Aᴴ = U₂ S V₂ᴴ` and return `U = V₂`, `Vᴴ = U₂ᴴ`.
    if m < n {
        let ah = conj_transpose(a, m, n); // n×m
        let (u2, s, v2h, info) = svd_with_info(&ah, n, m); // u2:n×k, v2h:k×m, k=m
        let k = s.len();
        let u = conj_transpose(&v2h, k, m); // (k×m)ᴴ = m×k = U
        let vt = conj_transpose(&u2, n, k); // (n×k)ᴴ = k×n = Vᴴ
        return (u, s, vt, info);
    }
    // Scale by the largest magnitude before forming Gram products. This avoids
    // underflow in f32/C32 phase normalization without changing singular
    // vectors; singular values are rescaled on return.
    let k = n;
    let matrix_scale = a.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
    let normalized = matrix_scale > 0.0 && matrix_scale.is_finite();
    let mut w: Vec<T> = if normalized {
        a.iter().map(|&x| x.div_real(matrix_scale)).collect()
    } else {
        a.to_vec()
    }; // m×n
    let mut accv = vec![T::zero(); checked_matrix_len(n, n, "svd")];
    for i in 0..n {
        accv[i * n + i] = T::one();
    }
    let eps = T::SVD_EPSILON;
    let max_sweeps = 60;
    let mut info = SvdInfo {
        converged: false,
        sweeps: 0,
        max_relative_off_diagonal: f64::INFINITY,
    };
    for sweep in 0..max_sweeps {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                // Hermitian Gram entries: alpha=<w_p,w_p>, beta=<w_q,w_q>,
                // gamma=<w_p,w_q>.
                let mut alpha = 0.0f64;
                let mut beta = 0.0f64;
                let mut gamma = T::zero();
                for i in 0..m {
                    let xp = w[i * n + p];
                    let xq = w[i * n + q];
                    alpha += xp.abs_sq();
                    beta += xq.abs_sq();
                    gamma += xp.conj() * xq;
                }
                let denom = (alpha * beta).sqrt();
                // Compute |gamma| directly. Squaring first can underflow,
                // corrupt the unit phase, and make the rotation non-unitary.
                let gmag = gamma.abs();
                if denom > 0.0 {
                    off = off.max(gmag / denom);
                }
                if gmag <= eps * denom || denom == 0.0 {
                    continue;
                }
                // Phase-align gamma to a positive real value.
                let phase = gamma.div_real(gmag);
                let cphase = phase.conj();
                let zeta = (beta - alpha) / (2.0 * gmag);
                // Use the reciprocal form when zeta² would overflow so a
                // representable small rotation is not rounded to zero.
                let zeta_squared = zeta * zeta;
                let t = if zeta_squared.is_finite() {
                    zeta.signum() / (zeta.abs() + (1.0 + zeta_squared).sqrt())
                } else {
                    let inverse_abs_zeta = 1.0 / zeta.abs();
                    zeta.signum() * inverse_abs_zeta
                        / (1.0 + (1.0 + inverse_abs_zeta * inverse_abs_zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                let (ct, st) = (T::from_real(c), T::from_real(s));
                // Apply the unitary rotation to columns p and q.
                for i in 0..m {
                    let xp = w[i * n + p];
                    let cxq = cphase * w[i * n + q];
                    w[i * n + p] = ct * xp - st * cxq;
                    w[i * n + q] = st * xp + ct * cxq;
                }
                for i in 0..n {
                    let vp = accv[i * n + p];
                    let cvq = cphase * accv[i * n + q];
                    accv[i * n + p] = ct * vp - st * cvq;
                    accv[i * n + q] = st * vp + ct * cvq;
                }
            }
        }
        info.sweeps = sweep + 1;
        info.max_relative_off_diagonal = off;
        if off <= eps {
            info.converged = true;
            break;
        }
    }
    // Singular values are column norms; normalized columns form U.
    let mut sv = vec![0.0f64; k];
    for j in 0..n {
        let mut nrm = 0.0f64;
        for i in 0..m {
            nrm += w[i * n + j].abs_sq();
        }
        sv[j] = nrm.sqrt();
    }
    // Sort singular values and corresponding U/V columns in descending order.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| sv[y].total_cmp(&sv[x]));
    let mut u = vec![T::zero(); checked_matrix_len(m, k, "svd")];
    let mut s = vec![0.0f64; k];
    let mut vt = vec![T::zero(); checked_matrix_len(k, n, "svd")];
    for (newj, &oldj) in order.iter().enumerate() {
        s[newj] = sv[oldj] * if normalized { matrix_scale } else { 1.0 };
        for i in 0..m {
            u[i * k + newj] = if sv[oldj] > 0.0 {
                w[i * n + oldj].div_real(sv[oldj])
            } else {
                T::zero()
            };
        }
        for i in 0..n {
            // Vᴴ[newj, i] = conj(V[i, oldj])
            vt[newj * n + i] = accv[i * n + oldj].conj();
        }
    }
    (u, s, vt, info)
}

/// Transposes a row-major matrix without conjugation.
pub fn transpose<T: Scalar>(a: &[T], r: usize, c: usize) -> Vec<T> {
    let len = checked_matrix_len(r, c, "transpose");
    assert_eq!(a.len(), len, "transpose: 尺寸不符");
    let mut t = vec![T::zero(); len];
    for i in 0..r {
        for j in 0..c {
            t[j * r + i] = a[i * c + j];
        }
    }
    t
}

/// Computes the conjugate transpose of a row-major matrix.
pub fn conj_transpose<T: Scalar>(a: &[T], r: usize, c: usize) -> Vec<T> {
    let len = checked_matrix_len(r, c, "conj_transpose");
    assert_eq!(a.len(), len, "conj_transpose: 尺寸不符");
    let mut t = vec![T::zero(); len];
    for i in 0..r {
        for j in 0..c {
            t[j * r + i] = a[i * c + j].conj();
        }
    }
    t
}

/// Multiplies small row-major matrices with a naive kernel.
pub fn matmul<T: Scalar>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    let a_len = checked_matrix_len(m, k, "matmul");
    let b_len = checked_matrix_len(k, n, "matmul");
    let out_len = checked_matrix_len(m, n, "matmul");
    assert_eq!(a.len(), a_len, "matmul: 左矩阵尺寸不符");
    assert_eq!(b.len(), b_len, "matmul: 右矩阵尺寸不符");
    let mut out = vec![T::zero(); out_len];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            // Skip only exact zeros. Squaring first can underflow and discard a
            // representable coefficient, breaking full-rank reconstruction.
            if aip.abs() == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += aip * b[p * n + j];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    // f64 tests.
    fn recon_err(a: &[f64], u: &[f64], s: &[f64], vt: &[f64], m: usize, n: usize) -> f64 {
        let k = s.len();
        let mut us = u.to_vec();
        for i in 0..m {
            for j in 0..k {
                us[i * k + j] *= s[j];
            }
        }
        let rec = matmul(&us, vt, m, k, n);
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..m * n {
            num += (a[i] - rec[i]).powi(2);
            den += a[i] * a[i];
        }
        (num / den.max(1e-300)).sqrt()
    }

    #[test]
    fn svd_reconstructs_random_matrices() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for &(m, n) in &[(5, 5), (8, 3), (3, 8), (10, 7), (1, 6), (6, 1), (12, 12)] {
            let a: Vec<f64> = (0..m * n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
            let (u, s, vt) = svd(&a, m, n);
            let k = m.min(n);
            assert_eq!(s.len(), k);
            for w in s.windows(2) {
                assert!(w[0] >= w[1] - 1e-12, "奇异值非降序: {:?}", s);
                assert!(w[1] >= -1e-12);
            }
            let e = recon_err(&a, &u, &s, &vt, m, n);
            assert!(e < 1e-9, "重构误差过大 m={m} n={n}: {e:e}");
        }
    }

    #[test]
    fn svd_singular_values_match_known() {
        let a = vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let (_u, s, _vt) = svd(&a, 3, 3);
        assert!((s[0] - 3.0).abs() < 1e-9);
        assert!((s[1] - 2.0).abs() < 1e-9);
        assert!((s[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn svd_columns_orthonormal() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (m, n) = (9, 6);
        let a: Vec<f64> = (0..m * n).map(|_| rng.gen::<f64>()).collect();
        let (u, s, vt) = svd(&a, m, n);
        let k = n;
        for j1 in 0..k {
            if s[j1] < 1e-9 {
                continue;
            }
            for j2 in j1..k {
                if s[j2] < 1e-9 {
                    continue;
                }
                let mut dotu = 0.0;
                for i in 0..m {
                    dotu += u[i * k + j1] * u[i * k + j2];
                }
                let mut dotv = 0.0;
                for i in 0..n {
                    dotv += vt[j1 * n + i] * vt[j2 * n + i];
                }
                let expect = if j1 == j2 { 1.0 } else { 0.0 };
                assert!((dotu - expect).abs() < 1e-8, "U 非正交 {j1},{j2}");
                assert!((dotv - expect).abs() < 1e-8, "V 非正交 {j1},{j2}");
            }
        }
    }

    // Complex64 cases.
    fn crand<R: Rng>(rng: &mut R) -> Complex64 {
        Complex64::new(rng.gen::<f64>() * 2.0 - 1.0, rng.gen::<f64>() * 2.0 - 1.0)
    }

    /// Relative Frobenius reconstruction error for a complex SVD.
    fn crecon_err(
        a: &[Complex64],
        u: &[Complex64],
        s: &[f64],
        vt: &[Complex64],
        m: usize,
        n: usize,
    ) -> f64 {
        let k = s.len();
        let mut us = u.to_vec();
        for i in 0..m {
            for j in 0..k {
                us[i * k + j] *= Complex64::new(s[j], 0.0);
            }
        }
        let rec = matmul(&us, vt, m, k, n);
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..m * n {
            num += (a[i] - rec[i]).norm_sqr();
            den += a[i].norm_sqr();
        }
        (num / den.max(1e-300)).sqrt()
    }

    #[test]
    fn svd_complex_reconstructs() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        for &(m, n) in &[(5, 5), (8, 3), (3, 8), (10, 7), (1, 4), (4, 1), (9, 9)] {
            let a: Vec<Complex64> = (0..m * n).map(|_| crand(&mut rng)).collect();
            let (u, s, vt) = svd(&a, m, n);
            assert_eq!(s.len(), m.min(n));
            for w in s.windows(2) {
                assert!(w[0] >= w[1] - 1e-12, "复数奇异值非降序");
            }
            let e = crecon_err(&a, &u, &s, &vt, m, n);
            assert!(e < 1e-9, "复数重构误差过大 m={m} n={n}: {e:e}");
        }
    }

    #[test]
    fn svd_complex_unitary_factors() {
        // Check Hermitian orthogonality of nonzero singular vectors.
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let (m, n) = (7, 5);
        let a: Vec<Complex64> = (0..m * n).map(|_| crand(&mut rng)).collect();
        let (u, s, vt) = svd(&a, m, n);
        let k = n;
        for j1 in 0..k {
            if s[j1] < 1e-9 {
                continue;
            }
            for j2 in j1..k {
                if s[j2] < 1e-9 {
                    continue;
                }
                // <U_j1, U_j2> = Σ conj(u_i,j1) u_i,j2
                let mut du = Complex64::new(0.0, 0.0);
                for i in 0..m {
                    du += u[i * k + j1].conj() * u[i * k + j2];
                }
                let mut dv = Complex64::new(0.0, 0.0);
                for i in 0..n {
                    dv += vt[j1 * n + i].conj() * vt[j2 * n + i];
                }
                let expect = if j1 == j2 { 1.0 } else { 0.0 };
                assert!(
                    (du.re - expect).abs() < 1e-8 && du.im.abs() < 1e-8,
                    "复U非正交"
                );
                assert!(
                    (dv.re - expect).abs() < 1e-8 && dv.im.abs() < 1e-8,
                    "复V非正交"
                );
            }
        }
    }

    #[test]
    fn svd_rank_deficient_no_denormal_garbage() {
        // Rank-three matrix that exposes underflow in `abs_sq().sqrt()`.
        let a = vec![
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.11688796813712088,
            -0.9215055759141204,
            0.0,
            -0.9715049740640438,
            0.9537606440778219,
            0.8086586615147335,
            0.9239325761487804,
            0.0,
            0.0,
            0.8768590852424984,
        ];
        let (u, s, vt) = svd(&a, 4, 4);
        let e = recon_err(&a, &u, &s, &vt, 4, 4);
        assert!(e < 1e-9, "秩亏矩阵重构误差过大（下溢回归）: {e:e}");
        // Vᴴ rows remain orthogonal.
        for j1 in 0..4 {
            if s[j1] < 1e-12 {
                continue;
            }
            let mut nrm = 0.0;
            for i in 0..4 {
                nrm += vt[j1 * 4 + i] * vt[j1 * 4 + i];
            }
            assert!(
                (nrm - 1.0).abs() < 1e-8,
                "Vᴴ 行非单位（旋转非酉回归）: {nrm}"
            );
        }
    }

    #[test]
    fn svd_extreme_zeta_keeps_a_representable_tiny_singular_direction() {
        // Nearly parallel columns with a representable 1e-160 orthogonal component.
        let a = vec![4.0e-2, 1.0e-157, 0.0, 1.0e-160];
        let (u, singular_values, vt, info) = svd_with_info(&a, 2, 2);
        assert!(info.converged, "极端 zeta 不应造成假不收敛: {info:?}");
        assert!(singular_values.iter().all(|value| value.is_finite()));
        assert!(
            (0.5e-160..=2.0e-160).contains(&singular_values[1]),
            "微小但可表示的奇异值丢失: {:?}",
            singular_values
        );

        let u_inner = u[0] * u[1] + u[2] * u[3];
        assert!(
            u_inner.abs() < 1.0e-12,
            "两个非零 U 列必须正交，内积为 {u_inner:e}"
        );

        let mut us = u.clone();
        for row in 0..2 {
            for column in 0..2 {
                us[row * 2 + column] *= singular_values[column];
            }
        }
        let reconstructed = matmul(&us, &vt, 2, 2, 2);
        assert!(reconstructed.iter().all(|value| value.is_finite()));
        for (actual, expected) in reconstructed.iter().zip(&a) {
            let scale = expected.abs().max(1.0e-300);
            assert!(
                (actual - expected).abs() <= 1.0e-12 * scale,
                "极端 zeta 重构失败: actual={actual:e}, expected={expected:e}"
            );
        }
    }

    #[test]
    fn svd_complex_hermitian_known_spectrum() {
        // diag(2i, 3, -1) has singular values {3, 2, 1}.
        let mut a = vec![Complex64::new(0.0, 0.0); 9];
        a[0] = Complex64::new(0.0, 2.0);
        a[4] = Complex64::new(3.0, 0.0);
        a[8] = Complex64::new(-1.0, 0.0);
        let (_u, s, _vt) = svd(&a, 3, 3);
        assert!((s[0] - 3.0).abs() < 1e-9);
        assert!((s[1] - 2.0).abs() < 1e-9);
        assert!((s[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "matmul: 左矩阵尺寸不符")]
    fn matmul_rejects_short_left_layout_before_indexing() {
        let _ = matmul(&[1.0], &[1.0, 2.0], 2, 1, 2);
    }

    #[test]
    fn matmul_keeps_nonzero_values_whose_square_underflows() {
        let tiny = 1.0e-200;
        assert_eq!(tiny * tiny, 0.0, "test precondition: square must underflow");

        let real = matmul(&[tiny], &[1.0e100], 1, 1, 1);
        assert!((real[0] - 1.0e-100).abs() < 1.0e-115);

        let ztiny = Complex64::new(tiny, -tiny);
        assert_eq!(
            ztiny.norm_sqr(),
            0.0,
            "test precondition: norm square must underflow"
        );
        let complex = matmul(&[ztiny], &[Complex64::new(1.0e100, 0.0)], 1, 1, 1);
        let want = Complex64::new(1.0e-100, -1.0e-100);
        assert!((complex[0] - want).norm() < 2.0e-115);
    }
}
