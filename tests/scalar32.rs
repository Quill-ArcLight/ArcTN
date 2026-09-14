//! Single-precision scalar coverage: representation/IO, GEMM execution and Jacobi SVD.

use arctn::linalg::{matmul, svd};
use arctn::network::load_tensors_bin;
use arctn::{
    contract_network, naive_einsum, CompiledContraction, DenseTensor, Scalar, TensorNetwork,
};
use num_complex::Complex32;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn matmul_network() -> TensorNetwork {
    TensorNetwork {
        name: "scalar32-matmul".into(),
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
        size_dict: [(0, 2), (1, 3), (2, 2)].into_iter().collect(),
    }
}

#[test]
fn scalar32_little_endian_io_and_magnitude_are_precise() {
    let real = -3.25f32;
    assert_eq!(<f32 as Scalar>::NBYTES, 4);
    assert_eq!(
        <f32 as Scalar>::from_le_bytes(&real.to_le_bytes()).to_bits(),
        real.to_bits()
    );

    let complex = Complex32::new(1.5, -2.25);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&complex.re.to_le_bytes());
    bytes.extend_from_slice(&complex.im.to_le_bytes());
    assert_eq!(<Complex32 as Scalar>::NBYTES, 8);
    assert_eq!(<Complex32 as Scalar>::from_le_bytes(&bytes), complex);
    assert_eq!(
        <Complex32 as Scalar>::conj(complex),
        Complex32::new(1.5, 2.25)
    );
    assert_eq!(
        <Complex32 as Scalar>::from_real(-7.0),
        Complex32::new(-7.0, 0.0)
    );

    // Squaring in f32 underflows, but Scalar norms accumulate in f64 so a nonzero
    // coefficient is not misclassified as zero by SVD/matmul control flow.
    let tiny = 1.0e-30f32;
    assert_eq!(tiny * tiny, 0.0, "test precondition");
    assert!(<f32 as Scalar>::abs_sq(tiny) > 0.0);
    let ztiny = Complex32::new(tiny, -tiny);
    assert_eq!(ztiny.norm_sqr(), 0.0, "test precondition");
    assert!(<Complex32 as Scalar>::abs_sq(ztiny) > 0.0);
    assert!(<Complex32 as Scalar>::abs(ztiny) > 0.0);
}

#[test]
fn scalar32_binary_loader_reads_f32_and_complex32_end_to_end() {
    let net = TensorNetwork {
        name: "scalar32-binary-io".into(),
        inputs: vec![vec![0]],
        output: vec![0],
        size_dict: [(0, 2)].into_iter().collect(),
    };
    let temp = std::env::temp_dir();
    let pid = std::process::id();

    let real_path = temp.join(format!("arctn-scalar32-{pid}-f32.bin"));
    let real_values = [1.25f32, -3.5];
    let real_bytes: Vec<u8> = real_values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(&real_path, real_bytes).expect("write f32 fixture");
    let real = load_tensors_bin::<f32>(&real_path, &net).expect("load f32 fixture");
    std::fs::remove_file(&real_path).expect("remove f32 fixture");
    assert_eq!(real[0].shape(), &[2]);
    assert_eq!(real[0].data(), &real_values);

    let complex_path = temp.join(format!("arctn-scalar32-{pid}-c32.bin"));
    let complex_values = [Complex32::new(0.5, -1.0), Complex32::new(2.25, 3.0)];
    let complex_bytes: Vec<u8> = complex_values
        .iter()
        .flat_map(|value| {
            value
                .re
                .to_le_bytes()
                .into_iter()
                .chain(value.im.to_le_bytes())
        })
        .collect();
    std::fs::write(&complex_path, complex_bytes).expect("write Complex32 fixture");
    let complex =
        load_tensors_bin::<Complex32>(&complex_path, &net).expect("load Complex32 fixture");
    std::fs::remove_file(&complex_path).expect("remove Complex32 fixture");
    assert_eq!(complex[0].shape(), &[2]);
    assert_eq!(complex[0].data(), &complex_values);
}

#[test]
fn f32_and_complex32_contract_gemm_match_naive() {
    let net = matmul_network();
    let path = vec![(0, 1)];
    let compiled = CompiledContraction::compile(&net, &path).expect("compile scalar32 network");

    let real = vec![
        DenseTensor::from_data(vec![2, 3], vec![1.0f32, -2.0, 0.5, 3.0, 4.0, -1.0]),
        DenseTensor::from_data(vec![3, 2], vec![2.0f32, 1.0, -1.0, 0.25, 0.5, -3.0]),
    ];
    let real_want = naive_einsum(&net, &real).expect("f32 naive");
    let real_compiled = compiled.execute(&real).expect("compiled f32 contraction");
    let real_got = contract_network(&net, real, &path).expect("f32 GEMM contraction");
    assert_eq!(real_got.shape(), &[2, 2]);
    assert!(real_got.max_abs_diff(&real_want) < 2.0e-6);
    assert!(real_got.max_abs_diff(&real_compiled) < 2.0e-6);

    let c = |re, im| Complex32::new(re, im);
    let complex = vec![
        DenseTensor::from_data(
            vec![2, 3],
            vec![
                c(1.0, 0.5),
                c(-2.0, 1.0),
                c(0.5, -0.25),
                c(3.0, -1.0),
                c(4.0, 0.75),
                c(-1.0, 2.0),
            ],
        ),
        DenseTensor::from_data(
            vec![3, 2],
            vec![
                c(2.0, -0.5),
                c(1.0, 1.0),
                c(-1.0, 0.25),
                c(0.25, -2.0),
                c(0.5, 1.5),
                c(-3.0, 0.5),
            ],
        ),
    ];
    let complex_want = naive_einsum(&net, &complex).expect("Complex32 naive");
    let complex_compiled = compiled
        .execute(&complex)
        .expect("compiled Complex32 contraction");
    let complex_got = contract_network(&net, complex, &path).expect("Complex32 GEMM contraction");
    assert!(complex_got.max_abs_diff(&complex_want) < 1.0e-5);
    assert!(complex_got.max_abs_diff(&complex_compiled) < 1.0e-5);
}

fn relative_reconstruction_error_f32(
    a: &[f32],
    u: &[f32],
    s: &[f64],
    vh: &[f32],
    m: usize,
    n: usize,
) -> f64 {
    let k = s.len();
    let mut us = u.to_vec();
    for i in 0..m {
        for j in 0..k {
            us[i * k + j] *= s[j] as f32;
        }
    }
    let reconstructed = matmul(&us, vh, m, k, n);
    let numerator: f64 = a
        .iter()
        .zip(reconstructed)
        .map(|(&x, y)| {
            let d = f64::from(x - y);
            d * d
        })
        .sum();
    let denominator: f64 = a
        .iter()
        .map(|&x| {
            let x = f64::from(x);
            x * x
        })
        .sum();
    (numerator / denominator).sqrt()
}

fn relative_reconstruction_error_c32(
    a: &[Complex32],
    u: &[Complex32],
    s: &[f64],
    vh: &[Complex32],
    m: usize,
    n: usize,
) -> f64 {
    let k = s.len();
    let mut us = u.to_vec();
    for i in 0..m {
        for j in 0..k {
            us[i * k + j] *= Complex32::new(s[j] as f32, 0.0);
        }
    }
    let reconstructed = matmul(&us, vh, m, k, n);
    let numerator: f64 = a
        .iter()
        .zip(reconstructed)
        .map(|(&x, y)| <Complex32 as Scalar>::abs_sq(x - y))
        .sum();
    let denominator: f64 = a.iter().map(|&x| <Complex32 as Scalar>::abs_sq(x)).sum();
    (numerator / denominator).sqrt()
}

#[test]
fn jacobi_svd_reconstructs_f32_and_complex32() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x32);
    for &(m, n) in &[(6, 4), (4, 6), (7, 7), (1, 5), (5, 1)] {
        let real: Vec<f32> = (0..m * n).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
        let (u, s, vh) = svd(&real, m, n);
        assert!(s.windows(2).all(|w| w[0] >= w[1]));
        let error = relative_reconstruction_error_f32(&real, &u, &s, &vh, m, n);
        assert!(error < 3.0e-5, "f32 SVD {m}x{n} relative error {error:e}");

        let complex: Vec<Complex32> = (0..m * n)
            .map(|_| {
                Complex32::new(
                    rng.gen_range(-1.0f32..1.0f32),
                    rng.gen_range(-1.0f32..1.0f32),
                )
            })
            .collect();
        let (u, s, vh) = svd(&complex, m, n);
        assert!(s.windows(2).all(|w| w[0] >= w[1]));
        let error = relative_reconstruction_error_c32(&complex, &u, &s, &vh, m, n);
        assert!(
            error < 6.0e-5,
            "Complex32 SVD {m}x{n} relative error {error:e}"
        );
    }
}

#[test]
fn scalar32_jacobi_factors_are_orthonormal_at_single_precision() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x532);
    let (m, n) = (9, 5);
    let real: Vec<f32> = (0..m * n).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
    let (u, singular, vh) = svd(&real, m, n);
    for left in 0..n {
        for right in left..n {
            let u_dot: f64 = (0..m)
                .map(|row| f64::from(u[row * n + left]) * f64::from(u[row * n + right]))
                .sum();
            let v_dot: f64 = (0..n)
                .map(|col| f64::from(vh[left * n + col]) * f64::from(vh[right * n + col]))
                .sum();
            let expected = if left == right { 1.0 } else { 0.0 };
            assert!((u_dot - expected).abs() < 2.0e-5);
            assert!((v_dot - expected).abs() < 2.0e-5);
        }
    }
    assert!(singular.windows(2).all(|w| w[0] >= w[1]));

    let complex: Vec<Complex32> = (0..m * n)
        .map(|_| {
            Complex32::new(
                rng.gen_range(-1.0f32..1.0f32),
                rng.gen_range(-1.0f32..1.0f32),
            )
        })
        .collect();
    let (u, singular, vh) = svd(&complex, m, n);
    for left in 0..n {
        for right in left..n {
            let hermitian_dot = |matrix: &[Complex32], rows: usize, cols: usize| {
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for row in 0..rows {
                    let x = matrix[row * cols + left];
                    let y = matrix[row * cols + right];
                    re += f64::from(x.re) * f64::from(y.re) + f64::from(x.im) * f64::from(y.im);
                    im += f64::from(x.re) * f64::from(y.im) - f64::from(x.im) * f64::from(y.re);
                }
                (re, im)
            };
            let (u_re, u_im) = hermitian_dot(&u, m, n);

            // Vᴴ rows are orthonormal; transpose the indexing in the same inner product.
            let mut v_re = 0.0f64;
            let mut v_im = 0.0f64;
            for col in 0..n {
                let x = vh[left * n + col];
                let y = vh[right * n + col];
                v_re += f64::from(x.re) * f64::from(y.re) + f64::from(x.im) * f64::from(y.im);
                v_im += f64::from(x.re) * f64::from(y.im) - f64::from(x.im) * f64::from(y.re);
            }
            let expected = if left == right { 1.0 } else { 0.0 };
            assert!((u_re - expected).abs() < 4.0e-5 && u_im.abs() < 4.0e-5);
            assert!((v_re - expected).abs() < 4.0e-5 && v_im.abs() < 4.0e-5);
        }
    }
    assert!(singular.windows(2).all(|w| w[0] >= w[1]));
}

#[test]
fn scalar32_matmul_keeps_nonzero_values_whose_square_underflows() {
    let tiny = 1.0e-30f32;
    let real = matmul(&[tiny], &[1.0e20f32], 1, 1, 1);
    assert!((real[0] - 1.0e-10).abs() < 1.0e-16);

    let ztiny = Complex32::new(tiny, -tiny);
    let complex = matmul(&[ztiny], &[Complex32::new(1.0e20, 0.0)], 1, 1, 1);
    let want = Complex32::new(1.0e-10, -1.0e-10);
    assert!((complex[0] - want).norm() < 2.0e-16);
}

#[test]
fn scalar32_svd_is_stable_for_nonzero_small_scale() {
    let scale = 1.0e-20f32;
    let real: Vec<f32> = [1.0, 2.0, -3.0, 0.5, 0.25, -1.5]
        .into_iter()
        .map(|x| x * scale)
        .collect();
    let (u, singular, vh) = svd(&real, 3, 2);
    assert!(u.iter().all(|x| x.is_finite()));
    assert!(singular.iter().all(|x| x.is_finite()));
    assert!(vh.iter().all(|x| x.is_finite()));
    let error = relative_reconstruction_error_f32(&real, &u, &singular, &vh, 3, 2);
    assert!(error < 3.0e-5, "small f32 SVD relative error {error:e}");

    let complex: Vec<Complex32> = [
        (1.0, 0.5),
        (2.0, -1.0),
        (-3.0, 0.25),
        (0.5, 1.5),
        (0.25, -2.0),
        (-1.5, 0.75),
    ]
    .into_iter()
    .map(|(re, im)| Complex32::new(re * scale, im * scale))
    .collect();
    let (u, singular, vh) = svd(&complex, 3, 2);
    assert!(u.iter().all(|x| x.re.is_finite() && x.im.is_finite()));
    assert!(singular.iter().all(|x| x.is_finite()));
    assert!(vh.iter().all(|x| x.re.is_finite() && x.im.is_finite()));
    let error = relative_reconstruction_error_c32(&complex, &u, &singular, &vh, 3, 2);
    assert!(
        error < 6.0e-5,
        "small Complex32 SVD relative error {error:e}"
    );
}
