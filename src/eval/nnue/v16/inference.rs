use super::arch::div_trunc;
use super::pairs::make_pawn_id;
use super::threats::{threat_luts, threat_orient};
use super::*;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum256_ps_avx2(v: std::arch::x86_64::__m256i) -> i32 {
    unsafe {
        use std::arch::x86_64::*;
        let hi128 = _mm256_extracti128_si256(v, 1);
        let lo128 = _mm256_castsi256_si128(v);
        let sum128 = _mm_add_epi32(lo128, hi128);
        let shuf = _mm_shuffle_epi32(sum128, 0x4E);
        let sum64 = _mm_add_epi32(sum128, shuf);
        let shuf2 = _mm_shuffle_epi32(sum64, 0x05);
        _mm_cvtsi128_si32(_mm_add_epi32(sum64, shuf2))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn fc0_avx2(arch: &SfnnArch, in_row: &[u8], fc0: &mut [i32; FC0_OUT]) {
    unsafe {
        use std::arch::x86_64::*;
        let ones = _mm256_set1_epi16(1);
        let mask = _mm256_set1_epi8(0x7f);
        let in_ptr = in_row.as_ptr() as *const __m256i;
        let chunks = in_row.len() / 32;
        let l1 = in_row.len();

        let mut o = 0;
        while o < FC0_OUT {
            let mut sum0 = _mm256_setzero_si256();
            let mut sum1 = _mm256_setzero_si256();
            let mut sum2 = _mm256_setzero_si256();
            let mut sum3 = _mm256_setzero_si256();

            let w0_ptr = arch.fc0_w.as_ptr().add(o * l1) as *const __m256i;
            let w1_ptr = arch.fc0_w.as_ptr().add((o + 1) * l1) as *const __m256i;
            let w2_ptr = arch.fc0_w.as_ptr().add((o + 2) * l1) as *const __m256i;
            let w3_ptr = arch.fc0_w.as_ptr().add((o + 3) * l1) as *const __m256i;

            for j in 0..chunks {
                let in_vec = _mm256_loadu_si256(in_ptr.add(j));
                let low = _mm256_and_si256(in_vec, mask);
                let high = _mm256_andnot_si256(mask, in_vec);

                let w0 = _mm256_loadu_si256(w0_ptr.add(j));
                let w1 = _mm256_loadu_si256(w1_ptr.add(j));
                let w2 = _mm256_loadu_si256(w2_ptr.add(j));
                let w3 = _mm256_loadu_si256(w3_ptr.add(j));

                let low32_0 = _mm256_madd_epi16(_mm256_maddubs_epi16(low, w0), ones);
                let high32_0 = _mm256_madd_epi16(_mm256_maddubs_epi16(high, w0), ones);
                sum0 = _mm256_add_epi32(sum0, _mm256_add_epi32(low32_0, high32_0));

                let low32_1 = _mm256_madd_epi16(_mm256_maddubs_epi16(low, w1), ones);
                let high32_1 = _mm256_madd_epi16(_mm256_maddubs_epi16(high, w1), ones);
                sum1 = _mm256_add_epi32(sum1, _mm256_add_epi32(low32_1, high32_1));

                let low32_2 = _mm256_madd_epi16(_mm256_maddubs_epi16(low, w2), ones);
                let high32_2 = _mm256_madd_epi16(_mm256_maddubs_epi16(high, w2), ones);
                sum2 = _mm256_add_epi32(sum2, _mm256_add_epi32(low32_2, high32_2));

                let low32_3 = _mm256_madd_epi16(_mm256_maddubs_epi16(low, w3), ones);
                let high32_3 = _mm256_madd_epi16(_mm256_maddubs_epi16(high, w3), ones);
                sum3 = _mm256_add_epi32(sum3, _mm256_add_epi32(low32_3, high32_3));
            }

            fc0[o] = arch.fc0_bias[o] + hsum256_ps_avx2(sum0);
            fc0[o + 1] = arch.fc0_bias[o + 1] + hsum256_ps_avx2(sum1);
            fc0[o + 2] = arch.fc0_bias[o + 2] + hsum256_ps_avx2(sum2);
            fc0[o + 3] = arch.fc0_bias[o + 3] + hsum256_ps_avx2(sum3);

            o += 4;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fc1_avx2(arch: &SfnnArch, concat_fc0: &[u8], fc1: &mut [i32; FC1_OUT]) {
    unsafe {
        use std::arch::x86_64::*;
        let ones = _mm256_set1_epi16(1);
        let in_ptr = concat_fc0.as_ptr() as *const __m256i;
        let in0 = _mm256_loadu_si256(in_ptr);
        let in1 = _mm256_loadu_si256(in_ptr.add(1));

        for (o, output) in fc1.iter_mut().enumerate() {
            let w_ptr = arch.fc1_w.as_ptr().add(o * 64) as *const __m256i;
            let w0 = _mm256_loadu_si256(w_ptr);
            let w1 = _mm256_loadu_si256(w_ptr.add(1));

            let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(in0, w0), ones);
            let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(in1, w1), ones);
            let sum = _mm256_add_epi32(p0, p1);

            *output = arch.fc1_bias[o] + hsum256_ps_avx2(sum);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fc2_avx2(arch: &SfnnArch, concat: &[u8]) -> i32 {
    unsafe {
        use std::arch::x86_64::*;
        let ones = _mm256_set1_epi16(1);
        let in_ptr = concat.as_ptr() as *const __m256i;
        let w_ptr = arch.fc2_w.as_ptr() as *const __m256i;

        let in0 = _mm256_loadu_si256(in_ptr);
        let in1 = _mm256_loadu_si256(in_ptr.add(1));
        let in2 = _mm256_loadu_si256(in_ptr.add(2));
        let in3 = _mm256_loadu_si256(in_ptr.add(3));

        let w0 = _mm256_loadu_si256(w_ptr);
        let w1 = _mm256_loadu_si256(w_ptr.add(1));
        let w2 = _mm256_loadu_si256(w_ptr.add(2));
        let w3 = _mm256_loadu_si256(w_ptr.add(3));

        let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(in0, w0), ones);
        let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(in1, w1), ones);
        let p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(in2, w2), ones);
        let p3 = _mm256_madd_epi16(_mm256_maddubs_epi16(in3, w3), ones);

        let sum = _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3));
        hsum256_ps_avx2(sum)
    }
}

pub(super) fn propagate(arch: &SfnnArch, l1: usize, input: &[u8]) -> i32 {
    if arch.is_rudi {
        #[cfg(target_arch = "x86_64")]
        let use_avx2 = has_avx2() && l1.is_multiple_of(32);
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2 = false;

        let mut fc0 = [0.0f32; FC0_OUT];
        if use_avx2 {
            let mut fc0_i32 = [0i32; FC0_OUT];
            #[cfg(target_arch = "x86_64")]
            unsafe {
                fc0_avx2(arch, &input[..l1], &mut fc0_i32);
            }
            for (output, &v) in fc0.iter_mut().zip(fc0_i32.iter()) {
                *output = (v as f32 / 16320.0).clamp(0.0, 1.0);
            }
        } else {
            for (o, output) in fc0.iter_mut().enumerate() {
                let mut sum = arch.fc0_bias[o];
                let row = o * l1;
                for (&input_value, &weight) in input.iter().zip(&arch.fc0_w[row..row + l1]) {
                    sum += i32::from(input_value) * i32::from(weight);
                }
                *output = (sum as f32 / 16320.0).clamp(0.0, 1.0);
            }
        }

        let mut pair = [0.0f32; FC1_IN];
        for (o, &c) in fc0.iter().enumerate() {
            pair[o] = c * c;
            pair[FC0_OUT + o] = c;
        }

        let mut fc1 = [0.0f32; FC1_OUT];
        for (o, output) in fc1.iter_mut().enumerate() {
            let mut sum = arch.fc1_bias[o] as f32 / 16320.0;
            let row = o * FC1_IN;
            for (&pair_value, &weight) in pair.iter().zip(&arch.fc1_w[row..row + FC1_IN]) {
                sum += pair_value * (weight as f32 / 64.0);
            }
            *output = sum.clamp(0.0, 1.0);
        }

        let mut out = arch.fc2_bias as f32 / 16320.0;
        for (&fc1_value, &weight) in fc1.iter().zip(&arch.fc2_w[..FC1_OUT]) {
            out += fc1_value * (weight as f32 / 64.0);
        }

        let score_cp = ((out - 2.80) * 100.0).clamp(-29000.0, 29000.0) as i32;
        return score_cp * OUTPUT_SCALE;
    }

    let mut fc0 = [0i32; FC0_OUT];
    let in_row = &input[..l1];

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = has_avx2() && l1 == 1024;
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    if use_avx2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            fc0_avx2(arch, in_row, &mut fc0);
        }
    } else {
        for (o, output) in fc0.iter_mut().enumerate() {
            let mut value = arch.fc0_bias[o];
            let row = o * l1;
            for (&input_value, &weight) in in_row.iter().zip(&arch.fc0_w[row..row + l1]) {
                value += i32::from(input_value) * i32::from(weight);
            }
            *output = value;
        }
    }

    let mut concat = [0u8; FC0_OUT * 2 + FC1_OUT * 2];
    for (i, &v) in fc0.iter().enumerate() {
        let sqr = (((v as i64 * v as i64) >> 21).min(127)) as u8;
        let clip = (v >> 7).clamp(0, 127) as u8;
        concat[i] = sqr;
        concat[FC0_OUT + i] = clip;
    }

    let mut fc1 = [0i32; FC1_OUT];
    let concat_fc0 = &concat[0..FC0_OUT * 2];

    if use_avx2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            fc1_avx2(arch, concat_fc0, &mut fc1);
        }
    } else {
        for (o, output) in fc1.iter_mut().enumerate() {
            let mut value = arch.fc1_bias[o];
            let row = o * (FC0_OUT * 2);
            for (&input_value, &weight) in
                concat_fc0.iter().zip(&arch.fc1_w[row..row + FC0_OUT * 2])
            {
                value += i32::from(input_value) * i32::from(weight);
            }
            *output = value;
        }
    }

    for (i, &v) in fc1.iter().enumerate() {
        let sqr = (((v as i64 * v as i64) >> 19).min(127)) as u8;
        let clip = (v >> 6).clamp(0, 127) as u8;
        concat[FC0_OUT * 2 + i] = sqr;
        concat[FC0_OUT * 2 + FC1_OUT + i] = clip;
    }

    let mut out = arch.fc2_bias;
    let dot = if use_avx2 && arch.fc2_w.len() == 128 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            fc2_avx2(arch, &concat)
        }
        #[cfg(not(target_arch = "x86_64"))]
        0
    } else {
        concat
            .iter()
            .zip(arch.fc2_w.iter())
            .map(|(&input_value, &weight)| i32::from(input_value) * i32::from(weight))
            .sum()
    };
    out += dot;
    let skip = fc0[FC0_OUT - 2] - fc0[FC0_OUT - 1];
    out += skip;

    let multiplier = 600 * OUTPUT_SCALE as i64;
    let denominator = 128 * 64 * 2;
    ((out as i64 * multiplier) / denominator) as i32
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn add_threat_w_i16_avx2(buf: &mut [i32; L1], w: &[i16]) {
    unsafe {
        use std::arch::x86_64::*;
        let b_ptr = buf.as_mut_ptr() as *mut __m256i;
        let w_ptr = w.as_ptr() as *const __m128i;
        let chunks = buf.len() / 8;
        for i in 0..chunks {
            let w_128 = _mm_loadu_si128(w_ptr.add(i));
            let w_256 = _mm256_cvtepi16_epi32(w_128);
            let b = _mm256_loadu_si256(b_ptr.add(i));
            _mm256_storeu_si256(b_ptr.add(i), _mm256_add_epi32(b, w_256));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn pairwise_transform_rudi_avx2(
    base_acc: &[i16],
    threat_buf: &[i32; L1],
    dst: &mut [u8],
    half: usize,
    use_threats: bool,
) {
    unsafe {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero_si256();
        let max_val = _mm256_set1_epi32(255);
        let mult257 = _mm256_set1_epi32(257);
        let base_ptr = base_acc.as_ptr();
        let threat_ptr = threat_buf.as_ptr() as *const __m256i;

        let n_chunks = half / 16;
        for chunk in 0..n_chunks {
            let j = chunk * 16;
            let b0_16 = _mm256_loadu_si256(base_ptr.add(j) as *const __m256i);
            let mut s0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_16));
            let mut s0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_16, 1));

            let b1_16 = _mm256_loadu_si256(base_ptr.add(j + half) as *const __m256i);
            let mut s1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_16));
            let mut s1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_16, 1));

            if use_threats {
                let t0_lo = _mm256_loadu_si256(threat_ptr.add(chunk * 2));
                let t0_hi = _mm256_loadu_si256(threat_ptr.add(chunk * 2 + 1));
                let t1_lo = _mm256_loadu_si256(threat_ptr.add((half / 8) + chunk * 2));
                let t1_hi = _mm256_loadu_si256(threat_ptr.add((half / 8) + chunk * 2 + 1));

                s0_lo = _mm256_add_epi32(s0_lo, t0_lo);
                s0_hi = _mm256_add_epi32(s0_hi, t0_hi);
                s1_lo = _mm256_add_epi32(s1_lo, t1_lo);
                s1_hi = _mm256_add_epi32(s1_hi, t1_hi);
            }

            let c0_lo = _mm256_min_epi32(_mm256_max_epi32(s0_lo, zero), max_val);
            let c0_hi = _mm256_min_epi32(_mm256_max_epi32(s0_hi, zero), max_val);
            let c1_lo = _mm256_min_epi32(_mm256_max_epi32(s1_lo, zero), max_val);
            let c1_hi = _mm256_min_epi32(_mm256_max_epi32(s1_hi, zero), max_val);

            let prod_lo = _mm256_mullo_epi32(c0_lo, c1_lo);
            let prod_hi = _mm256_mullo_epi32(c0_hi, c1_hi);

            let div_lo = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(prod_lo, mult257), mult257),
                16,
            );
            let div_hi = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(prod_hi, mult257), mult257),
                16,
            );

            let d_lo_0 = _mm256_castsi256_si128(div_lo);
            let d_lo_1 = _mm256_extracti128_si256(div_lo, 1);
            let d_hi_0 = _mm256_castsi256_si128(div_hi);
            let d_hi_1 = _mm256_extracti128_si256(div_hi, 1);

            let p0 = _mm_packs_epi32(d_lo_0, d_lo_1);
            let p1 = _mm_packs_epi32(d_hi_0, d_hi_1);

            let bytes = _mm_packus_epi16(p0, p1);
            _mm_storeu_si128(dst.as_mut_ptr().add(j) as *mut __m128i, bytes);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn pairwise_transform_threats_tiled_avx2(
    base_acc: &[i16],
    threat_w: &[i8],
    pair_w: &[i8],
    threat_features: &[usize],
    pair_features: &[usize],
    dst: &mut [u8],
    half: usize,
    clip_max: i32,
) {
    unsafe {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero_si256();
        let max_val = _mm256_set1_epi16(clip_max as i16);
        let base_ptr = base_acc.as_ptr();
        let threat_ptr = threat_w.as_ptr();
        let pair_ptr = pair_w.as_ptr();

        for chunk in (0..(half / 16)).step_by(4) {
            let offset0_a = chunk * 16;
            let offset0_b = offset0_a + 16;
            let offset0_c = offset0_a + 32;
            let offset0_d = offset0_a + 48;
            let offset1_a = offset0_a + half;
            let offset1_b = offset0_b + half;
            let offset1_c = offset0_c + half;
            let offset1_d = offset0_d + half;

            let mut sum0_a = _mm256_setzero_si256();
            let mut sum0_b = _mm256_setzero_si256();
            let mut sum0_c = _mm256_setzero_si256();
            let mut sum0_d = _mm256_setzero_si256();
            let mut sum1_a = _mm256_setzero_si256();
            let mut sum1_b = _mm256_setzero_si256();
            let mut sum1_c = _mm256_setzero_si256();
            let mut sum1_d = _mm256_setzero_si256();

            for &f in threat_features {
                let t = f - PSQ_DIMS;
                let w_base = threat_ptr.add(t * 1024);

                let raw0_ab = _mm256_loadu_si256(w_base.add(offset0_a) as *const __m256i);
                let w0_a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(raw0_ab));
                let w0_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(raw0_ab, 1));
                sum0_a = _mm256_add_epi16(sum0_a, w0_a);
                sum0_b = _mm256_add_epi16(sum0_b, w0_b);

                let raw0_cd = _mm256_loadu_si256(w_base.add(offset0_c) as *const __m256i);
                let w0_c = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(raw0_cd));
                let w0_d = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(raw0_cd, 1));
                sum0_c = _mm256_add_epi16(sum0_c, w0_c);
                sum0_d = _mm256_add_epi16(sum0_d, w0_d);

                let raw1_ab = _mm256_loadu_si256(w_base.add(offset1_a) as *const __m256i);
                let w1_a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(raw1_ab));
                let w1_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(raw1_ab, 1));
                sum1_a = _mm256_add_epi16(sum1_a, w1_a);
                sum1_b = _mm256_add_epi16(sum1_b, w1_b);

                let raw1_cd = _mm256_loadu_si256(w_base.add(offset1_c) as *const __m256i);
                let w1_c = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(raw1_cd));
                let w1_d = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(raw1_cd, 1));
                sum1_c = _mm256_add_epi16(sum1_c, w1_c);
                sum1_d = _mm256_add_epi16(sum1_d, w1_d);
            }

            if !pair_w.is_empty() {
                for &f in pair_features {
                    let t = f - PSQ_DIMS - THREAT_DIMS;
                    let w_base = pair_ptr.add(t * 1024);

                    let raw0_ab = _mm256_loadu_si256(w_base.add(offset0_a) as *const __m256i);
                    let w0_a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(raw0_ab));
                    let w0_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(raw0_ab, 1));
                    sum0_a = _mm256_add_epi16(sum0_a, w0_a);
                    sum0_b = _mm256_add_epi16(sum0_b, w0_b);

                    let raw0_cd = _mm256_loadu_si256(w_base.add(offset0_c) as *const __m256i);
                    let w0_c = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(raw0_cd));
                    let w0_d = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(raw0_cd, 1));
                    sum0_c = _mm256_add_epi16(sum0_c, w0_c);
                    sum0_d = _mm256_add_epi16(sum0_d, w0_d);

                    let raw1_ab = _mm256_loadu_si256(w_base.add(offset1_a) as *const __m256i);
                    let w1_a = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(raw1_ab));
                    let w1_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(raw1_ab, 1));
                    sum1_a = _mm256_add_epi16(sum1_a, w1_a);
                    sum1_b = _mm256_add_epi16(sum1_b, w1_b);

                    let raw1_cd = _mm256_loadu_si256(w_base.add(offset1_c) as *const __m256i);
                    let w1_c = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(raw1_cd));
                    let w1_d = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(raw1_cd, 1));
                    sum1_c = _mm256_add_epi16(sum1_c, w1_c);
                    sum1_d = _mm256_add_epi16(sum1_d, w1_d);
                }
            }

            let b0_a = _mm256_loadu_si256(base_ptr.add(offset0_a) as *const __m256i);
            let b1_a = _mm256_loadu_si256(base_ptr.add(offset1_a) as *const __m256i);
            let s0_a = _mm256_adds_epi16(b0_a, sum0_a);
            let s1_a = _mm256_adds_epi16(b1_a, sum1_a);
            let s0_clamp_a = _mm256_min_epi16(_mm256_max_epi16(s0_a, zero), max_val);
            let s0_shl_a = _mm256_slli_epi16(s0_clamp_a, 7);
            let s1_min_a = _mm256_min_epi16(s1_a, max_val);
            let mul_a = _mm256_mulhi_epi16(s0_shl_a, s1_min_a);
            let lo_a = _mm256_castsi256_si128(mul_a);
            let hi_a = _mm256_extracti128_si256(mul_a, 1);
            let bytes_a = _mm_packus_epi16(lo_a, hi_a);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset0_a) as *mut __m128i, bytes_a);

            let b0_b = _mm256_loadu_si256(base_ptr.add(offset0_b) as *const __m256i);
            let b1_b = _mm256_loadu_si256(base_ptr.add(offset1_b) as *const __m256i);
            let s0_b = _mm256_adds_epi16(b0_b, sum0_b);
            let s1_b = _mm256_adds_epi16(b1_b, sum1_b);
            let s0_clamp_b = _mm256_min_epi16(_mm256_max_epi16(s0_b, zero), max_val);
            let s0_shl_b = _mm256_slli_epi16(s0_clamp_b, 7);
            let s1_min_b = _mm256_min_epi16(s1_b, max_val);
            let mul_b = _mm256_mulhi_epi16(s0_shl_b, s1_min_b);
            let lo_b = _mm256_castsi256_si128(mul_b);
            let hi_b = _mm256_extracti128_si256(mul_b, 1);
            let bytes_b = _mm_packus_epi16(lo_b, hi_b);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset0_b) as *mut __m128i, bytes_b);

            let b0_c = _mm256_loadu_si256(base_ptr.add(offset0_c) as *const __m256i);
            let b1_c = _mm256_loadu_si256(base_ptr.add(offset1_c) as *const __m256i);
            let s0_c = _mm256_adds_epi16(b0_c, sum0_c);
            let s1_c = _mm256_adds_epi16(b1_c, sum1_c);
            let s0_clamp_c = _mm256_min_epi16(_mm256_max_epi16(s0_c, zero), max_val);
            let s0_shl_c = _mm256_slli_epi16(s0_clamp_c, 7);
            let s1_min_c = _mm256_min_epi16(s1_c, max_val);
            let mul_c = _mm256_mulhi_epi16(s0_shl_c, s1_min_c);
            let lo_c = _mm256_castsi256_si128(mul_c);
            let hi_c = _mm256_extracti128_si256(mul_c, 1);
            let bytes_c = _mm_packus_epi16(lo_c, hi_c);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset0_c) as *mut __m128i, bytes_c);

            let b0_d = _mm256_loadu_si256(base_ptr.add(offset0_d) as *const __m256i);
            let b1_d = _mm256_loadu_si256(base_ptr.add(offset1_d) as *const __m256i);
            let s0_d = _mm256_adds_epi16(b0_d, sum0_d);
            let s1_d = _mm256_adds_epi16(b1_d, sum1_d);
            let s0_clamp_d = _mm256_min_epi16(_mm256_max_epi16(s0_d, zero), max_val);
            let s0_shl_d = _mm256_slli_epi16(s0_clamp_d, 7);
            let s1_min_d = _mm256_min_epi16(s1_d, max_val);
            let mul_d = _mm256_mulhi_epi16(s0_shl_d, s1_min_d);
            let lo_d = _mm256_castsi256_si128(mul_d);
            let hi_d = _mm256_extracti128_si256(mul_d, 1);
            let bytes_d = _mm_packus_epi16(lo_d, hi_d);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset0_d) as *mut __m128i, bytes_d);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_psqt_avx2(
    dst: &mut [i32; N_BUCKETS],
    threat_psqt_w: &[i32],
    pair_psqt_w: &[i32],
    threat_features: &[usize],
    pair_features: &[usize],
) {
    unsafe {
        use std::arch::x86_64::*;
        let mut acc = _mm256_setzero_si256();
        let t_ptr = threat_psqt_w.as_ptr();
        for &f in threat_features {
            let t = f - PSQ_DIMS;
            let p = _mm256_loadu_si256(t_ptr.add(t * N_BUCKETS) as *const __m256i);
            acc = _mm256_add_epi32(acc, p);
        }
        if !pair_psqt_w.is_empty() {
            let p_ptr = pair_psqt_w.as_ptr();
            for &f in pair_features {
                let t = f - PSQ_DIMS - THREAT_DIMS;
                let p = _mm256_loadu_si256(p_ptr.add(t * N_BUCKETS) as *const __m256i);
                acc = _mm256_add_epi32(acc, p);
            }
        }
        _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, acc);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn pairwise_transform_threats_avx2(
    base_acc: &[i16],
    threat_buf: &[i32; L1],
    dst: &mut [u8],
    half: usize,
    clip_max: i32,
) {
    unsafe {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero_si256();
        let max_val = _mm256_set1_epi32(clip_max);
        let base_ptr = base_acc.as_ptr();
        let threat_ptr = threat_buf.as_ptr() as *const __m256i;

        for chunk in 0..(half / 16) {
            let j = chunk * 16;
            let b0_256 = _mm256_loadu_si256(base_ptr.add(j) as *const __m256i);
            let b0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_256));
            let b0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_256, 1));

            let b1_256 = _mm256_loadu_si256(base_ptr.add(j + half) as *const __m256i);
            let b1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_256));
            let b1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_256, 1));

            let t0_lo = _mm256_loadu_si256(threat_ptr.add(chunk * 2));
            let t0_hi = _mm256_loadu_si256(threat_ptr.add(chunk * 2 + 1));
            let t1_lo = _mm256_loadu_si256(threat_ptr.add(64 + chunk * 2));
            let t1_hi = _mm256_loadu_si256(threat_ptr.add(64 + chunk * 2 + 1));

            let s0_lo = _mm256_add_epi32(b0_lo, t0_lo);
            let s0_hi = _mm256_add_epi32(b0_hi, t0_hi);
            let s1_lo = _mm256_add_epi32(b1_lo, t1_lo);
            let s1_hi = _mm256_add_epi32(b1_hi, t1_hi);

            let c0_lo = _mm256_min_epi32(_mm256_max_epi32(s0_lo, zero), max_val);
            let c0_hi = _mm256_min_epi32(_mm256_max_epi32(s0_hi, zero), max_val);
            let c1_lo = _mm256_min_epi32(_mm256_max_epi32(s1_lo, zero), max_val);
            let c1_hi = _mm256_min_epi32(_mm256_max_epi32(s1_hi, zero), max_val);

            let prod_lo = _mm256_mullo_epi32(c0_lo, c1_lo);
            let prod_hi = _mm256_mullo_epi32(c0_hi, c1_hi);

            let div_lo = _mm256_srli_epi32(prod_lo, 9);
            let div_hi = _mm256_srli_epi32(prod_hi, 9);

            let d_lo_0 = _mm256_castsi256_si128(div_lo);
            let d_lo_1 = _mm256_extracti128_si256(div_lo, 1);
            let d_hi_0 = _mm256_castsi256_si128(div_hi);
            let d_hi_1 = _mm256_extracti128_si256(div_hi, 1);

            let p0 = _mm_packs_epi32(d_lo_0, d_lo_1);
            let p1 = _mm_packs_epi32(d_hi_0, d_hi_1);

            let bytes = _mm_packus_epi16(p0, p1);
            _mm_storeu_si128(dst.as_mut_ptr().add(j) as *mut __m128i, bytes);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn pairwise_transform_no_threats_avx2(
    base_acc: &[i16],
    dst: &mut [u8],
    half: usize,
    clip_max: i32,
) {
    unsafe {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero_si256();
        let max_val = _mm256_set1_epi32(clip_max);
        let base_ptr = base_acc.as_ptr();

        for chunk in 0..(half / 16) {
            let j = chunk * 16;
            let b0_256 = _mm256_loadu_si256(base_ptr.add(j) as *const __m256i);
            let b0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_256));
            let b0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_256, 1));

            let b1_256 = _mm256_loadu_si256(base_ptr.add(j + half) as *const __m256i);
            let b1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_256));
            let b1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_256, 1));

            let c0_lo = _mm256_min_epi32(_mm256_max_epi32(b0_lo, zero), max_val);
            let c0_hi = _mm256_min_epi32(_mm256_max_epi32(b0_hi, zero), max_val);
            let c1_lo = _mm256_min_epi32(_mm256_max_epi32(b1_lo, zero), max_val);
            let c1_hi = _mm256_min_epi32(_mm256_max_epi32(b1_hi, zero), max_val);

            let prod_lo = _mm256_mullo_epi32(c0_lo, c1_lo);
            let prod_hi = _mm256_mullo_epi32(c0_hi, c1_hi);

            let div_lo = _mm256_srli_epi32(prod_lo, 9);
            let div_hi = _mm256_srli_epi32(prod_hi, 9);

            let d_lo_0 = _mm256_castsi256_si128(div_lo);
            let d_lo_1 = _mm256_extracti128_si256(div_lo, 1);
            let d_hi_0 = _mm256_castsi256_si128(div_hi);
            let d_hi_1 = _mm256_extracti128_si256(div_hi, 1);

            let p0 = _mm_packs_epi32(d_lo_0, d_lo_1);
            let p1 = _mm_packs_epi32(d_hi_0, d_hi_1);

            let bytes = _mm_packus_epi16(p0, p1);
            _mm_storeu_si128(dst.as_mut_ptr().add(j) as *mut __m128i, bytes);
        }
    }
}
pub(super) fn eval_with_net(
    pos: &SfnnPosition,
    net: &Sfnn16Net,
    halfka_accs: [&[i16]; 2],
    psqt_accs: [&[i32; N_BUCKETS]; 2],
    stm: Side,
    bucket: usize,
) -> (i32, i32) {
    let l1 = net.l1;
    let half = l1 / 2;
    let clip_max: i32 = 255;
    let perspectives = [stm, stm.other()];

    let mut threat_lists = [[0usize; MAX_THREAT_ACTIVE]; 2];
    let mut threat_lens = [0usize; 2];
    let mut pair_lists = [[0usize; MAX_PAIR_ACTIVE]; 2];
    let mut pair_lens = [0usize; 2];

    if net.use_threats && maintenance_active() {
        let mut raw_threats = [(0u8, 0u8, 0u8, 0u8); 128];
        let mut n_raw_threats = 0usize;
        for_each_threat(pos, |attacker, from, to, attacked| {
            if n_raw_threats < 128 {
                raw_threats[n_raw_threats] = (attacker as u8, from as u8, to as u8, attacked as u8);
                n_raw_threats += 1;
            }
        });

        let luts = threat_luts();
        for (slot, &persp) in perspectives.iter().enumerate() {
            let ksq = pos.king_square(persp);
            let orient = threat_orient(persp, ksq);
            let flip_color = if persp == Side::Black { 8 } else { 0 };
            let mut len = 0;
            for &(attacker_raw, from_sf, to_sf, attacked_raw) in &raw_threats[..n_raw_threats] {
                let attacker = (attacker_raw as usize) ^ flip_color;
                let attacked = (attacked_raw as usize) ^ flip_color;
                let word = luts.lut1[attacker][attacked];
                let info = (word & 0xFF) as u8;
                let from = (from_sf as usize) ^ orient;
                let to = (to_sf as usize) ^ orient;
                let less_than = (from < to) as u8;
                if (info + less_than) & 2 != 0 {
                    continue;
                }
                let idx = (word >> 8)
                    + luts.offsets[attacker][from]
                    + luts.lut2[attacker][from][to] as u32;
                if (idx as usize) < THREAT_DIMS && len < MAX_THREAT_ACTIVE {
                    threat_lists[slot][len] = PSQ_DIMS + idx as usize;
                    len += 1;
                }
            }
            threat_lens[slot] = len;
        }

        if !net.transformer.pair_w.is_empty() || !net.transformer.pair_w_i16.is_empty() {
            let mut raw_pairs = [(Side::White, 0u8, 0u8, Side::White); 64];
            let mut n_raw_pairs = 0usize;
            for_each_pair(pos, |color, from, to, paired| {
                if n_raw_pairs < 64 {
                    raw_pairs[n_raw_pairs] = (color, from as u8, to as u8, paired);
                    n_raw_pairs += 1;
                }
            });

            for (slot, &persp) in perspectives.iter().enumerate() {
                let ksq = pos.king_square(persp);
                let flip = if persp == Side::Black { 56 } else { 0 };
                let left_files = (ksq & 7) < 4;
                let orient = flip ^ if left_files { 0 } else { 7 };
                let mut len = 0;
                for &(color, from_sf, to_sf, paired) in &raw_pairs[..n_raw_pairs] {
                    if len < MAX_PAIR_ACTIVE {
                        let from_oriented = (from_sf as usize) ^ orient;
                        let to_oriented = (to_sf as usize) ^ orient;
                        let color_oriented = if persp == Side::Black {
                            color.other()
                        } else {
                            color
                        };
                        let paired_oriented = if persp == Side::Black {
                            paired.other()
                        } else {
                            paired
                        };
                        let (Some(p1), Some(p2)) = (
                            make_pawn_id(color_oriented, from_oriented),
                            make_pawn_id(paired_oriented, to_oriented),
                        ) else {
                            continue;
                        };
                        let (a, b) = if p1 <= p2 { (p1, p2) } else { (p2, p1) };
                        let idx = (b * (b - 1)) / 2 + a + PSQ_DIMS + THREAT_DIMS;
                        pair_lists[slot][len] = idx;
                        len += 1;
                    }
                }
                pair_lens[slot] = len;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = has_avx2() && l1 == 1024 && !net.is_rudi;
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    let mut feats = [0u8; L1];
    let mut per_psqt = [[0i32; N_BUCKETS]; 2];

    if use_avx2 && half == 512 {
        for (slot, &persp) in perspectives.iter().enumerate() {
            let pi = persp as usize;
            let base_acc = halfka_accs[pi];
            let dst = &mut feats[slot * half..(slot + 1) * half];

            if net.use_threats {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    pairwise_transform_threats_tiled_avx2(
                        base_acc,
                        &net.transformer.threat_w,
                        &net.transformer.pair_w,
                        &threat_lists[slot][..threat_lens[slot]],
                        &pair_lists[slot][..pair_lens[slot]],
                        dst,
                        half,
                        clip_max,
                    );
                    accumulate_psqt_avx2(
                        &mut per_psqt[slot],
                        &net.transformer.threat_psqt_w,
                        &net.transformer.pair_psqt_w,
                        &threat_lists[slot][..threat_lens[slot]],
                        &pair_lists[slot][..pair_lens[slot]],
                    );
                }
            } else {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    pairwise_transform_no_threats_avx2(base_acc, dst, half, clip_max);
                }
            }
        }
    } else {
        for (slot, &persp) in perspectives.iter().enumerate() {
            let pi = persp as usize;
            let base_acc = halfka_accs[pi];
            let mut threat_buf = [0i32; L1];
            if net.use_threats {
                if net.is_rudi {
                    #[cfg(target_arch = "x86_64")]
                    let use_rudi_avx2 = has_avx2() && l1 == L1;
                    #[cfg(not(target_arch = "x86_64"))]
                    let use_rudi_avx2 = false;

                    for &f in &threat_lists[slot][..threat_lens[slot]] {
                        let t = f - PSQ_DIMS;
                        let base = t * l1;
                        let w_slice = &net.transformer.threat_w_i16[base..base + l1];
                        if use_rudi_avx2 {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                add_threat_w_i16_avx2(&mut threat_buf, w_slice);
                            }
                        } else {
                            for (acc, &w) in threat_buf.iter_mut().zip(w_slice) {
                                *acc += i32::from(w);
                            }
                        }
                    }
                    for &f in &pair_lists[slot][..pair_lens[slot]] {
                        let base = (f - PSQ_DIMS - THREAT_DIMS) * l1;
                        let w_slice = &net.transformer.pair_w_i16[base..base + l1];
                        if use_rudi_avx2 {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                add_threat_w_i16_avx2(&mut threat_buf, w_slice);
                            }
                        } else {
                            for (acc, &w) in threat_buf.iter_mut().zip(w_slice) {
                                *acc += i32::from(w);
                            }
                        }
                    }
                } else {
                    for &f in &threat_lists[slot][..threat_lens[slot]] {
                        let t = f - PSQ_DIMS;
                        let base = t * l1;
                        let w_slice = &net.transformer.threat_w[base..base + l1];
                        for (acc, &w) in threat_buf.iter_mut().zip(w_slice) {
                            *acc += i32::from(w);
                        }
                        let base_psqt = t * N_BUCKETS;
                        let p_slice =
                            &net.transformer.threat_psqt_w[base_psqt..base_psqt + N_BUCKETS];
                        for b in 0..N_BUCKETS {
                            per_psqt[slot][b] = per_psqt[slot][b].wrapping_add(p_slice[b]);
                        }
                    }
                    for &f in &pair_lists[slot][..pair_lens[slot]] {
                        let t = f - PSQ_DIMS - THREAT_DIMS;
                        let base = t * l1;
                        let w_slice = &net.transformer.pair_w[base..base + l1];
                        for (acc, &w) in threat_buf.iter_mut().zip(w_slice) {
                            *acc += i32::from(w);
                        }
                        let base_psqt = t * N_BUCKETS;
                        let p_slice =
                            &net.transformer.pair_psqt_w[base_psqt..base_psqt + N_BUCKETS];
                        for b in 0..N_BUCKETS {
                            per_psqt[slot][b] = per_psqt[slot][b].wrapping_add(p_slice[b]);
                        }
                    }
                }
            }
            let dst = &mut feats[slot * half..(slot + 1) * half];
            if net.is_rudi {
                #[cfg(target_arch = "x86_64")]
                let use_rudi_avx2 = has_avx2() && half.is_multiple_of(16);
                #[cfg(not(target_arch = "x86_64"))]
                let use_rudi_avx2 = false;

                if use_rudi_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        pairwise_transform_rudi_avx2(
                            base_acc,
                            &threat_buf,
                            dst,
                            half,
                            net.use_threats,
                        );
                    }
                } else {
                    for j in 0..half {
                        let mut s0 = i32::from(base_acc[j]);
                        let mut s1 = i32::from(base_acc[j + half]);
                        if net.use_threats {
                            s0 += threat_buf[j];
                            s1 += threat_buf[j + half];
                        }
                        let c0 = s0.clamp(0, clip_max);
                        let c1 = s1.clamp(0, clip_max);
                        dst[j] = ((c0 * c1) / 255) as u8;
                    }
                }
            } else {
                for j in 0..half {
                    let mut s0 = i32::from(base_acc[j]);
                    let mut s1 = i32::from(base_acc[j + half]);
                    if net.use_threats {
                        s0 += threat_buf[j];
                        s1 += threat_buf[j + half];
                    }
                    let c0 = s0.clamp(0, clip_max);
                    let c1 = s1.clamp(0, clip_max);
                    dst[j] = ((c0 * c1) / 512) as u8;
                }
            }
        }
    }
    let mut psqt = psqt_accs[stm as usize][bucket] - psqt_accs[stm.other() as usize][bucket];
    if net.use_threats {
        psqt = div_trunc(psqt + per_psqt[0][bucket] - per_psqt[1][bucket], 2);
    } else {
        psqt = div_trunc(psqt, 2);
    }
    let stack = &net.stacks[bucket];
    let positional = propagate(stack, l1, &feats[..l1]);
    if net.is_rudi {
        (0, div_trunc(positional, OUTPUT_SCALE))
    } else {
        (
            div_trunc(psqt, OUTPUT_SCALE),
            div_trunc(positional, OUTPUT_SCALE),
        )
    }
}
