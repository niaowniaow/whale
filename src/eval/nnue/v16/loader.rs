use super::arch::LEB128_MAGIC;
use super::*;

pub(super) fn read_u32_le(data: &[u8], pos: &mut usize) -> Result<u32, &'static str> {
    if *pos + 4 > data.len() {
        return Err("truncated u32");
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

pub(super) fn read_i32_le(data: &[u8], pos: &mut usize) -> Result<i32, &'static str> {
    Ok(read_u32_le(data, pos)? as i32)
}

pub(super) fn read_leb128_section<T, F>(
    data: &[u8],
    pos: &mut usize,
    count: usize,
    decode: F,
) -> Result<Vec<T>, &'static str>
where
    F: Fn(&[u8], &mut usize) -> Result<T, &'static str>,
{
    if *pos + LEB128_MAGIC.len() > data.len() {
        return Err("truncated leb128 magic");
    }
    if &data[*pos..*pos + LEB128_MAGIC.len()] != LEB128_MAGIC {
        return Err("bad leb128 magic");
    }
    *pos += LEB128_MAGIC.len();
    let _bytes = read_u32_le(data, pos)?;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(decode(data, pos)?);
    }
    Ok(out)
}

pub(super) fn decode_leb128_i64(
    data: &[u8],
    pos: &mut usize,
    bits: u32,
) -> Result<i64, &'static str> {
    let mut result: i64 = 0;
    let mut shift = 0u32;
    loop {
        if *pos >= data.len() {
            return Err("truncated leb128 payload");
        }
        let byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as i64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            if shift < bits && (byte & 0x40) != 0 {
                result |= !0i64 << shift;
            }
            break;
        }
        if shift >= bits + 7 {
            return Err("leb128 overflow");
        }
    }
    Ok(result)
}

pub(super) fn decode_leb128_i16(data: &[u8], pos: &mut usize) -> Result<i16, &'static str> {
    Ok(decode_leb128_i64(data, pos, 16)? as i16)
}

pub(super) fn decode_leb128_i32(data: &[u8], pos: &mut usize) -> Result<i32, &'static str> {
    Ok(decode_leb128_i64(data, pos, 32)? as i32)
}

#[derive(Clone, Debug)]
pub struct SfnnArch {
    pub fc0_bias: [i32; FC0_OUT],
    pub fc0_w: Vec<i8>,
    pub fc1_bias: [i32; FC1_OUT],
    pub fc1_w: Vec<i8>,
    pub fc2_bias: i32,
    pub fc2_w: [i8; FC0_OUT * 2 + FC1_OUT * 2],
    pub is_rudi: bool,
}

impl SfnnArch {
    pub(super) fn load(
        data: &[u8],
        pos: &mut usize,
        fc0_in: usize,
        expected_hash: u32,
    ) -> Result<Self, &'static str> {
        let hash = read_u32_le(data, pos)?;
        if hash != expected_hash {
            return Err("bad arch hash");
        }
        let mut fc0_bias = [0i32; FC0_OUT];
        for b in fc0_bias.iter_mut() {
            *b = read_i32_le(data, pos)?;
        }
        let padded0 = fc0_in.next_multiple_of(32);
        let mut raw0 = vec![0i8; FC0_OUT * padded0];
        for v in raw0.iter_mut() {
            if *pos >= data.len() {
                return Err("truncated fc0 weights");
            }
            *v = data[*pos] as i8;
            *pos += 1;
        }
        let fc0_w = raw0;

        let mut fc1_bias = [0i32; FC1_OUT];
        for b in fc1_bias.iter_mut() {
            *b = read_i32_le(data, pos)?;
        }
        let mut raw1 = vec![0i8; FC1_OUT * 64];
        for v in raw1.iter_mut() {
            if *pos >= data.len() {
                return Err("truncated fc1 weights");
            }
            *v = data[*pos] as i8;
            *pos += 1;
        }
        let fc1_w = raw1;

        let fc2_bias = read_i32_le(data, pos)?;
        let mut fc2_w = [0i8; FC1_IN + FC1_OUT * 2];
        for v in fc2_w.iter_mut() {
            if *pos >= data.len() {
                return Err("truncated fc2 weights");
            }
            *v = data[*pos] as i8;
            *pos += 1;
        }

        Ok(Self {
            fc0_bias,
            fc0_w,
            fc1_bias,
            fc1_w,
            fc2_bias,
            fc2_w,
            is_rudi: false,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct SfnnTransformer {
    pub bias: Vec<i16>,
    pub weights: Vec<i16>,
    pub threat_w: Vec<i8>,
    pub threat_w_i16: Vec<i16>,
    pub psqt_w: Vec<i32>,
    pub threat_psqt_w: Vec<i32>,
    pub pair_w: Vec<i8>,
    pub pair_w_i16: Vec<i16>,
    pub pair_psqt_w: Vec<i32>,
}

#[derive(Clone, Debug)]
pub struct Sfnn16Net {
    pub l1: usize,
    pub use_threats: bool,
    pub is_rudi: bool,
    pub transformer: SfnnTransformer,
    pub stacks: Vec<SfnnArch>,
}

impl Sfnn16Net {
    pub fn load_bytes(data: &[u8], use_threats: bool, l1: usize) -> Result<Self, &'static str> {
        let mut pos = 0;
        let version = read_u32_le(data, &mut pos)?;
        if version != SF_FILE_VERSION && version != SF17_FILE_VERSION {
            return Err("bad SFNN file version");
        }
        let file_hash = read_u32_le(data, &mut pos)?;
        if use_threats && file_hash != network_hash(true, l1 as u32) {
            return Err("bad SFNN file hash");
        }
        let desc_len = read_u32_le(data, &mut pos)? as usize;
        if pos + desc_len > data.len() {
            return Err("truncated description");
        }
        pos += desc_len;

        let thash = read_u32_le(data, &mut pos)?;
        if use_threats && thash != transformer_hash(true, l1 as u32) {
            return Err("bad transformer hash");
        }
        let bias = read_leb128_section(data, &mut pos, l1, decode_leb128_i16)?;

        let psq_inputs = PSQ_DIMS;
        let thr_inputs = if use_threats { THREAT_DIMS } else { 0 };
        let pair_inputs = if use_threats { PAIR_DIMS } else { 0 };

        let (weights, psqt_w, threat_w, threat_psqt_w, pair_w, pair_psqt_w) = if use_threats {
            let thr_bytes = thr_inputs * l1;
            let is_sf17 = pos + thr_bytes + LEB128_MAGIC.len() <= data.len()
                && &data[pos + thr_bytes..pos + thr_bytes + LEB128_MAGIC.len()] == LEB128_MAGIC;

            if is_sf17 {
                let tw = data[pos..pos + thr_bytes]
                    .iter()
                    .map(|&b| b as i8)
                    .collect::<Vec<i8>>();
                pos += thr_bytes;

                let tp =
                    read_leb128_section(data, &mut pos, thr_inputs * N_BUCKETS, decode_leb128_i32)?;

                let pair_bytes = pair_inputs * l1;
                if pos + pair_bytes > data.len() {
                    return Err("truncated pair weights");
                }
                let pw = data[pos..pos + pair_bytes]
                    .iter()
                    .map(|&b| b as i8)
                    .collect::<Vec<i8>>();
                pos += pair_bytes;

                let pp = read_leb128_section(
                    data,
                    &mut pos,
                    pair_inputs * N_BUCKETS,
                    decode_leb128_i32,
                )?;

                let w = read_leb128_section(data, &mut pos, psq_inputs * l1, decode_leb128_i16)?;
                let pw_psqt =
                    read_leb128_section(data, &mut pos, psq_inputs * N_BUCKETS, decode_leb128_i32)?;

                (w, pw_psqt, tw, tp, pw, pp)
            } else {
                let combined = read_leb128_section(
                    data,
                    &mut pos,
                    (thr_inputs + psq_inputs) * l1,
                    decode_leb128_i16,
                )?;
                let (t, m) = combined.split_at(thr_inputs * l1);
                let w = m.to_vec();
                let tw = t.iter().map(|&v| v as i8).collect::<Vec<i8>>();

                let combined_psqt = read_leb128_section(
                    data,
                    &mut pos,
                    (thr_inputs + psq_inputs) * N_BUCKETS,
                    decode_leb128_i32,
                )?;
                let (tp_slice, pw_slice) = combined_psqt.split_at(thr_inputs * N_BUCKETS);
                let pw_psqt = pw_slice.to_vec();
                let tp = tp_slice.to_vec();

                (w, pw_psqt, tw, tp, Vec::new(), Vec::new())
            }
        } else {
            let w = read_leb128_section(data, &mut pos, psq_inputs * l1, decode_leb128_i16)?;
            let pw_psqt =
                read_leb128_section(data, &mut pos, psq_inputs * N_BUCKETS, decode_leb128_i32)?;
            (w, pw_psqt, Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        let ahash = arch_hash(l1 as u32);
        let mut stacks = Vec::with_capacity(N_BUCKETS);
        for _ in 0..N_BUCKETS {
            stacks.push(SfnnArch::load(data, &mut pos, l1, ahash)?);
        }
        if pos != data.len() {
            return Err("trailing data after network");
        }

        Ok(Self {
            l1,
            use_threats,
            is_rudi: false,
            transformer: SfnnTransformer {
                bias,
                weights,
                threat_w,
                threat_w_i16: Vec::new(),
                psqt_w,
                threat_psqt_w,
                pair_w,
                pair_w_i16: Vec::new(),
                pair_psqt_w,
            },
            stacks,
        })
    }

    pub fn load_rudi(data: &[u8]) -> Result<Self, &'static str> {
        let raw_payload = if data.starts_with(b"RUDI") {
            let mut header_offset = 4;
            let payload_len = read_u32_le(data, &mut header_offset)? as usize;
            if !matches!(payload_len, 181_011_108 | 181_011_136)
                || data.len() != header_offset + payload_len
            {
                return Err("unsupported RUDI payload layout");
            }
            &data[header_offset..]
        } else if data.len() == 181_011_136 || data.len() == 181_011_108 {
            data
        } else {
            return Err("not a whale net");
        };

        let mut offset = 0;
        let l0w_slice = read_rudi_i16(raw_payload, &mut offset, BIG_INPUT_DIMS * L1)?;

        let l0b_slice = read_rudi_i16(raw_payload, &mut offset, L1)?;

        let l1w_slice = read_rudi_i8(raw_payload, &mut offset, L1 * N_BUCKETS * FC0_OUT)?;

        let l1b_slice = read_rudi_i32(raw_payload, &mut offset, N_BUCKETS * FC0_OUT)?;

        let l2w_slice = read_rudi_i8(raw_payload, &mut offset, FC1_IN * FC1_OUT)?;

        let l2b_slice = read_rudi_i32(raw_payload, &mut offset, FC1_OUT)?;

        let outw_slice = read_rudi_i8(raw_payload, &mut offset, FC1_OUT)?;

        let outb_slice = read_rudi_i32(raw_payload, &mut offset, 1)?;

        let psqt_slice = read_rudi_i32(raw_payload, &mut offset, N_BUCKETS * BIG_INPUT_DIMS)?;

        let weights = l0w_slice[..PSQ_DIMS * L1].to_vec();
        let threat_w_i16 = l0w_slice[PSQ_DIMS * L1..(PSQ_DIMS + THREAT_DIMS) * L1].to_vec();
        let pair_w_i16 = l0w_slice[(PSQ_DIMS + THREAT_DIMS) * L1..BIG_INPUT_DIMS * L1].to_vec();

        let bias = l0b_slice.to_vec();
        let psqt_w = psqt_slice[..PSQ_DIMS * N_BUCKETS].to_vec();
        let threat_psqt_w =
            psqt_slice[PSQ_DIMS * N_BUCKETS..(PSQ_DIMS + THREAT_DIMS) * N_BUCKETS].to_vec();
        let pair_psqt_w =
            psqt_slice[(PSQ_DIMS + THREAT_DIMS) * N_BUCKETS..BIG_INPUT_DIMS * N_BUCKETS].to_vec();

        let mut stacks = Vec::with_capacity(N_BUCKETS);
        for b in 0..N_BUCKETS {
            let mut fc0_bias = [0i32; FC0_OUT];
            fc0_bias.copy_from_slice(&l1b_slice[b * FC0_OUT..(b + 1) * FC0_OUT]);

            let mut fc0_w = vec![0i8; FC0_OUT * L1];
            for o in 0..FC0_OUT {
                for j in 0..L1 {
                    fc0_w[o * L1 + j] = l1w_slice[j * (N_BUCKETS * FC0_OUT) + b * FC0_OUT + o];
                }
            }

            let mut fc1_bias = [0i32; FC1_OUT];
            fc1_bias.copy_from_slice(&l2b_slice);

            let mut fc1_w = vec![0i8; FC1_OUT * FC1_IN];
            for o in 0..FC1_OUT {
                for j in 0..FC1_IN {
                    fc1_w[o * FC1_IN + j] = l2w_slice[j * FC1_OUT + o];
                }
            }

            let fc2_bias = outb_slice[0];
            let mut fc2_w = [0i8; FC0_OUT * 2 + FC1_OUT * 2];
            fc2_w[..FC1_OUT].copy_from_slice(&outw_slice);

            stacks.push(SfnnArch {
                fc0_bias,
                fc0_w,
                fc1_bias,
                fc1_w,
                fc2_bias,
                fc2_w,
                is_rudi: true,
            });
        }

        Ok(Self {
            l1: L1,
            use_threats: true,
            is_rudi: true,
            transformer: SfnnTransformer {
                bias,
                weights,
                threat_w: Vec::new(),
                threat_w_i16,
                psqt_w,
                threat_psqt_w,
                pair_w: Vec::new(),
                pair_w_i16,
                pair_psqt_w,
            },
            stacks,
        })
    }

    pub fn load_file(path: &str, use_threats: bool, l1: usize) -> Result<Self, &'static str> {
        let bytes = std::fs::read(path).map_err(|_| "cannot read network file")?;
        if (bytes.len() >= 4 && &bytes[0..4] == b"RUDI")
            || bytes.len() == 181_011_136
            || bytes.len() == 181_011_108
        {
            return Self::load_rudi(&bytes);
        }
        Self::load_bytes(&bytes, use_threats, l1)
    }
}
pub(super) fn read_rudi_i8(
    data: &[u8],
    offset: &mut usize,
    count: usize,
) -> Result<Vec<i8>, &'static str> {
    if *offset + count > data.len() {
        return Err("truncated whale payload");
    }
    let out = data[*offset..*offset + count]
        .iter()
        .map(|&b| b as i8)
        .collect();
    *offset += count;
    Ok(out)
}

#[allow(clippy::chunks_exact_to_as_chunks)]
pub(super) fn read_rudi_i16(
    data: &[u8],
    offset: &mut usize,
    count: usize,
) -> Result<Vec<i16>, &'static str> {
    let bytes = count.checked_mul(2).ok_or("bad whale length")?;
    if *offset + bytes > data.len() {
        return Err("truncated whale payload");
    }
    let mut out = Vec::with_capacity(count);
    for chunk in data[*offset..*offset + bytes].chunks_exact(2) {
        out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    *offset += bytes;
    Ok(out)
}

#[allow(clippy::chunks_exact_to_as_chunks)]
pub(super) fn read_rudi_i32(
    data: &[u8],
    offset: &mut usize,
    count: usize,
) -> Result<Vec<i32>, &'static str> {
    let bytes = count.checked_mul(4).ok_or("bad whale length")?;
    if *offset + bytes > data.len() {
        return Err("truncated whale payload");
    }
    let mut out = Vec::with_capacity(count);
    for chunk in data[*offset..*offset + bytes].chunks_exact(4) {
        out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    *offset += bytes;
    Ok(out)
}
