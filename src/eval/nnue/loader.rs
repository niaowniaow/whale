use std::alloc::{Layout, alloc_zeroed, handle_alloc_error};

use crate::eval::nnue::{ACC_SIZE, INPUT_SIZE};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetworkMetadata {
    pub input_size: usize,
    pub acc_size: usize,
    pub output_size: usize,
    pub bytes: usize,
    pub checksum: u64,
}

#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct Network {
    pub transformer_weights: [i16; INPUT_SIZE * ACC_SIZE],
    pub transformer_biases: [i16; ACC_SIZE],
    pub output_weights: [i16; ACC_SIZE * 2],
    pub output_bias: i16,
}

static EMBEDDED_NETWORK: Network =
    unsafe { std::mem::transmute(*include_bytes!("../../../resources/nnue.bin")) };

impl Network {
    #[inline(always)]
    pub fn get_embedded() -> &'static Self {
        &EMBEDDED_NETWORK
    }

    pub fn metadata(&self) -> NetworkMetadata {
        let bytes = std::mem::size_of_val(self);
        let checksum = Self::checksum_bytes(self as *const Self as *const u8, bytes);

        NetworkMetadata {
            input_size: INPUT_SIZE,
            acc_size: ACC_SIZE,
            output_size: 1,
            bytes,
            checksum,
        }
    }

    fn checksum_bytes(ptr: *const u8, len: usize) -> u64 {
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;

        let mut hash = FNV_OFFSET;
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    // TODO: for tests, refactor
    pub fn new_boxed() -> Box<Self> {
        unsafe {
            let layout = Layout::new::<Self>();
            let ptr = alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            Box::from_raw(ptr)
        }
    }

    pub fn randomize(&mut self) {
        use crate::common::random;
        for val in self.transformer_weights.iter_mut() {
            *val = random::next_i16_range(-10, 10);
        }
        for val in self.transformer_biases.iter_mut() {
            *val = random::next_i16_range(-10, 10);
        }
        for val in self.output_weights.iter_mut() {
            *val = random::next_i16_range(-10, 10);
        }
        self.output_bias = random::next_i16_range(-10, 10);
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;
        use std::slice::from_raw_parts;

        let mut file = File::create(path)?;
        let bytes = unsafe {
            from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        };
        file.write_all(bytes)?;
        Ok(())
    }

    pub fn snapshot_baseline(&self, path: &str) -> std::io::Result<()> {
        let metadata = self.metadata();
        let parent = std::path::Path::new(path)
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        std::fs::create_dir_all(parent)?;

        self.save_to_file(path)?;

        let metadata_text = format!(
            "{{\n  \"input_size\": {},\n  \"acc_size\": {},\n  \"output_size\": {},\n  \"bytes\": {},\n  \"checksum\": {}\n}}\n",
            metadata.input_size,
            metadata.acc_size,
            metadata.output_size,
            metadata.bytes,
            metadata.checksum
        );

        std::fs::write(format!("{}.meta", path), metadata_text)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_network_metadata_is_stable() {
        let network = Network::get_embedded();
        let metadata = network.metadata();

        assert_eq!(metadata.input_size, INPUT_SIZE);
        assert_eq!(metadata.acc_size, ACC_SIZE);
        assert_eq!(metadata.output_size, 1);
        assert!(metadata.bytes > 0);
        assert_ne!(metadata.checksum, 0);
    }

    #[test]
    fn boxed_network_starts_zeroed() {
        let net = Network::new_boxed();
        assert!(net.transformer_weights.iter().all(|&w| w == 0));
        assert!(net.transformer_biases.iter().all(|&b| b == 0));
        assert!(net.output_weights.iter().all(|&w| w == 0));
        assert_eq!(net.output_bias, 0);
    }

    #[test]
    fn randomize_stays_in_range() {
        let mut net = Network::new_boxed();
        net.randomize();
        for &w in net
            .transformer_weights
            .iter()
            .chain(net.transformer_biases.iter())
            .chain(net.output_weights.iter())
        {
            assert!((-10..=10).contains(&w));
        }
        assert!((-10..=10).contains(&net.output_bias));
        assert!(net.metadata().bytes > 0);
    }

    #[test]
    fn save_and_snapshot_roundtrip_in_tempdir() {
        let dir = std::env::temp_dir().join(format!("whale-nnue-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let network = Network::get_embedded();
        let path = dir.join("net.bin");
        let path_str = path.to_str().unwrap();
        network.save_to_file(path_str).unwrap();
        let saved = std::fs::read(&path).unwrap();
        assert_eq!(saved.len(), std::mem::size_of::<Network>());

        let snap = dir.join("sub").join("snap.bin");
        network.snapshot_baseline(snap.to_str().unwrap()).unwrap();
        let meta = std::fs::read_to_string(format!("{}.meta", snap.to_str().unwrap())).unwrap();
        assert!(meta.contains("\"input_size\""));
        assert!(meta.contains("\"checksum\""));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
