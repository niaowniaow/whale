#[cfg(test)]
use std::sync::Mutex;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

static ULONG_STATE: AtomicU64 = AtomicU64::new(1804289383);
static INT_STATE: AtomicI32 = AtomicI32::new(1804289383);

// Ensure tests don't interleave seed resets
#[cfg(test)]
static TEST_MUTEX: Mutex<()> = Mutex::new(());

// TODO: can be moved into build.rs
pub fn next_u64() -> u64 {
    let mut current = ULONG_STATE.load(Ordering::Relaxed);
    loop {
        let mut next = current;
        next ^= next << 13;
        next ^= next >> 7;
        next ^= next << 17;
        match ULONG_STATE.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => return next,
            Err(v) => current = v,
        }
    }
}

pub fn next_i32() -> i32 {
    let mut current = INT_STATE.load(Ordering::Relaxed);
    loop {
        let mut next = current;
        next ^= next << 13;
        next ^= next >> 17;
        next ^= next << 5;
        match INT_STATE.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return next,
            Err(v) => current = v,
        }
    }
}

pub fn reset_seed() {
    ULONG_STATE.store(1804289383, Ordering::Relaxed);
    INT_STATE.store(1804289383, Ordering::Relaxed);
}

pub fn next_i16_range(min: i16, max: i16) -> i16 {
    let range = (max - min + 1) as u64;
    min + (next_u64() % range) as i16
}

pub fn choose<T>(slice: &[T]) -> Option<&T> {
    if slice.is_empty() {
        None
    } else {
        let idx = (next_u64() as usize) % slice.len();
        Some(&slice[idx])
    }
}

#[cfg(test)]
mod tests {
    // TODO: improve these tests
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_unique_u64s() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset_seed();
        let mut generated = HashSet::new();
        for _ in 0..500 {
            let num = next_u64();
            assert!(
                generated.insert(num),
                "Collision detected for ulong number: {}",
                num
            );
        }
    }

    #[test]
    fn test_unique_i32s() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset_seed();
        let mut generated = HashSet::new();
        for _ in 0..500 {
            let num = next_i32();
            assert!(
                generated.insert(num),
                "Collision detected for int number: {}",
                num
            );
        }
    }

    #[test]
    fn test_next_i16_range() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset_seed();
        for _ in 0..500 {
            let val = next_i16_range(-10, 10);
            assert!((-10..=10).contains(&val));
        }
    }

    #[test]
    fn test_choose() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset_seed();
        let empty: [i32; 0] = [];
        assert_eq!(choose(&empty), None);

        let items = [10, 20, 30];
        for _ in 0..50 {
            let chosen = choose(&items);
            assert!(chosen.is_some());
            assert!(items.contains(chosen.unwrap()));
        }
    }

    #[test]
    fn test_reset_seed_reproduces_u64_sequence() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset_seed();
        let first: Vec<u64> = (0..20).map(|_| next_u64()).collect();
        reset_seed();
        let second: Vec<u64> = (0..20).map(|_| next_u64()).collect();
        assert_eq!(first, second);
    }

    #[test]
    fn test_reset_seed_reproduces_i32_sequence() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset_seed();
        let first: Vec<i32> = (0..20).map(|_| next_i32()).collect();
        reset_seed();
        let second: Vec<i32> = (0..20).map(|_| next_i32()).collect();
        assert_eq!(first, second);
    }

    #[test]
    fn test_next_i16_range_single_value_always_returns_bound() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset_seed();
        for _ in 0..50 {
            assert_eq!(next_i16_range(5, 5), 5);
            assert_eq!(next_i16_range(-3, -3), -3);
            assert_eq!(next_i16_range(0, 0), 0);
        }
    }

    #[test]
    fn test_next_i16_range_deterministic_after_reset() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset_seed();
        let first: Vec<i16> = (0..50).map(|_| next_i16_range(-10, 10)).collect();
        reset_seed();
        let second: Vec<i16> = (0..50).map(|_| next_i16_range(-10, 10)).collect();
        assert_eq!(first, second);
    }

    #[test]
    fn test_choose_single_element_always_returns_it() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset_seed();
        let items = [42];
        for _ in 0..20 {
            assert_eq!(choose(&items), Some(&42));
        }
    }

    #[test]
    fn test_choose_deterministic_after_reset() {
        let _guard = TEST_MUTEX.lock().unwrap();
        let items = [1, 2, 3, 4, 5];
        reset_seed();
        let first: Vec<i32> = (0..20).map(|_| *choose(&items).unwrap()).collect();
        reset_seed();
        let second: Vec<i32> = (0..20).map(|_| *choose(&items).unwrap()).collect();
        assert_eq!(first, second);
    }
}
