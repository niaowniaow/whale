use crate::world::intent::SearchIntent;
use crate::world::position_state::PositionState;

pub const FEATURES: [&str; 11] = [
    "eval_stm",
    "pressure",
    "opp_cpi",
    "own_cpi",
    "urgency",
    "risk",
    "musttry",
    "concession",
    "state",
    "intent",
    "side",
];

pub const WEIGHTS_PATH: &str = "heads_weights.json";

#[derive(Debug, Clone)]
pub struct HeadWeights {
    pub w: [f32; 11],
    pub b: f32,
    pub means: [f32; 11],
    pub stds: [f32; 11],
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
}

impl HeadWeights {
    pub fn predict(&self, features: [f32; 11]) -> f32 {
        let mut z = self.b;
        for (i, &feat) in features.iter().enumerate() {
            let std = if self.stds[i].abs() < 1e-9 {
                1.0
            } else {
                self.stds[i]
            };
            z += (feat - self.means[i]) / std * self.w[i];
        }
        sigmoid(z)
    }
}

pub fn state_id(state: PositionState) -> u8 {
    match state {
        PositionState::Defend => 0,
        PositionState::Stabilize => 1,
        PositionState::Improve => 2,
        PositionState::Press => 3,
        PositionState::Attack => 4,
        PositionState::Reset => 5,
        PositionState::Crush => 6,
        PositionState::Convert => 7,
    }
}

pub fn intent_id(intent: SearchIntent) -> u8 {
    match intent {
        SearchIntent::Survival => 0,
        SearchIntent::Stabilization => 1,
        SearchIntent::Improvement => 2,
        SearchIntent::Pressure => 3,
        SearchIntent::Attack => 4,
        SearchIntent::Verification => 5,
        SearchIntent::Recovery => 6,
        SearchIntent::Conversion => 7,
    }
}

pub fn urgency_id(urgency: crate::risk::Urgency) -> u8 {
    match urgency {
        crate::risk::Urgency::Low => 0,
        crate::risk::Urgency::Medium => 1,
        crate::risk::Urgency::High => 2,
        crate::risk::Urgency::MustTry => 3,
    }
}

fn parse_f32_array(content: &str, key: &str) -> Option<[f32; 11]> {
    let key_pos = content.find(key)?;
    let open = content[key_pos..].find('[')?;
    let abs_open = key_pos + open;
    let close_rel = content[abs_open..].find(']')?;
    let inner = &content[abs_open + 1..abs_open + close_rel];
    let mut out = [0f32; 11];
    let mut count = 0usize;
    for part in inner.split(',') {
        if count >= 11 {
            break;
        }
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        out[count] = part.parse::<f32>().ok()?;
        count += 1;
    }
    if count != 11 {
        return None;
    }
    Some(out)
}

fn parse_f32_scalar(content: &str, key: &str) -> Option<f32> {
    let key_pos = content.find(key)?;
    let colon_rel = content[key_pos..].find(':')?;
    let rest = content[key_pos + colon_rel + 1..].trim_start();
    let mut end = 0usize;
    for (i, ch) in rest.char_indices() {
        if ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == '+' || ch == 'e' || ch == 'E' {
            end = i + ch.len_utf8();
        } else {
            break;
        }
    }
    rest[..end].parse::<f32>().ok()
}

fn parse_names(content: &str) -> Option<bool> {
    let key_pos = content.find("\"features\"")?;
    let open = content[key_pos..].find('[')? + key_pos;
    let close = content[open..].find(']')? + open;
    let inner = &content[open + 1..close];
    let mut names = Vec::new();
    for part in inner.split(',') {
        names.push(part.trim().trim_matches('"').to_string());
    }
    if names.len() != FEATURES.len() {
        return Some(false);
    }
    for (a, b) in names.iter().zip(FEATURES.iter()) {
        if a != b {
            return Some(false);
        }
    }
    Some(true)
}

pub fn load_weights(path: &str) -> Result<HeadWeights, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
    if parse_names(&content) != Some(true) {
        return Err("feature list mismatch".to_string());
    }
    let w = parse_f32_array(&content, "\"w\"").ok_or("bad w".to_string())?;
    let means = parse_f32_array(&content, "\"means\"").ok_or("bad means".to_string())?;
    let stds = parse_f32_array(&content, "\"stds\"").ok_or("bad stds".to_string())?;
    let b = parse_f32_scalar(&content, "\"b\"").ok_or("bad b".to_string())?;
    Ok(HeadWeights { w, b, means, stds })
}

static CACHED: std::sync::OnceLock<std::sync::Mutex<Option<HeadWeights>>> =
    std::sync::OnceLock::new();

fn cache() -> &'static std::sync::Mutex<Option<HeadWeights>> {
    CACHED.get_or_init(|| std::sync::Mutex::new(None))
}

pub fn ensure_loaded() -> bool {
    if cache().lock().unwrap().is_some() {
        return true;
    }
    match load_weights(WEIGHTS_PATH) {
        Ok(weights) => {
            *cache().lock().unwrap() = Some(weights);
            true
        }
        Err(_) => false,
    }
}

pub fn predict_cached(features: [f32; 11]) -> Option<f32> {
    let guard = cache().lock().unwrap();
    guard.as_ref().map(|weights| weights.predict(features))
}

#[cfg(test)]
pub fn clear_cache_for_tests() {
    *cache().lock().unwrap() = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_weights() -> HeadWeights {
        HeadWeights {
            w: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            b: 0.0,
            means: [0.0; 11],
            stds: [1.0; 11],
        }
    }

    #[test]
    fn predict_is_bounded_and_monotone_in_eval() {
        let weights = sample_weights();
        let mut low = [0f32; 11];
        low[0] = -3.0;
        let mut high = [0f32; 11];
        high[0] = 3.0;
        let p_low = weights.predict(low);
        let p_high = weights.predict(high);
        assert!((0.0..=1.0).contains(&p_low));
        assert!((0.0..=1.0).contains(&p_high));
        assert!(p_high > p_low);
        assert!((weights.predict([0f32; 11]) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn loader_rejects_mismatched_features() {
        let dir = std::env::temp_dir().join(format!("whale-heads-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad_path = dir.join("bad.json");
        std::fs::write(&bad_path, r#"{"features":["a"],"w":[1.0],"b":0.0}"#).unwrap();
        assert!(load_weights(bad_path.to_str().unwrap()).is_err());
        let good_content = r#"{"features":["eval_stm","pressure","opp_cpi","own_cpi","urgency","risk","musttry","concession","state","intent","side"],"w":[1,0,0,0,0,0,0,0,0,0,0],"b":0.0,"means":[0,0,0,0,0,0,0,0,0,0,0],"stds":[1,1,1,1,1,1,1,1,1,1,1]}"#;
        let good_path = dir.join("good.json");
        std::fs::write(&good_path, good_content).unwrap();
        let loaded = load_weights(good_path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.w[0], 1.0);
        assert_eq!(loaded.b, 0.0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn ids_cover_all_variants() {
        use crate::risk::Urgency;
        assert_eq!(state_id(PositionState::Reset), 5);
        assert_eq!(intent_id(SearchIntent::Recovery), 6);
        assert_eq!(urgency_id(Urgency::MustTry), 3);
        assert_eq!(FEATURES.len(), 11);
    }
}
