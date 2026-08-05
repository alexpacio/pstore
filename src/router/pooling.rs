//! Mask-aware mean pooling, shared by the two encoder classifiers.
//!
//! Both checkpoints pool the encoder's per-token states into one vector before their
//! heads, and both were trained with the padded positions excluded. pstore classifies one
//! prompt at a time so nothing is padded in practice, but the arithmetic has to match the
//! reference implementations exactly or the scores drift — so the mask is honoured rather
//! than assumed away.

use candle_core::Tensor;

/// Mean-pool `(1, seq, hidden)` down to `(1, hidden)`, weighting by `mask`.
pub fn masked_mean(hidden: &Tensor, mask: &Tensor) -> Result<Tensor, String> {
    let m = mask
        .to_dtype(hidden.dtype())
        .and_then(|m| m.unsqueeze(candle_core::D::Minus1))
        .map_err(|e| format!("preparing mask: {e}"))?;
    let summed = hidden
        .broadcast_mul(&m)
        .and_then(|t| t.sum(1))
        .map_err(|e| format!("masked sum: {e}"))?;
    let counts = m
        .sum_keepdim(1)
        .and_then(|c| c.squeeze(candle_core::D::Minus1))
        // No sequence is entirely masked here, but clamp rather than risk NaN.
        .and_then(|c| c.clamp(1e-6, f32::MAX))
        .map_err(|e| format!("mask count: {e}"))?;
    summed
        .broadcast_div(&counts)
        .map_err(|e| format!("mean pool: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn masked_positions_are_excluded_from_the_mean() {
        let dev = Device::Cpu;
        // Two positions: the first all 1s, the second all 9s, with the second masked out.
        let hidden = Tensor::from_vec(vec![1.0f32, 1.0, 9.0, 9.0], (1, 2, 2), &dev).unwrap();
        let mask = Tensor::from_vec(vec![1u32, 0], (1, 2), &dev).unwrap();
        let pooled: Vec<f32> = masked_mean(&hidden, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(pooled, vec![1.0, 1.0], "the padded position leaked in");

        // With both positions live it is a plain mean.
        let mask = Tensor::from_vec(vec![1u32, 1], (1, 2), &dev).unwrap();
        let pooled: Vec<f32> = masked_mean(&hidden, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(pooled, vec![5.0, 5.0]);
    }

    #[test]
    fn an_all_masked_sequence_does_not_produce_nan() {
        let dev = Device::Cpu;
        let hidden = Tensor::zeros((1, 2, 2), DType::F32, &dev).unwrap();
        let mask = Tensor::from_vec(vec![0u32, 0], (1, 2), &dev).unwrap();
        let pooled: Vec<f32> = masked_mean(&hidden, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(pooled.iter().all(|v| v.is_finite()), "got {pooled:?}");
    }
}
