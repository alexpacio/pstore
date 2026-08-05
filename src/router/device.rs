//! Picking the compute device for the local classifiers.
//!
//! Metal on macOS, then CUDA, then CPU. Brick's own docs note the router runs fine on
//! CPU, so a GPU is an optimisation rather than a requirement — failing to acquire one
//! is never an error, just a different [`Backend`] in the status readout.

use std::fmt;

/// The backend actually in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Apple GPU via Metal.
    Metal,
    /// NVIDIA GPU via CUDA.
    Cuda,
    /// Plain CPU.
    Cpu,
}

impl Backend {
    /// Whether this is a GPU backend.
    pub fn is_gpu(self) -> bool {
        matches!(self, Backend::Metal | Backend::Cuda)
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Backend::Metal => "Metal (GPU)",
            Backend::Cuda => "CUDA (GPU)",
            Backend::Cpu => "CPU",
        })
    }
}

#[cfg(feature = "candle")]
mod real {
    use super::Backend;
    use candle_core::Device;

    /// Acquire the best available device.
    ///
    /// Each constructor returns `Err` when that backend was not compiled in or no
    /// such device exists, so the chain degrades quietly to CPU.
    pub fn pick() -> (Device, Backend) {
        #[cfg(target_os = "macos")]
        if let Ok(d) = Device::new_metal(0) {
            return (d, Backend::Metal);
        }
        if let Ok(d) = Device::new_cuda(0) {
            return (d, Backend::Cuda);
        }
        (Device::Cpu, Backend::Cpu)
    }
}

#[cfg(feature = "candle")]
pub use real::pick;

/// The backend that would be used, probed without keeping a device handle.
///
/// Answers even when the `candle` feature is off, so the UI always has something to
/// show.
pub fn probe() -> Backend {
    #[cfg(feature = "candle")]
    {
        pick().1
    }
    #[cfg(not(feature = "candle"))]
    {
        Backend::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_names_are_human_readable() {
        assert!(Backend::Metal.to_string().contains("Metal"));
        assert!(Backend::Cuda.to_string().contains("CUDA"));
        assert_eq!(Backend::Cpu.to_string(), "CPU");
    }

    #[test]
    fn gpu_classification_is_right() {
        assert!(Backend::Metal.is_gpu());
        assert!(Backend::Cuda.is_gpu());
        assert!(!Backend::Cpu.is_gpu());
    }

    #[test]
    fn probe_always_answers() {
        // The point of the fallback chain: there is always a usable backend.
        let b = probe();
        assert!(matches!(b, Backend::Metal | Backend::Cuda | Backend::Cpu));
    }
}
