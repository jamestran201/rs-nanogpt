use candle_core::{Device, Result};

/// Select the compute device based on the enabled backend feature.
///
/// Defaults to CPU. Build with `--features metal` (Apple GPU) or
/// `--features cuda` (NVIDIA, the cloud path) to target a GPU.
pub fn default_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0)
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        Device::new_metal(0)
    }
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        Ok(Device::Cpu)
    }
}
