use crate::engine::{Engine, GfElement, ShardsRefMut, GF_ORDER};

// The engine is chosen at compile time from the enabled target features.
// `Wide` needs a 16-byte table lookup instruction to be worthwhile: SSSE3 on
// x86(-64), `simd128` on WebAssembly, NEON on AArch64. Without one, `NoSimd`
// is faster than `Wide`'s scalar fallback.
#[cfg(any(
    target_feature = "ssse3",
    target_feature = "simd128",
    all(target_arch = "aarch64", target_feature = "neon"),
))]
type Inner = crate::engine::Wide;

#[cfg(not(any(
    target_feature = "ssse3",
    target_feature = "simd128",
    all(target_arch = "aarch64", target_feature = "neon"),
)))]
type Inner = crate::engine::NoSimd;

// ======================================================================
// DefaultEngine - PUBLIC

/// [`Engine`] chosen at compile time: [`Wide`] when the target has SSSE3,
/// WebAssembly `simd128` or NEON enabled, otherwise [`NoSimd`].
///
/// There is no runtime CPU detection. Enable the target feature when
/// building (for example `-C target-feature=+ssse3` or `+simd128`) to get
/// the SIMD engine.
///
/// [`Wide`]: crate::engine::Wide
/// [`NoSimd`]: crate::engine::NoSimd
pub struct DefaultEngine(Inner);

impl DefaultEngine {
    /// Creates new [`DefaultEngine`], initializing the underlying engine.
    pub fn new() -> Self {
        Self(Inner::new())
    }

    /// Whether the SIMD engine was selected at compile time.
    pub const SIMD: bool = cfg!(any(
        target_feature = "ssse3",
        target_feature = "simd128",
        all(target_arch = "aarch64", target_feature = "neon"),
    ));
}

// ======================================================================
// DefaultEngine - IMPL Default

impl Default for DefaultEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================
// DefaultEngine - IMPL Engine

impl Engine for DefaultEngine {
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        self.0.fft(data, pos, size, truncated_size, skew_delta);
    }

    fn ifft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        self.0.ifft(data, pos, size, truncated_size, skew_delta);
    }

    fn mul(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        self.0.mul(x, log_m);
    }

    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        Inner::eval_poly(erasures, truncated_size);
    }
}
