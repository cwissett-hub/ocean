/// Size of the internal render targets for a window, or `None` when the window has no area
/// (minimised). Never returns a zero dimension.
pub fn render_extent(window_w: u32, window_h: u32, scale: f32) -> Option<(u32, u32)> {
    if window_w == 0 || window_h == 0 {
        return None;
    }
    let s = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
    let dim = |v: u32| ((v as f32 * s).round() as u32).max(1);
    Some((dim(window_w), dim(window_h)))
}

/// Half-resolution extent, rounded up so every full-res pixel has a half-res parent.
pub fn half_extent(w: u32, h: u32) -> (u32, u32) {
    (w.div_ceil(2).max(1), h.div_ceil(2).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_size_window_has_no_extent() {
        assert_eq!(render_extent(0, 720, 1.0), None);
        assert_eq!(render_extent(1280, 0, 1.0), None);
    }

    #[test]
    fn scale_applies_and_never_hits_zero() {
        assert_eq!(render_extent(1280, 720, 1.0), Some((1280, 720)));
        assert_eq!(render_extent(1280, 720, 0.5), Some((640, 360)));
        assert_eq!(render_extent(1, 1, 0.5), Some((1, 1)));
    }

    #[test]
    fn half_extent_rounds_up() {
        assert_eq!(half_extent(1281, 721), (641, 361));
        assert_eq!(half_extent(1, 1), (1, 1));
    }
}
