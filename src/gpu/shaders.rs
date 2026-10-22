//! WGSL sources: embedded in the exe, overridden by `shaders/*.wgsl` on disk (next to the exe
//! or in the working directory) so F5 can hot-reload them. `// #include name.wgsl` lines are
//! expanded (each file at most once). Reloads are validated with naga and applied all-or-
//! nothing, so a typo never replaces a working shader.

use std::collections::HashMap;
use std::path::PathBuf;

/// Top-level shaders (one per pipeline family).
pub const SHADERS: &[&str] = &[
    "spectrum_update",
    "fft",
    "assemble",
    "foam",
    "ocean_gbuffer",
    "sky_luts",
    "shade",
    "reflect_ssr",
    "reflect_resolve",
    "post",
    "rq_smoke",
    "wave_query",
    "object_gbuffer",
    "rt_patch",
    "reflect_rt",
    "particles_sim",
    "particles_draw",
];

const EMBEDDED: &[(&str, &str)] = &[
    ("spectrum_update", include_str!("../../shaders/spectrum_update.wgsl")),
    ("fft", include_str!("../../shaders/fft.wgsl")),
    ("assemble", include_str!("../../shaders/assemble.wgsl")),
    ("foam", include_str!("../../shaders/foam.wgsl")),
    ("ocean_gbuffer", include_str!("../../shaders/ocean_gbuffer.wgsl")),
    ("sky_luts", include_str!("../../shaders/sky_luts.wgsl")),
    ("shade", include_str!("../../shaders/shade.wgsl")),
    ("reflect_ssr", include_str!("../../shaders/reflect_ssr.wgsl")),
    ("reflect_resolve", include_str!("../../shaders/reflect_resolve.wgsl")),
    ("post", include_str!("../../shaders/post.wgsl")),
    ("rq_smoke", include_str!("../../shaders/rq_smoke.wgsl")),
    ("wave_query", include_str!("../../shaders/wave_query.wgsl")),
    ("object_gbuffer", include_str!("../../shaders/object_gbuffer.wgsl")),
    ("rt_patch", include_str!("../../shaders/rt_patch.wgsl")),
    ("surfaces", include_str!("../../shaders/surfaces.wgsl")),
    ("reflect_rt", include_str!("../../shaders/reflect_rt.wgsl")),
    ("particles_sim", include_str!("../../shaders/particles_sim.wgsl")),
    ("particles_draw", include_str!("../../shaders/particles_draw.wgsl")),
    ("materials", include_str!("../../shaders/materials.wgsl")),
    ("frame_common", include_str!("../../shaders/frame_common.wgsl")),
    ("seabed_common", include_str!("../../shaders/seabed_common.wgsl")),
    ("sky_common", include_str!("../../shaders/sky_common.wgsl")),
    ("sky_sample", include_str!("../../shaders/sky_sample.wgsl")),
];

pub struct ShaderLibrary {
    dir: Option<PathBuf>,
    files: HashMap<String, String>,
    /// Why the disk copies were rejected at load, if they were.
    load_errors: Vec<(String, String)>,
}

/// Checks a fully expanded WGSL source the way wgpu will (parse + validate).
pub fn validate(src: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(src).map_err(|e| e.emit_to_string(src))?;
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .map(|_| ())
        .map_err(|e| e.emit_to_string(src))
}

/// Expands includes, then hoists `enable` directives to the top (WGSL wants them before any
/// declaration, and includes land above the including file's own lines).
fn expanded(files: &HashMap<String, String>, name: &str) -> Result<String, String> {
    let mut body = String::new();
    expand(files, name, &mut Vec::new(), &mut body)?;
    let (enables, rest): (Vec<&str>, Vec<&str>) = body.lines().partition(|l| l.trim_start().starts_with("enable "));
    let mut out = String::new();
    for e in enables {
        if !out.contains(e.trim()) {
            out.push_str(e.trim());
            out.push('\n');
        }
    }
    for l in rest {
        out.push_str(l);
        out.push('\n');
    }
    Ok(out)
}

fn expand(files: &HashMap<String, String>, name: &str, seen: &mut Vec<String>, out: &mut String) -> Result<(), String> {
    if seen.iter().any(|s| s == name) {
        return Ok(());
    }
    seen.push(name.to_string());
    let src = files.get(name).ok_or_else(|| format!("no shader named {name:?}"))?;
    for line in src.lines() {
        if let Some(inc) = line.trim().strip_prefix("// #include ") {
            expand(files, inc.trim().trim_end_matches(".wgsl"), seen, out)?;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    Ok(())
}

impl ShaderLibrary {
    /// Only the copies compiled into the exe.
    pub fn embedded() -> ShaderLibrary {
        ShaderLibrary {
            dir: None,
            files: EMBEDDED.iter().map(|(n, s)| (n.to_string(), s.to_string())).collect(),
            load_errors: Vec::new(),
        }
    }

    /// Embedded copies overridden by whatever `dir` holds — unless any of those fail
    /// validation (a half-edited or stale shader), in which case the embedded set is used
    /// and the errors are kept for the overlay. F5 can still pick up fixed files later.
    pub fn with_dir(dir: PathBuf) -> ShaderLibrary {
        let mut lib = ShaderLibrary::embedded();
        lib.dir = Some(dir);
        lib.load_errors = lib.reload_all();
        lib
    }

    /// Errors that made the disk shaders fall back to the embedded copies at load.
    pub fn load_errors(&self) -> &[(String, String)] {
        &self.load_errors
    }

    /// `shaders/` next to the exe, else in the working directory, else embedded only.
    pub fn discover() -> ShaderLibrary {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("shaders")));
        let cwd = std::env::current_dir().ok().map(|d| d.join("shaders"));
        match exe_dir.into_iter().chain(cwd).find(|d| d.is_dir()) {
            Some(d) => ShaderLibrary::with_dir(d),
            None => ShaderLibrary::embedded(),
        }
    }

    pub fn dir(&self) -> Option<&PathBuf> {
        self.dir.as_ref()
    }

    fn read_disk_from(&self, dir: &std::path::Path) -> HashMap<String, String> {
        EMBEDDED
            .iter()
            .filter_map(|(n, _)| {
                std::fs::read_to_string(dir.join(format!("{n}.wgsl")))
                    .ok()
                    .map(|s| (n.to_string(), s))
            })
            .collect()
    }

    /// Fully expanded source of a top-level shader.
    pub fn source(&self, name: &str) -> Result<String, String> {
        expanded(&self.files, name)
    }

    /// Re-reads every shader from disk. Returns (shader, error) for each one that failed; on
    /// any failure nothing changes.
    pub fn reload_all(&mut self) -> Vec<(String, String)> {
        let Some(dir) = self.dir.clone() else { return Vec::new() };
        let mut candidate = self.files.clone();
        candidate.extend(self.read_disk_from(&dir));
        let errors: Vec<(String, String)> = SHADERS
            .iter()
            .filter_map(|name| {
                expanded(&candidate, name)
                    .and_then(|out| validate(&out))
                    .err()
                    .map(|e| (name.to_string(), e))
            })
            .collect();
        if errors.is_empty() {
            self.files = candidate;
        }
        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ocean-shader-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn every_shipped_shader_validates() {
        let lib = ShaderLibrary::embedded();
        for name in SHADERS {
            let src = lib.source(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            validate(&src).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }

    #[test]
    fn includes_are_expanded_once_each() {
        let lib = ShaderLibrary::embedded();
        let src = lib.source("shade").unwrap();
        assert_eq!(src.matches("struct Frame {").count(), 1);
        assert!(src.contains("fn sky_radiance"));
        assert!(!src.contains("#include"));
    }

    #[test]
    fn disk_copy_overrides_embedded() {
        let d = temp_dir("override");
        let body = "@compute @workgroup_size(1) fn main() {}\n";
        std::fs::write(d.join("fft.wgsl"), body).unwrap();
        let lib = ShaderLibrary::with_dir(d.clone());
        assert_eq!(lib.source("fft").unwrap(), body);
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn reload_rejects_invalid_wgsl() {
        let d = temp_dir("reload");
        std::fs::write(d.join("fft.wgsl"), "@compute @workgroup_size(1) fn main() {}\n").unwrap();
        let mut lib = ShaderLibrary::with_dir(d.clone());
        let before = lib.source("fft").unwrap();
        std::fs::write(d.join("fft.wgsl"), "fn main( { this is not wgsl").unwrap();
        let errors = lib.reload_all();
        assert!(errors.iter().any(|(n, _)| n == "fft"), "{errors:?}");
        assert_eq!(
            lib.source("fft").unwrap(),
            before,
            "broken shader replaced the working one"
        );
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn enable_directives_are_hoisted_above_includes() {
        let src = ShaderLibrary::embedded().source("reflect_rt").unwrap();
        assert!(src.trim_start().starts_with("enable wgpu_ray_query;"), "{}", &src[..80]);
        assert_eq!(src.matches("enable wgpu_ray_query;").count(), 1);
    }

    #[test]
    fn broken_disk_shader_at_load_falls_back_to_embedded() {
        let d = temp_dir("broken-at-load");
        std::fs::write(d.join("fft.wgsl"), "fn main( { not wgsl").unwrap();
        let lib = ShaderLibrary::with_dir(d.clone());
        assert_eq!(
            lib.source("fft").unwrap(),
            ShaderLibrary::embedded().source("fft").unwrap()
        );
        assert!(
            lib.load_errors().iter().any(|(n, _)| n == "fft"),
            "{:?}",
            lib.load_errors()
        );
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn missing_disk_file_falls_back_to_embedded() {
        let d = temp_dir("missing");
        let lib = ShaderLibrary::with_dir(d.clone());
        assert!(lib.source("post").unwrap().contains("fn agx"));
        let _ = std::fs::remove_dir_all(d);
    }
}
