//! Settings panel and performance overlay (egui). The panel only edits `OceanParams`; every
//! decision about what a value means lives elsewhere.

use crate::gpu::timing::PassStats;
use crate::params::{self, OceanParams, ReflectionMode};

pub struct OverlayInfo {
    pub adapter: String,
    pub rt_reason: String,
    pub fps: f32,
    pub internal_extent: (u32, u32),
    pub stats: Vec<(&'static str, PassStats)>,
    pub total: PassStats,
    pub shader_errors: Vec<String>,
}

fn slider(ui: &mut egui::Ui, label: &str, v: &mut f32, range: (f32, f32), suffix: &str) {
    ui.add(egui::Slider::new(v, range.0..=range.1).text(label).suffix(suffix));
}

pub fn settings_panel(ctx: &egui::Context, p: &mut OceanParams, rt_available: bool) {
    egui::Window::new("Settings")
        .default_pos([12.0, 12.0])
        .resizable(false)
        .show(ctx, |ui| {
            ui.heading("Sea");
            slider(ui, "wind speed", &mut p.wind_speed, params::WIND_SPEED, " m/s");
            slider(ui, "wind direction", &mut p.wind_dir_deg, params::WIND_DIR_DEG, "°");
            slider(ui, "fetch", &mut p.fetch_km, params::FETCH_KM, " km");
            slider(ui, "swell", &mut p.swell_amount, params::SWELL_AMOUNT, "");
            slider(ui, "choppiness", &mut p.choppiness, params::CHOPPINESS, "");
            slider(ui, "foam", &mut p.foam_amount, params::FOAM_AMOUNT, "");
            slider(ui, "spray", &mut p.particle_amount, params::PARTICLE_AMOUNT, "");
            ui.separator();
            ui.heading("Sun");
            slider(
                ui,
                "elevation",
                &mut p.sun_elevation_deg,
                params::SUN_ELEVATION_DEG,
                "°",
            );
            slider(ui, "azimuth", &mut p.sun_azimuth_deg, params::SUN_AZIMUTH_DEG, "°");
            ui.separator();
            ui.heading("Rendering");
            ui.horizontal(|ui| {
                ui.label("reflections");
                ui.add_enabled_ui(rt_available, |ui| {
                    ui.radio_value(&mut p.reflection_mode, ReflectionMode::Rt, "ray traced")
                        .on_disabled_hover_text("This GPU/driver has no hardware ray query");
                });
                ui.radio_value(&mut p.reflection_mode, ReflectionMode::ScreenSpace, "screen space");
            });
            slider(ui, "render scale", &mut p.render_scale, params::RENDER_SCALE, "×");
            ui.checkbox(&mut p.vsync, "vsync");
            ui.checkbox(&mut p.show_overlay, "performance overlay (F2)");
            ui.separator();
            if ui.button("reset to defaults").clicked() {
                *p = OceanParams::default();
            }
            ui.small("F1 settings · F2 overlay · F5 reload shaders · Esc quit");
        });
}

pub fn overlay(ctx: &egui::Context, info: &OverlayInfo) {
    egui::Area::new(egui::Id::unique("perf overlay"))
        .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_min_width(250.0);
                ui.set_max_width(320.0);
                ui.label(egui::RichText::new(format!("{:.0} fps", info.fps)).strong());
                ui.small(&info.adapter);
                ui.small(&info.rt_reason);
                ui.small(format!(
                    "internal {}×{}",
                    info.internal_extent.0, info.internal_extent.1
                ));
                ui.separator();
                egui::Grid::new("passes").num_columns(3).show(ui, |ui| {
                    ui.small("pass");
                    ui.small("avg ms");
                    ui.small("p95 ms");
                    ui.end_row();
                    for (name, s) in &info.stats {
                        ui.small(*name);
                        ui.small(format!("{:.2}", s.avg_ms));
                        ui.small(format!("{:.2}", s.p95_ms));
                        ui.end_row();
                    }
                    ui.label(egui::RichText::new("GPU frame").strong());
                    ui.label(egui::RichText::new(format!("{:.2}", info.total.avg_ms)).strong());
                    ui.label(egui::RichText::new(format!("{:.2}", info.total.p95_ms)).strong());
                    ui.end_row();
                });
                for e in &info.shader_errors {
                    ui.separator();
                    ui.colored_label(egui::Color32::from_rgb(255, 120, 100), e);
                }
            });
        });
}

/// Shader errors on their own, for when the overlay is hidden.
pub fn shader_errors(ctx: &egui::Context, errors: &[String]) {
    egui::Window::new("Shader errors")
        .anchor(egui::Align2::RIGHT_TOP, [-10.0, 10.0])
        .show(ctx, |ui| {
            ui.set_max_width(420.0);
            for e in errors {
                ui.colored_label(egui::Color32::from_rgb(255, 120, 100), e);
            }
        });
}
