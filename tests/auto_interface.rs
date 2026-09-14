use arctn::auto::{
    auto_path_preset, auto_path_preset_to_size, auto_path_preset_to_size_with_mode,
    auto_path_preset_to_size_with_mode_and_objective, auto_path_preset_to_size_with_objective,
    auto_path_preset_with_objective, optimize, AutoPreset, SlicingMode,
};
use arctn::{PlannerObjective, TensorNetwork};

fn valid_network() -> TensorNetwork {
    TensorNetwork {
        name: "interface-validation".into(),
        inputs: vec![vec![19], vec![19]],
        output: vec![],
        size_dict: [(19, 2)].into_iter().collect(),
    }
}

#[test]
fn preset_and_mode_labels_round_trip() {
    assert_eq!(AutoPreset::default(), AutoPreset::Heavy);
    assert_eq!(SlicingMode::default(), SlicingMode::Fixed);
    for preset in [AutoPreset::Light, AutoPreset::Heavy] {
        assert_eq!(AutoPreset::from_label(preset.as_str()), Some(preset));
    }
    for mode in [SlicingMode::Fixed, SlicingMode::Dynamic] {
        assert_eq!(SlicingMode::from_label(mode.as_str()), Some(mode));
    }
    assert_eq!(AutoPreset::from_label("unknown"), None);
    assert_eq!(SlicingMode::from_label("unknown"), None);
}

#[test]
fn every_preset_entry_point_validates_network_before_loading_engine() {
    let mut net = valid_network();
    net.size_dict.insert(19, 0);
    let expected = net.validate().unwrap_err();
    let preset = AutoPreset::Light;
    let objective = PlannerObjective::FIXED;
    let results = [
        auto_path_preset(&net, preset, 0, None),
        auto_path_preset_with_objective(&net, preset, 0, None, objective),
        auto_path_preset_to_size(&net, preset, 0, None, 1),
        auto_path_preset_to_size_with_mode(&net, preset, 0, None, 1, SlicingMode::Dynamic),
        auto_path_preset_to_size_with_objective(&net, preset, 0, None, 1, objective, true),
        auto_path_preset_to_size_with_mode_and_objective(
            &net,
            preset,
            0,
            None,
            1,
            SlicingMode::Fixed,
            objective,
            false,
        ),
    ];
    for result in results {
        assert_eq!(result.unwrap_err(), expected);
    }
}

#[test]
fn invalid_deadlines_are_rejected_before_loading_engine() {
    let net = valid_network();
    for seconds in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = auto_path_preset(&net, AutoPreset::Heavy, 0, Some(seconds)).unwrap_err();
        assert!(error.contains("max_time"), "{error}");
    }
}

#[test]
fn zero_target_is_rejected_before_loading_engine() {
    let net = valid_network();
    let error = optimize(
        &net,
        AutoPreset::Light,
        0,
        None,
        PlannerObjective::FIXED,
        false,
        Some(0),
        SlicingMode::Dynamic,
    )
    .unwrap_err();
    assert!(error.contains("target_size"), "{error}");
}

#[test]
fn dynamic_slicing_requires_a_target_before_loading_engine() {
    let error = optimize(
        &valid_network(),
        AutoPreset::Light,
        0,
        None,
        PlannerObjective::FIXED,
        true,
        None,
        SlicingMode::Dynamic,
    )
    .unwrap_err();
    assert_eq!(error, "dynamic slicing requires target_size");
}
