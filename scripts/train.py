from neuro_pilot import NeuroPilot


def run_real_training():
    model = NeuroPilot("neuro_pilot/cfg/models/neuralPilot_yolo11.yaml", scale="s")

    # Training configuration overrides
    overrides = {
        "data": {
            "dataset_yaml": "v1/data.yaml",
            "batch_size": 32,
            "image_size": 320,
            "augment": {
                "rotate_deg": 10.0,
                "translate": 0.2,
                "scale": 0.2,
                "color_jitter": 0.01,
                "shear": 0.0,
                "perspective": 0.1,
                "mosaic": 0.0,
                "noise_prob": 0.0,
                "blur_prob": 0.05,
            },
        },
        "loss": {
            "lambda_det": 2.5,
            "lambda_traj": 7.5,
            "lambda_heatmap": 10.0,
            "lambda_cls": 2.0,
            "lambda_smooth": 0.1,
            "lambda_gate": 0.5,
            # FDAT Loss (Frenet-Decomposed Anisotropic Trajectory Loss)
            "use_fdat": True,
            "fdat_alpha_lane": 10.0,
            "fdat_beta_lane": 1.0,
            "fdat_alpha_inter": 5.0,
            "fdat_beta_inter": 3.0,
            "fdat_lambda_heading": 2.0,
            "fdat_lambda_endpoint": 5.0,
        },
        "trainer": {
            "experiment_name": "train_v1",
            "image_size": 320,
            "max_epochs": 100,
            "learning_rate": 1e-3,
            "optimizer": "AdamW",
            "lr_final": 0.01,
            "warmup_epochs": 3.0,
            "use_ema": True,
            "use_amp": True,
        },
    }

    model.train(**overrides)

if __name__ == "__main__":
    run_real_training()
