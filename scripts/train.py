from neuro_pilot import NeuroPilot


def run_real_training():
    model = NeuroPilot("neuro_pilot/cfg/models/neuralPilot.yaml", scale="n")

    model.train(
        data="/home/quynhthu/Downloads/data_final/data.yaml",

        # Override Trainer parameters
        epochs=10,
        batch=16,
        learning_rate=1e-3,  # Increased from 5e-4 for faster breakout
        patience=100,        # Alias for early_stop_patience

        # Loss Scaling
        lambda_traj=1,
        lambda_det=2,
        lambda_heatmap=1.5,
        lambda_gate=1.0,
        lambda_smooth=0.01,

        # Detection Sub-Losses
        box=2.5,
        cls_det=10.0,
        dfl=2.0,

        # Advanced FDAT Parameters
        use_fdat=True,
        use_uncertainty=False,
        fdat_alpha_lane=15.0,
        fdat_beta_lane=2.0,

        rotate_deg = 5.0,
        translate = 0.2,
        scale = 0.2,
        color_jitter = 0.0,
        shear = 0.0,
        hsv_h = 0.5,
        hsv_s = 0.3,
        hsv_v = 0.2,
        perspective = 0.1,
        mosaic = 0.0,
        noise_prob = 0.0,
        blur_prob = 0.05,
        experiment_name = "final_train",
        early_stop_patience = 100
    )

if __name__ == "__main__":
    run_real_training()
