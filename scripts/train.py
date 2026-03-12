from neuro_pilot import NeuroPilot


def run_real_training():
    model = NeuroPilot("neuro_pilot/cfg/models/neuralPilot.yaml", scale="n")

    model.train(
        data="/home/quynhthu/Downloads/data_final/data.yaml",

        # Override Trainer parameters
        epochs=100,
        batch=16,
        learning_rate=1e-3,  # Lowered from 5e-3 to stabilize convergence
        patience=100,

        # Loss Scaling - Balanced to prevent domination
        lambda_traj=2.0,     # Increased to give trajectory more "voice"
        lambda_det=1.0,
        lambda_heatmap=1.5,  # Lowered from 3.0 to prevent it from hogging gradients
        lambda_gate=0.5,
        lambda_smooth=0.1,

        # Detection Sub-Losses
        box=1.0,
        cls_det=1.0,
        dfl=1.0,

        # Advanced FDAT Parameters
        use_fdat=True,
        use_uncertainty=True, # CRITICAL: Enable uncertainty-aware weighting to balance 4 tasks
        fdat_alpha_lane=10.0,
        fdat_beta_lane=1.0,

        # Trainer settings
        use_amp=False,       # Keep disabled for high-precision trajectory gradients
        warmup_bias_lr=1e-4, # Start with a small bias LR to prevent initial weights "shock"
        warmup_epochs=2.0,   # Longer warmup for multi-task stability

        rotate_deg = 5.0,
        translate = 0.1,
        scale = 0.1,
        color_jitter = 0.1,
        shear = 0.0,
        hsv_h = 0.015,
        hsv_s = 0.4,
        hsv_v = 0.4,
        perspective = 0.05,
        mosaic = 0.0,
        noise_prob = 0.0,
        blur_prob = 0.05,
        experiment_name = "final_train_v2",
        early_stop_patience = 100
    )

if __name__ == "__main__":
    run_real_training()
