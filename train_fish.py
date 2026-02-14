"""
Training Script for Fish RL Agent

This script trains fish to avoid the bouncing ball using PPO
(Proximal Policy Optimization) from Stable-Baselines3.

Usage:
    python train_fish.py
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from fish_environment import SingleFishEnv


def make_env():
    """
    Create and wrap the environment.

    Returns:
        Wrapped environment ready for training
    """
    env = SingleFishEnv()
    env = Monitor(env)  # Monitor rewards and episode lengths
    return env


def train(
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    save_freq=50_000,
    model_save_path="models/fish_ppo",
    log_dir="logs/"
):
    """
    Train the fish agent using PPO.

    Args:
        total_timesteps: Total number of timesteps to train for
        learning_rate: Learning rate for the optimizer
        n_steps: Number of steps to collect before each update
        batch_size: Minibatch size for each gradient update
        n_epochs: Number of epochs to optimize over collected data
        gamma: Discount factor (how much to value future rewards)
        save_freq: Save model checkpoint every N timesteps
        model_save_path: Path to save the final model
        log_dir: Directory for TensorBoard logs
    """
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 70)
    print("FISH RL TRAINING - PPO")
    print("=" * 70)
    print(f"\nTraining Configuration:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Steps per update: {n_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs per update: {n_epochs}")
    print(f"  Discount factor (gamma): {gamma}")
    print(f"  Save frequency: {save_freq:,}")
    print()

    # Create vectorized environment (allows parallel collection)
    print("Creating environment...")
    env = DummyVecEnv([make_env])

    # Normalize observations and rewards (helps training stability)
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)
    # Note: We're not using normalization for now since our obs are already normalized

    print("Environment created successfully!")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print()

    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",  # Multi-Layer Perceptron (neural network)
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=1,  # Print training progress
        tensorboard_log=log_dir,
        device="auto"  # Use GPU if available, otherwise CPU
    )

    print("Model created successfully!")
    print(f"  Policy architecture: MlpPolicy (default: [64, 64] hidden layers)")
    print(f"  Device: {model.device}")
    print()

    # Create callbacks
    # Checkpoint callback: saves model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="models/checkpoints/",
        name_prefix="fish_ppo"
    )

    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print("\nMonitor training progress:")
    print(f"  1. Watch console output below")
    print(f"  2. Run TensorBoard: tensorboard --logdir={log_dir}")
    print(f"  3. Open browser to: http://localhost:6006")
    print()
    print("Key metrics to watch:")
    print("  - ep_rew_mean: Average episode reward (should increase)")
    print("  - ep_len_mean: Average episode length (should increase)")
    print("  - loss: Policy loss (may fluctuate, but should be stable)")
    print()
    print("Training begins now...")
    print("=" * 70)
    print()

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print()

    # Save final model
    print(f"Saving final model to: {model_save_path}")
    model.save(model_save_path)
    print("Model saved successfully!")
    print()

    # Save environment normalization stats if used
    # env.save(f"{model_save_path}_vecnormalize.pkl")

    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("1. View training graphs:")
    print(f"   tensorboard --logdir={log_dir}")
    print()
    print("2. Test the trained agent:")
    print(f"   python visualize_fish.py --model {model_save_path}")
    print()
    print("3. Train longer for better results:")
    print("   python train_fish.py  # Will train for 1M timesteps by default")
    print()


if __name__ == "__main__":
    # Quick test training (10K timesteps - just to verify everything works)
    # Change to 1_000_000 for full training

    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "FISH RL TRAINING SCRIPT" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    # Ask user for training duration
    print("Choose training duration:")
    print("  1. Quick test (10,000 timesteps - 2 minutes)")
    print("  2. Short training (100,000 timesteps - 20 minutes)")
    print("  3. Medium training (500,000 timesteps - 1.5 hours)")
    print("  4. Full training (1,000,000 timesteps - 3 hours)")
    print()

    try:
        choice = input("Enter choice (1-4) or press Enter for full training: ").strip()

        if choice == "1":
            timesteps = 10_000
            print("\n→ Quick test mode: 10K timesteps")
        elif choice == "2":
            timesteps = 100_000
            print("\n→ Short training: 100K timesteps")
        elif choice == "3":
            timesteps = 500_000
            print("\n→ Medium training: 500K timesteps")
        else:
            timesteps = 1_000_000
            print("\n→ Full training: 1M timesteps")

        print()
        input("Press Enter to start training...")
        print()

    except EOFError:
        # Non-interactive mode (default to quick test)
        timesteps = 10_000
        print("\n→ Non-interactive mode: Quick test (10K timesteps)")
        print()

    # Start training
    train(total_timesteps=timesteps)
