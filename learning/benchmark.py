"""
Benchmark script comparing old vs new neural architectures.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import old components
from learning.dqn import DQNAgent as OldDQNAgent
from simulated_game.agent.neural_policy import NeuralGridPolicy as OldNeuralPolicy

# Import new components
from learning.networks import ModernQNetwork, create_modern_q_network, ModernQNetworkConfig
from learning.world_model import WorldModel, create_world_model, WorldModelConfig
from learning.trainer import Trainer, create_trainer, TrainerConfig
from learning.policy import ModernNeuralPolicy, ModernPolicyConfig, NeuralGridPolicy as NewNeuralPolicy


def benchmark_inference_speed(
    model: torch.nn.Module,
    input_shape: tuple,
    num_iterations: int = 100,
    warmup: int = 20,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Benchmark model inference speed."""
    model.eval()
    model.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            x = torch.randn(input_shape, device=device)
            _ = model(x)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            x = torch.randn(input_shape, device=device)
            _ = model(x)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.perf_counter()
    
    total_time = end - start
    per_iter = total_time / num_iterations * 1000  # ms
    fps = num_iterations / total_time
    
    return {
        "total_time_ms": total_time * 1000,
        "per_iter_ms": per_iter,
        "fps": fps,
    }


def benchmark_training_step(
    trainer_fn,
    config: Dict[str, Any],
    num_steps: int = 20,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Benchmark training step speed."""
    trainer = trainer_fn(**config, device=device)
    
    # Fill replay buffer
    batch_size = config.get('batch_size', 64)
    for _ in range(batch_size * 10):
        state = np.random.randn(config['input_dim']).astype(np.float32)
        next_state = np.random.randn(config['input_dim']).astype(np.float32)
        action = np.random.randint(config['action_dim'])
        reward = np.random.randn()
        done = np.random.rand() < 0.1
        mask = np.ones(config['action_dim'], dtype=bool)
        next_mask = np.ones(config['action_dim'], dtype=bool)
        trainer.remember(state, action, reward, next_state, next_mask, done)
    
    # Warmup
    for _ in range(10):
        trainer.train_step()
    
    # Benchmark
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    losses = []
    for _ in range(num_steps):
        loss = trainer.train_step()
        if loss:
            losses.append(loss.get('total_loss', 0))
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.perf_counter()
    
    total_time = end - start
    per_step = total_time / num_steps * 1000
    
    return {
        "total_time_ms": total_time * 1000,
        "per_step_ms": per_step,
        "avg_loss": np.mean(losses) if losses else 0,
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def compare_q_networks(device: torch.device = torch.device("cpu")):
    """Compare old vs new Q-network architectures."""
    print("=" * 60)
    print("Q-NETWORK ARCHITECTURE COMPARISON")
    print("=" * 60)
    
    input_dim = 40
    action_dim = 6
    batch_size = 256
    
    # Old network
    print("\n--- Old Q-Network (3-layer MLP) ---")
    old_agent = OldDQNAgent(
        input_size=input_dim,
        action_count=action_dim,
        hidden_size=512,
        num_layers=3,
        device=device,
    )
    old_params = count_parameters(old_agent.online)
    print(f"Parameters: {old_params['total']:,} (trainable: {old_params['trainable']:,})")
    
    old_bench = benchmark_inference_speed(
        old_agent.online, (batch_size, input_dim), device=device
    )
    print(f"Inference: {old_bench['per_iter_ms']:.2f} ms/iter ({old_bench['fps']:.0f} FPS)")
    
    # New network (standard)
    print("\n--- New Q-Network (ResNet + MoE + C51) ---")
    new_config = ModernQNetworkConfig(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=512,
        num_residual_blocks=6,
        use_moe=True,
        use_distributional=True,
        num_atoms=51,
        use_spectral_norm=True,
        dropout=0.1,
    )
    new_net = ModernQNetwork(new_config).to(device)
    new_params = count_parameters(new_net)
    print(f"Parameters: {new_params['total']:,} (trainable: {new_params['trainable']:,})")
    
    new_bench = benchmark_inference_speed(new_net, (batch_size, input_dim), device=device)
    print(f"Inference: {new_bench['per_iter_ms']:.2f} ms/iter ({new_bench['fps']:.0f} FPS)")
    
    # New network (lightweight)
    print("\n--- New Q-Network (Lightweight, 4 blocks, no MoE) ---")
    light_config = ModernQNetworkConfig(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=512,
        num_residual_blocks=4,
        use_moe=False,
        use_distributional=False,
        use_spectral_norm=True,
        dropout=0.1,
    )
    light_net = ModernQNetwork(light_config).to(device)
    light_params = count_parameters(light_net)
    print(f"Parameters: {light_params['total']:,} (trainable: {light_params['trainable']:,})")
    
    light_bench = benchmark_inference_speed(light_net, (batch_size, input_dim), device=device)
    print(f"Inference: {light_bench['per_iter_ms']:.2f} ms/iter ({light_bench['fps']:.0f} FPS)")
    
    return {
        "old": {"params": old_params, "bench": old_bench},
        "new": {"params": new_params, "bench": new_bench},
        "light": {"params": light_params, "bench": light_bench},
    }


def compare_trainers(device: torch.device = torch.device("cpu")):
    """Compare old vs new training loops."""
    print("\n" + "=" * 60)
    print("TRAINING LOOP COMPARISON")
    print("=" * 60)
    
    config = {
        'input_dim': 40,
        'action_dim': 6,
        'batch_size': 256,
        'gamma': 0.99,
        'learning_rate': 3e-4,
    }
    
    # Old DQN training
    print("\n--- Old DQN Training ---")
    old_agent = OldDQNAgent(
        input_size=40,
        action_count=6,
        batch_size=256,
        replay_capacity=20000,
        warmup_steps=128,
        device=device,
    )
    
    # Fill replay
    for _ in range(5000):
        state = np.random.randn(40).astype(np.float32)
        next_state = np.random.randn(40).astype(np.float32)
        action = np.random.randint(6)
        reward = np.random.randn()
        done = np.random.rand() < 0.1
        mask = np.ones(6, dtype=bool)
        old_agent.remember(state, action, reward, next_state, mask, done)
    
    old_start = time.perf_counter()
    for _ in range(100):
        old_agent.train_step()
    old_time = time.perf_counter() - old_start
    print(f"100 steps: {old_time*1000:.1f} ms ({old_time/100*1000:.2f} ms/step)")
    
    # New trainer
    print("\n--- New Trainer (CQL + Mixed Precision + PER) ---")
    trainer_config = TrainerConfig(
        input_dim=40,
        action_dim=6,
        batch_size=256,
        gradient_accumulation_steps=2,
        gamma=0.99,
        learning_rate=3e-4,
        use_mixed_precision=(device.type == "cuda"),
        use_per=True,
        cql_alpha=1.0,
        compile_model=False,
    )
    new_trainer = Trainer(trainer_config, device=device)
    
    # Fill replay
    for _ in range(5000):
        state = np.random.randn(40).astype(np.float32)
        next_state = np.random.randn(40).astype(np.float32)
        action = np.random.randint(6)
        reward = np.random.randn()
        done = np.random.rand() < 0.1
        mask = np.ones(6, dtype=bool)
        new_trainer.remember(state, action, reward, next_state, mask, done)
    
    new_start = time.perf_counter()
    for _ in range(100):
        new_trainer.train_step()
    new_time = time.perf_counter() - new_start
    print(f"100 steps: {new_time*1000:.1f} ms ({new_time/100*1000:.2f} ms/step)")
    
    return {
        "old_time_ms": old_time * 1000,
        "new_time_ms": new_time * 1000,
    }


def test_world_model(device: torch.device = torch.device("cpu")):
    """Test World Model functionality."""
    print("\n" + "=" * 60)
    print("WORLD MODEL TEST")
    print("=" * 60)
    
    wm = create_world_model(
        obs_dim=40,
        action_dim=6,
        deter_dim=512,
        num_categoricals=32,
        num_classes=32,
        codebook_dim=256,
        hidden_dim=512,
        ensemble_size=5,
        imagination_horizon=15,
    ).to(device)
    
    params = count_parameters(wm)
    print(f"World Model Parameters: {params['total']:,} (trainable: {params['trainable']:,})")
    
    # Test observe
    batch_size = 32
    seq_len = 10
    obs = torch.randn(seq_len, batch_size, 40, device=device)
    actions = F.one_hot(torch.randint(0, 6, (seq_len, batch_size), device=device), 6).float()
    rewards = torch.randn(seq_len, batch_size, device=device)
    continues = torch.ones(seq_len, batch_size, device=device)
    
    # Warmup
    with torch.no_grad():
        _ = wm.observe(obs, actions, rewards, continues)
    
    # Benchmark observe
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    for _ in range(50):
        _ = wm.observe(obs, actions, rewards, continues)
    torch.cuda.synchronize() if device.type == "cuda" else None
    observe_time = time.perf_counter() - start
    print(f"Observe (50x seq_len={seq_len}, batch={batch_size}): {observe_time*1000:.1f} ms")
    
    # Test imagine
    policy = nn.Linear(512 + 32*32, 6).to(device)
    init_state = wm.initial_state(batch_size, device)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    for _ in range(50):
        _ = wm.imagine(policy, init_state, horizon=15)
    torch.cuda.synchronize() if device.type == "cuda" else None
    imagine_time = time.perf_counter() - start
    print(f"Imagine (50x horizon=15, batch={batch_size}): {imagine_time*1000:.1f} ms")
    
    return {"params": params, "observe_ms": observe_time*1000, "imagine_ms": imagine_time*1000}


def test_policy_compatibility():
    """Test that new policy is compatible with old interface."""
    print("\n" + "=" * 60)
    print("POLICY COMPATIBILITY TEST")
    print("=" * 60)
    
    # Create new policy with old interface
    policy = NewNeuralPolicy(
        seed=42,
        training=True,
        hidden_dim=256,
        num_residual_blocks=4,
        use_moe=False,
        use_distributional=False,
        batch_size=128,
        compile_model=False,
    )
    
    print(f"Policy created successfully")
    print(f"  ACTION_COUNT: {policy.ACTION_COUNT}")
    print(f"  FEATURE_COUNT: {policy.FEATURE_COUNT}")
    print(f"  Training mode: {policy.training}")
    
    # Test basic methods exist
    assert hasattr(policy, 'begin_episode')
    assert hasattr(policy, 'decide')
    assert hasattr(policy, 'observe_result')
    assert hasattr(policy, 'finish_episode')
    assert hasattr(policy, 'save')
    assert hasattr(policy, 'load')
    assert hasattr(policy, 'behavior_clone')
    
    print("All required methods present")
    
    # Test checkpoint save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = Path(f.name)
    
    try:
        policy.save(path, metadata={"test": "value"})
        print(f"Saved checkpoint to {path}")
        
        # Load into new policy
        policy2 = NewNeuralPolicy(seed=43, training=False)
        metadata = policy2.load(path)
        print(f"Loaded checkpoint, metadata: {metadata}")
        
        print("Checkpoint save/load: OK")
    finally:
        path.unlink(missing_ok=True)
    
    return {"compatible": True}


def run_full_comparison(device_str: str = "auto"):
    """Run all benchmarks and comparisons."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    print(f"\nUsing device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    
    results = {}
    
    # Compare Q-networks
    results['q_networks'] = compare_q_networks(device)
    
    # Compare trainers
    results['trainers'] = compare_trainers(device)
    
    # Test world model
    results['world_model'] = test_world_model(device)
    
    # Test policy compatibility
    results['policy_compat'] = test_policy_compatibility()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    old_params = results['q_networks']['old']['params']['total']
    new_params = results['q_networks']['new']['params']['total']
    light_params = results['q_networks']['light']['params']['total']
    
    old_fps = results['q_networks']['old']['bench']['fps']
    new_fps = results['q_networks']['new']['bench']['fps']
    light_fps = results['q_networks']['light']['bench']['fps']
    
    print(f"\nParameter Count:")
    print(f"  Old:     {old_params:>10,}")
    print(f"  New:     {new_params:>10,} ({new_params/old_params:.2f}x)")
    print(f"  Light:   {light_params:>10,} ({light_params/old_params:.2f}x)")
    
    print(f"\nInference Speed (FPS):")
    print(f"  Old:     {old_fps:>10.0f}")
    print(f"  New:     {new_fps:>10.0f} ({new_fps/old_fps:.2f}x)")
    print(f"  Light:   {light_fps:>10.0f} ({light_fps/old_fps:.2f}x)")
    
    print(f"\nTraining Step:")
    print(f"  Old:     {results['trainers']['old_time_ms']:>10.1f} ms")
    print(f"  New:     {results['trainers']['new_time_ms']:>10.1f} ms")
    
    print(f"\nWorld Model:")
    print(f"  Params:  {results['world_model']['params']['total']:>10,}")
    print(f"  Observe: {results['world_model']['observe_ms']:>10.1f} ms (50 batches)")
    print(f"  Imagine: {results['world_model']['imagine_ms']:>10.1f} ms (50 batches)")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark old vs new neural architectures")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--output", type=Path, help="Output JSON file for results")
    args = parser.parse_args()
    
    results = run_full_comparison(args.device)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")