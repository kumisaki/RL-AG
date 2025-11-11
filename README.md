## RL-AG

### Quick Start

```bash
cd RL-AG
PYTHONPATH=src python -m training.train_agent
PYTHONPATH=src python -m cli.evaluate --checkpoint checkpoints/ppo_step_0.pt
```

### TensorBoard logging

Training now emits scalars (average reward, done rate, PPO losses, entropy) through PyTorch's `SummaryWriter`. Install TensorBoard if you don't already have it:

```bash
pip install tensorboard
```

Launch TensorBoard pointed at the training log directory (defaults to `logs/` from `TrainingConfig`):

```bash
ssh -L 6006:localhost:6006 user@host
```

```bash
cd RL-AG
tensorboard --logdir logs --port 6006 --host localhost
```

Start a browser at the printed URL to watch metrics update live while `training.train_agent` runs.
