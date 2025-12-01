# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DoRobot is a robotics operating platform for robot control, data collection, and policy training. It combines:
- **DORA-RS**: Dataflow framework for real-time robot communication
- **LeRobot**: Policy training and dataset management (adapted from HuggingFace)
- **Ascend NPU Support**: Hardware acceleration for video encoding on Orange Pi/Ascend 310B

## Common Commands

### Environment Setup
```bash
# Automated setup (recommended)
bash scripts/setup_env.sh                    # Core only (data collection)
bash scripts/setup_env.sh --training         # With training deps
bash scripts/setup_env.sh --npu              # For Ascend NPU
bash scripts/setup_env.sh --cuda 12.4        # For CUDA GPU

# Manual install
conda create -n dorobot python=3.11 && conda activate dorobot
pip install -e .                             # Core only
pip install -e ".[training]"                 # With training
```

### Running Data Collection (SO101 Robot)
```bash
# Single command - starts DORA + CLI automatically
bash scripts/run_so101.sh

# With options
REPO_ID=my-dataset USE_NPU=1 bash scripts/run_so101.sh
```

### Training
```bash
python operating_platform/core/train.py \
  --dataset.repo_id="/path/to/dataset" \
  --policy.type=act \
  --policy.device=cuda  # or npu for Ascend
```

### Inference
```bash
python operating_platform/core/inference.py \
  --robot.type=so101 \
  --policy.path="/path/to/checkpoint"
```

### Testing
```bash
# Validate dataset format
python scripts/validate_dataset_format.py --dataset /path/to/dataset

# End-to-end async save test
python scripts/test_async_save_e2e.py
```

## Architecture

### Core Pipeline Flow
```
DORA Dataflow (hardware) → ZeroMQ IPC → CLI (main.py) → Dataset → Training
```

1. **DORA Nodes** (`operating_platform/robot/components/`): Camera and arm drivers run as separate processes, communicate via ZeroMQ IPC sockets
2. **Manipulator** (`operating_platform/robot/robots/so101_v1/manipulator.py`): Aggregates camera/joint data from DORA nodes
3. **Daemon** (`operating_platform/core/daemon.py`): Manages robot connection lifecycle
4. **Record** (`operating_platform/core/record.py`): Recording loop with async episode saving
5. **Dataset** (`operating_platform/dataset/dorobot_dataset.py`): HuggingFace-compatible dataset with video encoding

### Key Components

| Directory | Purpose |
|-----------|---------|
| `operating_platform/core/` | Main pipelines: main.py, record.py, train.py, inference.py |
| `operating_platform/robot/robots/` | Robot configurations (so101_v1, aloha_v1_TODO, pika_v1_TODO) |
| `operating_platform/robot/components/` | Hardware drivers (camera_opencv, arm_normal_so101_v1) |
| `operating_platform/dataset/` | Dataset management, image/video encoding |
| `operating_platform/policy/` | Policy implementations (ACT, Diffusion) |
| `operating_platform/utils/` | Utilities including video.py for NPU encoding |

### DORA Dataflow
Robot hardware is controlled via DORA dataflow YAML files:
- `operating_platform/robot/robots/so101_v1/dora_teleoperate_dataflow.yml`
- Nodes for cameras, arms communicate at 30fps via timer ticks
- ZeroMQ sockets at `/tmp/dora-zeromq-so101-*` bridge DORA to Python CLI

### Async Episode Saving
Episodes save asynchronously to avoid blocking recording:
- `AsyncEpisodeSaver` queues saves in background thread
- Thread-safe buffer swap with `_buffer_lock`
- Retry logic with exponential backoff
- See `docs/ASYNC_SAVE_IMPLEMENTATION.md` for details

### NPU Video Encoding
For Ascend 310B hardware (`operating_platform/utils/video.py`):
- Semaphore limits concurrent NPU encoding to 2 channels
- Retry with exponential backoff if channels busy
- Fallback to libx264 CPU after 30s timeout
- Configure with `set_npu_encoder_channels(n)`

## Important Patterns

### Resource Cleanup
All hardware components must handle cleanup on exit:
```python
import signal, atexit

def cleanup():
    # Close sockets, release cameras, etc.
    pass

signal.signal(signal.SIGINT, lambda s,f: (cleanup(), exit(0)))
signal.signal(signal.SIGTERM, lambda s,f: (cleanup(), exit(0)))
atexit.register(cleanup)
```

### ZeroMQ Lazy Initialization
ZeroMQ sockets should be created lazily in `connect()`, not at module import:
```python
_zmq_initialized = False

def _init_zmq():
    global _zmq_initialized
    if _zmq_initialized:
        return
    # Create sockets here
    _zmq_initialized = True

def connect():
    _init_zmq()
    # ...
```

### Thread-Safe Metadata
When multiple threads access dataset metadata, use locks:
```python
with self._meta_lock:
    self.total_episodes += 1
```

## Version History

See `docs/RELEASE.md` for detailed changelog. Key versions:
- V27: NPU semaphore + retry for video encoding
- V25: USB port cleanup, signal handlers
- V23: Unified launcher, combined camera display
- V16: Async save race condition fixes
- V15: Buffer lock for thread safety

## Hardware Notes

### SO101 Robot
- Leader arm: `/dev/ttyACM0`
- Follower arm: `/dev/ttyACM1`
- Head camera: `/dev/video0`
- Wrist camera: `/dev/video2`
- Connect in order: cameras first, then arms

### Ascend NPU (Orange Pi 20T)
- Source CANN: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- h264_ascend encoder has 2-4 hardware channels
- Set `USE_NPU=1` when running
