# Prometheus

> Repository description: Windows Rocket League reinforcement-learning workspace for training continuous-action PPO agents with attention, CUDA physics, transfer learning, an ImGui control panel, and RLBot runtime support.

Prometheus is a Windows-focused Rocket League reinforcement-learning workspace built on GigaLearnCPP, RLGymCPP, RocketSim, RocketSimCuda, RLBotCPP, and an ImGui control panel. It trains a continuous-action PPO policy with an attention head, optional CUDA physics, transfer learning from an older teacher checkpoint, live metrics, checkpoint management, and RLBot runtime support.

Author: **mitige**

French documentation is available in [README.fr.md](README.fr.md).

## What Prometheus Builds

The root CMake project builds two main executables:

| Target | Output | Purpose |
| --- | --- | --- |
| `Prometheus` | `build/Release/Prometheus.exe` | ImGui training frontend for configuring and launching training, transfer learning, rendering, checkpoints, rewards, and metrics. |
| `GigaLearnBot` | `build/Release/GigaLearnBot.exe` | Command-line learner and RLBot agent runtime. |

The source layout is:

| Path | Role |
| --- | --- |
| `src/ExampleMain.cpp` | CLI training, transfer learning, render mode, and RLBot agent entry point. |
| `src/gui/` | Prometheus ImGui frontend and JSON config handling. |
| `prometheus_config.json` | Main GUI/runtime config persisted by Prometheus. |
| `collision_meshes/` | RocketSim arena collision meshes required at runtime. |
| `GigaLearnCPP/` | Learning framework and RLGymCPP/RocketSim dependencies used by the learner. |
| `RocketSimCuda/` | CUDA physics backend and validation/benchmark code. |
| `RLBotCPP/` and `rlbot/` | RLBot C++ bridge and Python-side RLBot configuration. |

Generated folders such as `build/`, `checkpoints*/`, `wandb/`, `run_logs/`, `graphify-out/`, and local archives are intentionally ignored by Git.

## Requirements

Prometheus is intended for Windows with NVIDIA CUDA. CPU physics mode exists, but the default configuration uses CUDA.

Install:

- Windows 10/11
- Visual Studio 2022 with the Desktop development with C++ workload
- CMake 3.18 or newer
- Git
- Python 3.11 x64
- NVIDIA driver and CUDA Toolkit compatible with your LibTorch build
- LibTorch C++ distribution matching your CUDA version

The build scripts expect these environment variables:

| Variable | Required | Meaning |
| --- | --- | --- |
| `LIBTORCH_DIR` | Yes, unless `./libtorch` exists | Path to the extracted LibTorch folder containing `share/cmake/Torch/TorchConfig.cmake`. |
| `PROMETHEUS_PYTHON_HOME` | Optional | Path to Python 3.11. If omitted, scripts try to find `python` on `PATH`. |

Example in `cmd.exe`:

```bat
set "LIBTORCH_DIR=C:\tools\libtorch"
set "PROMETHEUS_PYTHON_HOME=C:\Users\you\AppData\Local\Programs\Python\Python311"
```

Example in PowerShell:

```powershell
$env:LIBTORCH_DIR = "C:\tools\libtorch"
$env:PROMETHEUS_PYTHON_HOME = "C:\Users\you\AppData\Local\Programs\Python\Python311"
```

## Build

From the repository root:

```bat
build.bat
```

The script configures CMake on the first run and then builds the Release configuration. If you change LibTorch, Python, CUDA, or CMake options after the first configure, delete `build/` and run `build.bat` again.

Manual CMake equivalent:

```bat
cmake -S . -B build -DCMAKE_PREFIX_PATH="%LIBTORCH_DIR%" -DPython_ROOT_DIR="%PROMETHEUS_PYTHON_HOME%" -DPython_EXECUTABLE="%PROMETHEUS_PYTHON_HOME%\python.exe"
cmake --build build --config Release -j
```

Useful CMake options:

| Option | Default | Description |
| --- | --- | --- |
| `PROMETHEUS_BUILD_GUI` | `ON` | Build the ImGui frontend target. |
| `GGL_ENABLE_ROCKETSIM_CUDA` | `ON` | Build and link the CUDA physics backend. |

## Run the GUI

```bat
run_gui.bat
```

The GUI reads and writes `prometheus_config.json`. Use it to configure:

- arena count, tick skip, PPO iteration size, minibatch size, epochs
- CUDA vs CPU mode
- continuous action size
- attention head dimensions, heads, blocks, and preprocessing/postprocessing layers
- reward list and weights
- checkpoint folder and save interval
- transfer-learning teacher and student checkpoint folders
- metrics and render behavior

Prometheus writes checkpoints into the configured checkpoint folder. Checkpoints are intentionally ignored by Git because they can be very large.

## Command-Line Modes

All CLI modes go through `GigaLearnBot.exe`.

```bat
run.bat
```

Default mode runs PPO training with the configured continuous-action attention architecture.

Common arguments:

| Command | Description |
| --- | --- |
| `run.bat train` | Normal PPO training. |
| `run.bat train cpu` | PPO training with RocketSim CPU physics instead of RocketSimCuda. |
| `run.bat train games=128` | Override the number of parallel arenas for this run. |
| `run.bat train metrics` | Enable Python/W&B-style metrics if the local environment supports them. |
| `run.bat train no-old` | Disable training against old policy versions. |
| `run.bat transfer` | Continuous transfer-learning loop. |
| `run.bat transfer-once` | One transfer-learning iteration. |
| `run.bat render` | Render/inference-oriented mode. |
| `run.bat agent` | Start the RLBot agent bridge. |

Convenience wrappers:

| Script | Equivalent |
| --- | --- |
| `run_render.bat` | `run.bat render` |
| `run_agent.bat` | `run.bat agent` |
| `rocketsimcpu.bat` | `run.bat train cpu` |
| `transferlearn.bat` | Transfer-learning helper with CUDA-related environment defaults. |

## Transfer Learning

The default transfer configuration expects a teacher checkpoint at:

```text
checkpoints/28188091392
```

That checkpoint is not suitable for Git and is ignored. To use transfer learning, place your teacher model folder there or update `transferOldModelsPath` in `prometheus_config.json` or the GUI. Student checkpoints default to:

```text
checkpoints_transfer
```

The student checkpoint folder must be separate from the teacher checkpoint folder.

## RLBot Mode

RLBot mode uses `rlbot/port.cfg` when present and defaults to port `23233` otherwise.

```bat
run_agent.bat
```

Before running RLBot mode:

1. Build Release.
2. Install the Python RLBot requirements in `rlbot/requirements.txt`.
3. Make sure Rocket League and the RLBot framework are set up locally.
4. Start the RLBot match/framework, then run the agent.

The policy loader uses the `checkpoints` folder and the same attention/continuous-action architecture as training.

## Configuration Reference

`prometheus_config.json` is safe to edit by hand. Important fields:

| Field | Meaning |
| --- | --- |
| `numArenas` | Number of parallel games/arenas. Lower this if VRAM or RAM is exhausted. |
| `useCuda` | Enables CUDA learner/device mode and RocketSimCuda physics. |
| `collisionMeshesDir` | Folder passed to RocketSim. Keep it as `collision_meshes` unless you move the meshes. |
| `checkpointDir` | Folder for PPO checkpoints. |
| `stepsPerIteration` | PPO rollout steps per iteration. |
| `minibatchSize` | PPO minibatch size; the GUI clamps it to divide the batch. |
| `policyLayerSizes`, `criticLayerSizes` | Comma-separated hidden sizes. |
| `useAttentionHead` | Enables the attention shared head. |
| `attentionDims`, `attentionHeads`, `attentionBlocks` | Main attention architecture knobs. `attentionDims` must be divisible by `attentionHeads`. |
| `transferLearning` | GUI run-state flag for transfer mode. |
| `transferOldModelsPath` | Teacher checkpoint folder. |
| `transferStudentCheckpointDir` | Student checkpoint output folder. |

## Public Repository Hygiene

This repository is prepared to keep only source, configuration, small runtime assets, and documentation in Git. Do not commit:

- `build/`
- `checkpoints*/`
- `wandb/`
- `run_logs/`
- `graphify-out/`
- `*.zip`
- `*.log`
- local LibTorch folders

No production secrets are required by the source tree. If you enable external metrics locally, keep credentials outside Git.

## Third-Party Code and Licenses

Prometheus includes or uses third-party components such as GigaLearnCPP, RLGymCPP, RocketSim, RocketSimCuda, RLBotCPP, pybind11, nlohmann/json, Bullet, GLFW, ImGui, and LibTorch. Keep the license files and notices inside those projects. The root project does not currently declare a separate license; add one only if you intend to grant public reuse rights for your own code.

## Troubleshooting

`LIBTORCH_DIR` is missing: set it to the extracted LibTorch folder, not to `share/cmake/Torch`.

Python cannot be found: set `PROMETHEUS_PYTHON_HOME` to your Python 3.11 install folder.

`Prometheus.exe` or `GigaLearnBot.exe` is missing: run `build.bat` first.

RocketSim cannot load meshes: make sure `collision_meshes/soccar/*.cmf` exists and that `collisionMeshesDir` points to `collision_meshes`.

CUDA build errors: check that Visual Studio, CUDA Toolkit, NVIDIA driver, and LibTorch CUDA versions are compatible. If CMake cached an old path, delete `build/` and configure again.

Out of memory: reduce `numArenas`, `stepsPerIteration`, model layer sizes, or attention dimensions.
