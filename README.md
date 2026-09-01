# finetrainers 🧪

Finetrainers is a work-in-progress library to support (accessible) training of diffusion models and various commonly used training algorithms.

> [!TIP]
> **This fork adds Apple Silicon (MPS) support** — single-device LoRA training on M-series Macs, no CUDA required. Validated models: LTX-Video 2B and Wan2.1 T2V 1.3B, with CPU↔MPS parity tests and saved performance baselines. Before running either recipe, follow the [Apple Silicon quickstart](./docs/apple_silicon.md#quickstart), including the required bitsandbytes MPS-fork build:
>
> ```bash
> bash examples/training/sft/ltx_video/crush_smol_lora/train_mps.sh   # LTX-Video 2B
> bash examples/training/sft/wan/crush_smol_lora/train_mps.sh        # Wan2.1 T2V 1.3B
> ```

<table align="center">
<tr>
  <td align="center"><video src="https://github.com/user-attachments/assets/aad07161-87cb-4784-9e6b-16d06581e3e5">Your browser does not support the video tag.</video></td>
  <td align="center"><video src="https://github.com/user-attachments/assets/c23d53e2-b422-4084-9156-3fce9fd01dad">Your browser does not support the video tag.</video></td>
</tr>
<tr>
  <th align="center">CogVideoX LoRA training as the first iteration of this project</th>
  <th align="center">Replication of PikaEffects</th>
</tr>
</table>

## Table of Contents

- [Quickstart](#quickstart)
- [Features](#features)
- [News](#news)
- [Support Matrix](#support-matrix)
- [Featured Projects](#featured-projects-)
- [Acknowledgements](#acknowledgements)

## Quickstart

Clone the repository, install the requirements with `pip install -r requirements.txt`, and install Diffusers from source with `pip install git+https://github.com/huggingface/diffusers`. The requirements specify `diffusers>=0.32.1`, but the Diffusers `main` branch is recommended for the latest features and bug fixes. The `finetrainers` `main` branch is also a development branch; use a release tag when you need stable behavior. Apple Silicon users must additionally install the bitsandbytes MPS fork as described in the [Apple Silicon quickstart](./docs/apple_silicon.md#quickstart).

Checkout to the latest stable release tag:

```bash
git fetch --all --tags
git checkout tags/v0.2.0
```

Follow the instructions in the [v0.2.0 release README](https://github.com/huggingface/finetrainers/tree/v0.2.0-release) for the latest stable release.

#### Using the main branch

To get started quickly with example training scripts on the main development branch, refer to the following:

- [LTX-Video Pika Effects Crush](./examples/training/sft/ltx_video/crush_smol_lora/)
- [CogVideoX Pika Effects Crush](./examples/training/sft/cogvideox/crush_smol_lora/)
- [Wan T2V Pika Effects Crush](./examples/training/sft/wan/crush_smol_lora/)

The following maintained datasets are useful for testing training quickly:

- [crush-smol](https://huggingface.co/datasets/finetrainers/crush-smol) — the small dataset used by the example recipes above
- [Finetrainers video-effects datasets](https://huggingface.co/collections/finetrainers/video-effects-datasets) — other small, captioned video datasets
- [OpenVid-1k split](https://huggingface.co/datasets/finetrainers/OpenVid-1k-split) — a larger WebDataset example
- [All Finetrainers datasets](https://huggingface.co/finetrainers/datasets)

Please check out [`docs/models`](./docs/models/) and [`examples/training`](./examples/training/) to learn more about supported models and reproducible training launch scripts. For a full list of training arguments, refer to [`docs/args`](./docs/args.md).

> [!IMPORTANT]
> PyTorch 2.5.1 or newer is recommended for training. Previous versions can lead to completely black videos, OOM errors, or other issues and are not tested. For fully reproducible training, use the environment documented in [environment.md](./docs/environment.md).

## Features

- DDP, FSDP-2 & HSDP, CP support
- Apple Silicon (MPS) single-device training — see [docs/apple_silicon.md](./docs/apple_silicon.md)
- LoRA and full-rank finetuning; Conditional Control training
- Memory-efficient single-GPU training
- Multiple attention backends supported - `flash`, `flex`, `sage`, `xformers` (see [attention](./docs/models/attention.md) docs)
- Auto-detection of commonly used dataset formats
- Combined image/video datasets, multiple chainable local/remote datasets, multi-resolution bucketing & more
- Memory-efficient precomputation support with/without on-the-fly precomputation for large scale datasets
- Standardized model specification format for training arbitrary models
- Fake FP8 training (QAT upcoming!)

## News

- 🔥 **2026-07-08**: Apple Silicon (MPS) support added in this fork — LTX-Video and Wan T2V LoRA train on M-series Macs, with CPU↔MPS parity tests and benchmark baselines!
- 🔥 **2025-04-25**: Support for different attention providers added!
- 🔥 **2025-04-21**: Wan I2V support added!
- 🔥 **2025-04-12**: Channel-concatenated control conditioning support added for CogView4 and Wan!
- 🔥 **2025-04-08**: `torch.compile` support added!
- 🔥 **2025-04-06**: Flux support added!
- 🔥 **2025-03-07**: CogView4 support added!
- 🔥 **2025-03-03**: Wan T2V support added!
- 🔥 **2025-03-03**: We have shipped a complete refactor to support multi-backend distributed training, better precomputation handling for big datasets, model specification format (externally usable for training custom models), FSDP & more.
- 🔥 **2025-02-12**: We have shipped a set of tooling to curate small and high-quality video datasets for fine-tuning. See [video-dataset-scripts](https://github.com/huggingface/video-dataset-scripts) documentation page for details!
- 🔥 **2025-02-12**: Check out [eisneim/ltx_lora_training_i2v_t2v](https://github.com/eisneim/ltx_lora_training_i2v_t2v/)! It builds off of `finetrainers` to support image to video training for LTX-Video and STG guidance for inference.
- 🔥 **2025-01-15**: Support for naive FP8 weight-casting training added! This allows training HunyuanVideo in under 24 GB upto specific resolutions.
- 🔥 **2025-01-13**: Support for T2V full-finetuning added! Thanks to [@ArEnSc](https://github.com/ArEnSc) for taking up the initiative!
- 🔥 **2025-01-03**: Support for T2V LoRA finetuning of [CogVideoX](https://huggingface.co/docs/diffusers/main/api/pipelines/cogvideox) added!
- 🔥 **2024-12-20**: Support for T2V LoRA finetuning of [Hunyuan Video](https://huggingface.co/docs/diffusers/main/api/pipelines/hunyuan_video) added! We would like to thank @SHYuanBest for his work on a training script [here](https://github.com/huggingface/diffusers/pull/10254).
- 🔥 **2024-12-18**: Support for T2V LoRA finetuning of [LTX Video](https://huggingface.co/docs/diffusers/main/api/pipelines/ltx_video) added!

## Support Matrix

The following trainers are currently supported:

- [SFT Trainer](./docs/trainer/sft_trainer.md)
- [Control Trainer](./docs/trainer/control_trainer.md)

> [!NOTE]
> The following numbers were obtained from the [v0.0.1 release](https://github.com/huggingface/finetrainers/tree/v0.0.1). The `main` branch is unstable at the moment and may use higher memory.

<div align="center">

|                 **Model Name**                 |   **Tasks**   | **Min. LoRA VRAM<sup>\*</sup>** | **Min. Full Finetuning VRAM<sup>^</sup>** |
| :--------------------------------------------: | :-----------: | :-----------------------------: | :---------------------------------------: |
|    [LTX-Video](./docs/models/ltx_video.md)     | Text-to-Video |              5 GB               |                   21 GB                   |
| [HunyuanVideo](./docs/models/hunyuan_video.md) | Text-to-Video |              32 GB              |                    OOM                    |
|   [CogVideoX-5b](./docs/models/cogvideox.md)   | Text-to-Video |              18 GB              |                   53 GB                   |
|          [Wan](./docs/models/wan.md)           | Text-to-Video |              TODO               |                   TODO                    |
|     [CogView4](./docs/models/cogview4.md)      | Text-to-Image |              TODO               |                   TODO                    |
|         [Flux](./docs/models/flux.md)          | Text-to-Image |              TODO               |                   TODO                    |

</div>

<sub><sup>\*</sup>Noted for training-only, no validation, at resolution `49x512x768`, rank 128, with pre-computation, using **FP8** weights & gradient checkpointing. Pre-computation of conditions and latents may require higher limits (but typically under 16 GB).</sub><br/>
<sub><sup>^</sup>Noted for training-only, no validation, at resolution `49x512x768`, with pre-computation, using **BF16** weights & gradient checkpointing.</sub>

If you would like to use a custom dataset, refer to the dataset preparation guide [here](./docs/dataset/README.md).

## Featured Projects 🔥

Checkout some amazing projects citing `finetrainers`:

- [Diffusion as Shader](https://github.com/IGL-HKUST/DiffusionAsShader)
- [SkyworkAI's SkyReels-A1](https://github.com/SkyworkAI/SkyReels-A1) & [SkyReels-A2](https://github.com/SkyworkAI/SkyReels-A2)
- [Aether](https://github.com/InternRobotics/Aether)
- [MagicMotion](https://github.com/quanhaol/MagicMotion)
- [eisneim's LTX Image-to-Video](https://github.com/eisneim/ltx_lora_training_i2v_t2v/)
- [wileewang's TransPixeler](https://github.com/wileewang/TransPixeler)
- [Feizc's Video-In-Context](https://github.com/feizc/Video-In-Context)

Checkout the following UIs built for `finetrainers`:

- [jbilcke's VideoModelStudio](https://github.com/jbilcke-hf/VideoModelStudio)
- [neph1's finetrainers-ui](https://github.com/neph1/finetrainers-ui)

## Acknowledgements

- `finetrainers` builds on top of & takes inspiration from great open-source libraries - `transformers`, `accelerate`, `torchtune`, `torchtitan`, `peft`, `diffusers`, `bitsandbytes`, `torchao` and `deepspeed` - to name a few.
- Some of the design choices of `finetrainers` were inspired by [`SimpleTuner`](https://github.com/bghira/SimpleTuner).
