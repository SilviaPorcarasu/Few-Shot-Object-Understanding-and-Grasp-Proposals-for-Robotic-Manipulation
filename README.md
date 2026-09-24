# Few-Shot Object Understanding and Grasp Proposals for Robotic Manipulation

Python code for RGB-D object segmentation, graspability heatmap inference,
3D reconstruction and three-finger grasp proposal generation.

## Dataset

The paper uses **GraspNet-1Billion** as the source for graspability supervision.
Download the RGB-D scenes and grasp annotations from the official dataset page:

**https://graspnet.net/datasets.html**

The GraspNet API is available at https://github.com/graspnet/graspnetAPI.
Follow the dataset's attribution and non-commercial redistribution terms.

The derived heatmap dataset described in the paper has no download link in the
supplied manuscript. Its generation and training scripts were not present in
the supplied code. This repository contains code only; input images, depth maps,
camera poses, calibration files, model weights and experiment results must be
provided separately.

## Requirements

Python **3.10 or later** is required by the source syntax. Dependencies:

- NumPy and SciPy
- Pillow, OpenCV and imageio
- Matplotlib
- PyTorch and torchvision
- Hugging Face Transformers with support for `Sam3Model` and `Sam3Processor`
- Optional: `pyrealsense2` for the RealSense camera utility

Install the dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the optional camera utility:

```bash
python -m pip install -r requirements-camera.txt
```

Use compatible PyTorch/torchvision builds for your CPU or CUDA environment.
The original environment lockfile was not supplied; package versions are not
pinned, and a complete clean-environment inference run has not been verified.

## Models

Perception uses `facebook/sam3` and, by default,
`IDEA-Research/grounding-dino-base`. The optional detector is
`google/owlvit-base-patch32`. Hugging Face model access is required as applicable.

Heatmap inference additionally requires the project's TorchScript checkpoint,
`grasp_model_v4.pt`, passed with `--heatmap-model`. A public download URL for this
checkpoint was not supplied. It cannot be replaced by the raw GraspNet dataset.

## Usage

Show the grasping commands:

```bash
cd grasp_work
python main.py --help
```

Run perception on your own image, supplying the heatmap checkpoint:

```bash
cd refactored_sam_pipeline-3
python run_pipeline.py \
  --heatmap-model /path/to/grasp_model_v4.pt \
  --input /path/to/rgb.png \
  --save-outputs \
  --output-dir outputs_demo
```

For 3D reconstruction, supply aligned depth and camera intrinsics; multi-view
reconstruction also needs the corresponding camera poses. Available options:

```bash
python run_pipeline.py --help
```

Generate grasps from existing perception outputs:

```bash
python main.py grasp-all --agg \
  --pipeline-root refactored_sam_pipeline-3 \
  --pipeline-outputs /path/to/perception_outputs \
  --output-root /path/to/grasp_results
```

Visualization and experiment-specific scripts expect separately supplied result
files and may require adjusting their default paths to match your data.
