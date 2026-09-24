# Data provenance and coverage

The accompanying paper cites **GraspNet-1Billion** (Fang et al., CVPR 2020).
The official data and annotation downloads are at:
https://graspnet.net/datasets.html

Section III describes generating dense graspability heatmaps from planar grasp
rectangles and object masks. Section VI reports 25,600 training images and 2,560
images in each validation and test split. The paper does not provide a download
URL for that derived heatmap dataset. These training data, their split manifests,
and the heatmap-generation/training scripts were not found in the supplied
workspace or its backup archive. This repository does not contain the complete
training dataset reported in the paper.

## Included local RGB-D captures

All 11 distinct captured frames found in the workspace are retained, each with
RGB PNG, depth NPY and camera-pose TXT files:

| Location under `grasp_work/` | Frames | Notes |
| --- | ---: | --- |
| `refactored_sam_pipeline-3/data/` | 4 | Intrinsics JSON included |
| `refactored_sam_pipeline-3/data_capture_4/` | 4 | No separate session intrinsics file was found |
| `grasp_pipeline/data_capture_5/` | 3 | Shared camera intrinsics available in `grasp_pipeline/camera_intrinsics.json`; session-specific calibration is not established |

The four frames in `refactored_sam_pipeline-2/data/` are exact duplicates retained
to preserve the earlier scripts' relative paths; do not count them as extra data.
The stored pose files contain six values (translation and rotation), accepted by
the reconstruction loader; they are not stored as 4x4 matrices. Depth units and
pose conventions must match the selected reconstruction options.
No additional labels or calibration values have been invented.

`dataset_manifest.csv` records capture paths, byte sizes and SHA-256 checksums.
Saved point clouds, masks, predicted heatmaps, candidate scores and figures are
**derived experiment outputs**, not additional independent captured frames.

## Selected experiment outputs

The publication snapshot retains both versions' `grasp_ready_objects`, the
canonical `outputs_full_pipeline_gpu_retry` runs, the version-3 capture-4
`outputs_capture_4_handcream_symmetric_v2_refacut` run used by the figure tools,
and `outputs_capture5_flex_fixed` / `outputs_capture5_baseline` runs.
These are the saved runs referenced by the code or useful for the comparisons;
this selection does not certify that every paper number is reproducible.
Other exploratory output folders remain in the original workspace.

## Capture review

The three `data_capture_5` RGB images show people in the laboratory background.
The original pixels are preserved. Permission to publish those images and any
derived full-scene visualizations needs to be confirmed by the authors.

## Attribution

The official GraspNet page states non-commercial redistribution conditions and
identifies the license as BY-NC-SA. Follow the upstream terms for upstream data
and any derived artifacts; public availability is not a blanket reuse license.
The included local captures must not be described as the original GraspNet data.
Their authorship and release license should be supplied by the project authors.

```bibtex
@inproceedings{fang2020graspnet,
  title={GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping},
  author={Fang, Hao-Shu and Wang, Chenxi and Gou, Minghao and Lu, Cewu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11444--11453},
  year={2020}
}
```
