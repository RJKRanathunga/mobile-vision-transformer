# Original MVT Tracker — Inference and Visualization Guide

**Platform:** Windows (Anaconda Prompt / Miniconda)  
**Repository:** [Original MVT](https://github.com/goutamyg/MVT)  
**Project:** `D:\Uni_Projects\movit-original`  
**Model:** MobileViT-S tracker, official GOT-10k epoch-100 checkpoint

This guide records the working inference procedure. The environment is already installed; there is no need to reinstall packages, retrain the model or convert the checkpoint for routine inference.

## 1. Activate the environment

Open **Anaconda Prompt** (or Miniconda Prompt) and execute:

```cmd
cd /d D:\Uni_Projects\movit-original
conda activate .\.conda
```

Verify the Python and CUDA environment if needed:

```cmd
python --version
python -c "import torch; import torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

Expected Python: `3.7.16`; PyTorch: `1.12.0`; Torchvision: `0.13.0`; CUDA availability: `True`. The environment uses the CUDA 11.3 build. Your separate Python 3.14 `.venv` is not used here.

## 2. Confirm the existing configuration (only needed if something changes)

### Official checkpoint

The checkpoint must be available at:

```text
output/checkpoints/train/mobilevit_track/mobilevit_256_128x1_got10k_ep100_cosine_annealing/MobileViT_Track_ep0100.pth.tar
```

If it needs to be restored from the original download, run these commands from the repository root:

```cmd
mkdir output\checkpoints\train\mobilevit_track\mobilevit_256_128x1_got10k_ep100_cosine_annealing
copy data\original_checkpoint\MobileViT_Track_ep0100.pth.tar output\checkpoints\train\mobilevit_track\mobilevit_256_128x1_got10k_ep100_cosine_annealing\
```

### Local paths

File: `lib/test/evaluation/local.py`

The relevant entries inside `local_env_settings()` are:

```python
settings.prj_dir = 'D:/Uni_Projects/movit-original'
settings.save_dir = 'D:/Uni_Projects/movit-original/output'
settings.got10k_path = 'D:/Uni_Projects/movit_object_tracking/data'
settings.network_path = 'D:/Uni_Projects/movit-original/output/test/networks'
settings.result_plot_path = 'D:/Uni_Projects/movit-original/output/test/result_plots'
settings.results_path = 'D:/Uni_Projects/movit-original/output/test/tracking_results'
settings.segmentation_path = 'D:/Uni_Projects/movit-original/output/test/segmentation_results'
```

Keep the other existing settings. The dataset path points to the parent of `val`, `test` and `train`, not directly to `val`. Update the path only if the dataset is moved.

### Experiment configuration

File: `experiments/mobilevit_track/mobilevit_256_128x1_got10k_ep100_cosine_annealing.yaml`

The `TEST` section should contain:

```yaml
TEST:
  DEVICE: cuda
  EPOCH: 100
  SEARCH_FACTOR: 4.0
  SEARCH_SIZE: 256
  TEMPLATE_FACTOR: 2.0
  TEMPLATE_SIZE: 128
```

These settings match the configuration used in the successful inference run. Do not alter the tracking algorithm or checkpoint.

## 3. Run inference on one sequence

From the project root, with `.conda` activated:

```cmd
python -u tracking/test.py --tracker_name mobilevit_track --tracker_param mobilevit_256_128x1_got10k_ep100_cosine_annealing --dataset_name got10k_val --sequence 0 --threads 0 --num_gpus 1 --debug 0
```

| Argument | Meaning |
|---|---|
| `--tracker_name mobilevit_track` | Original MVT tracker |
| `--tracker_param ...` | Official experiment configuration |
| `--dataset_name got10k_val` | GOT-10k validation split |
| `--sequence 0` | First sequence, `GOT-10k_Val_000001` |
| `--threads 0` | Sequential evaluation |
| `--num_gpus 1` | One GPU |
| `--debug 0` | Normal inference with saved results |

The successful run on November 25, 2026 reported **4.1203 FPS** and **17.760729 seconds** total execution time. These are measurements from that particular run, not guaranteed performance targets.

### Saved predictions

Results are stored under:

```text
output/test/tracking_results/mobilevit_track/mobilevit_256_128x1_got10k_ep100_cosine_annealing/
```

Look for `GOT-10k_Val_000001.txt` (bounding boxes in `x, y, width, height` format) and `GOT-10k_Val_000001_time.txt` (processing times), potentially within a `got10k` subfolder.

To locate the exact files:

```cmd
dir /s /b output\test\tracking_results\*.txt
```

**Rerunning a sequence:** the normal evaluation runner may skip a sequence whose prediction file already exists. If it does, move or rename that sequence's existing prediction and timing files to a backup location outside the results directory, then rerun the command. Keep earlier results if needed for comparisons.

## 4. Visualize saved inference results (OpenCV)

No inference needs to be repeated. If you already created `visualize_results.py` in the project root, run:

```cmd
python visualize_results.py
```

- Green bounding box: MVT prediction.
- Red bounding box: ground truth.
- Press **Q** to quit.

If the script is lost, recreate `visualize_results.py` using the following version. It includes a fallback for the result directory and accepts comma- or whitespace-separated annotation files. This is a standalone visualization helper, not a modification to MVT.

```python
import cv2
import numpy as np
from pathlib import Path

sequence = 'GOT-10k_Val_000001'

root = Path(__file__).resolve().parent
data_dir = Path('D:/Uni_Projects/movit_object_tracking/data/val') / sequence
result_dir = root / 'output/test/tracking_results/mobilevit_track' / (
    'mobilevit_256_128x1_got10k_ep100_cosine_annealing'
)

pred_path = result_dir / 'got10k' / (sequence + '.txt')
if not pred_path.is_file():
    pred_path = result_dir / (sequence + '.txt')
if not pred_path.is_file():
    raise FileNotFoundError('Prediction file not found: ' + str(pred_path))


def load_boxes(path):
    try:
        boxes = np.loadtxt(str(path), delimiter=',')
    except ValueError:
        boxes = np.loadtxt(str(path))
    return np.atleast_2d(boxes)


predictions = load_boxes(pred_path)
ground_truth = load_boxes(data_dir / 'groundtruth.txt')
frames = sorted(data_dir.glob('*.jpg'), key=lambda p: int(p.stem))

if not frames or not (len(frames) == len(predictions) == len(ground_truth)):
    raise ValueError('Frame / prediction / ground-truth count mismatch: '
                     '{} / {} / {}'.format(len(frames), len(predictions), len(ground_truth)))

try:
    for i, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise IOError('Cannot read frame: ' + str(frame_path))

        # Draw ground truth in red and prediction in green.
        for box, color in [(ground_truth[i], (0, 0, 255)),
                           (predictions[i], (0, 255, 0))]:
            if np.all(np.isfinite(box)) and box[2] > 0 and box[3] > 0:
                x, y, w, h = map(int, box)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(frame, 'Frame: {}/{}'.format(i+1, len(frames)),
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.imshow('Original MVT - Tracking Results', frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
```

The playback interval is approximately 33 ms per frame (nominal 30 FPS, excluding rendering overhead); it does not measure model inference speed.

## 5. Visualize live inference (Visdom)

Open a second Anaconda Prompt:

```cmd
cd /d D:\Uni_Projects\movit-original
conda activate .\.conda
python -m visdom.server
```

Open [http://localhost:8097](http://localhost:8097) in your browser. Keep the server running.

In the first prompt (with `.conda` active), run:

```cmd
python -u tracking/test.py --tracker_name mobilevit_track --tracker_param mobilevit_256_128x1_got10k_ep100_cosine_annealing --dataset_name got10k_val --sequence 0 --threads 0 --num_gpus 1 --debug 1
```

The debug dashboard can display tracking frames, template and search crops, and score maps. Space pauses/resumes; Right Arrow advances a frame while paused. Stop the server with Ctrl+C in its terminal when finished. Use `--debug 0` for benchmark timing and saving predictions; visualization adds overhead and debug runs are not intended to save the normal evaluation results.

## 6. Run the entire validation split (optional)

```cmd
python -u tracking/test.py --tracker_name mobilevit_track --tracker_param mobilevit_256_128x1_got10k_ep100_cosine_annealing --dataset_name got10k_val --threads 0 --num_gpus 1 --debug 0
```

To use the official test split instead, replace `got10k_val` with `got10k_test`. The test split must be present. Official test-set accuracy is generally obtained through the GOT-10k evaluation server. Inference itself does not calculate AO or success rates.

## 7. Troubleshooting and reproducibility

| Issue | Action |
|---|---|
| `conda` not recognized | Open Anaconda Prompt rather than ordinary PowerShell. |
| `No module named 'torch._six'` | Verify the `.conda` environment is active and PyTorch is 1.12.0. Do not use the modern `.venv` for this baseline. |
| Missing TensorBoard imports | Consult the original `mvt_pyenv.yml`; its relevant pins are `tb-nightly==2.12.0a20230113`, `tensorboardx==2.6`, `protobuf==3.20.3`. |
| Checkpoint not found | Verify the checkpoint location in section 2. |
| Dataset not found | Verify `settings.got10k_path` and `val/list.txt`. |
| Torchvision image-extension or floor-division warning | These warnings were non-fatal in the successful run. Investigate if a later run fails. |
| `Not Yet Implemented!` | This message appeared in the successful run; its exact source was not established. |
| No visualization window | Check that OpenCV with GUI support is installed and the prediction/annotation files exist. |

To record the current package versions for future reproducibility:

```cmd
conda list > mvt_conda_packages.txt
python -m pip freeze > mvt_pip_freeze.txt
```

The original `mvt_pyenv.yml` is a Linux environment specification; do not apply it unchanged on Windows. Preserve original package pins where available and keep platform-specific adjustments documented.

**Scope:** The single-sequence inference was confirmed by the successful terminal output, and the user confirmed visualization worked. The fallback-enhanced playback script above has not been executed on the user's machine. The full validation run and accuracy evaluation are separate, optional steps not confirmed by this run.
