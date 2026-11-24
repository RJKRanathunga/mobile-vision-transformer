# MVT reporting and presentation figures

These scripts do not change training, evaluation or model code. Install the plotting dependency in the existing virtual environment:

```powershell
python -m pip install -r requirements-reporting.txt
```

## Local training curves

```powershell
python scripts/plot_training.py --metrics checkpoints/mvt_got10k/metrics.jsonl
```

Change the metrics path for a different run. PNGs (300 DPI), vector PDFs and CSV go to `reports/training/<run>/`. Validation is only plotted where logged; use `--val-interval 1` in `train.py` for more frequent validation. Graphs can be regenerated after each completed epoch. Duplicate epochs use the latest row; no loss curves are invented. The IoU is pairwise training IoU, not full-sequence tracking AO. The original paper does not include training loss curves.

## Published paper plots (no training required)

```powershell
python scripts/plot_paper_results.py
```

`reports/paper/` contains charts and CSVs for Tables 1–3 (GOT-10k, TrackingNet, NfS30, LaSOT, speed, parameters and backbone fusion ablation) and reconstructed Figure 2. All are **published reference results**, not your measurements. Source: Gopal and Amer, *Mobile Vision Transformer-based Visual Object Tracking*, BMVC 2023, https://papers.bmvc2023.org/0800.pdf (pp. 7–9). MixFormer-L CPU speed is reported as `<5` and not plotted as an exact number.

## Local tracker comparisons

Evaluate checkpoints using distinct names and the same data split, sequence list, device and settings, e.g.:

```powershell
python evaluate_mvt.py --weights checkpoints/mvt_got10k/best.pth --subset val --device cuda --name MVT_trained
python scripts/compare_trackers.py --reports reports/GOT-10k/MVT_official_full/performance.json reports/GOT-10k/MVT_trained/performance.json --protocol "GOT-10k full val; same GPU and settings"
```

Use actual paths if your toolkit differs. This reads `overall` metrics from toolkit reports, exports accuracy, FPS, success curves (when supplied by the toolkit), and CSV. Legacy scalar SR is treated as SR0.50. SR0.75 is read from an explicit metric or the exact 0.75 threshold in the upstream toolkit’s uniform success-curve grid; otherwise omitted, never interpolated. The toolkit reports may not encode dataset split or hardware; you must verify the common protocol yourself. Do not compare local validation scores with paper test scores as equivalent. The script can also compare two independently trained ablation variants if you implement them; it does not create or train a no-fusion architecture.

## Local LaSOT robustness

Download the official LaSOT dataset/test list and sequence-level attributes separately. Create your own CSV from the official annotations, not invented tags. Columns: `sequence,path,attributes`. Example format only (attribute field is a placeholder, not a verified annotation):

```csv
sequence,path,attributes
example-1,example/example-1,ARC FM
```

Each path is relative to `--data-root`, with `groundtruth.txt` and `img/00000001.jpg` etc. Use the official 280 test sequences for a full test experiment. The script validates the manifest structure, not the official membership or labels. `FOC`, `OV`, etc. are sequence-level tags here, not frame-level masks.

```powershell
python scripts/evaluate_robustness.py --manifest data/lasot_test_attributes.csv --data-root data/LaSOT --weights pretrained_models/MobileViT_Track_ep0100_weights_only.pth --device cuda --name MVT_official --output-dir reports/robustness/MVT_official
```

Alternatively use `--predictions-dir` instead of `--weights` to score existing comma-separated xywh predictions (`<sequence>.txt`, one row per GT frame including initialization). Output contains per-sequence metrics, per-attribute FR, protocol JSON and PNG/PDF. Actual model runs are not tested by the patch installer. Valid GT boxes only; initialization excluded. Failure is IoU=0. Frame-weighted FR is plotted and both frame-weighted and sequence-mean FR are exported. Exact paper visibility and aggregation policy is not established, so this is a documented local reproduction, not guaranteed numerically identical. No LaSOT data or annotations are bundled.

## Limitations

- Only local files are installed. No remote changes, downloads, model training or package installation happen automatically.
- The existing training loop is untouched; rerun the plotter to refresh reports.
- The paper's Figure 1 architecture drawing is not generated; Figure 2 and numeric tables are reconstructed.
- Real backbone ablation requires code/configuration and separate training for the no-fusion variant; only the published ablation graph and generic local comparison are included here.
