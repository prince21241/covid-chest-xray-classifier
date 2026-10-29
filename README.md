# COVID vs Normal chest X-ray classifier

Educational experiment. This project trains an ImageNet-pretrained ResNet-18 to separate COVID chest X-rays from Normal chest X-rays in the COVID-19 Radiography Database.

It is not a clinical diagnostic tool. Do not use it to decide whether someone has an infection, and do not treat a model score as a calibrated probability of infection. The project does not target a particular accuracy.

## Limitations

Read these before trusting a number the model prints.

**Source bias.** The dataset authors built the COVID class from several public collections (BIMCV/PadChest, a German medical-school repository, SIRM, Eurorad, GitHub, Figshare, and others). Most Normal images come from the RSNA pneumonia challenge, with a smaller share from a Kaggle pneumonia collection. The model can learn those source differences — equipment, contrast, markers, or layout — instead of learning COVID. A high score on this split is not evidence the model would work in a hospital.

**Patient independence cannot be verified** with the metadata shipped in this copy of the database. The workbooks have `FILE NAME`, `FORMAT`, `SIZE`, and `URL` only. Those are file details and shared links to source datasets, not patient identifiers. Filenames such as `COVID-1.png` are sequential image names, and this project does not treat them as patient IDs. Exact duplicate files are kept in the same split. Different images of the same patient can still land in training and test. If that happened, the test score is an optimistic picture of performance on new patients.

**COVID versus Normal only.** Lung Opacity and Viral Pneumonia folders are ignored. Mask images are ignored. The model has never been asked to recognize those other patterns, and a Normal prediction only means "closer to the Normal images in this dataset than to the COVID images."

**Held-out test set.** Validation data chooses the decision threshold. The test set is scored once after that choice is frozen. It is not a tuning set. If you change the model after seeing the test score, that number is no longer an honest final result.

## Dataset layout this project expects

Point `--data-dir` at the folder that contains `COVID` and `Normal`, or leave it empty when this project sits next to the data:

```text
COVID-19_Radiography_Dataset/
  COVID/
    images/          used
    masks/           not used
  Normal/
    images/          used
    masks/           not used
  Lung_Opacity/      not used
  Viral Pneumonia/   not used
  COVID.metadata.xlsx
  Normal.metadata.xlsx
```

The image folders in this workspace contain 3,616 COVID PNGs and 10,192 Normal PNGs, plus a matching mask PNG for each. Only the image folders are model inputs.

Preparation of this copy opened all 13,808 of those images. None failed to open, and every opened image is 299×299. The metadata `SIZE` value `256*256` disagreed with the pixels for every file, so the opened size is what the model uses. There were 35 exact-duplicate groups (47 extra copies of the same pixels) and no duplicate pair with conflicting labels. The saved split has 9,667 training images, 2,072 validation images, and 2,069 test images. Run `prepare` again if you replace the files; do not assume these counts for a different download.

## Project layout

```text
covid_cxr/
  prepare.py          validate images, hashes, metadata, and the split
  splits.py           keep duplicate and patient groups intact
  preprocessing.py    RGB letterbox to 224×224 and ImageNet normalization
  train.py            head training, then the final residual block
  evaluate.py         validation threshold, then one test evaluation
  predict.py          one image, same preprocessing as evaluation
  metrics.py          COVID is the positive class
requirements.txt
```

Commands all start with `python -m covid_cxr`.

## Set up on your computer

Use Python 3.10 or newer.

```powershell
cd C:\Credit-Card-Fraud-Detector\covid-chest-xray-classifier
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate with `source .venv/bin/activate`.

CPU-only PyTorch (no NVIDIA GPU):

```powershell
pip install numpy Pillow matplotlib scikit-learn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

NVIDIA GPU: install the CUDA build shown at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), then install the rest:

```powershell
pip install -r requirements.txt
```

Do not run that PyPI install on top of a CPU-only torch you just installed, or pip may replace it with a different build. Kaggle notebooks already have a GPU build; do not install the CPU index there.

The first training run downloads ImageNet ResNet-18 weights. That needs a network connection.

## Set up a Kaggle notebook with a GPU

1. Create a notebook and add the COVID-19 Radiography Database as input.
2. In notebook settings, set the accelerator to GPU.
3. Upload this project (the `covid_cxr` package, `requirements.txt`, and this README). Kaggle already provides PyTorch, torchvision, Pillow, and scikit-learn. Run `pip install -r requirements.txt` only if an import fails.
4. Open the Input panel and copy the directory that contains `COVID` and `Normal`. A common path is:

```text
/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset
```

The folder name follows the dataset slug, so use the path you see if it differs. Input data is read-only. Write results under `/kaggle/working`.

```bash
python -m covid_cxr prepare \
  --data-dir /kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset \
  --output-dir /kaggle/working/outputs

python -m covid_cxr train --smoke-test --output-dir /kaggle/working/outputs
python -m covid_cxr train --full --output-dir /kaggle/working/outputs --batch-size 32
python -m covid_cxr evaluate --output-dir /kaggle/working/outputs
```

Turn on notebook internet so the pretrained weights can download. If the GPU runs out of memory, rerun full training with `--batch-size 16` or `--batch-size 8`. On Linux you can try `--num-workers 2`. Leave `--num-workers` at `0` on Windows.

## How to start training

From the project folder, with the virtual environment activated:

```powershell
python -m covid_cxr prepare
python -m covid_cxr train --smoke-test
python -m covid_cxr train --full
python -m covid_cxr evaluate
python -m covid_cxr predict --image "C:\path\to\one-xray.png"
```

`prepare` is safe to run more than once. When the dataset path, seed, and files are unchanged, it reuses `outputs/splits/manifest.csv`. Pass `--resplit` after you add, remove, or replace images.

`train` with no extra flag does not train. `--smoke-test` runs one epoch on 16 training images and 8 validation images from each class and writes `outputs/smoke/`. That only checks that the pipeline runs. It is not a result. `--full` trains on the whole training split and writes `outputs/checkpoints/best.pt`.

A full run on a CPU can take many hours. Use a GPU for that run. The smoke test is the run to start with on a CPU.

If automatic discovery cannot see the data, pass the folder yourself:

```powershell
python -m covid_cxr prepare --data-dir "C:\path\to\COVID-19_Radiography_Dataset"
```

You can also set the `CXR_DATA_DIR` environment variable. An explicit `--data-dir` is used as given and is not replaced by a different folder.

## What preparation does

1. Finds `COVID/images` and `Normal/images`. Mask directories are counted and then skipped. Other class folders are named in the report and skipped.
2. Opens every X-ray, records its pixel size, and drops files that cannot be opened.
3. Hashes each valid file. Exact copies stay in one split. Copies that have both labels are excluded and listed as `conflicting_labels`. Keeping them would force the model to learn two answers for the same pixels.
4. Reads the metadata workbooks. A patient-level split is used only when a column is actually a patient or subject identifier and it covers the images. File names are never used as patient IDs. When that identifier is missing, the report states that patient independence cannot be verified.
5. Builds a 70% / 15% / 15% split, stratified by label. The split unit is the duplicate group, or the patient when an identifier exists. A group that crosses a 70% or 15% boundary stays entirely on one side, so the percentages are approximate.
6. Saves `outputs/splits/manifest.csv` and `outputs/splits/split_report.json`. The seed defaults to 42.

Normal is the larger class in this database (10,192 image files versus 3,616 COVID image files before validity checks). An always-Normal classifier gets an accuracy equal to the Normal share of whatever split it is scored on. Accuracy by itself can look strong for a model that rarely predicts COVID. Sensitivity and the baseline exist so that number is not the whole story.

## What training does

The network is ResNet-18 initialized from ImageNet (`IMAGENET1K_V1`). The original 1,000-way layer is replaced with one output. COVID is label 1. The training loss is weighted binary cross-entropy. The positive weight is `number of Normal training images / number of COVID training images`, computed from the training split only.

Images are converted to RGB, resized so the long side is 224, and padded so the short side stays visible. Nothing is cropped away to make the square. Pixels are normalized with the pretrained model's ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

Training images also get a random rotation of up to 10 degrees and a translation of up to 5% of the image. There is no random crop, no horizontal flip, and no vertical flip. Validation and test images use the resize, padding, and normalization only. The same evaluation transform is saved inside the checkpoint and used for prediction.

Phase 1 trains the new classification head with the backbone frozen. Phase 2 unfreezes the final residual block (`layer4`) and the head, with a smaller learning rate. Frozen batch-norm layers stay in evaluation mode so their running statistics do not move. `--head-only` skips phase 2.

Early stopping watches validation loss, with patience of 3 epochs in each phase. The checkpoint on disk is the one with the lowest validation loss, across both phases. The test split is not loaded during training.

Useful flags:

| Flag | Default | Role |
| --- | --- | --- |
| `--batch-size` | 32 | Images per step |
| `--lr` | 0.001 | Head learning rate |
| `--lr-finetune` | 0.0001 | Learning rate after `layer4` is unfrozen |
| `--epochs` | 5 | Maximum head epochs |
| `--finetune-epochs` | 5 | Maximum fine-tuning epochs |
| `--patience` | 3 | Epochs without a better validation loss |
| `--seed` | 42 | Shuffle, augmentation, and weight initialization |
| `--data-dir` | auto | Dataset location |

These defaults are a starting budget, not a recipe that guarantees a good model. Curves are written to `outputs/plots/training_curves.png` and `outputs/metrics/history.csv`. The validation ROC-AUC on that plot is a monitor. It is not the test result, and it is not used to pick the decision threshold.

Training uses CUDA when PyTorch can see a GPU, and CPU otherwise.

## What evaluation does

```powershell
python -m covid_cxr evaluate
```

The best checkpoint is run on the validation set. The threshold is the value from 0.01 to 0.99 that maximizes COVID F1. Ties prefer a higher Youden index (sensitivity + specificity − 1), then the threshold closer to 0.5. That threshold is written into the checkpoint and then left alone.

The test set is scored once with that frozen threshold. COVID is the positive class. The saved metrics are accuracy, sensitivity (COVID recall), specificity, precision, F1, ROC-AUC, PR-AUC, and a confusion matrix. `outputs/metrics/test_metrics.json` also includes an always-Normal baseline on the same test labels. The confusion matrix figure is `outputs/plots/confusion_matrix_test.png`.

Validation metrics are in the JSON file so you can see how the threshold was chosen. They are not a final score. A smoke-test checkpoint is refused here.

## What prediction does

```powershell
python -m covid_cxr predict --image "C:\path\to\one-xray.png"
```

The image goes through the evaluation preprocessing stored in the checkpoint: RGB, letterbox to 224×224, ImageNet normalization, no augmentation. The printed COVID value is a model score (a sigmoid of the network output). It is not a calibrated probability of infection. The predicted label is the class on the positive side of the frozen threshold. If you predict before evaluation, the command says so and uses 0.5 until a validation threshold exists.

The checkpoint stores the class mapping (Normal = 0, COVID = 1), the preprocessing configuration, and, after evaluation, the threshold.

## Checks you can rerun

Unit tests do not train a model:

```powershell
python -m unittest discover -s tests -v
```

## Citation

If you use the images, cite the dataset authors, as their included README requests:

- M.E.H. Chowdhury et al., “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, vol. 8, 2020, pp. 132665–132676.
- T. Rahman et al., “Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images,” arXiv:2012.02238, 2020.
