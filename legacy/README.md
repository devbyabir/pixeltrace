# Legacy notebooks

The original TensorFlow/Keras Colab notebooks, kept with their saved outputs as the "before" state of this project. Code and outputs are otherwise unchanged; removed for publication: Colab account metadata, the dataset's private download link, and one output cell that showed dataset images (licence unknown).

| Notebook | Reported test accuracy | Notes |
|---|---|---|
| `01_custom_cnn.ipynb` | 93.25 % | 500 epochs; ROC-AUC computed from hard labels |
| `02_efficientnet_b7.ipynb` | 99.61 % | Test set used as validation |
| `03_mobilenet_actually_resnet50.ipynb` | 98.96 % | Despite the name, trains ResNet50 |
| `04_resnet50.ipynb` | 96.61 % | Test set used as validation; no ResNet preprocessing |
| `05_vgg16.ipynb` | 96.74 % | Test set used as validation; no VGG preprocessing |
| `06_vit_unfinished.ipynb` | n/a | ViT from scratch, never executed |

**Don't cite these numbers.** All notebooks split with `dataset.take()/skip()` on an `image_dataset_from_directory(shuffle=True)` dataset, which reshuffles every epoch, so train and test overlap. On top of that, every image in the dataset exists twice, and file metadata alone predicts the label with 97.7 % accuracy. See the main [README](../README.md) for the full audit and the corrected pipeline.
