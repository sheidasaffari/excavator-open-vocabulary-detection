# Data layout (data are not distributed)

Place or symlink private data into these directories. Image and label filenames
must share the same stem. Labels use normalized YOLO bounding-box format.

```text
data/
├── mocs/{images,labels}/
├── acid/{images,labels}/
├── cis/{images,labels}/
├── positive_control/{images,labels}/
├── confusers/images/
└── excavision/
    └── train.txt
```

Only empty `.gitkeep` placeholders are committed. Excavision, MOCS, ACID, CIS,
the positive-control images, and confuser images are not uploaded.
