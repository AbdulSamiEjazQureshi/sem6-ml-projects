# Kaggle Dataset Drop Folder

Put real Kaggle leaf datasets here when internet/Kaggle credentials are available.

Recommended datasets:

- PlantVillage-style plant disease datasets for disease labels.
- Leaf classification/tree species datasets for tree identity.
- Plant Pathology apple leaf datasets for real-world apple disease photos.

Expected folder format:

```text
leafguard/data/kaggle/
  Apple___healthy/
    image1.jpg
    image2.png
  Apple___rust/
    image3.jpg
  Tomato___leaf_spot/
    image4.jpg
```

Class folder naming rule:

- `Species___Condition` gives both tree/crop identity and disease condition.
- If only one folder name exists, it is used as the condition label and species remains unknown.

JPG/PNG training requires Pillow in the active Python environment. BMP files work without Pillow.
