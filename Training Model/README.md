# Training Model

This folder contains the training pipeline for the **SIBI Sign Language Recognition** project. Three CNN architectures were experimented with: **ResNet50**, **VGG16**, and **MobileNetV2**.

---

## Model Training

Each model was trained using transfer learning with pretrained ImageNet weights. After training, the models were initially saved in **Keras `.h5` format**.

To prepare the models for deployment, the `.h5` files were then converted into **TensorFlow SavedModel format** using the following scripts:

* `mobilenetv2/h5_to_savedmodel_mobilenetv2.py`
* `resnet50/h5_to_savedmodel_resnet50.py`
* `vgg16/h5_to_savedmodel_vgg16.py`

---

## Conversion for Web Deployment

Since the final application runs in the browser using **TensorFlow.js**, the SavedModel needed to be converted again into **TensorFlow.js format**.

The conversion pipeline used in this project:

Training → `.h5` model → **TensorFlow SavedModel** → **TensorFlow.js model**

Among the three architectures tested, **ResNet50 achieved the highest performance**, therefore it was selected as the final model and converted to TensorFlow.js.

The conversion process is implemented in:

`savedmodel_to_js_resnet50.ipynb`

The final TensorFlow.js model used by the web application is located in:

`web_app/model_web/`

---

## Model Performance

| Model           | Validation Accuracy | Test Accuracy |
| --------------- | ------------------- | ------------- |
| **ResNet50**    | 0.9955              | 0.9933        |
| **VGG16**       | 0.9933              | 0.9910        |
| **MobileNetV2** | 0.9933              | 0.9865        |

---

## Summary

Three CNN architectures were trained and evaluated. **ResNet50** produced the best results and was selected as the final model for deployment in the browser-based inference system using **TensorFlow.js**.
