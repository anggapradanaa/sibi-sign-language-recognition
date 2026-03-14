```markdown
# Training Model

This folder contains the training pipeline for the SIBI sign language recognition model.  
Three CNN architectures were experimented with: **ResNet50, VGG16, and MobileNetV2**.

## Model Training

Each model was trained using transfer learning with a pretrained backbone.  
After training, the models were initially saved in **Keras `.h5` format**.

For compatibility with deployment tools, the `.h5` models were then converted into the **TensorFlow SavedModel format** using scripts such as:

- `h5_to_savedmodel_resnet50.py`
- `h5_to_savedmodel_vgg16.py`
- `h5_to_savedmodel_mobilenetv2.py`

## Model Conversion for Web Deployment

Since the web application uses **TensorFlow.js**, the SavedModel was converted again into **TensorFlow.js format**.  

Among the three trained architectures, **ResNet50 achieved the highest performance**, therefore it was selected as the final model and converted to TensorFlow.js for the web application.

The conversion process is implemented in:

```

savedmodel_to_js_resnet50.ipynb

```

The final model used by the web application is stored in the `web_app/model_web` directory.

## Model Performance

| Model | Validation Accuracy | Test Accuracy |
|------|---------------------|---------------|
| ResNet50 | 0.9955 | 0.9933 |
| VGG16 | 0.9933 | 0.9910 |
| MobileNetV2 | 0.9933 | 0.9865 |

## Summary

Training workflow:

```

Training → .h5 model → SavedModel → TensorFlow.js model

```

ResNet50 was selected as the final model due to its superior performance and was deployed in the browser-based inference system.
```