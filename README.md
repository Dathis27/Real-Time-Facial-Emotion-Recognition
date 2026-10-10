# Real-Time Facial Emotion Recognition (FER-2013 + CNN)

## Setup in VS Code
1. Install Python 3.9–3.11 and the VS Code **Python** extension.
2. Open this folder (File > Open Folder).
3. Terminal > New Terminal, then:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows   (macOS/Linux: source venv/bin/activate)
   pip install -r requirements.txt
   ```
   Select the interpreter: Ctrl+Shift+P > "Python: Select Interpreter" > venv.
4. Download the dataset: https://www.kaggle.com/datasets/msambare/fer2013
   and extract it into `dataset/` so that `dataset/train/happy/...` and `dataset/test/happy/...` exist.

## Train
```
python train.py
```
Produces `emotion_model.keras`, `labels.json`, `training_curves.png`.
Expect ~60–67% test accuracy (FER-2013 is noisy; human agreement is ~65%).
A GPU speeds this up a lot; on CPU, lower EPOCHS in train.py.

## Run live
```
python realtime.py
```
q = quit, s = screenshot.

## Tips
- Good front lighting and a face roughly facing the camera help most.
- For higher accuracy: transfer learning (MobileNetV2 on 96x96) or the FER+ labels.
##   py -3.13 realtime.py