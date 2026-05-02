# AI Goalkeeper
Real-Time Ball Interception Using Edge AI on Raspberry Pi

## Quick Links
- 📄 [Full Project Report](report.md)
- 📁 [Source Code](src/)
- 🧠 [Model](model/best.onnx)

## Repository Structure

```
ai-goalkeeper/
├── src/
│   ├── goalkeeper_ai.py    ← main system (detection + prediction + motor)
│   ├── calib.py            ← one-time pixel→cm calibration tool
│   ├── snapshots.py        ← burst capture for dataset collection
│   └── photos.py           ← background image capture
├── model/
│   └── best.onnx           ← deployed ONNX model
├── training/               ← training config, weights and result plots
├── figures/                ← images used in report
├── report.md               ← full project report
└── requirements.txt
```

- Course Link - https://www.samy101.com/edge-ai-26/
