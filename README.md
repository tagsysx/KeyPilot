# KeyPilot: Vision-Language Typing Agent

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

KeyPilot is a vision-language typing agent that activates keyboard intelligence by jointly understanding the visual context of the user's screen and the linguistic context of conversations. The system predicts both the next input intent (e.g., text, symbol, emoji, or numeric) and the optimal keyboard layout, grounding predictions in multimodal reasoning.

## 🌟 Key Features

- **Multimodal Understanding**: Combines visual context (screen content) with linguistic context (conversation history)
- **Intent Prediction**: Predicts next input type (text, symbol, emoji, numeric, etc.)
- **Layout Optimization**: Recommends optimal keyboard layout based on context
- **On-Device Deployment**: Distills large VLMs into lightweight models for mobile devices
- **Multi-Agent Data Generation**: Generates diverse training data using ChatGPT, DeepSeek, and other LLMs
- **LoRA Fine-tuning**: Efficient parameter-efficient training with quantization support

## 📋 Research Background

Current mobile keyboards rely mainly on short-range textual prediction, anticipating the next word from recent keystrokes while ignoring higher-level intent, language shifts, and on-screen context. This narrow view prevents them from predicting upcoming input types or adapting keyboard layouts intelligently, leading to frequent switching and disrupted typing flow.

KeyPilot addresses this by leveraging fine-tuned vision-language models (VLMs) to understand both what's on screen and what's being typed, enabling intelligent keyboard adaptation.

## 🏗️ Project Structure

```
KeyPilot/
├── src/keypilot/          # Main source code
│   ├── models/            # VLM models and distillation
│   ├── data/              # Dataset and preprocessing
│   ├── training/          # Training pipeline
│   ├── evaluation/        # Evaluation metrics
│   ├── agents/            # Multi-agent data generation
│   └── utils/             # Configuration and logging
├── data/                  # Dataset storage
│   ├── raw/               # Raw data
│   ├── processed/         # Processed data
│   ├── train/             # Training data
│   ├── val/               # Validation data
│   └── test/              # Test data
├── models/                # Saved models
│   ├── pretrained/        # Pre-trained base models
│   ├── finetuned/         # Fine-tuned models
│   └── distilled/         # Distilled on-device models
├── configs/               # Configuration files
├── experiments/           # Training experiments
├── logs/                  # Training logs
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks
└── .temp/                 # Temporary files

```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/KeyPilot.git
cd KeyPilot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys (for data generation):
```bash
export OPENAI_API_KEY="your-openai-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
```

### Data Generation

Generate training data using multi-agent pipeline:

```python
from keypilot.agents import MultiAgentPipeline
from keypilot.utils import KeyPilotConfig

# Load configuration
config = KeyPilotConfig()

# Create multi-agent pipeline
pipeline = MultiAgentPipeline(
    num_agents=4,
    intent_classes=config.intent_classes,
    layout_types=config.layout_types,
    use_chatgpt=True,
    use_deepseek=True,
)

# Generate dataset
pipeline.generate_dataset(
    total_samples=10000,
    output_path="data/raw/generated_data.json",
)
```

### Data Preprocessing

Split dataset into train/val/test:

```python
from keypilot.data import DataPreprocessor

preprocessor = DataPreprocessor(
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
)

train_path, val_path, test_path = preprocessor.split_dataset(
    data_path="data/raw/generated_data.json",
    output_dir="data/processed",
)
```

### Model Training

Train the KeyPilot model:

```python
from keypilot import KeyPilotVLM, KeyPilotTrainer
from keypilot.data import KeyPilotDataset
from keypilot.utils import KeyPilotConfig

# Load configuration
config = KeyPilotConfig()

# Initialize model
model = KeyPilotVLM(
    config=config.model,
    intent_classes=config.intent_classes,
    layout_types=config.layout_types,
)

# Load datasets
train_dataset = KeyPilotDataset(
    data_path="data/processed/train.json",
    processor=model.processor,
    intent_classes=config.intent_classes,
    layout_types=config.layout_types,
)

val_dataset = KeyPilotDataset(
    data_path="data/processed/val.json",
    processor=model.processor,
    intent_classes=config.intent_classes,
    layout_types=config.layout_types,
)

# Train model
trainer = KeyPilotTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=config.training,
)

trainer.train()
```

### Model Evaluation

Evaluate trained model:

```python
from keypilot import KeyPilotVLM, KeyPilotEvaluator
from keypilot.data import KeyPilotDataset
from keypilot.utils import ModelConfig

# Load model
config = ModelConfig()
model = KeyPilotVLM.from_pretrained(
    model_dir="experiments/best_model",
    config=config,
)

# Load test dataset
test_dataset = KeyPilotDataset(
    data_path="data/processed/test.json",
    processor=model.processor,
    intent_classes=model.intent_classes,
    layout_types=model.layout_types,
)

# Evaluate
evaluator = KeyPilotEvaluator(model, test_dataset)
metrics = evaluator.evaluate()
evaluator.save_results(metrics, ".temp/evaluation_results.json")
```

### Inference

Make predictions with trained model:

```python
from PIL import Image
from keypilot import KeyPilotVLM
from keypilot.utils import ModelConfig

# Load model
model = KeyPilotVLM.from_pretrained(
    model_dir="experiments/best_model",
    config=ModelConfig(),
)

# Load screen image
image = Image.open("screen_capture.png")

# Conversation text
conversation = "Hey, how are you? I'm doing great! What about"

# Predict
intent, layout = model.predict(image, conversation)
print(f"Predicted intent: {intent}")
print(f"Optimal layout: {layout}")
```

## 🔧 Configuration

Configuration is managed through the `KeyPilotConfig` class. You can customize:

- Model architecture and quantization
- Training hyperparameters
- Data generation settings
- Distillation parameters

Example configuration file (`configs/default.yaml`):

```yaml
model:
  base_model: "microsoft/phi-3-vision-128k-instruct"
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  load_in_4bit: true

training:
  num_epochs: 10
  batch_size: 4
  learning_rate: 2e-4
  use_wandb: true
  wandb_project: "keypilot"

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  num_agents: 4
```

## 📊 Model Performance

Performance metrics on test set:

| Metric | Intent Prediction | Layout Prediction |
|--------|------------------|-------------------|
| Accuracy | TBD | TBD |
| Precision | TBD | TBD |
| Recall | TBD | TBD |
| F1 Score | TBD | TBD |

*Note: Train your model and run evaluation to populate these metrics.*

## 🔬 Model Distillation

Distill large models for on-device deployment:

```python
from keypilot.models import DistillationTrainer
from keypilot.utils import DistillationConfig

# Load teacher and student models
teacher_model = KeyPilotVLM.from_pretrained("experiments/best_model", config)
student_model = KeyPilotVLM(student_config, intent_classes, layout_types)

# Distill
distiller = DistillationTrainer(
    teacher_model=teacher_model,
    student_model=student_model,
    config=DistillationConfig(),
)

# Training code here...
distiller.save_student("models/distilled/student_model")
```

## 🧪 Testing

Run tests:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_models.py -v
```

## 📝 Citation

If you use KeyPilot in your research, please cite:

```bibtex
@article{keypilot2024,
  title={KeyPilot: A Vision-Language Typing Agent for Intelligent Keyboard Prediction},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Vision-language model based on [Phi-3 Vision](https://huggingface.co/microsoft/phi-3-vision-128k-instruct)
- LoRA implementation from [PEFT](https://github.com/huggingface/peft)
- Data generation powered by ChatGPT and DeepSeek

## 📧 Contact

For questions or collaborations, please open an issue or contact [your-email@example.com].

## 🗺️ Roadmap

- [x] Initial project setup
- [x] Multi-agent data generation pipeline
- [x] Vision-language model architecture
- [x] Training and evaluation pipeline
- [ ] Collect and label real mobile typing data
- [ ] Train production model
- [ ] Model quantization and optimization
- [ ] Mobile SDK development
- [ ] iOS/Android keyboard integration
- [ ] User study and evaluation

---

**Note**: This is a research project. The model requires training on domain-specific data before deployment.

