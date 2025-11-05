# KeyPilot: Vision-Language Typing Agent

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**⚠️ Note: This project is under active development. Features, APIs, and documentation may change. Contributions are welcome!**

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
├── results/               # Training results and checkpoints
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

4. Run the quickstart example:
```bash
python examples/quickstart.py
```

This will demonstrate model initialization, forward pass, and prediction capabilities.

5. Set up API keys (for data generation, optional):
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

Train the KeyPilot model with the provided script:

```bash
python scripts/train.py --config configs/model_config.yaml --use_wandb
```

Or programmatically:

```python
import yaml
from keypilot.models import create_keypilot_model, KeyPilotLoss

# Load configuration
with open('configs/model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model
model = create_keypilot_model(config['model'])

# Print model summary
summary = model.get_model_summary()
print(f"Total parameters: {summary['total_parameters']:,}")
print(f"Model size (INT8): {summary['model_size_mb_int8']:.2f} MB")

# Create loss function
criterion = KeyPilotLoss(**config['training']['loss'])

# TODO: Implement dataloaders and training loop
# See scripts/train.py for full training implementation
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
    model_dir="results/best_model",
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
import torch
from PIL import Image
from keypilot.models import KeyPilotVLM
from keypilot.utils.vocabulary import KeyPilotVocabulary

# Load model
model = KeyPilotVLM(
    vocab_size=32000,
    d_model=256,
    num_tasks=3,
    num_layouts=5,
    num_experts=5,
    pretrained_backbone=True
)
model.eval()

# Load tokenizer
vocab = KeyPilotVocabulary()
vocab.load("models/tokenizer/tokenizer.json")

# Prepare inputs
image = torch.randn(1, 3, 512, 256)  # Replace with actual screen capture
text = "Hey, how are you? I'm doing great! What about"
encoded = vocab.encode(text, max_length=64)
input_ids = torch.tensor([encoded['input_ids']])
attention_mask = torch.tensor([encoded['attention_mask']])

# Predict
predictions = model.predict(
    image=image,
    input_ids=input_ids,
    attention_mask=attention_mask,
    temperature=0.9
)

print(f"Task: {predictions[0]['task']}")
print(f"Layout: {predictions[0]['layout']}")
print(f"Confidence: {predictions[0]['task_confidence']:.3f}")
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

## 📊 Model Architecture

KeyPilot uses a lightweight Vision-Language architecture optimized for on-device deployment:

### Encoder (~7.5M parameters, ≤19ms)
- **Visual Backbone**: MobileViT-XXS (α=0.75) - 1.3M params
- **ROI Segmentation**: SAM-Lite with 4 fixed regions - 0.6M params
- **Global Features**: Lightweight projection - 0.3M params
- **Text Encoder**: mBERT-tiny (2 layers, multilingual) - 4.5M params
- **Cross-Former**: Single-layer fusion - 0.8M params

### Decoder (~3.2M parameters, <10ms)
- **Task Router**: 2-layer MLP (error/completion/suggestion) - 0.1M params
- **Layout Router**: 2-layer Transformer (EN/ZH/SYM/EMOJI/NUM) - 0.3M params
- **Language MoE**: 5 expert decoders with sparse activation - 2.0M params
- **Vocabulary**: Shared 32K multilingual BPE - 0.8M params

### Total: ~10.7M parameters (< 50MB after INT8 quantization)

For detailed design rationale, see [docs/module_selection.md](docs/module_selection.md).

## 📊 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Total Latency | < 50ms | End-to-end on mobile NPU |
| Encoder Latency | ≤ 19ms | Vision-language fusion |
| Decoder Latency | < 10ms | Task output generation |
| Model Size | < 50MB | After INT8 quantization |
| RAM Usage | < 100MB | Peak memory |
| Task Accuracy | > 90% | Error/completion/suggestion |
| Layout Accuracy | > 95% | EN/ZH/SYM/EMOJI/NUM |

*Note: Train your model and run evaluation to measure actual performance.*

## 🔬 Model Distillation

Distill large models for on-device deployment:

```python
from keypilot.models import DistillationTrainer
from keypilot.utils import DistillationConfig

# Load teacher and student models
teacher_model = KeyPilotVLM.from_pretrained("results/best_model", config)
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
- [x] Design principles and architecture documentation
- [x] Module selection and component design
- [x] Encoder implementation (MobileViT, SAM-Lite, mBERT-tiny)
- [x] Decoder implementation (Task/Layout routers, MoE)
- [x] Complete KeyPilot VLM model
- [x] Training script and configuration
- [x] Unit tests and quickstart example
- [ ] Multilingual tokenizer training (32K BPE)
- [ ] Dataset preparation and data loaders
- [ ] Multi-agent data generation pipeline
- [ ] Model training and validation
- [ ] Collect and label real mobile typing data
- [ ] Model quantization (INT8) and optimization
- [ ] ONNX/TensorRT export for mobile
- [ ] Mobile SDK development (iOS/Android)
- [ ] Keyboard integration and testing
- [ ] User study and evaluation

---

**Note**: This is a research project. The model requires training on domain-specific data before deployment.

