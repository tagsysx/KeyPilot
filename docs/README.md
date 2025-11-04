# KeyPilot Documentation

This directory contains comprehensive documentation for the KeyPilot project.

## Documents

### [Design Principles](design_principles.md)
Core design philosophy, problem formulation, and architectural decisions for KeyPilot. This document covers:
- Core functionalities (Auto-Completion, Error Correction, Layout Switching, Proactive Suggestion)
- System workflow and architecture
- Mathematical formulation of the intelligent IME problem
- Performance requirements and constraints
- Future extension possibilities

### [Method](method.md)
Detailed technical method for KeyPilot's implementation. This document covers:
- Vision-Language Encoder architecture (MobileViT, dual-path encoding, cross-modal fusion)
- Task-Specific Decoders (MoE architecture, task/layout routing)
- Training objectives (encoder loss, decoder loss, distillation loss)
- Model specifications and performance metrics

### [Module Selection Guide](module_selection.md)
Comprehensive guide for selecting components for the Vision-Language Encoder:
- Visual Backbone options (MobileViT-XXS, alternatives)
- ROI Segmentation (SAM-Lite, MobileSAM)
- Global Visual Features (MobileCLIP, alternatives)
- Text Encoder options (RoBERTa-tiny, DistilBERT)
- Cross-Modal Fusion (Transformer configurations)
- Parameter budget analysis and recommendations
- Implementation priorities and pretrained weights sources

## Additional Documentation

For more information, please refer to:
- [Main README](../README.md) - Project overview and quick start guide
- [API Documentation](api.md) - Detailed API reference (coming soon)
- [Training Guide](training.md) - Model training instructions (coming soon)
- [Deployment Guide](deployment.md) - On-device deployment guide (coming soon)

## Contributing to Documentation

When contributing documentation:
1. Use clear, concise language
2. Include code examples where applicable
3. Keep mathematical notation consistent
4. Add diagrams for complex concepts
5. Update this index when adding new documents

