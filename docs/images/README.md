# Documentation Images

This directory contains images and diagrams used in the KeyPilot documentation.

## Images

### architecture.png
The main system architecture diagram showing:
- Vision-Language Encoder (left): MobileViT, MobileSAM, MobileCLIP, TinyLM components
- Task-Specific Decoder (right): Task Router, Layout Router, Language Router with specialized experts

**Note**: Please replace the placeholder file with the actual architecture diagram image.

## Adding New Images

To add new images to the documentation:

1. Save the image file to this directory (`docs/images/`)
2. Reference it in markdown files using relative path:
   ```markdown
   ![Alt text](images/filename.png)
   ```
3. Add a description in this README

## Supported Formats

- PNG (recommended for diagrams)
- JPG/JPEG (for photos)
- SVG (for vector graphics)
- GIF (for animations)

