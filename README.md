# 🌿 Patapim Net

A vision-language pipeline for automated plant health diagnostics. This project combines CNN-based species and disease classifiers with a large language model (LLM) to generate natural language care reports, all wrapped in a simple web interface.

## Project Structure

- **LLM & RAG/** – Natural language generation modules for care report generation.
- **Plant Classification/** – Colab notebook for identifying plant species using a CNN.
- **Plant Disease/** – Colab notebook for detecting plant diseases from leaf images using a CNN.
- **website/** – Frontend web interface built with Next.js.

## How to Use

### 1. Run the Classifiers (Colab)
The disease and species classification models are designed to be run in Google Colab:

- Open `Plant Classification Model` and `646ProjectDisease` notebooks.
- Upload and process plant leaf images.
- Export the model predictions.

### 2. Launch the Web Interface

To view the full system end-to-end, run the web interface locally:

```bash
npm install
npm run dev
```

The main backend functional code is in website/app/api/analyze.