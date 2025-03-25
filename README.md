AI-Powered Plant Health Detector: A Multimodal Vision & Language System
Problem Statement

Diagnosing plant health issues like improper watering, sunlight exposure, and nutrient deficiencies requires expertise. This project develops an AI-powered Plant Health Detector using computer vision and NLP to assess plant health from an image and provide care recommendations. The system generalizes to unseen species through meta-learning and domain adaptation.
Related Work & Dataset Availability

Previous studies focus on plant disease classification (e.g., PlantVillage). Hyperspectral imaging aids water/nutrient detection but requires special equipment. Our model integrates vision-based assessment and NLP-driven recommendations.

We will use:

Plant Identification: 

PlantNet, 
iNaturalist, 
LeafSnap

Health & Disease: 

PlantVillage, 
CVPPP Leaf Segmentation

Water/Nutrient Estimation: 

RGB-NIR datasets

Methodology

Plant Classification: Train CNNs or Transformers (e. g. EfficientNet & ViT) for species identification.
Water/Nutrient Analysis: Use deep learning and color-based feature extraction.
Sun Exposure Estimation: Implement shadow analysis and color degradation detection.
NLP-Based Recommendations: Fine-tune T5, GPT-4, BERT for care suggestions.


Generalization to Unseen Species: Apply meta-learning and contrastive learning.
Evaluation Metrics
Classification Accuracy (Top-1, Top-5 accuracy on unseen species)
Water/Nutrient Prediction (Mean Absolute Error vs. ground-truth values)
Sunlight Estimation (Comparison with dataset metadata)
NLP Evaluation (BLEU, ROUGE scores for recommendations)

Road Map & Timeline
Week 1: Dataset collection
Weeks 2-3: Plant classification model
Weeks 4-5: Water/nutrient estimation
Week 6: Sun exposure detection
Week 7: NLP-based recommendations
Week 8: Model integration and testing
Week 9: Final evaluation and report preparation
Expected Outcomes
A functional AI system that analyzes plant health and generates accurate care recommendations, generalizing to unseen plant species.
AI Assistance Disclosure: Parts of this proposal were AI-generated and refined by our team.
