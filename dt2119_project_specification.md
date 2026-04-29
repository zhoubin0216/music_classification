# Project Specification

**Group Number:** proj 6  
**Member Names:** Bin Zhou, Jiahang Lyu

## Preliminary Project Title

**Cross-Cultural Music Genre Classification using Deep Audio Models**

## Type of Project

Experimental Project

## Main Problem Statement

Music genre classification is a common task in Music Information Retrieval (MIR), but most existing datasets and models mainly focus on Western music genres, limiting their ability to generalize to non-Western traditions.

This project aims to build a cross-cultural music genre classification system by combining Western and Eastern music genres into a unified dataset. The goal is to investigate whether modern deep audio models can effectively distinguish genres across different musical traditions and analyze similarities and confusions between them.

## Data

The project will use a balanced dataset of **18 genres**.

### Western genres (10) - GTZAN Dataset

- Pop
- Rock
- Metal
- Hip-hop
- Blues
- Country
- Reggae
- Disco
- Classical
- Jazz

### Eastern genres (8) - Saraga Dataset + public online collections

- Peking Opera
- Chinese Instrumental
- C-pop
- Enka
- J-pop
- K-pop
- Carnatic Vocal
- Indian Instrumental

All audio will be standardized and segmented into short clips for training.

## Models / Algorithms

The project will use a pretrained deep audio representation model: **MERT (Music Encoder Representation Transformer)**. Extracted audio features will be used as input to a simple classifier for genre prediction.

## Evaluation

Performance will be evaluated using:

- Accuracy
- F1-score
- Confusion Matrix

Additional analysis includes:

- Western-only vs cross-cultural training
- Vocal vs instrumental genre recognition
- Genre embedding visualization using t-SNE

## Key References

1. Tzanetakis, G., & Cook, P. (2002). *Musical genre classification of audio signals*. IEEE.
2. Sturm, B. L. (2013). *The GTZAN dataset*. arXiv.
3. Srinivasamurthy, A., et al. (2021). *Saraga: Open datasets for Indian art music*. EMR.
4. Li, Y., et al. (2023). *MERT: Acoustic music understanding model*. arXiv.
5. Choi, K., et al. (2017). *Transfer learning for music classification tasks*. ISMIR.
