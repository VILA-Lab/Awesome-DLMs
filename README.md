# Awesome Diffusion Language Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

## Table of Contents
- [Must-Read](#must-read)
- [Surveys](#surveys)
- [Diffusion Foundation](#diffusion-foundation)
- [Discrete DLMs](#discrete-dlms)
- [Continuous DLMs](#continuous-dlms)
- [Multimodal DLMs](#multimodal-dlms)
- [Training Strategies](#training-strategies)
- [Inference Optimization](#inference-optimization)
- [Applications](#applications)
- [Resources](#resources)



## Must-Read
D3PM: [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.03006)  

LLaDA: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.09992)
[![Website](https://img.shields.io/badge/Website-9cf)](https://ml-gsai.github.io/LLaDA-demo/)
[![Star](https://img.shields.io/github/stars/ML-GSAI/LLaDA.svg?style=social&label=Star)](https://github.com/ML-GSAI/LLaDA)

Diffusion-LM
Diffusion-Bert
Fast-dllm



## Surveys
<!-- add ours here! -->
[16 Jun 2025] [Discrete Diffusion in Large Language and Multimodal Models: A Survey](https://arxiv.org/abs/2506.13759)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.13759) [![Star](https://img.shields.io/github/stars/LiQiiiii/DLLM-Survey.svg?style=social&label=Star)](https://github.com/LiQiiiii/DLLM-Survey)

[28 Feb 2025] [Diffusion models in text generation: a survey](https://peerj.com/articles/cs-1905/) (PeerJ Computer Science)  

[5 Jun 2023] [An Overview of Diffusion Models for Text Generation](https://ieeexplore.ieee.org/document/10159911) (MIPRO)  

[24 May 2023] [A Survey of Diffusion Models in Natural Language Processing](https://arxiv.org/abs/2305.14671)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.14671)

[13 Mar 2023] [Diffusion Models in NLP: A Survey](https://arxiv.org/abs/2303.07576)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.07576)

[12 Mar 2023] [Diffusion Models for Non-autoregressive Text Generation: A Survey](https://arxiv.org/abs/2303.06574) (IJCAI 2023)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.06574) [![Star](https://img.shields.io/github/stars/RUCAIBox/Awesome-Text-Diffusion-Models.svg?style=social&label=Star)](https://github.com/RUCAIBox/Awesome-Text-Diffusion-Models)

## Diffusion Foundation

## Discrete DLMs
[10 Jul 2025] [Bayesian Discrete Diffusion Beats Autoregressive Perplexity](https://arxiv.org/abs/2507.07586v1)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.07586v1) [![Star](https://img.shields.io/github/stars/mercury0100/bayesradd.svg?style=social&label=Star)](https://github.com/mercury0100/bayesradd)

[17 Jun 2025] [LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs](https://arxiv.org/abs/2506.14429)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.14429)
[![Star](https://img.shields.io/github/stars/OpenMOSS/LongLLaDA.svg?style=social&label=Star)](https://github.com/OpenMOSS/LongLLaDA)

[2 Apr 2025] [Dream 7B](https://hkunlp.github.io/blog/2025/dream/)  
[![Website](https://img.shields.io/badge/Website-9cf)](https://hkunlp.github.io/blog/2025/dream/)
[![Star](https://img.shields.io/github/stars/HKUNLP/Dream.svg?style=social&label=Star)](https://github.com/HKUNLP/Dream)

[14 Feb 2025] [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.09992)
[![Website](https://img.shields.io/badge/Website-9cf)](https://ml-gsai.github.io/LLaDA-demo/)
[![Star](https://img.shields.io/github/stars/ML-GSAI/LLaDA.svg?style=social&label=Star)](https://github.com/ML-GSAI/LLaDA) 






## Continuous DLMs

## Multimodal DLMs

[29 May 2025] [Muddit: Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model](https://arxiv.org/abs/2505.23606)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.23606) [![Star](https://img.shields.io/github/stars/M-E-AGI-Lab/Muddit.svg?style=social&label=Star)](https://github.com/M-E-AGI-Lab/Muddit) 

[26 May 2025] [FUDOKI: Discrete Flow-based Unified Understanding and Generation via Kinetic-Optimal Velocities](https://arxiv.org/abs/2505.20147)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.20147) [![Website](https://img.shields.io/badge/Website-9cf)](https://fudoki-hku.github.io/)

[22 May 2025] [LaViDa: A Large Diffusion Language Model for Multimodal Understanding](https://arxiv.org/abs/2505.16839)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.16839) [![Website](https://img.shields.io/badge/Website-9cf)](https://jacklishufan.github.io/LaViDa/) [![Star](https://img.shields.io/github/stars/jacklishufan/LaViDa.svg?style=social&label=Star)](https://github.com/jacklishufan/LaViDa) 


[22 May 2025] [Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding](https://arxiv.org/abs/2505.16990)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.16990) [![Star](https://img.shields.io/github/stars/yu-rp/Dimple.svg?style=social&label=Star)](https://github.com/yu-rp/Dimple) 

[22 May 2025] [LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning](https://arxiv.org/abs/2505.16933)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.16933) [![Website](https://img.shields.io/badge/Website-9cf)](https://ml-gsai.github.io/LLaDA-V-demo/) [![Star](https://img.shields.io/github/stars/ML-GSAI/LLaDA-V.svg?style=social&label=Star)](https://github.com/ML-GSAI/LLaDA-V) 

[21 May 2025] [MMaDA: Multimodal Large Diffusion Language Models](https://arxiv.org/abs/2505.15809)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.15809) [![Star](https://img.shields.io/github/stars/Gen-Verse/MMaDA.svg?style=social&label=Star)](https://github.com/Gen-Verse/MMaDA) 





## Training Strategies
[25 May 2025] [LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models](https://arxiv.org/abs/2505.19223)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.19223)
[![Website](https://img.shields.io/badge/Website-9cf)](https://ml-gsai.github.io/LLaDA-1.5-Demo/)
[![Star](https://img.shields.io/github/stars/ML-GSAI/LLaDA-1.5.svg?style=social&label=Star)](https://github.com/ML-GSAI/LLaDA-1.5)

[15 May 2025] [Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](https://arxiv.org/abs/2505.10446)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.10446)

[16 Apr 2025] [d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning](https.arxiv.org/abs/2504.12216)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2504.12216)
[![Website](https://img.shields.io/badge/Website-9cf)](https://dllm-reasoning.github.io/)
[![Star](https://img.shields.io/github/stars/dllm-reasoning/d1.svg?style=social&label=Star)](https://github.com/dllm-reasoning/d1)

[2 Apr 2025] [Dream 7B](https://hkunlp.github.io/blog/2025/dream/)  
[![Website](https://img.shields.io/badge/Website-9cf)](https://hkunlp.github.io/blog/2025/dream/)
[![Star](https://img.shields.io/github/stars/HKUNLP/Dream.svg?style=social&label=Star)](https://github.com/HKUNLP/Dream)

## Inference Optimization
[12 Jun 2025] [Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles](https://arxiv.org/abs/2506.10848)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.10848)

[2 Jun 2025] [Esoteric Language Models](https://arxiv.org/abs/2506.01928)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.01928)

[31 May 2025] [Accelerating Diffusion LLMs via Adaptive Parallel Decoding](https://arxiv.org/abs/2506.00413)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.00413)

[30 May 2025] [Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking](https://arxiv.org/abs/2505.24857)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.24857)

[28 May 2025] [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.23481)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.23481)
[![Website](https://img.shields.io/badge/Website-9cf)](https://nvlabs.github.io/Fast-dLLM/)
[![Star](https://img.shields.io/github/stars/NVlabs/Fast-dLLM.svg?style=social&label=Star)](https://github.com/NVlabs/Fast-dLLM)

[27 May 2025] [Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion](https://arxiv.org/abs/2505.21467)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.21467)

[23 May 2025] [Variational Autoencoding Discrete Diffusion with Enhanced Dimensional Correlations Modeling](https://arxiv.org/abs/2505.17384)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.17384)

[22 May2025] [Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding](https://arxiv.org/abs/2505.16990)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.16990) [![Star](https://img.shields.io/github/stars/yu-rp/Dimple.svg?style=social&label=Star)](https://github.com/yu-rp/Dimple) 

[22 May 2025] [dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](https://arxiv.org/abs/2505.16962)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.16962)
[![Star](https://img.shields.io/github/stars/maomaocun/dLLM-cache.svg?style=social&label=Star)](https://github.com/maomaocun/dLLM-cache)

[21 May 2025] [dKV-Cache: The Cache for Diffusion Language Models](https://arxiv.org/abs/2505.15781)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.15781)


## Applications


## Resources
[bansky-cl/diffusion-nlp-paper-arxiv](https://github.com/bansky-cl/diffusion-nlp-paper-arxiv) 
 [![Star](https://img.shields.io/github/stars/bansky-cl/diffusion-nlp-paper-arxiv?style=social)](https://github.com/bansky-cl/diffusion-nlp-paper-arxiv)

[bansky-cl/Diffusion-LM-Papers](https://github.com/bansky-cl/Diffusion-LM-Papers) [![Star](https://img.shields.io/github/stars/bansky-cl/Diffusion-LM-Papers?style=social)](https://github.com/bansky-cl/Diffusion-LM-Papers)

[yczhou001/Awesome-Diffusion-LLM](https://github.com/yczhou001/Awesome-Diffusion-LLM) [![Star](https://img.shields.io/github/stars/yczhou001/Awesome-Diffusion-LLM?style=social)](https://github.com/yczhou001/Awesome-Diffusion-LLM)

[StevenYuan666/Awesome-Diffusion-Models-for-NLP](https://github.com/StevenYuan666/Awesome-Diffusion-Models-for-NLP) [![Star](https://img.shields.io/github/stars/StevenYuan666/Awesome-Diffusion-Models-for-NLP?style=social)](https://github.com/StevenYuan666/Awesome-Diffusion-Models-for-NLP)

[LiQiiiii/DLLM-Survey](https://github.com/LiQiiiii/DLLM-Survey) [![Star](https://img.shields.io/github/stars/LiQiiiii/DLLM-Survey?style=social)](https://github.com/LiQiiiii/DLLM-Survey)

[ML-GSAI/Diffusion-LLM-Papers](https://github.com/ML-GSAI/Diffusion-LLM-Papers) [![Star](https://img.shields.io/github/stars/ML-GSAI/Diffusion-LLM-Papers?style=social)](https://github.com/ML-GSAI/Diffusion-LLM-Papers)

[AoiDragon/Awesome-Text-Diffusion-Models](https://github.com/AoiDragon/Awesome-Text-Diffusion-Models) [![Star](https://img.shields.io/github/stars/AoiDragon/Awesome-Text-Diffusion-Models?style=social)](https://github.com/AoiDragon/Awesome-Text-Diffusion-Models)

