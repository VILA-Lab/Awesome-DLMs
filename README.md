# Awesome Diffusion Language Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

## Table of Contents
- [Playground](#playground)
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


## Playground
[Mecury](https://chat.inceptionlabs.ai/) [![Static Badge](https://img.shields.io/badge/📰-Demo-green)](https://chat.inceptionlabs.ai/)

[LLaDA](https://huggingface.co/spaces/multimodalart/LLaDA) [![deploy](https://img.shields.io/badge/Hugging%20Face-Demo-yellow)](https://huggingface.co/spaces/multimodalart/LLaDA)


[MMaDA](https://huggingface.co/spaces/Gen-Verse/MMaDA) [![deploy](https://img.shields.io/badge/Hugging%20Face-Demo-yellow)](https://huggingface.co/spaces/Gen-Verse/MMaDA)

[Dream](https://huggingface.co/spaces/multimodalart/Dream) [![deploy](https://img.shields.io/badge/Hugging%20Face-Demo-yellow)](https://huggingface.co/spaces/multimodalart/Dream)


## Must-Read
D3PM: [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.03006)  

LLaDA: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.09992)
[![Website](https://img.shields.io/badge/Website-9cf)](https://ml-gsai.github.io/LLaDA-demo/)
[![Star](https://img.shields.io/github/stars/ML-GSAI/LLaDA.svg?style=social&label=Star)](https://github.com/ML-GSAI/LLaDA)

[Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.23481)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.23481)
[![Website](https://img.shields.io/badge/Website-9cf)](https://nvlabs.github.io/Fast-dLLM/)
[![Star](https://img.shields.io/github/stars/NVlabs/Fast-dLLM.svg?style=social&label=Star)](https://github.com/NVlabs/Fast-dLLM)



## Surveys
<!-- add ours here! -->
[16 Jun 2025] [Discrete Diffusion in Large Language and Multimodal Models: A Survey](https://arxiv.org/abs/2506.13759)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.13759) [![Star](https://img.shields.io/github/stars/LiQiiiii/DLLM-Survey.svg?style=social&label=Star)](https://github.com/LiQiiiii/DLLM-Survey)

[23 Feb 2024] [Diffusion models in text generation: a survey](https://peerj.com/articles/cs-1905/) (PeerJ Computer Science)  

[5 Jun 2023] [An Overview of Diffusion Models for Text Generation](https://ieeexplore.ieee.org/document/10159911) (MIPRO)  

[24 May 2023] [A Survey of Diffusion Models in Natural Language Processing](https://arxiv.org/abs/2305.14671)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.14671)

[13 Mar 2023] [Diffusion Models in NLP: A Survey](https://arxiv.org/abs/2303.07576)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.07576)

[12 Mar 2023] [Diffusion Models for Non-autoregressive Text Generation: A Survey](https://arxiv.org/abs/2303.06574) (IJCAI 2023)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.06574) [![Star](https://img.shields.io/github/stars/RUCAIBox/Awesome-Text-Diffusion-Models.svg?style=social&label=Star)](https://github.com/RUCAIBox/Awesome-Text-Diffusion-Models)

## Diffusion Foundation
[7 Sep 2022] [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.03003) [![Star](https://img.shields.io/github/stars/gnobitab/RectifiedFlow.svg?style=social&label=Star)](https://github.com/gnobitab/RectifiedFlow)

[26 Nov 2020] [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2011.13456) [![Star](https://img.shields.io/github/stars/yang-song/score_sde.svg?style=social&label=Star)](https://github.com/yang-song/score_sde)


[12 Mar 2023] [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2010.02502) [![Star](https://img.shields.io/github/stars/ermongroup/ddim.svg?style=social&label=Star)](https://github.com/ermongroup/ddim)

[19 Jun 2020] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2006.11239) [![Website](https://img.shields.io/badge/Website-9cf)](https://hojonathanho.github.io/diffusion/) [![Star](https://img.shields.io/github/stars/hojonathanho/diffusion.svg?style=social&label=Star)](https://github.com/hojonathanho/diffusion)

[12 Jul 2019] [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1907.05600) [![Star](https://img.shields.io/github/stars/ermongroup/ncsn.svg?style=social&label=Star)](https://github.com/ermongroup/ncsn)

[12 Mar 2015] [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1503.03585)


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

[7 Jul 2021] [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.03006)  






## Continuous DLMs

## Multimodal DLMs

[29 May 2025] [Muddit: Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model](https://arxiv.org/abs/2505.23606)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.23606) [![Star](https://img.shields.io/github/stars/M-E-AGI-Lab/Muddit.svg?style=social&label=Star)](https://github.com/M-E-AGI-Lab/Muddit) 

[26 May 2025] [FUDOKI: Discrete Flow-based Unified Understanding and Generation via Kinetic-Optimal Velocities](https://arxiv.org/abs/2505.20147)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.20147) [![Website](https://img.shields.io/badge/Website-9cf)](https://fudoki-hku.github.io/)

[22 May 2025] [LaViDa: A Large Diffusion Language Model for Multimodal Understanding](https://arxiv.org/abs/2505.16839)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.16839) [![Website](https://img.shields.io/badge/Website-9cf)](https://homepage.jackli.org/projects/lavida/) [![Star](https://img.shields.io/github/stars/jacklishufan/LaViDa.svg?style=social&label=Star)](https://github.com/jacklishufan/LaViDa) 


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
[6 Jul 2025] [Unveiling the Potential of Diffusion Large Language Model in Controllable Generation](https://arxiv.org/abs/2507.04504)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.04504)

[23 Jun 2025] [Plan for Speed -- Dilated Scheduling for Masked Diffusion Language Models](https://arxiv.org/abs/2506.19037)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.19037)

[12 Jun 2025] [Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles](https://arxiv.org/abs/2506.10848)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.10848) [![Star](https://img.shields.io/github/stars/liangrunflora/slow-fast-sampling.svg?style=social&label=Star)](https://github.com/liangrunflora/slow-fast-sampling)


[12 Jun 2025] [The Diffusion Duality](https://arxiv.org/abs/2506.10892)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.10892) [![Star](https://img.shields.io/github/stars/s-sahoo/duo.svg?style=social&label=Star)](https://github.com/s-sahoo/duo)


[2 Jun 2025] [Esoteric Language Models](https://arxiv.org/abs/2506.01928)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.01928)

[31 May 2025] [Accelerating Diffusion LLMs via Adaptive Parallel Decoding](https://arxiv.org/abs/2506.00413)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.00413)

[30 May 2025] [DLM-One: Diffusion Language Models for One-Step Sequence Generation](https://arxiv.org/abs/2506.00290)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.00290)

[30 May 2025] [Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking](https://arxiv.org/abs/2505.24857)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.24857)

[28 May 2025] [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.23481) 
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.23481)
[![Website](https://img.shields.io/badge/Website-9cf)](https://nvlabs.github.io/Fast-dLLM/)
[![Star](https://img.shields.io/github/stars/NVlabs/Fast-dLLM.svg?style=social&label=Star)](https://github.com/NVlabs/Fast-dLLM)

[28 May 2025] [DINGO: Constrained Inference for Diffusion LLMs](https://arxiv.org/abs/2505.23061)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.23061)

[27 May 2025] [Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion](https://arxiv.org/abs/2505.21467)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.21467)

[22 May 2025] [Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding](https://arxiv.org/abs/2505.16990)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.16990) [![Star](https://img.shields.io/github/stars/yu-rp/Dimple.svg?style=social&label=Star)](https://github.com/yu-rp/Dimple) 

[22 May 2025] [dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](https://arxiv.org/abs/2505.16962)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.16962)
[![Star](https://img.shields.io/github/stars/maomaocun/dLLM-cache.svg?style=social&label=Star)](https://github.com/maomaocun/dLLM-cache)

[21 May 2025] [dKV-Cache: The Cache for Diffusion Language Models](https://arxiv.org/abs/2505.15781)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.15781) [![Star](https://img.shields.io/github/stars/horseee/dkv-cache.svg?style=social&label=Star)](https://github.com/horseee/dkv-cache)




[1 Mar 2025] [Remasking Discrete Diffusion Models with Inference-Time Scaling](https://arxiv.org/abs/2503.00307)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.00307)


[12 Aug 2024] [Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion](https://arxiv.org/abs/2408.05636) (NAACL 2025)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.05636)

## Applications
[11 Jun 2025] [Debunk and Infer: Multimodal Fake News Detection via Diffusion-Generated Evidence and LLM Reasoning](https://arxiv.org/abs/2506.21557)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.21557)


[27 Feb 2025] [EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models](https://arxiv.org/abs/2502.19765)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.19765)

[5 Jan 2025] [DiffusionAttacker: Diffusion-Driven Prompt Manipulation for LLM Jailbreak](https://arxiv.org/abs/2412.17522)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.17522)

[5 Nov 2024] [DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](https://arxiv.org/abs/2411.03250)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.03250)



[30 Oct 2024] [Private Synthetic Text Generation with Diffusion Models](https://arxiv.org/abs/2410.22971)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.22971) [![Star](https://img.shields.io/github/stars/trusthlt/private-synthetic-text-generation.svg?style=social&label=Star)](https://github.com/trusthlt/private-synthetic-text-generation)

[23 Sep 2024] [An Effective Deployment of Diffusion LM for Data Augmentation in Low-Resource Sentiment Classification](https://arxiv.org/abs/2409.03203)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.03203) [![Star](https://img.shields.io/github/stars/johnnychanv/diffusioncls.svg?style=social&label=Star)](https://github.com/johnnychanv/diffusioncls)


[14 Sep 2024] [Towards Diverse and Efficient Audio Captioning via Diffusion Models](https://arxiv.org/abs/2409.09401)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.09401)

[10 Sep 2024] [Table-to-Text Generation with Pretrained Diffusion Models](https://arxiv.org/abs/2409.13739) (IEEE 2024)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.13739)


[24 Oct 2023] [ScanDL: A Diffusion Model for Generating Synthetic Scanpaths on Texts](https://arxiv.org/abs/2310.15587)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.15587) [![Star](https://img.shields.io/github/stars/dili-lab/scandl.svg?style=social&label=Star)](https://github.com/dili-lab/scandl)

[16 Oct 2023] [ForceGen: End-to-end de novo protein generation based on nonlinear mechanical unfolding responses using a protein language diffusion model](https://arxiv.org/abs/2310.10605)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.10605)


[4 Sep 2023] [ParaGuide: Guided Diffusion Paraphrasers for Plug-and-Play Textual Style Transfer](https://arxiv.org/abs/2308.15459)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.15459) [![Star](https://img.shields.io/github/stars/zacharyhorvitz/ParaGuide.svg?style=social&label=Star)](https://github.com/zacharyhorvitz/ParaGuide)

[17 Aug 2023] [Enhancing Phrase Representation by Information Bottleneck Guided Text Diffusion Process for Keyphrase Extraction](https://arxiv.org/abs/2308.08739)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.08739)

[31 Jul 2023] [XDLM: Cross-lingual Diffusion Language Model for Machine Translation](https://arxiv.org/abs/2307.13560)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.13560)

[26 Jul 2023] [Founding a mathematical diffusion model in linguistics. The case study of German syntactic features in the North-Eastern Italian dialects](https://arxiv.org/abs/2307.14291)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.14291)

[14 Jun 2023] [DiffuDetox: A Mixed Diffusion Model for Text Detoxification](https://arxiv.org/abs/2306.08505)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.08505)



[5 Jun 2023] [PLANNER: Generating Diversified Paragraph via Latent Language Diffusion Model](https://arxiv.org/abs/2306.02531)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.02531)

[2 Jun 2023] [DiffusEmp: A Diffusion Model-Based Framework with Multi-Grained Control for Empathetic Response Generation](https://arxiv.org/abs/2306.01657)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.01657)

[31 May 2023] [Fine-grained Text Style Transfer with Diffusion-Based Language Models](https://arxiv.org/abs/2305.19512) (RepL4NLP 2023)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.19512) [![Star](https://img.shields.io/github/stars/lvyiwei1/diffuseq_styleptb.svg?style=social&label=Star)](https://github.com/lvyiwei1/diffuseq_styleptb)

[22 May 2023] [DiffusionNER: Boundary Diffusion for Named Entity Recognition](https://arxiv.org/abs/2305.13289) (ACL 2023)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.13289)

[13 May 2023] [Fine-grained Text Style Transfer with Diffusion-Based Language Models](https://arxiv.org/abs/2305.08111) (RepL4NLP 2023)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.08111)

[9 Jan 2023] [ROIC-DM: Robust Text Inference and Classification via Diffusion Model](https://arxiv.org/abs/2401.03514)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.03514)

[1 Nov 2023] [CodeFusion: A Pre-trained Diffusion Model for Code Generation](https://arxiv.org/abs/2310.17680)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.17680)

[26 Oct 2023] [DiffS2UT: A Semantic Preserving Diffusion Model for Textless Direct Speech-to-Speech Translation](https://arxiv.org/abs/2310.17570)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.17570)


## Resources
[bansky-cl/diffusion-nlp-paper-arxiv](https://github.com/bansky-cl/diffusion-nlp-paper-arxiv) 
 [![Star](https://img.shields.io/github/stars/bansky-cl/diffusion-nlp-paper-arxiv?style=social)](https://github.com/bansky-cl/diffusion-nlp-paper-arxiv)

[bansky-cl/Diffusion-LM-Papers](https://github.com/bansky-cl/Diffusion-LM-Papers) [![Star](https://img.shields.io/github/stars/bansky-cl/Diffusion-LM-Papers?style=social)](https://github.com/bansky-cl/Diffusion-LM-Papers)

[yczhou001/Awesome-Diffusion-LLM](https://github.com/yczhou001/Awesome-Diffusion-LLM) [![Star](https://img.shields.io/github/stars/yczhou001/Awesome-Diffusion-LLM?style=social)](https://github.com/yczhou001/Awesome-Diffusion-LLM)

[StevenYuan666/Awesome-Diffusion-Models-for-NLP](https://github.com/StevenYuan666/Awesome-Diffusion-Models-for-NLP) [![Star](https://img.shields.io/github/stars/StevenYuan666/Awesome-Diffusion-Models-for-NLP?style=social)](https://github.com/StevenYuan666/Awesome-Diffusion-Models-for-NLP)

[LiQiiiii/DLLM-Survey](https://github.com/LiQiiiii/DLLM-Survey) [![Star](https://img.shields.io/github/stars/LiQiiiii/DLLM-Survey?style=social)](https://github.com/LiQiiiii/DLLM-Survey)

[ML-GSAI/Diffusion-LLM-Papers](https://github.com/ML-GSAI/Diffusion-LLM-Papers) [![Star](https://img.shields.io/github/stars/ML-GSAI/Diffusion-LLM-Papers?style=social)](https://github.com/ML-GSAI/Diffusion-LLM-Papers)

[AoiDragon/Awesome-Text-Diffusion-Models](https://github.com/AoiDragon/Awesome-Text-Diffusion-Models) [![Star](https://img.shields.io/github/stars/AoiDragon/Awesome-Text-Diffusion-Models?style=social)](https://github.com/AoiDragon/Awesome-Text-Diffusion-Models)

[kuleshov-group/awesome-discrete-diffusion-models](https://github.com/kuleshov-group/awesome-discrete-diffusion-models) [![Star](https://img.shields.io/github/stars/kuleshov-group/awesome-discrete-diffusion-models?style=social)](https://github.com/kuleshov-group/awesome-discrete-diffusion-models)

[Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/)

[Mecury](https://www.inceptionlabs.ai/introducing-mercury) 
[![Arxiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.17298)
