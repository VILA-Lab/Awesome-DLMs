# Awesome Diffusion Language Models 
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![](https://img.shields.io/github/last-commit/VILA-Lab/Awesome-DLMs?color=green)

## Table of Contents
- [Playground](#playground)
- [Must-Read](#must-read)
- [Surveys](#surveys)
- [Diffusion Foundation](#diffusion-foundation)
    <!-- tocheck -->
- [Discrete DLMs](#discrete-dlms)   
- [Continuous DLMs](#continuous-dlms)
- [Multimodal DLMs](#multimodal-dlms)
- [Training Strategies](#training-strategies)
- [Inference Optimization](#inference-optimization)
- [Applications](#applications)
    <!-- tocheck -->
- [Resources](#resources)


## Playground
- [Mercury](https://chat.inceptionlabs.ai/) [![Static Badge](https://img.shields.io/badge/📰-Demo-green)](https://chat.inceptionlabs.ai/)

- [LLaDA](https://huggingface.co/spaces/multimodalart/LLaDA) [![deploy](https://img.shields.io/badge/Hugging%20Face-Demo-yellow)](https://huggingface.co/spaces/multimodalart/LLaDA)

- [MMaDA](https://huggingface.co/spaces/Gen-Verse/MMaDA) [![deploy](https://img.shields.io/badge/Hugging%20Face-Demo-yellow)](https://huggingface.co/spaces/Gen-Verse/MMaDA)

- [Dream](https://huggingface.co/spaces/multimodalart/Dream) [![deploy](https://img.shields.io/badge/Hugging%20Face-Demo-yellow)](https://huggingface.co/spaces/multimodalart/Dream)


## Must-Read
D3PM: [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.03006)  

LLaDA: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.09992)
[![Website](https://img.shields.io/badge/Website-9cf)](https://ml-gsai.github.io/LLaDA-demo/)
[![Star](https://img.shields.io/github/stars/ML-GSAI/LLaDA.svg?style=social&label=Star)](https://github.com/ML-GSAI/LLaDA)

[Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573) (ICLR 2025)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.09573) 
[![Star](https://img.shields.io/github/stars/kuleshov-group/bd3lms.svg?style=social&label=Star)](https://github.com/kuleshov-group/bd3lms)

[Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.22618)
[![Website](https://img.shields.io/badge/Website-9cf)](https://nvlabs.github.io/Fast-dLLM/)
[![Star](https://img.shields.io/github/stars/NVlabs/Fast-dLLM.svg?style=social&label=Star)](https://github.com/NVlabs/Fast-dLLM)



## Surveys
<!-- add ours here! -->
[16 Jun 2025] [Discrete Diffusion in Large Language and Multimodal Models: A Survey](https://arxiv.org/abs/2506.13759)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.13759)
[![Star](https://img.shields.io/github/stars/LiQiiiii/DLLM-Survey.svg?style=social&label=Star)](https://github.com/LiQiiiii/DLLM-Survey)

[23 Feb 2024] [Diffusion models in text generation: a survey](https://peerj.com/articles/cs-1905/) (PeerJ Computer Science)  

[29 Jun 2023] [An Overview of Diffusion Models for Text Generation](https://ieeexplore.ieee.org/document/10159911) (MIPRO)  

[24 May 2023] [A Survey of Diffusion Models in Natural Language Processing](https://arxiv.org/abs/2305.14671)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.14671)

[14 Mar 2023] [Diffusion Models in NLP: A Survey](https://arxiv.org/abs/2303.07576)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.07576)

[12 Mar 2023] [Diffusion Models for Non-autoregressive Text Generation: A Survey](https://arxiv.org/abs/2303.06574) (IJCAI 2023)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.06574)
[![Star](https://img.shields.io/github/stars/RUCAIBox/Awesome-Text-Diffusion-Models.svg?style=social&label=Star)](https://github.com/RUCAIBox/Awesome-Text-Diffusion-Models)

## Diffusion Foundation

[7 Sep 2022] [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)   (ICLR 2023)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2209.03003)
[![Star](https://img.shields.io/github/stars/gnobitab/RectifiedFlow.svg?style=social&label=Star)](https://github.com/gnobitab/RectifiedFlow)

[26 Nov 2020] [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) (ICLR 2021)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2011.13456)
[![Star](https://img.shields.io/github/stars/yang-song/score_sde.svg?style=social&label=Star)](https://github.com/yang-song/score_sde)

[6 Oct 2020] [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (ICLR 2021)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2010.02502)
[![Star](https://img.shields.io/github/stars/ermongroup/ddim.svg?style=social&label=Star)](https://github.com/ermongroup/ddim)

[19 Jun 2020] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (NeurIPS 2020)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2006.11239)
[![Website](https://img.shields.io/badge/Website-9cf)](https://hojonathanho.github.io/diffusion/)
[![Star](https://img.shields.io/github/stars/hojonathanho/diffusion.svg?style=social&label=Star)](https://github.com/hojonathanho/diffusion)

[12 Jul 2019] [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) (NeurIPS 2019)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1907.05600)
[![Star](https://img.shields.io/github/stars/ermongroup/ncsn.svg?style=social&label=Star)](https://github.com/ermongroup/ncsn)

[12 Mar 2015] [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) (ICML 2015)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1503.03585)
[![Star](https://img.shields.io/github/stars/Sohl-Dickstein/Diffusion-Probabilistic-Models.svg?style=social&label=Star)](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models)



## Discrete DLMs

[25 Jul 2025] [Jailbreaking Large Language Diffusion Models: Revealing Hidden Safety Flaws in Diffusion-Based Text Generation](https://arxiv.org/abs/2507.19227v1)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.19227v1)

[15 Jul 2025] [The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs](https://arxiv.org/abs/2507.11097v1))<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.11097v1)
[![Star](https://img.shields.io/github/stars/hojonathanho/diffusion.svg?style=social&label=Star)](https://github.com/hojonathanho/diffusion)

[10 Jul 2025] [Your Absorbing Discrete Diffusion Secretly Models the Bayesian Posterior](https://arxiv.org/abs/2507.07586)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.07586)
[![Star](https://img.shields.io/github/stars/mercury0100/bayesradd.svg?style=social&label=Star)](https://github.com/mercury0100/bayesradd)

[7 Jul 2025] [Review, Remask, Refine (R3): Process-Guided Block Diffusion for Text Generation](https://arxiv.org/abs/2507.08018v1) (ICML 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.08018v1)

[6 Jul 2025] [Efficient perplexity bound and ratio matching in discrete diffusion language models](https://arxiv.org/abs/2507.04341) (ICLR 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.04341)
[![Star](https://img.shields.io/github/stars/MetaDialog-Research/PBRC.svg?style=social&label=Star)](https://github.com/MetaDialog-Research/PBRC)

[2 Jul 2025] [Discrete Diffusion Models for Language Generation](https://arxiv.org/abs/2507.07050)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.07050)
[![Star](https://img.shields.io/github/stars/AshenWELI/Discrete-Diffusion-Models-for-Language-Genaration.svg?style=social&label=Star)](https://github.com/AshenWELI/Discrete-Diffusion-Models-for-Language-Genaration)

[17 Jun 2025] [LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs](https://arxiv.org/abs/2506.14429)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.14429)
[![Star](https://img.shields.io/github/stars/OpenMOSS/LongLLaDA.svg?style=social&label=Star)](https://github.com/OpenMOSS/LongLLaDA)

[12 Jun 2025] [The Diffusion Duality](https://arxiv.org/abs/2506.10892) (ICML 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.10892)
[![Website](https://img.shields.io/badge/Website-9cf)](https://s-sahoo.com/duo/)
[![Star](https://img.shields.io/github/stars/s-sahoo/duo.svg?style=social&label=Star)](https://github.com/s-sahoo/duo)

[2 Jun 2025] [Esoteric Language Models](https://arxiv.org/abs/2506.01928)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.01928)
[![Website](https://img.shields.io/badge/Website-9cf)](https://s-sahoo.com/Eso-LMs/)
[![Star](https://img.shields.io/github/stars/s-sahoo/Eso-LMs.svg?style=social&label=Star)](https://github.com/s-sahoo/Eso-LMs)

[25 May 2025] [LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models](https://arxiv.org/abs/2505.19223)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.19223)
[![Website](https://img.shields.io/badge/Website-9cf)](https://ml-gsai.github.io/LLaDA-1.5-Demo/)
[![Star](https://img.shields.io/github/stars/ML-GSAI/LLaDA-1.5.svg?style=social&label=Star)](https://github.com/ML-GSAI/LLaDA-1.5)

[24 May 2025] [Anchored Diffusion Language Model](https://arxiv.org/abs/2505.18456)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.18456)

[21 May 2025] [Diffusion vs. Autoregressive Language Models: A Text Embedding Perspective](https://arxiv.org/abs/2505.15045)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.15045)

[20 May 2025] [CtrlDiff: Boosting Large Diffusion Language Models with Dynamic Block Prediction and Controllable Generation](https://arxiv.org/abs/2505.14455)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.14455)

[9 May 2025] [Insertion Language Models: Sequence Generation with Arbitrary-Position Insertions](https://arxiv.org/abs/2505.05755)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.05755)

[22 Apr 2025] [Target Concrete Score Matching: A Holistic Framework for Discrete Diffusion](https://arxiv.org/abs/2504.16431) (ICML 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2504.16431)

[2 Apr 2025] [Dream 7B](https://hkunlp.github.io/blog/2025/dream/)<br>
[![Website](https://img.shields.io/badge/Website-9cf)](https://hkunlp.github.io/blog/2025/dream/)
[![Star](https://img.shields.io/github/stars/HKUNLP/Dream.svg?style=social&label=Star)](https://github.com/HKUNLP/Dream)

[16 Mar 2025] [State Fourier Diffusion Language Model (SFDLM): A Scalable, Novel Iterative Approach to Language Modeling](https://arxiv.org/abs/2503.17382)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.17382)

[12 Mar 2025] [Constrained Discrete Diffusion](https://arxiv.org/abs/2503.09790)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.09790)


[12 Mar 2025] [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573) (ICLR 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.09573)
[![Website](https://img.shields.io/badge/Website-9cf)](https://m-arriola.com/bd3lms)
[![Star](https://img.shields.io/github/stars/kuleshov-group/bd3lms.svg?style=social&label=Star)](https://github.com/kuleshov-group/bd3lms)

[11 Mar 2025] [Understanding the Quality-Diversity Trade-off in Diffusion Language Models](https://arxiv.org/abs/2503.10683)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.10683)
[![Star](https://img.shields.io/github/stars/zzbuzzard/guidediffuseq.svg?style=social&label=Star)](https://github.com/zzbuzzard/guidediffuseq)

[6 Mar 2025] [Generalized Interpolating Discrete Diffusion](https://arxiv.org/abs/2503.04482) (ICML 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.04482)
[![Star](https://img.shields.io/github/stars/dvruette/gidd.svg?style=social&label=Star)](https://github.com/dvruette/gidd)

[14 Feb 2025] [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.09992)
[![Website](https://img.shields.io/badge/Website-9cf)](https://ml-gsai.github.io/LLaDA-demo/)
[![Star](https://img.shields.io/github/stars/ML-GSAI/LLaDA.svg?style=social&label=Star)](https://github.com/ML-GSAI/LLaDA) 

[13 Feb 2025] [Theoretical Benefit and Limitation of Diffusion Language Model](https://arxiv.org/abs/2502.09622)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.09622)

[13 Feb 2025] [Non-Markovian Discrete Diffusion with Causal Language Models](https://arxiv.org/abs/2502.09767)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.09767)

[10 Feb 2025] [Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions](https://arxiv.org/abs/2502.06768) (ICML 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.06768)

[10 Nov 2024] [Conditional [MASK] Discrete Diffusion Language Model](https://arxiv.org/abs/2411.06438v5)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.06438v5)

[28 Oct 2024] [Beyond Autoregression: Fast LLMs via Self-Distillation Through Time](https://arxiv.org/abs/2410.21035) (ICLR 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.21035)
[![Website](https://img.shields.io/badge/Website-9cf)](https://jdeschena.github.io/sdtt-blog/)
[![Star](https://img.shields.io/github/stars/jdeschena/sdtt.svg?style=social&label=Star)](https://github.com/jdeschena/sdtt)

[28 Oct 2024] [Energy-Based Diffusion Language Models for Text Generation](https://arxiv.org/abs/2410.21357) (ICLR 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.21357)
[![Star](https://img.shields.io/github/stars/MinkaiXu/Energy-Diffusion-LLM.svg?style=social&label=Star)](https://github.com/MinkaiXu/Energy-Diffusion-LLM)

[24 Oct 2024] [Scaling up Masked Diffusion Models on Text](https://arxiv.org/abs/2410.18514) (ICLR 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.18514)
[![Star](https://img.shields.io/github/stars/ml-gsai/smdm.svg?style=social&label=Star)](https://github.com/ml-gsai/smdm)

[23 Oct 2024] [Scaling Diffusion Language Models via Adaptation from Autoregressive Models](https://arxiv.org/abs/2410.17891) (ICLR 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.17891)
[![Star](https://img.shields.io/github/stars/hkunlp/diffullama.svg?style=social&label=Star)](https://github.com/hkunlp/diffullama)

[18 Oct 2024] [Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning](https://arxiv.org/abs/2410.14157) (ICLR 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.14157)
[![Star](https://img.shields.io/github/stars/HKUNLP/diffusion-vs-ar.svg?style=social&label=Star)](https://github.com/HKUNLP/diffusion-vs-ar)

[8 Oct 2024] (DDPD) [Think While You Generate: Discrete Diffusion with Planned Denoising](https://arxiv.org/abs/2410.06264) (ICLR 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.06264)
[![Star](https://img.shields.io/github/stars/liusulin/DDPD.svg?style=social&label=Star)](https://github.com/liusulin/DDPD)

[4 Sep 2024] [Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling](https://arxiv.org/abs/2409.02908) (ICLR 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.02908)

[22 Jul 2024] [Discrete Flow Matching](https://arxiv.org/abs/2407.15595) (NeurIPS 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.15595)

[10 Jul 2024][Promises, Outlooks and Challenges of Diffusion Language Modeling](https://arxiv.org/abs/2406.11473)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.11473)

[11 Jun 2024] (MDLM) [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) (NeurIPS 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.07524)
[![Website](https://img.shields.io/badge/Website-9cf)](https://s-sahoo.com/mdlm/)
[![Star](https://img.shields.io/github/stars/kuleshov-group/mdlm.svg?style=social&label=Star)](https://github.com/kuleshov-group/mdlm)

[6 Jun 2024] (RADD) [Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data](https://arxiv.org/abs/2406.03736) (ICLR 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.03736)

[6 Jun 2024] (MD4) [Simplified and Generalized Masked Diffusion for Discrete Data](https://arxiv.org/abs/2406.04329) (NeurIPS 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.04329)
[![Star](https://img.shields.io/github/stars/google-deepmind/md4.svg?style=social&label=Star)](https://github.com/google-deepmind/md4)

[7 Feb 2024] [Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design](https://arxiv.org/abs/2402.04997) (ICML 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.04997)

[30 Jan 2024] [Transfer Learning for Text Diffusion Models](https://arxiv.org/abs/2401.17181)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.17181)

[25 Oct 2023] (SEDD) [Discrete Diffusion Language Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834) (ICML 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.16834)
[![Star](https://img.shields.io/github/stars/louaaron/Score-Entropy-Discrete-Diffusion.svg?style=social&label=Star)](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)

[15 Oct 2023] [FiLM: Fill-in Language Models for Any-Order Generation](https://arxiv.org/abs/2310.09930)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.09930)
[![Star](https://img.shields.io/github/stars/shentianxiao/FiLM.svg?style=social&label=Star)](https://github.com/shentianxiao/FiLM)

[23 Aug 2023] [Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning](https://arxiv.org/abs/2308.12219)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.12219)
[![Star](https://img.shields.io/github/stars/yegcjs/diffusionllm.svg?style=social&label=Star)](https://github.com/yegcjs/diffusionllm)

[30 May 2023] [Likelihood-Based Diffusion Language Models](https://arxiv.org/abs/2305.18619) (NeurIPS 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.18619)
[![Star](https://img.shields.io/github/stars/igul222/plaid.svg?style=social&label=Star)](https://github.com/igul222/plaid)

[6 May 2023] [Diffusion-NAT: Self-Prompting Discrete Diffusion for Non-Autoregressive Text Generation](https://arxiv.org/abs/2305.04044) (EACL 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.04044) [![Star](https://img.shields.io/github/stars/Lancelot39/DiffusionNAT.svg?style=social&label=Star)](https://github.com/Lancelot39/DiffusionNAT)

[11 Feb 2023] [A Reparameterized Discrete Diffusion Model for Text Generation](https://arxiv.org/abs/2302.05737) (COLM 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.05737)
[![Star](https://img.shields.io/github/stars/HKUNLP/reparam-discrete-diffusion.svg?style=social&label=Star)](https://github.com/HKUNLP/reparam-discrete-diffusion)

[28 Nov 2022] [DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models](https://arxiv.org/abs/2211.15029) (ACL 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.15029)
[![Star](https://img.shields.io/github/stars/Hzfinfdu/Diffusion-BERT.svg?style=social&label=Star)](https://github.com/Hzfinfdu/Diffusion-BERT)

<!-- [2 Nov 2022] [Concrete Score Matching: Generalized Score Matching for Discrete Data](https://arxiv.org/abs/2211.00802) (NeurIPS 2022)
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.00802) -->

[30 Oct 2022] [DiffusER: Discrete Diffusion via Edit-based Reconstruction](https://arxiv.org/abs/2210.16886) (ICLR 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.16886)

<!-- [30 May 2022] [A Continuous Time Framework for Discrete Denoising Models](https://arxiv.org/abs/2205.14987) (NeurIPS 2022)  
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.14987)  -->

[13 Dec 2021] (SUNDAE) [Step-unrolled Denoising Autoencoders for Text Generation](https://arxiv.org/abs/2112.06749) (ICLR 2022)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2112.06749)

[7 Jul 2021] [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) (NeurIPS 2021)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2107.03006)

[10 Feb 2021] [Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions](https://arxiv.org/abs/2102.05379) (NeurIPS 2021)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2102.05379)
[![Star](https://img.shields.io/github/stars/didriknielsen/argmax_flows.svg?style=social&label=Star)](https://github.com/didriknielsen/argmax_flows)


## Continuous DLMs
[26 Jun 2025] [Compressed and Smooth Latent Space for Text Diffusion Modeling](https://arxiv.org/abs/2506.21170)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.21170)

[28 May 2025] [Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes](https://arxiv.org/abs/2505.22165) (ACL 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.22165)

[24 May 2025] [Smoothie: Smoothing Diffusion on Token Embeddings for Text Generation](https://arxiv.org/abs/2505.18853)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.18853)
[![Star](https://img.shields.io/github/stars/ashaba1in/smoothie.svg?style=social&label=Star)](https://github.com/ashaba1in/smoothie)

[19 Feb 2025] [TESS 2: A Large-Scale Generalist Diffusion Language Model](https://arxiv.org/abs/2502.13917)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.13917)
[![Star](https://img.shields.io/github/stars/hamishivi/tess-2.svg?style=social&label=Star)](https://github.com/hamishivi/tess-2)

[15 Dec 2024] [Segment-Level Diffusion: A Framework for Controllable Long-Form Generation with Diffusion Language Models](https://arxiv.org/abs/2412.11333) (ACL 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.11333)

[17 Oct 2024] [Meta-DiffuB: A Contextualized Sequence-to-Sequence Text Diffusion Model with Meta-Exploration](https://arxiv.org/abs/2410.13201) (NeurIPS 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.13201)
[![Star](https://img.shields.io/github/stars/meta-diffub/meta-diffub.svg?style=social&label=Star)](https://github.com/meta-diffub/meta-diffub)

[8 Aug 2024] [Diffusion Guided Language Modeling](https://arxiv.org/abs/2408.04220) (ACL Findings 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.04220)
[![Star](https://img.shields.io/github/stars/justinlovelace/diffusion-guided-lm.svg?style=social&label=Star)](https://github.com/justinlovelace/diffusion-guided-lm) 

[May 2024] [Effective Integration of Text Diffusion and Pre-Trained Language Models with Linguistic Easy-First Schedule](https://aclanthology.org/2024.lrec-main.493/) (LREC-COLING 2024)

[17 Mar 2024] [Language Rectified Flow: Advancing Diffusion Language Generation with Probabilistic Flows](https://arxiv.org/abs/2403.16995) (NAACL 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.16995)

[14 Mar 2024] [LDSeq: Latent Diffusion Models for Sequence to Sequence Text Generation](https://dl.acm.org/doi/10.1145/3638584.3638617) (CSAI 23)

[Mar 2024] [Flow Matching for Conditional Text Generation in a Few Sampling Steps](https://aclanthology.org/2024.eacl-short.33/) (EACL 2024)  

[29 Feb 2024] [TEncDM: Understanding the Properties of Diffusion Model in the Space of Language Model Encodings](https://arxiv.org/abs/2402.19097)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.19097)
[![Star](https://img.shields.io/github/stars/M0RJIQUE/tencdm.svg?style=social&label=Star)](https://github.com/M0RJIQUE/tencdm)

[29 Feb 2024] [Generating, Reconstructing, and Representing Discrete and Continuous Data: Generalized Diffusion with Learnable Encoding-Decoding](https://arxiv.org/abs/2402.19009) (ICML 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.19009)

[31 Oct 2023] [LADIDA: Latent Diffusion for Document Generation with Sequential Decoding](https://neurips.cc/virtual/2023/74876) (NeurIPS Workshop 2023)  

[18 Oct 2023] [InfoDiffusion: Information Entropy Aware Diffusion Process for Non-Autoregressive Text Generation](https://arxiv.org/abs/2310.11976) (EMNLP 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.11976)
[![Star](https://img.shields.io/github/stars/rzhwang/infodiffusion.svg?style=social&label=Star)](https://github.com/rzhwang/infodiffusion)

[09 Oct 2023] [DiffuSeq-v2: Bridging Discrete and Continuous Text Spaces for Accelerated Seq2Seq Diffusion Models](https://arxiv.org/abs/2310.05793) (EMNLP 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.05793)
[![Star](https://img.shields.io/github/stars/Shark-NLP/DiffuSeq.svg?style=social&label=Star)](https://github.com/Shark-NLP/DiffuSeq)

[26 Jul 2023] [How Does Diffusion Influence Pretrained Language Models on Out-of-Distribution Data?](https://arxiv.org/abs/2307.13949) (ECAI 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.13949)
[![Star](https://img.shields.io/github/stars/maybelizzy/diffusion_ood_robustness.svg?style=social&label=Star)](https://github.com/maybelizzy/diffusion_ood_robustness)

[19 May 2023] [DiffuSIA: A Spiral Interaction Architecture for Encoder-Decoder Text Diffusion](https://arxiv.org/abs/2305.11517)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.11517)

[16 May 2023] [AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation](https://arxiv.org/abs/2305.09515) (NeurIPS 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.09515)
[![Star](https://img.shields.io/github/stars/microsoft/ProphetNet.svg?style=social&label=Star)](https://github.com/microsoft/ProphetNet/tree/master/AR-diffusion)

[15 May 2023] [TESS: Text-to-Text Self-Conditioned Simplex Diffusion](https://arxiv.org/abs/2305.08379) (EACL 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.08379)
[![Star](https://img.shields.io/github/stars/allenai/tess-diffusion.svg?style=social&label=Star)](https://github.com/allenai/tess-diffusion)

[25 Apr 2023] [Glyphdiffusion: Text generation as image generation](https://arxiv.org/abs/2304.12519)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.12519)

[10 Apr 2023] [A Cheaper and Better Diffusion Language Model with Soft-Masked Noise](https://arxiv.org/abs/2304.04746) (EMNLP 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.04746)
[![Star](https://img.shields.io/github/stars/amazon-science/masked-diffusion-lm.svg?style=social&label=Star)](https://github.com/amazon-science/masked-diffusion-lm)

[20 Feb 2023] [Dinoiser: Diffused conditional sequence learning by manipulating noises](https://arxiv.org/abs/2302.10025) (TCAL 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.10025)
[![Star](https://img.shields.io/github/stars/yegcjs/DINOISER.svg?style=social&label=Star)](https://github.com/yegcjs/DINOISER)

[22 Dec 2022] (GENIE) [Text Generation with Diffusion Language Models: A Pre-training Approach with Continuous Paragraph Denoise](https://arxiv.org/abs/2212.11685) (ICML 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.11685)
[![Star](https://img.shields.io/github/stars/microsoft/ProphetNet.svg?style=social&label=Star)](https://github.com/microsoft/ProphetNet/tree/master/GENIE)

[20 Dec 2022] [Seqdiffuseq: Text diffusion with encoder-decoder transformers](https://arxiv.org/abs/2212.10325) (NAACL 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.10325)[![Star](https://img.shields.io/github/stars/Yuanhy1997/SeqDiffuSeq.svg?style=social&label=Star)](https://github.com/Yuanhy1997/SeqDiffuSeq)

[19 Dec 2022] [Latent Diffusion for Language Generation](https://arxiv.org/abs/2212.09462) (NeurIPS 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.09462)
[![Star](https://img.shields.io/github/stars/justinlovelace/latent-diffusion-for-language.svg?style=social&label=Star)](https://github.com/justinlovelace/latent-diffusion-for-language)

[19 Dec 2022] (Difformer) [Empowering Diffusion Models on the Embedding Space for Text Generation](https://arxiv.org/abs/2212.09412) (NAACL 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.09412)
[![Star](https://img.shields.io/github/stars/zhjgao/difformer.svg?style=social&label=Star)](https://github.com/zhjgao/difformer)

[28 Nov 2022] [Continuous diffusion for categorical data](https://arxiv.org/abs/2211.15089)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.15089)

[8 Nov 2022] [Self-conditioned Embedding Diffusion for Text Generation](https://arxiv.org/abs/2211.04236)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.04236)

[31 Oct 2022] [SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control](https://arxiv.org/abs/2210.17432) (ACL 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.17432)
[![Star](https://img.shields.io/github/stars/xhan77/ssd-lm.svg?style=social&label=Star)](https://github.com/xhan77/ssd-lm)

[17 Oct 2022] [DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models](https://arxiv.org/abs/2210.08933) (ICLR 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.08933)
[![Star](https://img.shields.io/github/stars/Shark-NLP/DiffuSeq.svg?style=social&label=Star)](https://github.com/Shark-NLP/DiffuSeq)

[1 Aug 2022] [Composable Text Controls in Latent Space with ODEs](https://arxiv.org/abs/2208.00638) (EMNLP 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2208.00638)
[![Star](https://img.shields.io/github/stars/guangyliu/LatentOps.svg?style=social&label=Star)](https://github.com/guangyliu/LatentOps)

[13 Jun 2022] [Latent Diffusion Energy-Based Model for Interpretable Text Modeling](https://arxiv.org/abs/2206.05895) (ICML 2022)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.05895)
[![Star](https://img.shields.io/github/stars/yuPeiyu98/Latent-Diffusion-EBM.svg?style=social&label=Star)](https://github.com/yuPeiyu98/Latent-Diffusion-EBM)

[27 May 2022] [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217) (NeurIPS 2022)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.14217)
[![Star](https://img.shields.io/github/stars/XiangLi1999/Diffusion-LM.svg?style=social&label=Star)](https://github.com/XiangLi1999/Diffusion-LM)




## Multimodal DLMs

[29 May 2025] [Muddit: Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model](https://arxiv.org/abs/2505.23606)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.23606)
[![Star](https://img.shields.io/github/stars/M-E-AGI-Lab/Muddit.svg?style=social&label=Star)](https://github.com/M-E-AGI-Lab/Muddit) 

[26 May 2025] [FUDOKI: Discrete Flow-based Unified Understanding and Generation via Kinetic-Optimal Velocities](https://arxiv.org/abs/2505.20147)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.20147)
[![Website](https://img.shields.io/badge/Website-9cf)](https://fudoki-hku.github.io/)
[![Star](https://img.shields.io/github/stars/fudoki-hku/FUDOKI.svg?style=social&label=Star)](https://github.com/fudoki-hku/FUDOKI)

[22 May 2025] [LaViDa: A Large Diffusion Language Model for Multimodal Understanding](https://arxiv.org/abs/2505.16839)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.16839)
[![Website](https://img.shields.io/badge/Website-9cf)](https://homepage.jackli.org/projects/lavida/)
[![Star](https://img.shields.io/github/stars/jacklishufan/LaViDa.svg?style=social&label=Star)](https://github.com/jacklishufan/LaViDa) 


[22 May 2025] [Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding](https://arxiv.org/abs/2505.16990)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.16990)
[![Star](https://img.shields.io/github/stars/yu-rp/Dimple.svg?style=social&label=Star)](https://github.com/yu-rp/Dimple) 

[22 May 2025] [LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning](https://arxiv.org/abs/2505.16933)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.16933)
[![Website](https://img.shields.io/badge/Website-9cf)](https://ml-gsai.github.io/LLaDA-V-demo/)
[![Star](https://img.shields.io/github/stars/ML-GSAI/LLaDA-V.svg?style=social&label=Star)](https://github.com/ML-GSAI/LLaDA-V) 

[21 May 2025] [MMaDA: Multimodal Large Diffusion Language Models](https://arxiv.org/abs/2505.15809)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.15809)
[![Star](https://img.shields.io/github/stars/Gen-Verse/MMaDA.svg?style=social&label=Star)](https://github.com/Gen-Verse/MMaDA) 

[26 Mar 2025] [Unified Multimodal Discrete Diffusion](https://arxiv.org/abs/2503.20853)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.20853)
[![Website](https://img.shields.io/badge/Website-9cf)](https://unidisc.github.io)
[![Star](https://img.shields.io/github/stars/alexanderswerdlow/unidisc.svg?style=social&label=Star)](https://github.com/alexanderswerdlow/unidisc)



## Training Strategies
[7 Jul 2025] [wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](https://arxiv.org/abs/2507.08838)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.08838) 
[![Star](https://img.shields.io/github/stars/xiaohangt/wd1.svg?style=social&label=Star)](https://github.com/xiaohangt/wd1)

[25 Jun 2025] [DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation](https://arxiv.org/abs/2506.20639)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.20639)
[![Star](https://img.shields.io/github/stars/apple/ml-diffucoder.svg?style=social&label=Star)](https://github.com/apple/ml-diffucoder)

[25 May 2025] [LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models](https://arxiv.org/abs/2505.19223) <br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.19223)
[![Website](https://img.shields.io/badge/Website-9cf)](https://ml-gsai.github.io/LLaDA-1.5-Demo/)
[![Star](https://img.shields.io/github/stars/ML-GSAI/LLaDA-1.5.svg?style=social&label=Star)](https://github.com/ML-GSAI/LLaDA-1.5)

[21 May 2025] [MMaDA: Multimodal Large Diffusion Language Models](https://arxiv.org/abs/2505.15809)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.15809)
[![Star](https://img.shields.io/github/stars/Gen-Verse/MMaDA.svg?style=social&label=Star)](https://github.com/Gen-Verse/MMaDA) 

[15 May 2025] [Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](https://arxiv.org/abs/2505.10446)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.10446)

[16 Apr 2025] [d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2504.12216)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2504.12216)
[![Website](https://img.shields.io/badge/Website-9cf)](https://dllm-reasoning.github.io/)
[![Star](https://img.shields.io/github/stars/dllm-reasoning/d1.svg?style=social&label=Star)](https://github.com/dllm-reasoning/d1)

[2 Apr 2025] [Dream 7B](https://hkunlp.github.io/blog/2025/dream/)<br>
[![Website](https://img.shields.io/badge/Website-9cf)](https://hkunlp.github.io/blog/2025/dream/)
[![Star](https://img.shields.io/github/stars/HKUNLP/Dream.svg?style=social&label=Star)](https://github.com/HKUNLP/Dream)

[3 Feb 2025] [Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods](https://arxiv.org/abs/2502.01384)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.01384)
[![Star](https://img.shields.io/github/stars/ozekri/SEPO.svg?style=social&label=Star)](https://github.com/ozekri/SEPO)

[Jan 2025] [Addressing the Training-Inference Discrepancy in Discrete Diffusion for Text Generation](https://aclanthology.org/2025.coling-main.477/) (COLING 2025)<br>
[![Star](https://img.shields.io/github/stars/aistairc/text-diff-2step-loss.svg?style=social&label=Star)](https://github.com/aistairc/text-diff-2step-loss)

[23 Oct 2024] [Scaling Diffusion Language Models via Adaptation from Autoregressive Models](https://arxiv.org/abs/2410.17891) (ICLR 2025) <br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.17891)
[![Star](https://img.shields.io/github/stars/hkunlp/diffullama.svg?style=social&label=Star)](https://github.com/hkunlp/diffullama)

[17 Oct 2024] [Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design](https://arxiv.org/abs/2410.13643) (ICLR 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.13643) [![Star](https://img.shields.io/github/stars/ChenyuWang-Monica/DRAKES.svg?style=social&label=Star)](https://github.com/ChenyuWang-Monica/DRAKES)

[19 Feb 2024] [Text Diffusion with Reinforced Conditioning](https://arxiv.org/abs/2402.14843)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.14843)


[12 Feb 2024] [Diffusion of Thought: Chain-of-Thought Reasoning in Diffusion Language Models](https://arxiv.org/abs/2402.07754) (NeurIPS 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.07754)
[![Star](https://img.shields.io/github/stars/hkunlp/diffusion-of-thoughts.svg?style=social&label=Star)](https://github.com/hkunlp/diffusion-of-thoughts)

[8 May 2023] [Can Diffusion Model Achieve Better Performance in Text Generation? Bridging the Gap between Training and Inference!](https://arxiv.org/abs/2305.04465) (ACL 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.04465)
[![Star](https://img.shields.io/github/stars/ZetangForward/Bridge_Gap_Diffusion.svg?style=social&label=Star)](https://github.com/ZetangForward/Bridge_Gap_Diffusion)



## Inference Optimization
[11 Jul 2025] [Inference-Time Scaling of Diffusion Language Models with Particle Gibbs Sampling](https://arxiv.org/abs/2507.08390v1)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.08390v1) 

[6 Jul 2025] [Unveiling the Potential of Diffusion Large Language Model in Controllable Generation](https://arxiv.org/abs/2507.04504)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2507.04504)

[23 Jun 2025] [Plan for Speed -- Dilated Scheduling for Masked Diffusion Language Models](https://arxiv.org/abs/2506.19037)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.19037)

[12 Jun 2025] [Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles](https://arxiv.org/abs/2506.10848)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.10848)
[![Star](https://img.shields.io/github/stars/liangrunflora/slow-fast-sampling.svg?style=social&label=Star)](https://github.com/liangrunflora/slow-fast-sampling)


[12 Jun 2025] [The Diffusion Duality](https://arxiv.org/abs/2506.10892) (ICML 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.10892)
[![Website](https://img.shields.io/badge/Website-9cf)](https://s-sahoo.com/duo/)
[![Star](https://img.shields.io/github/stars/s-sahoo/duo.svg?style=social&label=Star)](https://github.com/s-sahoo/duo)

[2 Jun 2025] [Esoteric Language Models](https://arxiv.org/abs/2506.01928)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.01928)
[![Website](https://img.shields.io/badge/Website-9cf)](https://s-sahoo.com/Eso-LMs/)
[![Star](https://img.shields.io/github/stars/s-sahoo/Eso-LMs.svg?style=social&label=Star)](https://github.com/s-sahoo/Eso-LMs)

[31 May 2025] [Accelerating Diffusion LLMs via Adaptive Parallel Decoding](https://arxiv.org/abs/2506.00413)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.00413)

[30 May 2025] [DLM-One: Diffusion Language Models for One-Step Sequence Generation](https://arxiv.org/abs/2506.00290)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.00290)

[30 May 2025] [Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking](https://arxiv.org/abs/2505.24857)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.24857)

[28 May 2025] [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.22618)
[![Website](https://img.shields.io/badge/Website-9cf)](https://nvlabs.github.io/Fast-dLLM/)
[![Star](https://img.shields.io/github/stars/NVlabs/Fast-dLLM.svg?style=social&label=Star)](https://github.com/NVlabs/Fast-dLLM)

[28 May 2025] [DINGO: Constrained Inference for Diffusion LLMs](https://arxiv.org/abs/2505.23061)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.23061)

[27 May 2025] [Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion](https://arxiv.org/abs/2505.21467)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.21467)

[22 May 2025] [Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding](https://arxiv.org/abs/2505.16990)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.16990)
[![Star](https://img.shields.io/github/stars/yu-rp/Dimple.svg?style=social&label=Star)](https://github.com/yu-rp/Dimple) 

[17 May 2025] [dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](https://arxiv.org/abs/2506.06295)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.06295)
[![Star](https://img.shields.io/github/stars/maomaocun/dLLM-cache.svg?style=social&label=Star)](https://github.com/maomaocun/dLLM-cache)

[21 May 2025] [dKV-Cache: The Cache for Diffusion Language Models](https://arxiv.org/abs/2505.15781)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.15781)
[![Star](https://img.shields.io/github/stars/horseee/dkv-cache.svg?style=social&label=Star)](https://github.com/horseee/dkv-cache)

[1 Mar 2025] [Remasking Discrete Diffusion Models with Inference-Time Scaling](https://arxiv.org/abs/2503.00307) (ICLR 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.00307)
[![Website](https://img.shields.io/badge/Website-9cf)](https://remdm.github.io/)
[![Star](https://img.shields.io/github/stars/kuleshov-group/remdm.svg?style=social&label=Star)](https://github.com/kuleshov-group/remdm)

[11 Oct 2024] [Distillation of Discrete Diffusion through Dimensional Correlations](https://arxiv.org/abs/2410.08709) (ICML 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.08709)
[![Star](https://img.shields.io/github/stars/sony/di4c.svg?style=social&label=Star)](https://github.com/sony/di4c)

[8 Oct 2024] (DDPD) [Think While You Generate: Discrete Diffusion with Planned Denoising](https://arxiv.org/abs/2410.06264)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.06264)
[![Star](https://img.shields.io/github/stars/liusulin/DDPD.svg?style=social&label=Star)](https://github.com/liusulin/DDPD)

[Nov 2024] [Enable Fast Sampling for Seq2Seq Text Diffusion](https://aclanthology.org/2024.findings-emnlp.497/) (EMNLP Findings 2024)<br>
[![Anthology](https://img.shields.io/badge/Anthology-000000.svg)](https://aclanthology.org/2024.findings-emnlp.497/)
[![Star](https://img.shields.io/github/stars/Peacer68/FMSeq.svg?style=social&label=Star)](https://github.com/Peacer68/FMSeq)

[10 Aug 2024] [Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion](https://arxiv.org/abs/2408.05636) (NAACL 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.05636)

[May 2024] [Few-shot Temporal Pruning Accelerates Diffusion Models for Text Generation](https://aclanthology.org/2024.lrec-main.637/) (LREC-COLING 2024)<br>
[![Anthology](https://img.shields.io/badge/Anthology-000000.svg)](https://aclanthology.org/2024.lrec-main.637/)

[15 Mar 2024] [Utilizing Latent Diffusion Model to Accelerate Sampling Speed and Enhance Text Generation Quality](https://www.mdpi.com/2079-9292/13/6/1093)

[15 Feb 2024] [Quantized Embedding Vectors for Controllable Diffusion Language Models](https://arxiv.org/abs/2402.10107)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.10107)

[3 Jun 2024] [Unlocking Guidance for Discrete State-Space Diffusion and Flow Models](https://arxiv.org/abs/2406.01572) (ICLR 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.01572)
[![Star](https://img.shields.io/github/stars/hnisonoff/discrete_guidance.svg?style=social&label=Star)](https://github.com/hnisonoff/discrete_guidance)

[09 Oct 2023] [DiffuSeq-v2: Bridging Discrete and Continuous Text Spaces for Accelerated Seq2Seq Diffusion Models](https://arxiv.org/abs/2310.05793) (EMNLP 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.05793)
[![Star](https://img.shields.io/github/stars/Shark-NLP/DiffuSeq.svg?style=social&label=Star)](https://github.com/Shark-NLP/DiffuSeq)

[24 May 2023] [David helps Goliath: Inference-Time Collaboration Between Small Specialized and Large General Diffusion LMs](https://arxiv.org/abs/2305.14771) (NAACL 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.14771)

[18 May 2023] [Diffusion Language Models Generation Can Be Halted Early](https://arxiv.org/abs/2305.10818)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.10818)



## Applications
[26 Jun 2025] [DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation](https://arxiv.org/abs/2506.20639)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.20639) [![Star](https://img.shields.io/github/stars/apple/ml-diffucoder.svg?style=social&label=Star)](https://github.com/apple/ml-diffucoder)

[11 Jun 2025] [Debunk and Infer: Multimodal Fake News Detection via Diffusion-Generated Evidence and LLM Reasoning](https://arxiv.org/abs/2506.21557)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.21557)

[28 May 2025] [CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models](https://arxiv.org/abs/2505.22869) (ICML 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.22869)[![Star](https://img.shields.io/github/stars/yinjunbo/cfpgen.svg?style=social&label=Star)](https://github.com/yinjunbo/cfpgen)

[27 Feb 2025] [EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models](https://arxiv.org/abs/2502.19765) (ACL 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2502.19765)

[31 Jan 2025] [TermDiffuSum: A Term-guided Diffusion Model for Extractive Summarization of Legal Documents](https://aclanthology.org/2025.coling-main.216/) (COLING 2025)<br>
[![Star](https://img.shields.io/github/stars/huaand/TermDiffuSum-.svg?style=social&label=Star)](https://github.com/huaand/TermDiffuSum-)

[1 Jan 2025] [DiffETM: Diffusion Process Enhanced Embedded Topic Model](https://arxiv.org/abs/2501.00862) (ICASSP 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2501.00862)

[23 Dec 2024] [DiffusionAttacker: Diffusion-Driven Prompt Manipulation for LLM Jailbreak](https://arxiv.org/abs/2412.17522)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2412.17522)

[5 Nov 2024] [DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](https://arxiv.org/abs/2411.03250) (ACL 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.03250)
[![Star](https://img.shields.io/github/stars/bytedance/DiffLM.svg?style=social&label=Star)](https://github.com/bytedance/DiffLM)

[30 Oct 2024] [Private Synthetic Text Generation with Diffusion Models](https://arxiv.org/abs/2410.22971) (NAACL 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.22971)
[![Star](https://img.shields.io/github/stars/trusthlt/private-synthetic-text-generation.svg?style=social&label=Star)](https://github.com/trusthlt/private-synthetic-text-generation)

[17 Oct 2024] [Text-Guided Multi-Property Molecular Optimization with a Diffusion Language Model](https://arxiv.org/abs/2410.13597)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.13597)

[17 Oct 2024] [DPLM-2: A Multimodal Diffusion Protein Language Model](https://arxiv.org/abs/2410.13782)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.13782)
[![Website](https://img.shields.io/badge/Website-9cf)](https://bytedance.github.io/dplm/dplm-2)
[![Star](https://img.shields.io/github/stars/bytedance/dplm.svg?style=social&label=Star)](https://github.com/bytedance/dplm)

[17 Oct 2024] [Fine-Tuning Discrete Diffusion Models via Reward Optimization with Applications to DNA and Protein Design](https://arxiv.org/abs/2410.13643) (ICLR 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.13643)
[![Star](https://img.shields.io/github/stars/ChenyuWang-Monica/DRAKES.svg?style=social&label=Star)](https://github.com/ChenyuWang-Monica/DRAKES)

[10 Oct 2024] [Steering Masked Discrete Diffusion Models via Discrete Denoising Posterior Prediction](https://arxiv.org/abs/2410.08134) (ICLR 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.08134)

[14 Sep 2024] [Towards Diverse and Efficient Audio Captioning via Diffusion Models](https://arxiv.org/abs/2409.09401) (DAC-Interspeech25)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.09401)

[10 Sep 2024] [Table-to-Text Generation with Pretrained Diffusion Models](https://arxiv.org/abs/2409.13739) (IEEE 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.13739)

[5 Sep 2024] [An Effective Deployment of Diffusion LM for Data Augmentation in Low-Resource Sentiment Classification](https://arxiv.org/abs/2409.03203) (EMNLP 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.03203)
[![Star](https://img.shields.io/github/stars/johnnychanv/diffusioncls.svg?style=social&label=Star)](https://github.com/johnnychanv/diffusioncls)

[Aug 2024] [DiffusPoll: Conditional Text Diffusion Model for Poll Generation](https://aclanthology.org/2024.findings-acl.54/) (ACL 2024)<br>
[![Star](https://img.shields.io/github/stars/bansky-cl/DiffusPoll.svg?style=social&label=Star)](https://github.com/bansky-cl/DiffusPoll)

[25 Jun 2024] [Discrete Diffusion Language Model for Efficient Text Summarization](https://arxiv.org/abs/2407.10998) (NAACL 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.10998)

[16 Apr 2024] [LaDiC: Are Diffusion Models Really Inferior to Autoregressive Counterparts for Image-to-Text Generation?](https://arxiv.org/abs/2404.10763) (NAACL 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.10763)
[![Star](https://img.shields.io/github/stars/wangyuchi369/LaDiC.svg?style=social&label=Star)](https://github.com/wangyuchi369/LaDiC)

[13 Apr 2024] [Improved Paraphrase Generation via Controllable Latent Diffusion](https://arxiv.org/abs/2404.08938)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.08938)
[![Star](https://img.shields.io/github/stars/NJUNLP/ld4pg.svg?style=social&label=Star)](https://github.com/NJUNLP/ld4pg)

[10 Apr 2024] [DiffusionDialog: A Diffusion Model for Diverse Dialog Generation with Latent Space](https://arxiv.org/abs/2404.06760) (LREC-COLING 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.06760)
[![Star](https://img.shields.io/github/stars/Jxxiang99/DiffusionDialog.svg?style=social&label=Star)](https://github.com/Jxxiang99/DiffusionDialog)

[10 Apr 2024] [Diffuwords: A Contrastive Diffusion Model for Lexically Constrained Text Generation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4790017) (SSRN 2024 Apr)

[28 Mar 2024] [Benchmarking Diffusion Models for Machine Translation](https://aclanthology.org/2024.eacl-srw.25/) (EACL 2024)

[26 Mar 2024] [Improving Iteration-based Non-Autoregressive Language Model With Time Step Awareness](https://www.researchgate.net/publication/379326543_Improving_Iteration-based_Non-Autoregressive_Language_Model_With_Time_Step_Awareness) (ICPADS 2023)

[28 Feb 2024] [Diffusion Language Models Are Versatile Protein Learners](https://arxiv.org/abs/2402.18567) (ICML 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.18567)
[![Star](https://img.shields.io/github/stars/bytedance/dplm.svg?style=social&label=Star)](https://github.com/bytedance/dplm)

[26 Feb 2024] [DiffuCOMET: Contextual Commonsense Knowledge Diffusion](https://arxiv.org/abs/2402.17011) (ACL 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.17011)
[![Star](https://img.shields.io/github/stars/silin159/diffucomet.svg?style=social&label=Star)](https://github.com/silin159/diffucomet)

[24 Feb 2024] [IPED: An Implicit Perspective for Relational Triple Extraction based on Diffusion Model](https://arxiv.org/abs/2403.00808) (NAACL 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.00808)
[![Star](https://img.shields.io/github/stars/girlsuuu/IPED.svg?style=social&label=Star)](https://github.com/girlsuuu/IPED)

[23 Feb 2024] [Let's Rectify Step by Step: Improving Aspect-based Sentiment Analysis with Diffusion Models](https://arxiv.org/abs/2402.15289) (LREC-COLING 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.15289)
[![Star](https://img.shields.io/github/stars/Qlb6x/DiffusionABSA.svg?style=social&label=Star)](https://github.com/Qlb6x/DiffusionABSA)

[20 Feb 2024] [Text-Guided Molecule Generation with Diffusion Language Model](https://arxiv.org/abs/2402.13040) (AAAI 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.13040)
[![Star](https://img.shields.io/github/stars/Deno-V/tgm-dlm.svg?style=social&label=Star)](https://github.com/Deno-V/tgm-dlm)

[16 Feb 2024] [Rethinking Human-like Translation Strategy: Integrating Drift-Diffusion Model with Large Language Models for Machine Translation](https://arxiv.org/abs/2402.10699)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.10699)

[11 Jan 2024] [MDM: Meta diffusion model for hard-constrained text generation](https://www.sciencedirect.com/science/article/abs/pii/S0950705123008973) (Knowledge-Based Systems)

[Dec 2023] [DiffusionSL: Sequence Labeling via Tag Diffusion Process](https://aclanthology.org/2023.findings-emnlp.860/) (EMNLP 2023)

[19 Dec 2023] [IPAD: Iterative, Parallel, and Diffusion-based Network for Scene Text Recognition](https://arxiv.org/abs/2312.11923) (IJCV 2025)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.11923)

[12 Dec 2023] [DiffuVST: Narrating Fictional Scenes with Global-History-Guided Denoising Models](https://arxiv.org/abs/2312.07066) (EMNLP 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.07066)

[3 Dec 2023] [DiffuCom: A novel diffusion model for comment generation](https://www.sciencedirect.com/science/article/abs/pii/S0950705123008195) (Knowledge-Based Systems)

[Dec 2023] [DiffusionRet: Diffusion-Enhanced Generative Retriever using Constrained Decoding](https://aclanthology.org/2023.findings-emnlp.638/) (EMNLP 2023)<br>
[![Star](https://img.shields.io/github/stars/jpthu17/DiffusionRet.svg?style=social&label=Star)](https://github.com/jpthu17/DiffusionRet)

[16 Nov 2023] [P^3SUM: Preserving Author's Perspective in News Summarization with Diffusion Language Models](https://arxiv.org/abs/2311.09741) (NAACL 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.09741)
[![Star](https://img.shields.io/github/stars/lyh6560new/P3Sum.svg?style=social&label=Star)](https://github.com/lyh6560new/P3Sum)

[31 Oct 2023] [LADIDA: Latent Diffusion for Document Generation with Sequential Decoding](https://neurips.cc/virtual/2023/74876) (NeurIPS Workshop 2023)  

[26 Oct 2023] [DiffS2UT: A Semantic Preserving Diffusion Model for Textless Direct Speech-to-Speech Translation](https://arxiv.org/abs/2310.17570) (EMNLP 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.17570)

[24 Oct 2023] [ScanDL: A Diffusion Model for Generating Synthetic Scanpaths on Texts](https://arxiv.org/abs/2310.15587) (EMNLP 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.15587)
[![Star](https://img.shields.io/github/stars/dili-lab/scandl.svg?style=social&label=Star)](https://github.com/dili-lab/scandl)

[23 Oct 2023] [DeTiME: Diffusion-Enhanced Topic Modeling using Encoder-decoder based LLM](https://arxiv.org/abs/2310.15296) (EMNLP 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.15296)
[![Star](https://img.shields.io/github/stars/amazon-science/text_generation_diffusion_llm_topic.svg?style=social&label=Star)](https://github.com/amazon-science/text_generation_diffusion_llm_topic)

[21 Oct 2023] [Context-Aware Prompt for Generation-based Event Argument Extraction with Diffusion Models](https://dl.acm.org/doi/abs/10.1145/3583780.3614820) (CIKM 2023)

[16 Oct 2023] [ForceGen: End-to-end de novo protein generation based on nonlinear mechanical unfolding responses using a protein language diffusion model](https://arxiv.org/abs/2310.10605) (ScienceAdvances)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.10605)

[29 Aug 2023] [ParaGuide: Guided Diffusion Paraphrasers for Plug-and-Play Textual Style Transfer](https://arxiv.org/abs/2308.15459) (AAAI 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.15459)
[![Star](https://img.shields.io/github/stars/zacharyhorvitz/ParaGuide.svg?style=social&label=Star)](https://github.com/zacharyhorvitz/ParaGuide)

[17 Aug 2023] [Enhancing Phrase Representation by Information Bottleneck Guided Text Diffusion Process for Keyphrase Extraction](https://arxiv.org/abs/2308.08739) (LREC-COLING 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.08739)

[25 Jul 2023] [XDLM: Cross-lingual Diffusion Language Model for Machine Translation](https://arxiv.org/abs/2307.13560)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.13560)
[![Star](https://img.shields.io/github/stars/Amayama/XDLM.svg?style=social&label=Star)](https://github.com/Amayama/XDLM)

[9 Jul 2023] [Controllable Conversation Generation with Conversation Structures via Diffusion Models](https://aclanthology.org/2023.findings-acl.454/) (ACL 2023)

[14 Jun 2023] [PoetryDiffusion: Towards Joint Semantic and Metrical Manipulation in Poetry Generation](https://arxiv.org/abs/2306.08456) (AAAI 2024)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.08456)

[14 Jun 2023] [DiffuDetox: A Mixed Diffusion Model for Text Detoxification](https://arxiv.org/abs/2306.08505) (ACL 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.08505)
[![Star](https://img.shields.io/github/stars/D3Mlab/diffu-detox.svg?style=social&label=Star)](https://github.com/D3Mlab/diffu-detox)

[5 Jun 2023] [PLANNER: Generating Diversified Paragraph via Latent Language Diffusion Model](https://arxiv.org/abs/2306.02531) (NeurIPS 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.02531)
[![Star](https://img.shields.io/github/stars/apple/ml-planner.svg?style=social&label=Star)](https://github.com/apple/ml-planner)

[2 Jun 2023] [DiffusEmp: A Diffusion Model-Based Framework with Multi-Grained Control for Empathetic Response Generation](https://arxiv.org/abs/2306.01657) (ACL 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.01657)
[![Star](https://img.shields.io/github/stars/surika/DiffusEmp.svg?style=social&label=Star)](https://github.com/surika/DiffusEmp)

[31 May 2023] [Fine-grained Text Style Transfer with Diffusion-Based Language Models](https://arxiv.org/abs/2305.19512) (RepL4NLP 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.19512)
[![Star](https://img.shields.io/github/stars/lvyiwei1/diffuseq_styleptb.svg?style=social&label=Star)](https://github.com/lvyiwei1/diffuseq_styleptb)

[31 May 2023] [Protein Design with Guided Discrete Diffusion](https://arxiv.org/abs/2305.20009) (NeurIPS 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.20009)
[![Star](https://img.shields.io/github/stars/ngruver/NOS.svg?style=social&label=Star)](https://github.com/ngruver/NOS)

[22 May 2023] [Dior-CVAE: Pre-trained Language Models and Diffusion Priors for Variational Dialog Generation](https://arxiv.org/abs/2305.15025) (EMNLP 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.15025)
[![Star](https://img.shields.io/github/stars/UKPLab/dior-cvae.svg?style=social&label=Star)](https://github.com/UKPLab/dior-cvae)

[22 May 2023] [DiffusionNER: Boundary Diffusion for Named Entity Recognition](https://arxiv.org/abs/2305.13298) (ACL 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.13298)
[![Star](https://img.shields.io/github/stars/tricktreat/DiffusionNER.svg?style=social&label=Star)](https://github.com/tricktreat/DiffusionNER)

[2 May 2023] [DiffuSum: Generation Enhanced Extractive Summarization with Diffusion](https://arxiv.org/abs/2305.01735) (ACL 2023)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.01735)
[![Star](https://img.shields.io/github/stars/hpzhang94/DiffuSum.svg?style=social&label=Star)](https://github.com/hpzhang94/DiffuSum)

[7 Jan 2023] [ROIC-DM: Robust Text Inference and Classification via Diffusion Model](https://arxiv.org/abs/2401.03514)<br>
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.03514)

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

[Mercury](https://www.inceptionlabs.ai/introducing-mercury) 
[![Arxiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.17298)

