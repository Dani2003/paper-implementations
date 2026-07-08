# Paper Implementations 📚

> Most tutorials teach you how to use deep learning techniques. Very few explain why they actually work.

This repository contains detailed implementations of machine learning research papers, focusing on understanding the mathematics and engineering behind each method.

---

## 🔬 Current Implementation

### LoRA: Low-Rank Adaptation of Large Language Models

**Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)  
**Notebook**: [`notebooks/01_LoRA_From_Scratch.ipynb`](notebooks/01_LoRA_From_Scratch.ipynb)

#### What is LoRA?

Instead of fine-tuning all parameters in a massive model, LoRA learns a small, low-rank adaptation that can be added on top. The insight:

```
Output = Base Model Output + (α/r) × B × A × Input
```

Where:
- **Base Model** (frozen): 124M parameters in GPT-2
- **LoRA Adaptation** (trainable): Only 1.3M parameters
- **Result**: 99% parameter reduction with minimal performance loss

#### What I Implemented

✓ **Custom LoRA Linear Layer** - Built from scratch without using pre-built adapters  
✓ **Proper Initialization** - Kaiming uniform for A, zeros for B (critical for stability)  
✓ **Base Weight Freezing** - Ensured frozen weights stay frozen during training  
✓ **Model Surgery** - Injected LoRA into GPT-2's attention layers  
✓ **Gradient Verification** - Explicit proof that gradients flow ONLY through LoRA weights  

#### Key Technical Moments

- **Initialization strategy**: Why zeros for B ensures ΔW starts at zero
- **Gradient freezing**: Ensuring no accidental backprop into base weights
- **Scaling factor (α/r)**: Normalizing across different rank configurations
- **Attention layer targeting**: Why c_attn is the optimal injection point

#### Results

```
Model                  Total Parameters    Trainable Parameters    Efficiency
─────────────────────────────────────────────────────────────────────────────
GPT-2 (Baseline)       124M                124M                    100%
GPT-2 + LoRA (r=4)     124M                1.3M                    0.98%
```

---

## 🚀 Quick Start

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU (8GB+ VRAM recommended)
- Transformers library

### Installation

```bash
# Clone the repository
git clone https://github.com/Dani2003/paper-implementations.git
cd paper-implementations

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/01_LoRA_From_Scratch.ipynb
```

### Running the Notebook

The notebook walks you through:
1. GPU setup and verification
2. Custom LoRA layer implementation
3. Applying LoRA to GPT-2
4. Full training loop with gradient verification
5. Parameter efficiency analysis

**Expected runtime**: ~5-10 minutes (with GPU)

---

## 📖 Understanding the Implementation

Each cell in the notebook includes detailed explanation of:
- **What's happening** - The mechanics of that step
- **Why it's done this way** - The reasoning behind each choice
- **How it aligns with the paper** - Verification against the original research

For deeper analysis, see [`docs/LoRA_Analysis.md`](docs/LoRA_Analysis.md).

---

## 💡 Why Implement Papers?

- **Reading ≠ Understanding** - Papers are easy to skim but hard to grasp
- **Implementation reveals challenges** - Every detail becomes visible during coding
- **Bugs teach more than documentation** - Errors force deeper understanding
- **You can't explain what you haven't built** - Building forces clarity

---

## 🔍 How This Implementation Differs From Others

| Aspect | This Implementation | Standard Approach |
|--------|-------------------|-------------------|
| **Initialization** | Kaiming + Zeros (paper-exact) | Often random |
| **Gradient verification** | Explicit checks | Assumed correct |
| **Code clarity** | Annotated explanations | Library abstractions |
| **Parameter counting** | Detailed reporting | Basic numbers |

---

## 📚 What's Next?

Future papers to implement:
- [ ] Attention Is All You Need (Vaswani et al., 2017)
- [ ] Adapter Modules (Houlsby et al., 2019)
- [ ] Flash Attention (Dao et al., 2022)
- [ ] Prefix Tuning (Li & Liang, 2021)

Each addition strengthens understanding of modern LLM fine-tuning techniques.

---

## 🤝 Contributing & Feedback

I'm always open to feedback from people who have implemented LoRA themselves. If you notice:
- Improvements to make it closer to the original paper
- Clearer ways to explain concepts
- Performance optimizations
- Bugs or edge cases

Please open an issue or start a discussion. Learning from the community makes the implementation better.

---

## 📖 References

- **Original Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Paper PDF**: [arxiv.org/pdf/2106.09685.pdf](https://arxiv.org/pdf/2106.09685.pdf)
- **Hugging Face PEFT**: [github.com/huggingface/peft](https://github.com/huggingface/peft) (for production use)

---

## 🛠️ Technical Stack

- **PyTorch** - Deep learning framework
- **Transformers (Hugging Face)** - Pre-trained models
- **Jupyter** - Interactive notebooks
- **Python 3.8+** - Programming language

---

## 📋 Repository Structure

```
paper-implementations/
├── README.md                           # This file
├── requirements.txt                    # Dependencies
├── .gitignore                         # Git ignore rules
│
├── notebooks/
│   └── 01_LoRA_From_Scratch.ipynb     # Main implementation
│
├── docs/
│   ├── LoRA_Analysis.md               # Detailed analysis
│   └── LoRA_Paper_Summary.md          # Paper highlights
│
└── implementations/                    # Optional: Python modules
    ├── lora.py                        # Reusable LoRA class
    └── __init__.py
```

---

## ✨ Key Learnings

**From implementing LoRA, you'll understand:**

1. **Why low-rank decomposition works** - It's not magic, it's math
2. **How to freeze model weights** - Critical for parameter efficiency
3. **Initialization strategies** - Why zeros for B, Kaiming for A matters
4. **Gradient verification** - Proving your implementation is correct
5. **Parameter counting** - Measuring real efficiency gains

---

## 📞 Questions?

- **"How do I run this?"** - See Quick Start section
- **"Why did you initialize B with zeros?"** - See docs/LoRA_Analysis.md
- **"Can I use this in production?"** - Yes, but consider using [PEFT](https://github.com/huggingface/peft) for robustness
- **"How does this compare to adapters?"** - Great question for a future implementation!

---

## 📊 Performance Notes

**Current Setup**:
- Model: GPT-2 (124M parameters)
- Rank: r=4
- Target Modules: c_attn (attention layers)
- Learning Rate: 1e-4
- Optimizer: AdamW

**Expected Training Time**:
- Per epoch: ~2-3 seconds (GPU)
- 100 epochs: ~4-5 minutes

---

## 📌 Citation

If this implementation helps your research, feel free to reference:

```
@misc{paper_implementations_lora,
  title={LoRA: Low-Rank Adaptation from Scratch},
  author={Dani2003},
  url={https://github.com/Dani2003/paper-implementations},
  year={2024}
}
```

---

## 🎯 The Philosophy

> Reimplementing papers like this forces you to move beyond "using AI" and toward understanding how these methods are actually built.

The goal isn't to compete with production libraries. It's to build deep understanding of foundational techniques so you can invent the next generation.

---

**Always learning. Always building.** 🚀

---

Last updated: 2024  
Status: LoRA complete | Extending to more papers
