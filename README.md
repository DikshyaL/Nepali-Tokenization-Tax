# Nepali Tokenization Tax

**Why does it cost more to use AI in Nepali than in English, and what causes it?**

Large language models read text as *tokens*, and they are billed, limited and slowed down by token count. The same sentence often needs far more tokens in some languages than in English. This extra cost is called the **tokenization tax**.

This project does two things:

1. **Measures** the tokenization tax for Nepali across 11 modern LLM tokenizers and 12 languages.
2. **Explains** it, identifying two mechanisms and testing them with controlled experiments.

The central question: Nepali and Hindi share the Devanagari script, so is Nepali's extra cost caused by the **script**, or by **how little Nepali data** tokenizers are trained on?

---

## Main findings

**1. Nepali pays a tax in every tokenizer tested.** On the same 2,009 sentences, Nepali needs **1.5× to 4.8×** as many tokens as English, depending on the tokenizer.

**2. Mechanism 1, the splitting rule.** Before learning their vocabulary, tokenizers cut text into chunks with a fixed rule (pre-tokenization). Older rules keep only Unicode *letters* together, but Devanagari vowel signs (matras such as ा and े) are Unicode *marks*. These rules therefore cut every Devanagari word at its vowel signs, so whole words can never become tokens. Under this rule, Nepali and Hindi cost the same (Nepali/Hindi ≈ 1.00), and adding more Nepali data barely helps.

**3. Mechanism 2, data share.** When the splitting rule keeps words whole, a language's cost depends on how much of it the tokenizer saw during training. In a controlled experiment, increasing Nepali's share of the training data **caused** its token count to fall by **37%**, with the first 1% of data alone removing **19%**.

**4. With equal data, Nepali is cheaper than Hindi.** Given the same amount of training data, Nepali needs fewer tokens than Hindi (ratio 0.88). The 11–19% Nepali-over-Hindi gap in strong production tokenizers therefore reflects how little Nepali they were trained on, not anything inherent to the language.

---

## Part 1: Observational study (11 tokenizers)

Parity = tokens needed for a language ÷ tokens needed for the same text in English (1.00 = same cost as English).

| Tokenizer | Vocab | Splitting rule | Hindi parity | Nepali parity | Nepali / Hindi |
|---|---:|---|---:|---:|---:|
| Gemma 3 | 262k | Keeps words whole | 1.31 | 1.53 | 1.17 |
| GPT-4o (o200k) | 200k | Keeps words whole | 1.58 | 1.62 | 1.03 |
| Llama 4 | 201k | Keeps words whole | 1.65 | 1.83 | 1.11 |
| Mistral Nemo | 131k | Keeps words whole | 1.80 | 2.14 | 1.19 |
| Mistral Small 3.1 | 131k | Keeps words whole | 1.80 | 2.14 | 1.19 |
| Kimi K2 | 163k | Keeps words whole | 2.75 | 2.80 | 1.02 |
| DeepSeek-V3 | 128k | Keeps words whole | 2.81 | 2.98 | 1.06 |
| Llama 3.2 | 128k | Breaks words | 2.53 | 2.59 | 1.02 |
| Qwen3 | 151k | Breaks words | 4.43 | 4.42 | 1.00 |
| GPT-4 (cl100k) | 100k | Breaks words | 4.78 | 4.77 | 1.00 |
| GLM-4.5 | 151k | Breaks words | 4.77 | 4.77 | 1.00 |

**What the table shows:**

- **Tokenizers that break words** have Nepali/Hindi between **0.997 and 1.023**: the two languages cost the same, because both are reduced to the same small shared fragments.
- **Tokenizers that keep words whole** have Nepali/Hindi between **1.016 and 1.19**. Keeping words whole is *necessary* for a large gap, but not *sufficient*: the gap also depends on how much Devanagari vocabulary the tokenizer learned.
- **Vocabulary size is not destiny.** Similar-sized vocabularies give very different results; what a vocabulary is spent on matters more than its size.

The splitting-rule difference, in one example:

```python
import regex
word = "नेपालमा"   # "in Nepal"
regex.findall(r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+", word)              # old rule
# ['न', 'े', 'प', 'ा', 'लम', 'ा']
regex.findall(r"[\p{L}\p{M}]+|\p{N}+|[^\s\p{L}\p{M}\p{N}]+", word)  # keeps marks
# ['नेपालमा']
```

*(Simplified versions of the real pre-tokenization rules, showing the key difference.)*

---

## Part 2: Experiment 1, does data share cause the tax?

The observational study suggests data scarcity drives the gap, but real tokenizers differ in many ways at once. To test causation, I trained byte-level BPE tokenizers myself and changed **only one thing**: Nepali's share of the training data.

**Setup:**

- Training text: Wikipedia in English, Hindi and Nepali, 20M characters in total
- Hindi fixed at 20%; Nepali varied from 0% to 30%
- Algorithm: byte-level BPE, 32k vocabulary
- 7 Nepali shares × 3 random seeds = 21 tokenizers, plus 2 ablation tokenizers (23 in total)
- Evaluation: FLORES-200 (2,009 parallel sentences), never seen in training

**Results** (averages over 3 seeds; variation between seeds was ±0.001–0.008):

| Nepali share | Nepali parity | Hindi parity | Nepali / Hindi | Devanagari tokens in vocab |
|---:|---:|---:|---:|---:|
| 0% | 1.708 | 1.279 | 1.336 | 8,254 |
| 1% | 1.378 | 1.275 | 1.080 | 8,628 |
| 2% | 1.314 | 1.271 | 1.034 | 8,973 |
| 5% | 1.229 | 1.261 | 0.974 | 9,880 |
| 10% | 1.148 | 1.247 | 0.921 | 11,245 |
| 20% | 1.073 | 1.221 | 0.879 | 13,514 |
| 30% | 1.025 | 1.196 | 0.857 | 15,600 |

**What it shows:**

- **Data share causes the tax.** Nepali's token count falls by **37%** from 0% to 30% share, and it falls at every single step (Spearman ρ = −1.0 across the seven settings). Hindi's token count, whose data never changed, moves only about 2%.
- **The first data matters most.** Going from 0% to just 1% Nepali cuts Nepali's tokens by **19%**; going from 1% all the way to 30% saves only about 18 points more. Even a small, deliberate amount of a language in tokenizer training removes most of its tax.
- **Nepali catches up with Hindi on much less data.** Nepali/Hindi crosses 1.00 between 2% and 5% Nepali, while Hindi has 20%.
- **Spillover.** Marathi, never in the training data, becomes about 6% cheaper as Nepali's share rises: related Devanagari languages share many pieces.

**Ablation: the splitting rule.** I repeated two settings with the word-breaking (GPT-2 style) splitting rule:

| Splitting rule | Nepali tokens, 5% → 20% share | Nepali / Hindi |
|---|---|---|
| Keeps words whole | 70,091 → 62,731 (**−10.5%**) | 0.97 → 0.88 |
| Breaks words | 173,595 → 172,540 (**−0.6%**) | 1.04 → 1.04 |

With the word-breaking rule, quadrupling Nepali's data barely helps, matching what GPT-4, Qwen3 and GLM-4.5 show in Part 1. The observational study and the controlled experiment agree.

*A note on the numbers: parity is a ratio to English, and English's share of the training data shrinks as Nepali's grows, which slightly raises English token counts. Causal claims above therefore use raw token counts, which measure each language directly.*

---

## Limitations

- **Narrow controlled setup:** one algorithm (byte-level BPE), one vocabulary size (32k), and a 20M-character training corpus, much smaller than production tokenizers use.
- **Domain mismatch:** tokenizers were trained on Wikipedia but evaluated on FLORES news text. This affects all languages, and conclusions rely on comparisons between languages.
- **Hindi fixed at 20%:** Hindi's share was never varied.
- **Untested explanation:** why Nepali is cheaper than Hindi with equal data is not yet established. One hypothesis is grammar: Hindi writes postpositions such as में and को as separate words, while Nepali attaches them (मा, को), letting them merge into words.
- **Cost, not quality:** these results measure how many tokens text needs, not how well models understand or translate it.
- **Closed tokenizers excluded:** tokenizers that are not publicly available could not be tested.

---

## Next steps

- Robustness checks: a larger vocabulary (64k) and a different Hindi share.
- **Experiment 2:** extend an existing LLM tokenizer with Nepali-specific tokens and measure how much the tax drops.
- Test whether a higher tokenization tax goes along with lower translation quality, and which evaluation metrics are reliable for Nepali.
- Romanized Nepali and Nepali–English code-mixed text.

---

## Running it yourself

Everything runs in [Google Colab](https://colab.research.google.com/) for free, with no Hugging Face login needed.

1. Open a notebook from `notebooks/` in Colab (File → Upload notebook).
2. Run all cells. The observational study takes a few minutes; Experiment 1 takes longer because it trains 23 tokenizers.
3. Training text and trained tokenizers are not stored in this repository because of their size. The Experiment 1 notebook downloads and rebuilds them.

---

## References

- Petrov, A., La Malfa, E., Torr, P. H. S., & Bibi, A. (2023). *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS 2023.
- Ahia, O., Kumar, S., Gonen, H., Kasai, J., Mortensen, D. R., Smith, N. A., & Tsvetkov, Y. (2023). *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models.* EMNLP 2023.
- Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL 2021.
- NLLB Team (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation.* (Source of FLORES-200.)

---

## License

Code: MIT License. FLORES-200 data is released by Meta under CC BY-SA 4.0; see the [FLORES repository](https://github.com/facebookresearch/flores). Wikipedia text is available under CC BY-SA.

## Author

**Dikshya Lamichhane** — Computer Engineering, Kathmandu University
[GitHub](https://github.com/DikshyaL) · [LinkedIn](https://www.linkedin.com/in/dikshya-lamichhane) · lamichhaned001@gmail.com

Feedback and collaboration on Nepali and low-resource language NLP are very welcome.
