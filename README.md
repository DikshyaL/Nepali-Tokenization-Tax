# Nepali Tokenization Tax

**Why does it cost more to use AI in Nepali than in English, and what causes it?**

Large language models read text as *tokens*, and they are billed, limited and slowed down by token count. The same sentence often needs far more tokens in some languages than in English. This extra cost is called the **tokenization tax**.

This project does two things:

1. **Measures** the tokenization tax for Nepali across 11 modern LLM tokenizers and 12 languages.
2. **Explains** it, separating three causes and testing them: the UTF-8 encoding, the tokenizer's splitting rule, and how little Nepali data tokenizers are trained on.

The central question: Nepali and Hindi share the Devanagari script and almost the same UTF-8 length, so what makes Nepali cost more?

---

## Main findings

**1. Nepali pays a tax in every tokenizer tested.** On the same 2,009 sentences, Nepali needs **1.53× to 4.77×** as many tokens as English, depending on the tokenizer.

**2. UTF-8 sets a starting gap, and the tokenizer decides how much of it is paid.** Each Devanagari character takes 3 bytes in UTF-8 and each English letter takes 1. The Nepali *byte premium* (bytes needed relative to English for the same content) is **2.57**, the same as Hindi's 2.56. For identical text and bytes, Nepali's token ratio ranges from 1.53 to 4.77 across tokenizers, so the tokenizer, not the encoding alone, drives the difference. Measured in bytes, Gemma 3, GPT-4o, Llama 4 and Mistral beat the byte premium; measured in **characters**, no tokenizer does, because a Devanagari character takes three bytes but is not three times the information.

**3. Mechanism 1, the splitting rule.** Before learning their vocabulary, tokenizers cut text into chunks with a fixed rule (pre-tokenization). Some rules keep only Unicode *letters* together, but Devanagari vowel signs (matras such as ा and े) are Unicode *marks*, so these rules cut words at their vowel signs. In GPT-4 (cl100k), Llama 3.2, GLM-4.5 and Qwen3, only **8.5% to 9.6%** of Nepali words reach the tokenizer whole, against **89.5% to 97.6%** in the other seven. In those four, Nepali and Hindi cost the same (ratio 0.997 to 1.023). The rule caps what merges can learn, while the vocabulary decides the level: Llama 3.2 costs 2.59 for Nepali against 4.77 for GPT-4.

**4. Mechanism 2, data share.** When the splitting rule keeps words whole, a language's cost depends on how much of it the tokenizer saw during training. In a controlled experiment, increasing Nepali's share of the training data **caused** its token count to fall by **37%**, with the first 1% of data alone removing **19%**.

**5. With equal data, Nepali is cheaper than Hindi.** Given the same amount of training data, Nepali needs fewer tokens than Hindi (ratio 0.88). In the seven production tokenizers that keep words whole, Nepali costs about 2% to 19% more than Hindi, so that gap reflects how little Nepali they were trained on, not anything inherent to the language.

---

## Part 1: Observational study (11 tokenizers)

Parity = tokens needed for a language ÷ tokens needed for the same text in English (1.00 = same cost as English). "Words kept whole" is the share of Nepali words that reach the tokenizer as one chunk after pre-tokenization (first 400 sentences).

| Tokenizer | Vocab | Splitting rule | Hindi parity | Nepali parity | Nepali / Hindi | Nepali words kept whole |
|---|---:|---|---:|---:|---:|---:|
| Gemma 3 | 262k | Keeps words whole | 1.31 | 1.53 | 1.17 | 89.5% |
| GPT-4o (o200k) | 200k | Keeps words whole | 1.58 | 1.62 | 1.03 | 97.4% |
| Llama 4 | 201k | Keeps words whole | 1.65 | 1.83 | 1.11 | 97.4% |
| Mistral Nemo | 131k | Keeps words whole | 1.80 | 2.14 | 1.19 | 96.3% |
| Mistral Small 3.1 | 131k | Keeps words whole | 1.80 | 2.14 | 1.19 | 96.3% |
| Kimi K2 | 163k | Keeps words whole | 2.75 | 2.80 | 1.02 | 97.4% |
| DeepSeek-V3 | 128k | Keeps words whole | 2.81 | 2.98 | 1.06 | 97.6% |
| Llama 3.2 | 128k | Breaks words | 2.53 | 2.59 | 1.02 | 9.6% |
| Qwen3 | 151k | Breaks words | 4.43 | 4.42 | 1.00 | 8.5% |
| GPT-4 (cl100k) | 100k | Breaks words | 4.78 | 4.77 | 1.00 | 9.6% |
| GLM-4.5 | 151k | Breaks words | 4.77 | 4.77 | 1.00 | 9.6% |

![Tokenization tax across languages and tokenizers](images/heatmap.png)

**What the table shows:**

- **Tokenizers that break words** have Nepali/Hindi between **0.997 and 1.023**. For GPT-4, GLM-4.5 and Qwen3 the 95% confidence interval includes 1.00, so Nepali and Hindi cost the same; for Llama 3.2 the ratio is 1.023 with an interval of [1.017, 1.030], a small but real difference.
- **Tokenizers that keep words whole** have Nepali/Hindi between **1.016 and 1.19**, and every 95% interval excludes 1.00. Keeping words whole is *necessary* for a large gap, but not *sufficient*: the gap also depends on how much Devanagari vocabulary the tokenizer learned.
- **Per character, the gap is larger.** After adjusting for text length (tokens per character), all 11 tokenizers charge Nepali 3% to 23% more than Hindi, including the three at 1.00.
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

![Word integrity versus Nepali cost](images/word_integrity_vs_parity.png)

---

## Part 1b: How much of the cost is UTF-8?

A **premium** is how much longer a language's text is than English for the same content.

| Language | Byte premium | Character premium | Word premium |
|---|---:|---:|---:|
| Hindi | 2.56 | 1.00 | 1.17 |
| Marathi | 2.69 | 1.01 | 0.87 |
| Nepali | 2.57 | 0.96 | 0.85 |
| Maithili | 2.54 | 0.98 | 1.12 |
| Bhojpuri | 2.48 | 0.97 | 1.17 |
| Sanskrit | 2.64 | 0.98 | 0.79 |
| Bengali | 2.62 | 0.98 | 0.89 |
| Tamil | 3.19 | 1.17 | 0.77 |
| Urdu | 1.76 | 0.99 | 1.27 |
| Chinese | 0.92 | 0.33 | 0.10 |
| Spanish | 1.21 | 1.19 | 1.19 |

A tokenizer that used one token per byte would have a parity equal to the byte premium (2.57 for Nepali). The **excess** is Nepali parity divided by the premium: above 1, the tokenizer adds cost beyond the length difference; below 1, it has learned merges that recover part of it.

| Tokenizer | Nepali parity | Excess over bytes | Excess over characters | Excess over words |
|---|---:|---:|---:|---:|
| Gemma 3 | 1.53 | 0.60 | 1.59 | 1.80 |
| GPT-4o (o200k) | 1.62 | 0.63 | 1.69 | 1.91 |
| Llama 4 | 1.83 | 0.71 | 1.91 | 2.15 |
| Mistral Nemo | 2.14 | 0.83 | 2.23 | 2.52 |
| Mistral Small 3.1 | 2.14 | 0.83 | 2.23 | 2.52 |
| Llama 3.2 | 2.59 | 1.01 | 2.70 | 3.05 |
| Kimi K2 | 2.80 | 1.09 | 2.92 | 3.29 |
| DeepSeek-V3 | 2.98 | 1.16 | 3.10 | 3.51 |
| Qwen3 | 4.42 | 1.72 | 4.60 | 5.20 |
| GPT-4 (cl100k) | 4.77 | 1.86 | 4.97 | 5.61 |
| GLM-4.5 | 4.77 | 1.86 | 4.97 | 5.61 |

Because the premium is a constant, dividing by it does not change the tokenizer ranking; what changes is the reference line. In bytes, Gemma 3, GPT-4o, Llama 4 and Mistral fall below it. In characters and words, none do, so even the best tokenizer here charges Nepali more per character than English.

![Nepali cost against the byte premium](images/nepali_vs_byte_premium.png)

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
- **Bytes are not information:** the byte premium overstates how long Nepali is, which is why the byte, character and word views are reported together.
- **Word integrity is an approximate check:** it uses the first 400 sentences per language, and some tokenizers (for example Gemma 3, which also splits some English words) do not separate words cleanly.
- **Untested explanation:** why Nepali is cheaper than Hindi with equal data is not yet established. One hypothesis is grammar: Hindi writes postpositions such as में and को as separate words, while Nepali attaches them (मा, को), letting them merge into words.
- **Cost, not quality:** these results measure how many tokens text needs, not how well models understand or translate it.
- **Formal text only:** FLORES-200 is professionally translated, formal text.
- **Closed tokenizers excluded:** tokenizers that are not publicly available could not be tested.

---

## Next steps

- A monolingual Nepali tokenizer, following the design of Arnett et al. (2025), to compare with the multilingual ones.
- A small morphological-alignment check on the Nepali UD treebank (UD_Nepali-BK has only 69 sentences, so this would be exploratory).
- Robustness checks: a larger vocabulary (64k) and a different Hindi share.
- **Experiment 2:** extend an existing LLM tokenizer with Nepali-specific tokens and measure how much the tax drops.
- Test whether a higher tokenization tax goes along with lower translation quality, and which evaluation metrics are reliable for Nepali.
- Romanized Nepali and Nepali–English code-mixed text.

---

## Results files

- `token_tax_results.csv`: parity, tokens per word, characters per token and bytes per token for every tokenizer and language
- `nepali_hindi_ci.csv`: Nepali/Hindi ratio with 95% bootstrap intervals and a length-adjusted ratio
- `pretokenizer_check.csv`: how each tokenizer splits a short Nepali phrase before merging
- `word_integrity.csv`: share of words that stay whole after pre-tokenization
- `unit_premiums.csv` and `excess_over_units.csv`: byte, character and word premiums, and each tokenizer's excess over them
- `images/`: all charts

---

## Running it yourself

Everything runs in [Google Colab](https://colab.research.google.com/) for free, with no Hugging Face login needed.

1. Open `nepali_tokenization_tax.ipynb` in Colab (File → Upload notebook, or use the badge at the top of the notebook).
2. Run all cells. The observational study takes a few minutes; Experiment 1 takes longer because it trains 23 tokenizers.
3. Training text and trained tokenizers are not stored in this repository because of their size. The Experiment 1 notebook downloads and rebuilds them.

---

## References

- Petrov, A., La Malfa, E., Torr, P. H. S., & Bibi, A. (2023). *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS 2023.
- Ahia, O., Kumar, S., Gonen, H., Kasai, J., Mortensen, D. R., Smith, N. A., & Tsvetkov, Y. (2023). *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models.* EMNLP 2023.
- Arnett, C., Chang, T. A., Biderman, S., & Bergen, B. K. (2025). *Explaining and Mitigating Crosslingual Tokenizer Inequities.* NeurIPS 2025. (Defines the byte premium.)
- Arnett, C., Hudspeth, M., & O'Connor, B. (2025). *Evaluating Morphological Alignment of Tokenizers in 70 Languages.* ICML 2025 Tokenization Workshop.
- Hudspeth, M., Burns, P. J., & O'Connor, B. (2026). *Contextual morphologically-guided tokenization for Latin encoder models.* EACL 2026.
- Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL 2021.
- NLLB Team (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation.* (Source of FLORES-200.)

---

## License

Code: MIT License. FLORES-200 data is released by Meta under CC BY-SA 4.0; see the [FLORES repository](https://github.com/facebookresearch/flores). Wikipedia text is available under CC BY-SA.

## Author

**Dikshya Lamichhane** — Computer Engineering, Kathmandu University
[GitHub](https://github.com/DikshyaL) · [LinkedIn](https://www.linkedin.com/in/dikshya-lamichhane) · lamichhaned001@gmail.com

Feedback and collaboration on Nepali and low-resource language NLP are very welcome.
