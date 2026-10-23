# Nepali Tokenization Tax

**How much more does it cost to use AI in Nepali than in English?**

Large language models read text as *tokens*, and they are billed, limited and slowed down by token count. Because tokenizers are built mostly from English data, the same sentence often costs far more tokens in other languages. This extra cost is called the **tokenization tax**.

This project measures the tokenization tax for **Nepali** across 11 modern tokenizers, and asks a specific question: Nepali and Hindi share the same script (Devanagari), so **is the tax caused by the script itself, or by how little Nepali data exists online?**

![English vs Hindi vs Nepali](images/english_hindi_nepali.png)

---

## Key findings

**1. Nepali pays a tax in every tokenizer tested.** The same 2,009 sentences cost between **1.53×** and **4.77×** as many tokens in Nepali as in English, depending on the tokenizer. That is more than a threefold difference in cost for identical content, purely from tokenizer design.

**2. In weak tokenizers, the script is the bottleneck.** In tokenizers with little Devanagari knowledge (GPT-4's cl100k, GLM-4.5, Qwen3), Nepali and Hindi cost almost exactly the same (Nepali/Hindi ratio ≈ 1.00). Their text is split into raw byte fragments, so cost depends on byte length, not on how much data each language had.

**3. In strong multilingual tokenizers, data scarcity takes over.** In tokenizers that learned real Devanagari subwords (Gemma 3, Mistral Nemo, Llama 4), Nepali costs **11–19% more than Hindi**, even though both use the same script. Better tokenizers make Nepali much cheaper in absolute terms, but they also reveal the data gap that weaker tokenizers were hiding.

**4. Vocabulary size is not destiny.** Qwen3 and GLM-4.5 (~151k tokens) are about as costly for Nepali as GPT-4's older 100k vocabulary, while GPT-4o (~200k) and Llama 4 (~201k), nearly identical in size, differ noticeably. What a vocabulary is spent on matters more than how large it is.

**5. Same script, different costs.** Across six Devanagari languages (Hindi, Marathi, Nepali, Maithili, Bhojpuri, Sanskrit), weak tokenizers price them all about the same, while strong tokenizers spread them apart, roughly in line with how much online text each language has.

![Devanagari languages](images/devanagari.png)

---

## Results

Parity = tokens needed for the text in that language ÷ tokens needed for the same text in English (1.00 = same cost as English). Sorted from best to worst for Nepali.

| Tokenizer | Vocab size | Hindi parity | Nepali parity | Nepali / Hindi |
|---|---:|---:|---:|---:|
| Gemma 3 | 262k | 1.31 | 1.53 | 1.17 |
| GPT-4o (o200k) | 200k | 1.58 | 1.62 | 1.03 |
| Llama 4 | 201k | 1.65 | 1.83 | 1.11 |
| Mistral Nemo | 131k | 1.80 | 2.14 | 1.19 |
| Mistral Small 3.1 | 131k | 1.80 | 2.14 | 1.19 |
| Llama 3.2 | 128k | 2.53 | 2.59 | 1.02 |
| Kimi K2 | 163k | 2.75 | 2.80 | 1.02 |
| DeepSeek-V3 | 128k | 2.81 | 2.98 | 1.06 |
| Qwen3 | 151k | 4.43 | 4.42 | 1.00 |
| GPT-4 (cl100k) | 100k | 4.78 | 4.77 | 1.00 |
| GLM-4.5 | 151k | 4.77 | 4.77 | 1.00 |

Full results for all 12 languages, including tokens per word, characters per token and bytes per token, are in [`token_tax_results.csv`](token_tax_results.csv).

![Vocabulary size vs tokenization tax](images/vocab_vs_tax.png)

**Sanity checks.** Mistral Nemo and Mistral Small 3.1 produce identical results, as expected since they share the same tokenizer (Tekken). GLM-4.5 produces exactly the same token counts as GPT-4's cl100k on the sample Devanagari sentences, suggesting it handles Devanagari in essentially the same way.

---

## Method

**Data.** [FLORES-200](https://github.com/facebookresearch/flores), a professionally translated parallel benchmark from Meta. All `dev` and `devtest` sentences were used: **2,009 sentences per language**, identical in meaning across languages, which makes direct cost comparison fair.

**Languages (12).**

| Group | Languages |
|---|---|
| Baseline | English |
| Devanagari script | Hindi, Marathi, Nepali, Maithili, Bhojpuri, Sanskrit |
| Other scripts (context) | Bengali, Tamil, Urdu, Chinese (Simplified), Spanish |

**Tokenizers (11).**

| Name | Source |
|---|---|
| GPT-4 (cl100k) | `tiktoken` — `cl100k_base` |
| GPT-4o (o200k) | `tiktoken` — `o200k_base` |
| Llama 3.2 | `unsloth/Llama-3.2-1B` |
| Llama 4 | `unsloth/Llama-4-Scout-17B-16E-Instruct` |
| Qwen3 | `Qwen/Qwen3-0.6B` |
| DeepSeek-V3 | `deepseek-ai/DeepSeek-V3` |
| Mistral Nemo | `mistralai/Mistral-Nemo-Base-2407` |
| Mistral Small 3.1 | `unsloth/Mistral-Small-3.1-24B-Instruct-2503` (loaded with `fix_mistral_regex=True`) |
| Gemma 3 | `unsloth/gemma-3-1b-it` |
| Kimi K2 | `moonshotai/Kimi-K2-Instruct` (requires `trust_remote_code=True`) |
| GLM-4.5 | `zai-org/GLM-4.5` |

Only the tokenizer files are downloaded, not the models themselves. Special tokens are excluded from all counts.

**Metrics.**

- **Parity**: total tokens for a language ÷ total tokens for English on the same sentences.
- **Nepali / Hindi ratio**: Nepali parity ÷ Hindi parity, isolating the cost difference between two languages that share a script.
- **Tokens per word**, **characters per token**, **bytes per token**: how much text each token carries. (Tokens per word is not meaningful for Chinese, which does not separate words with spaces.)
- **Per-sentence parity**: computed for every sentence to show how consistent the tax is, not just its average.

---

## Running it yourself

The whole analysis runs in [Google Colab](https://colab.research.google.com/) for free, with no Hugging Face login needed.

1. Open `nepali_tokenization_tax.ipynb` in Colab (File → Upload notebook, or use the badge below once this repo is public).
2. Run the cells from top to bottom.
3. Measuring all tokenizers takes a few minutes. Results are saved to `token_tax_results.csv`.

<!-- Replace YOUR_USERNAME and YOUR_REPO with your GitHub details -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/nepali_tokenization_tax.ipynb)

To add a language, add its FLORES-200 code to the `LANGS` dictionary (for example `"ben_Beng": "Bengali"`). To add a tokenizer, add its Hugging Face repo name to `TOKENIZERS`.

---

## Repository structure

```
├── nepali_tokenization_tax.ipynb   # full analysis notebook
├── token_tax_results.csv           # results for all tokenizers and languages
├── images/                         # charts used in this README
│   ├── english_hindi_nepali.png
│   ├── devanagari.png
│   ├── vocab_vs_tax.png
│   └── heatmap.png
└── README.md
```

---

## Limitations

- **Formal text only.** FLORES-200 contains news-style, formal translations. Everyday Nepali, romanized Nepali (Nepali typed in Latin letters) and Nepali–English code-mixed text may behave very differently.
- **Cost, not quality.** These results measure how many tokens text needs, not how well models understand it. Higher fragmentation is often linked to weaker performance, but that was not tested here.
- **Translation effects.** Translations naturally vary in length, so small parity differences (a few percent) should not be over-interpreted.
- **Closed tokenizers excluded.** Tokenizers that are not publicly available (for example those of Claude or Gemini) could not be tested.

---

## Future work

- Measure the tax for **romanized Nepali** and **Nepali–English code-mixed** text.
- Test whether a higher tokenization tax corresponds to **lower model performance** on Nepali tasks.
- Extend an existing tokenizer with **Nepali-specific tokens** and measure how much the tax drops.

---

## References

- Petrov, A., La Malfa, E., Torr, P. H. S., & Bibi, A. (2023). *Language Model Tokenizers Introduce Unfairness Between Languages.* NeurIPS 2023.
- Ahia, O., Kumar, S., Gonen, H., Kasai, J., Mortensen, D. R., Smith, N. A., & Tsvetkov, Y. (2023). *Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models.* EMNLP 2023.
- Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL 2021.
- NLLB Team (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation.* (Source of FLORES-200.)

---

## License

Code: MIT License. FLORES-200 data is released by Meta under CC BY-SA 4.0; see the [FLORES repository](https://github.com/facebookresearch/flores) for details.

## Author

**Dikshya Lamichhane** — [GitHub](https://github.com/DikshyaL) · [Email](mailto:lamichhanedikshya7@gmail.com)

Feedback, questions and collaboration on Nepali and low-resource language NLP are very welcome.
