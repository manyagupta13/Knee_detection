"""Label reports with a local multilingual LLM, replacing the regex lexicon.

Why this exists (PLAN.md §2.3 and the measured numbers):

  * the rule-based labeler reaches ~0.80 agreement with gold and covers 5 of
    ~10 languages, leaving 23% of the corpus (el, ru/bg, hr, fr, pl) unlabeled;
  * it fills 12,842 of 52,884 label cells - 24% of the matrix;
  * its precision ranges 0.56 to 0.90 per column, and three lexicon passes
    failed to lift Contusion above 0.56.

Every image-model gain is capped by that. An instruct model reads all ten
languages natively, handles negation and hedging as language rather than as
regex, and can fill close to the whole matrix.

Runs OFFLINE at inference time by construction: this is a TRAIN-ONLY step. The
test set has no reports, so nothing here can leak into submission.

    python llmlabel.py --data-dir <data> --out llm_labels.csv \\
        --model Qwen/Qwen2.5-7B-Instruct --limit 50
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from config import ID_COLUMN, TARGET_COLUMNS

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Short, unambiguous descriptions. The model is told to answer per finding with
# 1 / 0 / null, and crucially that "not mentioned" is null, NOT 0 - absence of
# mention is not evidence of absence (PLAN.md §2.3).
FINDING_DESCRIPTIONS: dict[str, str] = {
    "ACL": "anterior cruciate ligament tear or rupture (partial or complete)",
    "MCL": "medial collateral ligament tear, sprain or rupture",
    "Medial Meniscus": "medial meniscus tear (NOT plain degeneration without a tear)",
    "Lateral Meniscus": "lateral meniscus tear (NOT plain degeneration without a tear)",
    "Medial OA": "osteoarthritis or cartilage loss in the MEDIAL femorotibial compartment",
    "Lateral OA": "osteoarthritis or cartilage loss in the LATERAL femorotibial compartment",
    "PF OA": "patellofemoral osteoarthritis, chondromalacia patellae, or patellar/trochlear cartilage loss",
    "Effusion": "joint effusion / excess intra-articular fluid",
    "Synovitis": "synovitis or synovial thickening/proliferation",
    "Baker's": "Baker's cyst / popliteal cyst",
    "Contusion": "bone contusion, bone bruise, or traumatic bone marrow oedema",
    "Fracture": "any fracture, including stress, insufficiency, avulsion or osteochondral",
}

SYSTEM_PROMPT = (
    "You are a musculoskeletal radiologist extracting structured findings from "
    "knee MRI reports. Reports may be in any language (English, Spanish, "
    "Turkish, Greek, German, Dutch, Croatian, Russian, French, Polish). "
    "Answer ONLY with a JSON object."
)

INSTRUCTIONS = """For each finding, answer:
  1    = the report states this finding IS present
  0    = the report states this finding is ABSENT, normal, or intact
  null = the report does not mention it, or is genuinely equivocal
         (e.g. "possible", "cannot exclude", "trace")

Rules:
- "not mentioned" is null, never 0.
- Anchor laterality carefully: medial vs lateral vs patellofemoral are separate
  findings. The "medial patellar facet" belongs to PATELLOFEMORAL, not medial.
- Meniscal degeneration WITHOUT a tear is not a meniscus tear (answer 0 or null).
- Mild/small findings that are definitely present are 1, not null.

Findings:
{findings}

Report:
\"\"\"{report}\"\"\"

Answer with JSON only, keys exactly as listed, values 1, 0 or null."""


def build_prompt(report: str, columns: Sequence[str]) -> str:
    findings = "\n".join(f"- {c}: {FINDING_DESCRIPTIONS.get(c, c)}" for c in columns)
    text = str(report)[:6000]  # keep well inside context; reports are short
    return INSTRUCTIONS.format(findings=findings, report=text)


def parse_response(text: str, columns: Sequence[str]) -> dict[str, float | None]:
    """Extract the JSON object and coerce to 1.0/0.0/None. Never raises."""
    out: dict[str, float | None] = {c: None for c in columns}
    if not text:
        return out
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return out
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return out
    if not isinstance(data, dict):
        return out

    lowered = {str(k).strip().lower(): v for k, v in data.items()}
    for col in columns:
        value = lowered.get(col.strip().lower())
        if value is None:
            continue
        if isinstance(value, bool):
            out[col] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            # Compare the float itself, not int(value): int(0.5) == 0 would turn
            # a model's hedge into a confident negative.
            if float(value) in (0.0, 1.0):
                out[col] = float(value)
        elif isinstance(value, str):
            v = value.strip().lower()
            if v in ("1", "yes", "true", "present"):
                out[col] = 1.0
            elif v in ("0", "no", "false", "absent"):
                out[col] = 0.0
    return out


def label_reports_llm(
    reports: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    text_col: str = "report_text",
    id_col: str = ID_COLUMN,
    columns: Sequence[str] = TARGET_COLUMNS,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    limit: int | None = None,
    dtype: str = "float16",
    verbose: bool = True,
) -> pd.DataFrame:
    """Label reports with a local instruct model. Returns a labels DataFrame
    indexed by StudyInstanceUID with NaN where the model declined to commit."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    columns = list(columns)
    df = reports if limit is None else reports.head(limit)
    index = pd.Index(df[id_col].astype(str), name=id_col)
    out = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, dtype),
        device_map="auto",
    )
    model.eval()

    texts = list(df[text_col])
    uids = list(index)
    t0 = time.time()

    for start in range(0, len(texts), batch_size):
        chunk_texts = texts[start : start + batch_size]
        chunk_uids = uids[start : start + batch_size]
        prompts = [
            tok.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(t, columns)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for t in chunk_texts
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=3072).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,           # deterministic: this is extraction
                pad_token_id=tok.pad_token_id,
            )
        for uid, seq in zip(chunk_uids, generated):
            reply = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for col, value in parse_response(reply, columns).items():
                if value is not None:
                    out.loc[uid, col] = value

        done = start + len(chunk_texts)
        if verbose and (done % (batch_size * 10) == 0 or done >= len(texts)):
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(texts) - done) / max(rate, 1e-9)
            print(f"  {done}/{len(texts)} @ {rate:.2f}/s  eta {eta/60:.0f} min")

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="llm_labels.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from data import load_tables

    tables = load_tables(args.data_dir, split="train")
    if tables.reports is None:
        raise SystemExit("no reports found")
    labels = label_reports_llm(
        tables.reports, model_name=args.model,
        batch_size=args.batch_size, limit=args.limit,
    )
    labels.to_csv(args.out)
    print(f"wrote {args.out}: {int(labels.notna().to_numpy().sum())} labeled cells")


if __name__ == "__main__":
    main()
