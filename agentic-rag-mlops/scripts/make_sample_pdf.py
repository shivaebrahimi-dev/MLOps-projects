"""Generate data/sample-knowledge/evaluation-metrics-findings.pdf (the sample PDF).

The content summarises the worked examples of the author's Homework 3 presentation on evaluation
metrics. Run with: uv run python scripts/make_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "data" / "sample-knowledge"
OUT_FILE = OUT / "evaluation-metrics-findings.pdf"

PAGES: list[list[tuple[str, str]]] = [
    [
        ("h1", "Evaluation Metrics Findings (Homework 3)"),
        (
            "p",
            "This document summarises the worked examples from the course presentation on evaluation "
            "metrics across statistical machine learning, deep learning, generative AI, software "
            "engineering and MLOps. All numbers were computed from seeded synthetic data or data sets "
            "bundled with scikit-learn.",
        ),
        ("h2", "Classification: accuracy lies when positives are rare"),
        (
            "p",
            "In the fraud-detection example the model reached 97.9% accuracy but a recall of only 0.18, "
            "because positives were rare and the model missed most frauds. Precision, recall, F1 and "
            "PR-AUC describe such imbalanced problems better than accuracy.",
        ),
        (
            "p",
            "Choosing the decision threshold from the cost of errors, instead of using 0.5, cut the "
            "total cost by 52% in the same example. ROC-AUC can look good on imbalanced data while "
            "PR-AUC exposes weak performance on the rare class.",
        ),
        ("h2", "Regression and probability quality"),
        (
            "p",
            "Adding a single outlier multiplied RMSE by 3.8 but MAE only by 1.6, because RMSE squares "
            "large errors. Log loss, the Brier score and reliability bins measure whether predicted "
            "probabilities are calibrated.",
        ),
    ],
    [
        ("h2", "Generative AI: word overlap is not meaning"),
        (
            "p",
            "BLEU and ROUGE gave a wrong answer that reused the reference wording a score of 0.88 and "
            "a correct paraphrase only 0.18. Overlap metrics therefore cannot judge factual "
            "correctness.",
        ),
        (
            "p",
            "For an LLM or RAG system the presentation recommends measuring faithfulness (are the "
            "claims supported by the retrieved context), answer relevance, retrieval quality such as "
            "recall at k, the hallucination rate, safety, latency and cost.",
        ),
        ("h2", "LLM-as-a-Judge must be checked against humans"),
        (
            "p",
            "An LLM judge agreed with human labels on 84% of the items, but Cohen's kappa was only "
            "0.64 once chance agreement was removed, and the judge showed a leniency bias: it passed "
            "answers that humans failed. A judge should be trusted only after its agreement and kappa "
            "with human labels have been measured.",
        ),
        ("h2", "Reliability and delivery"),
        (
            "p",
            "In the latency example the median (p50) was 120 ms but the 99th percentile (p99) was "
            "1,114 ms, so service-level objectives should be set on tail latency. The four DORA "
            "metrics are deployment frequency, lead time for changes, change failure rate and time to "
            "restore service.",
        ),
        ("h2", "Drift monitoring"),
        (
            "p",
            "Over six simulated weeks the Population Stability Index (PSI) and the Kolmogorov-Smirnov "
            "statistic rose before accuracy fell, so drift metrics can warn before a model's quality "
            "visibly decays.",
        ),
    ],
]


def build() -> Path:
    styles = getSampleStyleSheet()
    style = {"h1": styles["Title"], "h2": styles["Heading2"], "p": styles["BodyText"]}
    story = []
    for i, page in enumerate(PAGES):
        if i:
            story.append(PageBreak())
        for kind, text in page:
            story.append(Paragraph(text, style[kind]))
            story.append(Spacer(1, 6))
    OUT.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_FILE),
        pagesize=A4,
        title="Evaluation Metrics Findings",
        author="Course student",
        invariant=1,  # reproducible bytes: no creation timestamp or random document id
    )
    doc.build(story)
    return OUT_FILE


if __name__ == "__main__":
    print(f"wrote {build()}")
