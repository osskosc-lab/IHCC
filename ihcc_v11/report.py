from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def write_markdown_report(
    output_path: str | Path,
    config: dict[str, Any],
    gates: pd.DataFrame,
    decision: str,
    basis_recovery: pd.DataFrame,
) -> None:
    primary = basis_recovery[basis_recovery["scenario"].isin([1, 2, 3])]
    minimal_f1 = float(primary["minimal_f1"].mean())
    unpruned_f1 = float(primary["unpruned_f1"].mean())
    lines = [
        "# IHCC v1.1 Phase 1 preregistered results",
        "",
        f"- Profile: `{config['profile']}`",
        f"- Confirmatory: `{bool(config.get('confirmatory', False))}`",
        f"- Seeds: `{config['sampling']['seeds']}`",
        f"- Decision: `{decision}`",
        f"- Primary minimal-basis macro F1: `{minimal_f1:.4f}`",
        f"- Unpruned macro F1: `{unpruned_f1:.4f}`",
        "",
        "## Falsification gates",
        "",
        "| Gate | Status | Value | Criterion |",
        "|---|---|---|---|",
    ]
    for row in gates.itertuples(index=False):
        lines.append(f"| {row.gate} | {row.status} | {row.value} | {row.criterion} |")
    lines.extend(
        [
            "",
            "## Scope boundary",
            "",
            "This result concerns only the configured finite intervention set and finite candidate-state class. It is not evidence for an absolute non-Markov entity, a quantum mechanism, Orch OR, or consciousness.",
            "",
        ]
    )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#60656f"))
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"Page {document.page}")
    canvas.restoreState()


def write_pdf_report(
    output_path: str | Path,
    config: dict[str, Any],
    gates: pd.DataFrame,
    decision: str,
    basis_recovery: pd.DataFrame,
) -> None:
    output = Path(output_path)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="IHCC_Title",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=colors.HexColor("#14395b"),
            alignment=TA_CENTER,
            spaceAfter=9 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="IHCC_Heading",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#14395b"),
            spaceBefore=5 * mm,
            spaceAfter=3 * mm,
        )
    )
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="IHCC v1.1 Phase 1 Report",
        author="IHCC Research Project",
    )
    primary = basis_recovery[basis_recovery["scenario"].isin([1, 2, 3])]
    minimal_f1 = float(primary["minimal_f1"].mean())
    unpruned_f1 = float(primary["unpruned_f1"].mean())
    story = [
        Paragraph("IHCC v1.1 Phase 1", styles["IHCC_Title"]),
        Paragraph("Synthetic boundary recovery and falsification audit", styles["Heading3"]),
        Spacer(1, 5 * mm),
        Table(
            [
                ["Profile", config["profile"]],
                ["Confirmatory", str(bool(config.get("confirmatory", False)))],
                ["Seeds", str(config["sampling"]["seeds"])],
                ["Decision", decision],
                ["Minimal-basis macro F1", f"{minimal_f1:.4f}"],
                ["Unpruned macro F1", f"{unpruned_f1:.4f}"],
                ["Generated (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")],
            ],
            colWidths=[52 * mm, 105 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e9f0f6")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1f2933")),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#aab7c4")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            ),
        ),
        Paragraph("Interpretation boundary", styles["IHCC_Heading"]),
        Paragraph(
            "This report evaluates an estimator inside a fixed synthetic intervention and state-class scope. "
            "It does not support an absolute non-Markov entity, a quantum mechanism, Orch OR, or consciousness.",
            styles["BodyText"],
        ),
        PageBreak(),
        Paragraph("Falsification gates", styles["IHCC_Heading"]),
    ]

    gate_data = [["Gate", "Status", "Observed value"]]
    for row in gates.itertuples(index=False):
        gate_data.append([row.gate, row.status, Paragraph(str(row.value), styles["BodyText"])])
    story.append(
        Table(
            gate_data,
            repeatRows=1,
            colWidths=[18 * mm, 24 * mm, 115 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14395b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aab7c4")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            ),
        )
    )
    story.extend(
        [
            Paragraph("Decision semantics", styles["IHCC_Heading"]),
            Paragraph(
                "Only the confirmatory profile can yield SUPPORTED_IN_SYNTHETIC_SCOPE. "
                "Smoke and pilot runs are diagnostic or provisional, even if every numerical gate passes.",
                styles["BodyText"],
            ),
        ]
    )
    doc.build(story, onFirstPage=_page_number, onLaterPages=_page_number)

    reader = PdfReader(str(output))
    if len(reader.pages) < 2:
        raise RuntimeError("generated report is unexpectedly short")
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    if "IHCC v1.1 Phase 1" not in extracted or "Falsification gates" not in extracted:
        raise RuntimeError("generated report failed text validation")
