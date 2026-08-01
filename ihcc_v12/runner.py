from __future__ import annotations

import copy
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ihcc_v12.core import (
    best_exponential_projection,
    calibrate_beta,
    pooled_input_second_moment,
    powerlaw_kernel,
    theoretical_oracle_gain,
)

REQUIRED_OUTPUTS = (
    "projection_floor.csv",
    "oracle_gain_calibration.csv",
    "seed_level_ood_gain.csv",
    "budget_gradient.csv",
    "threshold_operating_curve.csv",
    "metric_sensitivity.csv",
    "reset_error_detection.csv",
    "environment_transport.csv",
    "python_version_comparison.csv",
    "v11_regression_gates.csv",
    "v12_falsification_gates.csv",
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, profile: str) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if profile not in raw.get("profiles", {}):
        raise ValueError(f"unknown profile: {profile}")
    resolved = {key: value for key, value in raw.items() if key != "profiles"}
    resolved = _deep_merge(resolved, raw["profiles"][profile])
    resolved["profile"] = profile
    return resolved


def projection_preflight(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    experiment = config["experiment"]
    kernel_cfg = config["kernel"]
    projection_cfg = config["projection"]
    probabilities = config["sampling"]["train_probabilities"]
    rows: list[dict[str, Any]] = []

    for length in experiment["sensitivity_lengths"]:
        sigma_u = pooled_input_second_moment(int(length), probabilities)
        for rho in kernel_cfg["sensitivity_rho"]:
            kernel = powerlaw_kernel(int(length), float(rho), float(kernel_cfg["tau0"]))
            for budget in experiment["budgets"]:
                result = best_exponential_projection(
                    kernel,
                    int(budget),
                    sigma_u,
                    multistarts=int(projection_cfg["multistarts"]),
                    seed=int(experiment["base_seed"]) + 1000 * int(length) + 100 * int(budget) + int(round(10 * float(rho))),
                    lambda_min=float(projection_cfg["lambda_min"]),
                    lambda_max=float(projection_cfg["lambda_max"]),
                    tolerance=float(projection_cfg["tolerance"]),
                )
                rows.append(
                    {
                        "length": int(length),
                        "rho": float(rho),
                        "budget": int(budget),
                        "relative_projection_error": result.relative_error,
                        "residual_variance": result.residual_variance,
                        "lambdas": json.dumps(result.lambdas.tolist()),
                        "weights": json.dumps(result.weights.tolist()),
                        "converged": result.converged,
                        "multistarts": result.starts,
                        "projection_floor": float(experiment["projection_floor"]),
                        "passes_floor": bool(result.relative_error >= float(experiment["projection_floor"])),
                    }
                )

    frame = pd.DataFrame(rows).sort_values(["length", "rho", "budget"]).reset_index(drop=True)
    primary = frame[
        (frame["length"] == int(experiment["primary_length"]))
        & (np.isclose(frame["rho"], float(kernel_cfg["rho"])))
        & (frame["budget"] == int(experiment["primary_budget"]))
    ]
    if len(primary) != 1:
        raise RuntimeError("primary projection row is not unique")
    primary_row = primary.iloc[0].to_dict()
    residual_variance = float(primary_row["residual_variance"])
    target_gain = float(experiment["target_oracle_gain"])
    sigma = float(kernel_cfg["sigma"])
    beta = calibrate_beta(target_gain, sigma, residual_variance)
    calibrated_gain = theoretical_oracle_gain(beta, sigma, residual_variance)
    calibration = pd.DataFrame(
        [
            {
                "length": int(experiment["primary_length"]),
                "rho": float(kernel_cfg["rho"]),
                "budget": int(experiment["primary_budget"]),
                "residual_variance": residual_variance,
                "sigma": sigma,
                "target_oracle_gain": target_gain,
                "calibrated_beta": beta,
                "reconstructed_oracle_gain": calibrated_gain,
            }
        ]
    )
    return frame, calibration, primary_row


def _blocked_table(name: str, reason: str) -> pd.DataFrame:
    return pd.DataFrame([{"status": "NOT_RUN", "reason": reason, "artifact": name}])


def evaluate_preflight_gates(config: dict[str, Any], primary: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    floor = float(config["experiment"]["projection_floor"])
    error = float(primary["relative_projection_error"])
    x1_pass = error >= floor
    rows = [
        {
            "gate": "X1",
            "name": "projection error floor",
            "status": "PASS" if x1_pass else "FAIL",
            "value": f"E_proj,3={error:.12g}",
            "criterion": f">={floor:.6g}",
        }
    ]
    reason = "X1 must pass before outcome generation under the locked preregistration"
    for gate, name in [
        ("X2", "out-of-class history detection"),
        ("X3", "in-class negative control"),
        ("X4", "state-budget gradient"),
        ("X5", "threshold operating characteristics"),
        ("X6", "distance metric sensitivity"),
        ("X7", "environment-direction transport"),
        ("X8", "Python 3.11/3.12 reproducibility"),
    ]:
        rows.append(
            {
                "gate": gate,
                "name": name,
                "status": "BLOCKED" if not x1_pass else "PENDING",
                "value": reason if not x1_pass else "downstream implementation required",
                "criterion": "see preregistration",
            }
        )
    decision = "DESIGN_BLOCKED_X1" if not x1_pass else "PREFLIGHT_PASSED_DOWNSTREAM_PENDING"
    return pd.DataFrame(rows), decision


def _write_markdown(
    path: Path,
    config: dict[str, Any],
    primary: dict[str, Any],
    calibration: pd.DataFrame,
    gates: pd.DataFrame,
    decision: str,
) -> None:
    error = float(primary["relative_projection_error"])
    floor = float(config["experiment"]["projection_floor"])
    beta = float(calibration.iloc[0]["calibrated_beta"])
    lines = [
        "# IHCC v1.2 Phase 2A preflight report",
        "",
        f"- Profile: `{config['profile']}`",
        f"- Python: `{platform.python_version()}`",
        f"- Decision: `{decision}`",
        f"- Primary condition: `T={config['experiment']['primary_length']}, rho={config['kernel']['rho']}, B={config['experiment']['primary_budget']}`",
        f"- Best relative projection error: `{error:.12g}`",
        f"- Locked projection floor: `{floor:.6g}`",
        f"- Calibrated beta for oracle gain 0.05: `{beta:.6g}`",
        "",
        "## Interpretation",
        "",
    ]
    if decision == "DESIGN_BLOCKED_X1":
        lines.extend(
            [
                "The locked power-law positive control does not satisfy the generation-before-analysis gate.",
                "Because the optimized three-exponential class approximates the finite kernel far better than the preregistered 5% floor, outcome generation and X2-X8 testing were stopped rather than silently changing the kernel or threshold.",
            ]
        )
    else:
        lines.append("X1 passed. Downstream outcome-generation experiments may proceed without changing the locked positive control.")
    lines.extend(["", "## Gates", "", "| Gate | Status | Value | Criterion |", "|---|---|---|---|"])
    for row in gates.itertuples(index=False):
        lines.append(f"| {row.gate} | {row.status} | {row.value} | {row.criterion} |")
    lines.extend(
        [
            "",
            "## Scope boundary",
            "",
            "This preflight concerns a finite synthetic kernel and the fixed class F_3. It does not establish representation-invariant non-Markovianity, a quantum mechanism, Orch OR, consciousness, or a soul-information particle.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_pdf(
    path: Path,
    config: dict[str, Any],
    primary: dict[str, Any],
    gates: pd.DataFrame,
    decision: str,
) -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="JPTitle",
            parent=styles["Title"],
            fontName="HeiseiKakuGo-W5",
            fontSize=19,
            leading=25,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#14395b"),
            spaceAfter=7 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="JPBody",
            parent=styles["BodyText"],
            fontName="HeiseiKakuGo-W5",
            fontSize=9.2,
            leading=14,
        )
    )
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="IHCC v1.2 Phase 2A Report",
        author="IHCC Research Project",
    )
    error = float(primary["relative_projection_error"])
    floor = float(config["experiment"]["projection_floor"])
    story = [
        Paragraph("IHCC v1.2 Phase 2A", styles["JPTitle"]),
        Paragraph("状態クラス外残差の正対照再構築 - 生成前監査報告", styles["JPBody"]),
        Spacer(1, 5 * mm),
        Table(
            [
                ["実行プロファイル", config["profile"]],
                ["Python", platform.python_version()],
                ["主要条件", f"T={config['experiment']['primary_length']}, rho={config['kernel']['rho']}, B={config['experiment']['primary_budget']}"],
                ["最良射影誤差", f"{error:.12g}"],
                ["事前固定下限", f"{floor:.6g}"],
                ["判定", decision],
                ["生成時刻 UTC", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")],
            ],
            colWidths=[48 * mm, 112 * mm],
            style=TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "HeiseiKakuGo-W5"),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e9f0f6")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#aab7c4")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            ),
        ),
        Spacer(1, 7 * mm),
    ]
    if decision == "DESIGN_BLOCKED_X1":
        story.append(
            Paragraph(
                "主要べき乗核は、最適化された3指数状態クラスにより事前固定した5%下限より大幅に良く近似された。したがって、正対照としての生成条件X1は不成立であり、結果を見て核・閾値を変更することを避けるため、未知介入予測を含む下流実験を停止した。",
                styles["JPBody"],
            )
        )
    gate_data = [["Gate", "Status", "Observed / reason"]]
    for row in gates.itertuples(index=False):
        gate_data.append([row.gate, row.status, Paragraph(str(row.value), styles["JPBody"])])
    story.extend(
        [
            Spacer(1, 7 * mm),
            Table(
                gate_data,
                repeatRows=1,
                colWidths=[17 * mm, 24 * mm, 119 * mm],
                style=TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), "HeiseiKakuGo-W5"),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14395b")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aab7c4")),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                ),
            ),
            Spacer(1, 7 * mm),
            Paragraph(
                "本判定は有限合成核と固定状態クラスF3に限定される。表現不変の非Markov性、量子機構、Orch OR、意識、魂情報粒を支持しない。",
                styles["JPBody"],
            ),
        ]
    )
    document.build(story)
    reader = PdfReader(str(path))
    if len(reader.pages) < 1:
        raise RuntimeError("generated PDF has no pages")


def run_preflight(config_path: str | Path, profile: str, output_dir: str | Path) -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path, profile)
    projection, calibration, primary = projection_preflight(config)
    gates, decision = evaluate_preflight_gates(config, primary)

    projection.to_csv(output / "projection_floor.csv", index=False)
    calibration.to_csv(output / "oracle_gain_calibration.csv", index=False)
    blocked_reason = "preregistered X1 projection floor failed" if decision == "DESIGN_BLOCKED_X1" else "downstream runner not yet invoked"
    for filename in REQUIRED_OUTPUTS:
        path = output / filename
        if path.exists():
            continue
        if filename == "v12_falsification_gates.csv":
            gates.to_csv(path, index=False)
        elif filename == "python_version_comparison.csv":
            pd.DataFrame(
                [
                    {
                        "python_version": platform.python_version(),
                        "implementation": platform.python_implementation(),
                        "decision": decision,
                        "primary_projection_error": float(primary["relative_projection_error"]),
                        "status": "SELF_RECORDED",
                    }
                ]
            ).to_csv(path, index=False)
        else:
            _blocked_table(filename, blocked_reason).to_csv(path, index=False)

    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    _write_markdown(output / "IHCC_v1.2_Phase2A_Report_JA.md", config, primary, calibration, gates, decision)
    _write_pdf(output / "IHCC_v1.2_Phase2A_Report_JA.pdf", config, primary, gates, decision)
    (output / "decision.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "profile": profile,
                "python": platform.python_version(),
                "primary_projection_error": float(primary["relative_projection_error"]),
                "projection_floor": float(config["experiment"]["projection_floor"]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return decision
