#!/usr/bin/env python3
"""Regenerate the 43-system LaTeX table from the frozen local CSV authority."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "tables/confirmed_43_provenance.csv"
OUTPUT = ROOT / "tables/confirmed_43_provenance.tex"

# The first compact citation describes object-level lens confirmation. These
# amendments add the publications supporting the representative redshift
# values displayed in the final two columns.
REDSHIFT_CITATION_AMENDMENTS = {
    "0442m235_7484": ["Nakane2025"],
    "1219p105_7293": ["Tran2022"],
    "2026p047_4274": ["Tran2022"],
    "2349p167_2044": ["Tran2022", "Barone2026"],
    "3108m062_2639": ["Tran2022", "Barone2026"],
    "3277p137_6699": ["Tran2022", "Barone2026"],
    "3418m020_1357": ["Sand2005"],
}


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def evidence_keys(row: dict[str, str]) -> str:
    keys = [key for key in row["evidence_keys"].split(",") if key]
    for key in REDSHIFT_CITATION_AMENDMENTS.get(row["galaxy_id"], []):
        if key not in keys:
            keys.append(key)
    return ",".join(keys)


def main() -> None:
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 43 or len({row["galaxy_id"] for row in rows}) != 43:
        raise RuntimeError("local provenance authority is not exactly 43 rows")

    body = [
        (
            rf"\texttt{{{latex_escape(row['galaxy_id'])}}} & "
            rf"{latex_escape(row['confirmation_category'])} & "
            rf"\citep{{{evidence_keys(row)}}} & "
            rf"{row['lens_redshift_display']} & "
            rf"{row['source_redshift_display']} \\"
        )
        for row in rows
    ]
    OUTPUT.write_text(
        "\n".join(
            [
                r"\subsection{Object-level provenance of the 43-system endpoint}",
                "",
                r"\label{app:confirmed_provenance}",
                r"The selected endpoint is not a uniformly dual-spectroscopic",
                r"sample. Table~\ref{tab:confirmed_43_provenance}",
                r"lists the object identifier, confirmation route, principal",
                r"object-level publication, and one representative recorded",
                r"foreground and source redshift for every evaluated system.",
                r"Multiple measurements and the complete evidence record are provided in",
                r"\texttt{confirmed\_43\_provenance.csv} in the reproducibility materials;",
                r"``--'' means that no",
                r"representative role-assigned redshift is recorded. The",
                r"confirmation-route column identifies whether displayed values",
                r"are secure, probable, low-confidence, or unverified.",
                "",
                r"\begin{table*}",
                r"\centering",
                r"\caption{Object-level provenance of the selected 43-system endpoint.}",
                r"\label{tab:confirmed_43_provenance}",
                r"{\scriptsize",
                r"\setlength{\tabcolsep}{3pt}",
                r"\renewcommand{\arraystretch}{0.92}",
                r"\begin{tabular}{p{0.14\textwidth}p{0.28\textwidth}p{0.24\textwidth}rr}",
                r"\toprule",
                r"Identifier & Confirmation route & Principal evidence & $z_{\rm foreground}$ & $z_{\rm source}$ \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabular}",
                r"}",
                r"\end{table*}",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
