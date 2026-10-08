#!/usr/bin/env python3
"""Verify the manuscript and canonical public evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

ACTIVE_TEX = [
    "01_abstract.tex",
    "02_introduction.tex",
    "03_data_samples_detector.tex",
    "04_injection_models_experiments.tex",
    "05_analysis_design_statistical_methods.tex",
    "06_results.tex",
    "07_discussion.tex",
    "08_conclusions.tex",
    "09_declarations.tex",
    "A2_supplementary_analyses.tex",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_text(command: list[str], *, cwd: Path = ROOT) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True)


def abstract_word_count() -> int:
    text = (ROOT / "01_abstract.tex").read_text(encoding="utf-8")
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", " ", text)
    text = re.sub(r"\\[A-Za-z]+(?:\[[^]]*\])?", " ", text)
    text = re.sub(r"[{}$\\_^~]", " ", text)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))


def abstract_paragraph_count() -> int:
    text = (ROOT / "01_abstract.tex").read_text(encoding="utf-8")
    match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return 0
    body = re.sub(r"%.*", "", match.group(1)).strip()
    return len([paragraph for paragraph in re.split(r"\n\s*\n", body) if paragraph])


def pdf_pages(path: Path) -> int:
    output = run_text(["pdfinfo", str(path)])
    match = re.search(r"^Pages:\s+(\d+)$", output, re.MULTILINE)
    if not match:
        raise RuntimeError(f"page count unavailable: {path}")
    return int(match.group(1))


def main() -> None:
    problems: list[str] = []

    count = abstract_word_count()
    if not 180 <= count <= 250:
        problems.append(f"abstract word count {count} is outside 180--250")
    if abstract_paragraph_count() != 1:
        problems.append("abstract is not a single paragraph")
    abstract_text = " ".join(
        (ROOT / "01_abstract.tex").read_text(encoding="utf-8").split()
    )
    for phrase in (
        "operational detector's pooled penultimate features",
        "49,500-scene grid with no visibility cut",
        "On 43 fixed targets",
        "expectation-image recovery varies non-monotonically",
        "18.6 per cent",
        "provide a practical basis",
    ):
        if phrase not in abstract_text:
            problems.append(f"refined abstract phrase missing: {phrase}")
    for phrase in (
        "0.980",
        "fixed-head",
        "99-cell",
        "7/12",
        "8/12",
        "unique morphology cause",
        "certify physical realism",
        "estimate real-lens completeness",
    ):
        if phrase in abstract_text:
            problems.append(f"unrequested abstract inventory present: {phrase}")

    main_text = (ROOT / "main.tex").read_text(encoding="utf-8")
    if "Linear-probe diagnostics" not in main_text:
        problems.append("final four-representation title is missing")

    active_text = "\n".join(
        (ROOT / name).read_text(encoding="utf-8") for name in ACTIVE_TEX
    )
    active_normalized = " ".join(active_text.split())
    required_phrases = [
        "reveal differences between the selected populations",
        "four feature protocols",
        "pooled penultimate",
        "0.991",
        "0.958",
        "0.979",
        "0.981",
        "0.980",
        "+0.023",
        "1,710 detections among 49,500",
        "39/43",
        "3,096",
        "+0.560290",
        "1.20",
        "cross-population geographic links",
        "8/12",
        "source-effect interval exclusion is policy-dependent",
        "The matched controls locate differences that predate injection",
        "source-construction pipelines",
        "Across all eight magnitudes",
        "comparisons between these checkpoint policies are exploratory",
        "documented general scientific eligibility rule that independently selects those 43",
        "Exact footprint polygons are unavailable",
        "Twenty had an exploratory feature-analysis-development role",
        r"retained without the \(z<20\) requirement",
        "do not document a scientific reason for this exemption",
        "needed to recheck those two eligibility criteria row by row",
        "repeated seeds probe sensitivity to training stochasticity",
        "local-pixel criterion is met in 1/16 cases",
        "shape criterion in 0/16",
    ]
    for phrase in required_phrases:
        if phrase not in active_normalized:
            problems.append(f"required manuscript phrase missing: {phrase}")

    forbidden_claims = [
        "morphological barrier",
        "feature-space separation explains",
        "domain separation explains",
        "linear-probe AUC proves",
        "population completeness is",
        "model-independent replication",
        "32.5--45",
        "clip 20",
        "30/43",
    ]
    lowered = active_normalized.lower()
    for phrase in forbidden_claims:
        if phrase.lower() in lowered:
            problems.append(f"forbidden or unsupported claim present: {phrase}")

    citation_text = active_text + (
        ROOT / "tables/confirmed_43_provenance.tex"
    ).read_text(encoding="utf-8")
    cited_keys: set[str] = set()
    for match in re.finditer(
        r"\\cite\w*(?:\[[^\]]*\]){0,2}\{([^}]*)\}", citation_text
    ):
        cited_keys.update(key.strip() for key in match.group(1).split(","))
    bibliography_keys = set(
        re.findall(
            r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}",
            main_text,
        )
    )
    if bibliography_keys - cited_keys:
        problems.append(
            "uncited bibliography entries remain: "
            + ", ".join(sorted(bibliography_keys - cited_keys))
        )
    if cited_keys - bibliography_keys:
        problems.append(
            "missing bibliography entries: "
            + ", ".join(sorted(cited_keys - bibliography_keys))
        )

    evidence_index = " ".join(
        (ROOT / "SUPPORTING_EVIDENCE_INDEX.md")
        .read_text(encoding="utf-8")
        .split()
    )
    for phrase in (
        "PUBLIC_STUDY_AUTHORITY.json",
        "PUBLIC_REPORTING_SUMMARY.json",
        "Large third-party or project-generated pixel payloads are not stored in Git",
    ):
        if phrase not in evidence_index:
            problems.append(f"supporting-evidence scope missing: {phrase}")

    public_authority_path = (
        REPO
        / "results_v2/canonical_test_host_repair_v1/PUBLIC_STUDY_AUTHORITY.json"
    )
    reporting_summary_path = (
        REPO
        / "results_v2/canonical_test_host_repair_v1/PUBLIC_REPORTING_SUMMARY.json"
    )
    public_authority = json.loads(public_authority_path.read_text(encoding="utf-8"))
    reporting_summary_relative = (
        "results_v2/canonical_test_host_repair_v1/"
        "PUBLIC_REPORTING_SUMMARY.json"
    )
    if (
        public_authority["artifacts"].get(reporting_summary_relative)
        != sha256(reporting_summary_path)
    ):
        problems.append("public study authority does not bind reporting summary")
    if sha256(public_authority_path) not in evidence_index:
        problems.append("supporting-evidence index has stale public-authority hash")

    table_registry = json.loads(
        (ROOT / "tables/sources.json").read_text(encoding="utf-8")
    )["tables"]["confirmed_43_provenance"]
    registered_paths = {
        table_registry["generator"]: table_registry["generator_sha256"],
        table_registry["input"]: table_registry["input_sha256"],
        **table_registry["outputs"],
    }
    for relative, expected in registered_paths.items():
        path = REPO / relative
        if not path.exists() or sha256(path) != expected:
            problems.append(f"provenance table registry mismatch: {relative}")

    for name in ("manuscript.pdf",):
        if not (ROOT / name).exists():
            problems.append(f"missing PDF: {name}")
    if not (ROOT / "SUPPORTING_EVIDENCE_INDEX.md").exists():
        problems.append("missing compact supporting-evidence index")

    log_problems: dict[str, list[str]] = {}
    for log_name in ("main.log",):
        path = ROOT / log_name
        if not path.exists():
            continue
        matches = [
            line
            for line in path.read_text(errors="replace").splitlines()
            if re.search(
                r"Overfull|undefined references|Rerun to get|error:",
                line,
                re.IGNORECASE,
            )
        ]
        if matches:
            log_problems[log_name] = matches
    if log_problems:
        problems.append("TeX logs contain layout or reference problems")

    receipt = {
        "schema_version": 1,
        "status": "manuscript_checks_pass" if not problems else "fail",
        "abstract_words": count,
        "pdf_pages": {
            name: pdf_pages(ROOT / name)
            for name in ("manuscript.pdf",)
            if (ROOT / name).exists()
        },
        "sha256": {
            name: sha256(ROOT / name)
            for name in (
                "manuscript.pdf",
                *ACTIVE_TEX,
                "main.tex",
                "SUPPORTING_EVIDENCE_INDEX.md",
                "build_documents.py",
                "generate_confirmed_43_provenance_table.py",
                "verify_submission.py",
                "tables/confirmed_43_provenance.csv",
                "tables/confirmed_43_provenance.tex",
                "tables/sources.json",
            )
            if (ROOT / name).exists()
        },
        "scientific_authority_sha256": {
            str(public_authority_path.relative_to(REPO)): sha256(
                public_authority_path
            ),
            str(reporting_summary_path.relative_to(REPO)): sha256(
                reporting_summary_path
            ),
        },
        "log_problems": log_problems,
        "problems": problems,
    }
    (ROOT / "VERIFICATION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(ROOT / "VERIFICATION_RECEIPT.json")
    if problems:
        raise SystemExit("\n".join(problems))


if __name__ == "__main__":
    main()
