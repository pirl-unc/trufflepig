"""Lightweight contracts for analyze pipeline orchestration.

These classes deliberately avoid importing pandas, matplotlib, or the
large reference datasets. They are safe to import from both the current
``pirlygenes`` CLI and a future standalone ``trufflepig`` runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnalyzeConfig:
    """User-facing analyze options after CLI parsing.

    The fields mirror the public ``pirlygenes analyze`` arguments. Keeping
    them in one immutable object makes each pipeline step depend on a
    documented contract instead of a long positional argument list.
    """

    input_path: str
    output_dir: str = "pirlygenes-output"
    output_image_prefix: str | None = None
    aggregate_gene_expression: bool = False
    genes: str | None = None
    transcripts: str | None = None
    label_genes: str | None = None
    gene_name_col: str | None = None
    gene_id_col: str | None = None
    sample_id_col: str | None = None
    sample_id_value: str | None = None
    output_dpi: int = 300
    plot_height: float = 14.0
    plot_aspect: float = 1.4
    cancer_type: str | None = None
    sample_mode: str = "auto"
    tumor_context: str = "auto"
    site_hint: str | None = None
    met_site: str | None = None
    decomposition_templates: str | None = None
    hla_types: str | None = None
    fusions: str | None = None
    alterations: str | None = None
    alignment_qc: str | None = None
    expression_qc_rescue: str = "auto"
    expression_qc_remove_noncoding: bool = False
    therapy_target_top_k: int = 10
    therapy_target_tpm_threshold: float = 30.0
    deprecated_figures: bool = False
    force: bool = False

    def template_overrides(self) -> list[str]:
        if self.decomposition_templates is None:
            return []
        return [
            token.strip()
            for token in str(self.decomposition_templates).split(",")
            if token.strip()
        ]

    def hla_type_list(self) -> list[str]:
        """Normalized class-I HLA alleles supplied for TCR-style gating."""
        from ..hla import parse_hla_types

        return parse_hla_types(self.hla_types)

    def fusion_path_list(self) -> list[str]:
        """Fusion evidence files supplied for loose parsing."""
        from ..fusions import split_fusion_paths

        return split_fusion_paths(self.fusions)

    def alteration_input_list(self) -> list[str]:
        """Alteration evidence files or inline calls supplied for loose parsing."""
        from ..alterations import split_alteration_inputs

        return split_alteration_inputs(self.alterations)

    def public_dict(self) -> dict[str, Any]:
        """JSON-safe representation for manifests and provenance files."""
        payload = asdict(self)
        payload["hla_type_list"] = self.hla_type_list()
        payload["fusion_path_list"] = self.fusion_path_list()
        payload["alteration_input_list"] = self.alteration_input_list()
        return payload


@dataclass(frozen=True)
class InputResolution:
    """Resolved expression inputs used by the runner."""

    gene_input: str
    transcript_input: str | None
    aggregate_gene_expression: bool
    input_level: str
    notes: tuple[str, ...] = ()

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalyzePaths:
    """All run-specific filesystem naming in one place."""

    out_dir: Path
    prefix_base: str
    sample_display_id: str

    @property
    def prefix(self) -> str:
        return str(self.out_dir / self.prefix_base)

    def file(self, suffix: str) -> str:
        """Return ``<prefix>-<suffix>`` for a run artifact."""
        clean_suffix = str(suffix).lstrip("-")
        return f"{self.prefix}-{clean_suffix}"

    def public_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "prefix_base": self.prefix_base,
            "prefix": self.prefix,
            "sample_display_id": self.sample_display_id,
        }


@dataclass(frozen=True)
class AnalyzeArtifact:
    """One emitted file and the pipeline role it serves."""

    path: str
    kind: str
    step: str
    role: str
    description: str = ""
    include_in_pdf: bool = False
    move_to_figures: bool = False
    audit_only: bool = False

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepRecord:
    """Structured trace entry for a pipeline step."""

    name: str
    status: str = "pending"
    message: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalyzeRun:
    """Mutable per-run state shared across analyze sub-steps."""

    config: AnalyzeConfig
    inputs: InputResolution
    paths: AnalyzePaths
    artifacts: list[AnalyzeArtifact] = field(default_factory=list)
    steps: dict[str, StepRecord] = field(default_factory=dict)

    def note_step(
        self,
        name: str,
        *,
        status: str = "completed",
        message: str = "",
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> StepRecord:
        record = self.steps.get(name) or StepRecord(name=name)
        record.status = status
        record.message = message
        if inputs:
            record.inputs.update(inputs)
        if outputs:
            record.outputs.update(outputs)
        if warnings:
            record.warnings.extend(warnings)
        self.steps[name] = record
        return record

    def add_artifact(
        self,
        path: str | Path,
        *,
        kind: str,
        step: str,
        role: str,
        description: str = "",
        include_in_pdf: bool = False,
        move_to_figures: bool = False,
        audit_only: bool = False,
    ) -> str:
        artifact = AnalyzeArtifact(
            path=str(path),
            kind=kind,
            step=step,
            role=role,
            description=description,
            include_in_pdf=include_in_pdf,
            move_to_figures=move_to_figures,
            audit_only=audit_only,
        )
        self.artifacts.append(artifact)
        return artifact.path

    def public_manifest(self) -> dict[str, Any]:
        return {
            "config": self.config.public_dict(),
            "inputs": self.inputs.public_dict(),
            "paths": self.paths.public_dict(),
            "steps": {
                name: record.public_dict()
                for name, record in sorted(self.steps.items())
            },
            "artifacts": [artifact.public_dict() for artifact in self.artifacts],
        }
