"""Tests for enhanced exporter features: last_learned_date, required_inputs."""

from __future__ import annotations

from pathlib import Path

from agenthandover_worker.exporter import IndexGenerator


def _sample_template_with_vars(
    slug: str = "test_workflow",
    title: str = "Test Workflow",
    confidence_avg: float = 0.88,
    variables: list | None = None,
) -> dict:
    return {
        "slug": slug,
        "title": title,
        "steps": [
            {"step": "click", "target": "Submit button", "selector": None,
             "parameters": {}, "confidence": 0.9},
        ],
        "variables": variables or [],
        "confidence_avg": confidence_avg,
        "episode_count": 5,
        "apps_involved": ["Chrome"],
    }


# ------------------------------------------------------------------
# 1. Last learned date in index
# ------------------------------------------------------------------


class TestLastLearnedDate:
    def test_index_contains_last_learned(self, tmp_path: Path) -> None:
        sops_dir = tmp_path / "sops"
        sops_dir.mkdir()
        gen = IndexGenerator()

        entries = [_sample_template_with_vars()]
        content = gen.generate_index(sops_dir, entries)

        assert "**Last learned:**" in content


# ------------------------------------------------------------------
# 2. Required inputs in index
# ------------------------------------------------------------------


class TestRequiredInputs:
    def test_index_contains_required_inputs(self, tmp_path: Path) -> None:
        sops_dir = tmp_path / "sops"
        sops_dir.mkdir()
        gen = IndexGenerator()

        entries = [_sample_template_with_vars(
            variables=[
                {"name": "customer_name", "type": "string", "example": "John"},
                {"name": "order_id", "type": "number", "example": "123"},
            ],
        )]
        content = gen.generate_index(sops_dir, entries)

        assert "**Required inputs:**" in content
        assert "`customer_name` (string)" in content
        assert "`order_id` (number)" in content

    def test_no_required_inputs_when_no_variables(self, tmp_path: Path) -> None:
        sops_dir = tmp_path / "sops"
        sops_dir.mkdir()
        gen = IndexGenerator()

        entries = [_sample_template_with_vars(variables=[])]
        content = gen.generate_index(sops_dir, entries)

        assert "**Required inputs:**" not in content
