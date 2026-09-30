"""Tests for cli.py command wiring (Phase 5b soft-scoring registration)."""

from __future__ import annotations

import krm.cli as cli


class TestSoftCommandWiring:
    def test_commands_dict_contains_soft(self):
        assert "soft" in cli.COMMANDS

    def test_cmd_all_orders_soft_between_phase5_and_model(self, monkeypatch, capsys):
        def _noop(config_path=None):
            return None

        for name in (
            "cmd_collect",
            "cmd_classify",
            "cmd_phase25",
            "cmd_roles",
            "cmd_skills",
            "cmd_characteristics",
            "cmd_soft",
            "cmd_model",
            "cmd_validate",
            "cmd_report",
        ):
            monkeypatch.setattr(cli, name, _noop)

        cli.cmd_all(None)
        out = capsys.readouterr().out

        p5 = out.index("Phase 5: Map to characteristics")
        p5b = out.index("Phase 5b: Soft competences")
        p6 = out.index("Phase 6: Build models")
        assert p5 < p5b < p6
