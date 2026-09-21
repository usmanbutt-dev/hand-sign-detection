"""Integrity tests for the static portfolio output."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path


class AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.assets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        attribute = "href" if tag == "link" else "src"
        if tag in {"link", "script", "img"} and values.get(attribute):
            self.assets.append(values[attribute] or "")


def test_all_local_site_assets_exist():
    docs = Path("docs")
    parser = AssetParser()
    parser.feed((docs / "index.html").read_text(encoding="utf-8"))

    for value in parser.assets:
        if not value.startswith(("http://", "https://", "#", "mailto:")):
            assert (docs / value).is_file(), value


def test_published_metrics_equal_training_artifact():
    measured = json.loads(Path("artifacts/classifier_metrics.json").read_text())
    published = json.loads(Path("docs/assets/classifier_metrics.json").read_text())

    assert published == measured


def test_pages_workflow_publishes_docs_directory():
    workflow = Path(".github/workflows/pages.yml").read_text(encoding="utf-8")

    assert "path: docs" in workflow
    assert "actions/deploy-pages@" in workflow
