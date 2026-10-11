"""Hermes-local: a Hub mine must let readers in while it pages the palace."""

from mempalace import palace as palace_pkg


class _Pages:
    def __init__(self, total):
        self.total = total

    def count(self):
        return self.total

    def get(self, limit, offset, include):
        ids = list(range(offset, min(offset + limit, self.total)))
        return {"ids": ids, "metadatas": [{"i": i} for i in ids]}


def test_paged_metadata_yields_between_pages():
    calls = []
    with palace_pkg.mine_yield_hook(lambda: calls.append(1)):
        rows = list(palace_pkg._paged_metadata(_Pages(2500)))
    assert len(rows) == 2500
    assert len(calls) == 3


def test_paged_metadata_without_hook_is_unchanged():
    assert len(list(palace_pkg._paged_metadata(_Pages(10)))) == 10
