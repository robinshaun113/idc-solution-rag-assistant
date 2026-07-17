from langchain_core.documents import Document

from evidence import (
    attach_evidence_metadata,
    backfill_legacy_evidence_metadata,
    evidence_payload,
    make_evidence_id,
)


def test_evidence_id_is_stable_and_content_sensitive():
    first = make_evidence_id("standard.pdf", 1, 2, "温度 18℃~27℃")
    second = make_evidence_id("standard.pdf", 1, 2, "温度 18℃~27℃")
    changed = make_evidence_id("standard.pdf", 1, 2, "温度 20℃~27℃")
    assert first == second
    assert first != changed


def test_evidence_payload_exposes_one_based_page():
    docs = [Document(page_content="条款内容", metadata={"source": "/tmp/a.pdf", "page": 0})]
    attach_evidence_metadata(docs)
    payload = evidence_payload(docs[0])
    assert payload["source"] == "a.pdf"
    assert payload["page"] == 1
    assert payload["evidence_id"].startswith("ev_")
    assert payload["preview"] == "条款内容"


def test_legacy_backfill_does_not_depend_on_retrieval_rank():
    a = Document(page_content="同一条款", metadata={"source": "a.pdf", "page": 1})
    b = Document(page_content="其他条款", metadata={"source": "a.pdf", "page": 2})
    backfill_legacy_evidence_metadata([a, b])
    first_id = a.metadata["evidence_id"]

    a_again = Document(page_content="同一条款", metadata={"source": "a.pdf", "page": 1})
    backfill_legacy_evidence_metadata([b, a_again])
    assert a_again.metadata["evidence_id"] == first_id
