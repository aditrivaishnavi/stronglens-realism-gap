from scripts.verify_public_renderer_evidence import verify


def test_renderer_evidence_bundle_is_hash_closed() -> None:
    result = verify()
    assert result["status"] == "renderer_public_evidence_verified"
    assert result["sample_rows"] == 142
