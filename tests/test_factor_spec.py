from pathlib import Path

from fu_alpha_research.factor_spec import load_selected


def test_factor_spec_views(tmp_path: Path):
    path = tmp_path / "selected.txt"
    path.write_text("mom_2\ntsz_bop\ncsz_macd_3_13\ncsr_cpos\n")
    spec = load_selected(path)
    assert spec.n_factors == 4
    assert spec.by_view["raw"] == ["mom_2"]
    assert spec.by_view["tsz"] == ["bop"]
    assert spec.by_view["csz"] == ["macd_3_13"]
    assert spec.by_view["csr"] == ["cpos"]
