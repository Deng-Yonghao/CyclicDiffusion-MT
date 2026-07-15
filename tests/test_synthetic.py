# tests/test_synthetic.py
import torch, pytest
from cyclicdiffusion_mt.data.synthetic import (
    compute_quality_score,
    SyntheticMultiTargetBuilder,
)


class TestQualityScore:
    def test_perfect(self):
        s = compute_quality_score(0.0, 500.0, 0.0)
        assert 0.9 < s <= 1.0

    def test_bad_clash(self):
        s = compute_quality_score(100.0, 500.0, 0.0)
        assert s < 0.5

    def test_bad_contact(self):
        s = compute_quality_score(0.0, 10.0, 0.0)
        assert s < 0.5

    def test_bad_docking(self):
        s = compute_quality_score(0.0, 500.0, 50.0)
        assert s < 0.5

    def test_range(self):
        for _ in range(100):
            s = compute_quality_score(
                torch.rand(1).item() * 50,
                torch.rand(1).item() * 1000,
                torch.rand(1).item() * 20,
            )
            assert 0.0 <= s <= 1.0


class TestSyntheticBuilder:
    @pytest.fixture
    def peptide_data(self):
        return {
            "peptide_torsions": torch.randn(8, 7),
            "peptide_aa_types": torch.randint(0, 25, (8,)),
            "cyclo_mode": 0,
            "dG_rosetta": -32.0,
            "confidence": 1.0,
        }

    @pytest.fixture
    def target_pool(self):
        return [
            {
                "coords": torch.randn(50, 14, 3),
                "sequence": torch.randint(0, 25, (50,)),
                "name": "target_A",
            },
            {
                "coords": torch.randn(60, 14, 3),
                "sequence": torch.randint(0, 25, (60,)),
                "name": "target_B",
            },
            {
                "coords": torch.randn(45, 14, 3),
                "sequence": torch.randint(0, 25, (45,)),
                "name": "target_C",
            },
        ]

    def test_build_one_target(self, peptide_data, target_pool):
        builder = SyntheticMultiTargetBuilder(max_targets=1)
        results = builder.build(peptide_data, target_pool[:1])
        assert len(results) == 1
        entry = results[0]
        assert "peptide_torsions" in entry
        assert "target_coords" in entry
        assert len(entry["target_coords"]) == 1

    def test_build_multi_target(self, peptide_data, target_pool):
        builder = SyntheticMultiTargetBuilder(max_targets=3)
        results = builder.build(peptide_data, target_pool)
        assert len(results) >= 1
        entry = results[0]
        assert "confidence" in entry
        assert 0.0 <= entry["confidence"] <= 1.0

    def test_output_is_manifest_format(self, peptide_data, target_pool):
        builder = SyntheticMultiTargetBuilder(max_targets=2)
        results = builder.build(peptide_data, target_pool[:2])
        for entry in results:
            assert isinstance(entry["peptide_torsions"], torch.Tensor)
            assert isinstance(entry["peptide_aa_types"], torch.Tensor)
            assert isinstance(entry["target_coords"], list)
            assert "cyclo_mode" in entry
            assert "confidence" in entry

    def test_empty_pool(self, peptide_data):
        builder = SyntheticMultiTargetBuilder(max_targets=3)
        results = builder.build(peptide_data, [])
        assert len(results) == 0
