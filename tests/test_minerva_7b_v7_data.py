import hashlib
import json
from array import array
from pathlib import Path

import pytest

from sonnet_corpus.canonical_corpus_reader import CanonicalTextUnit
from sonnet_training.minerva_7b_v7_data import (
    BROADER_ROLES,
    CountedUnit,
    EncodedDocument,
    PoolSpec,
    build_broader_split_rows,
    build_exact_stage_plan,
    build_pool_specs,
    compare_encoded_builds,
    count_canonical_units,
    encode_pool,
    encoded_content_identity,
    load_training_data_policy,
    render_encoded_data_markdown,
    select_broader_validation,
    write_broader_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "data/metadata/minerva_7b_v7_training_data_policy_v1.json"
COMPOSITION_POLICY_PATH = (
    ROOT / "data/metadata/minerva_7b_v7_composition_policy_v1.json"
)


class FakeTokenizer:
    eos_token_id = 2

    def __len__(self):
        return 512

    def __call__(self, text, **kwargs):
        return {"input_ids": [3 + ord(character) % 100 for character in text]}


class TextPathOnlyReader:
    def read_text(self, unit):  # pragma: no cover - a failed safety invariant
        raise AssertionError("text-path documents must not ask the corpus reader")


class OneUnitReader:
    def __init__(self, unit):
        self.units = (unit,)

    def read_text(self, unit):
        return unit.unit_id


def _unit(
    unit_id: str,
    *,
    role: str,
    kind: str = "broader",
    author: str = "",
    training_eligible: bool = True,
) -> CanonicalTextUnit:
    text = unit_id
    return CanonicalTextUnit(
        unit_id=unit_id,
        unit_kind=kind,
        source_group="source",
        source_id=unit_id,
        title=unit_id,
        author=author,
        source_archive="archive",
        source_url=f"https://example.test/{unit_id}",
        epoch_bucket="16th_century",
        final_role=role,
        attribution_id=f"attr:{unit_id}",
        activation_status=(
            "active_training" if training_eligible else "protected_v6_validation_test"
        ),
        training_eligible=training_eligible,
        storage_kind="synthetic",
        storage_path=f"data/{unit_id}.txt",
        byte_start=0,
        byte_end=len(text.encode("utf-8")),
        logical_character_count=len(text),
        logical_byte_count=len(text.encode("utf-8")),
        logical_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        physical_file_sha256="0" * 64,
    )


def _counted(
    unit_id: str,
    *,
    role: str,
    tokens: int,
    component: str = "",
    kind: str = "broader",
    split: str = "",
    training_eligible: bool = True,
) -> CountedUnit:
    unit = _unit(
        unit_id,
        role=role,
        kind=kind,
        author=component,
        training_eligible=training_eligible,
    )
    return CountedUnit(
        unit=unit,
        text_tokens=tokens - 1,
        training_tokens=tokens,
        component_key=component if kind == "broader" else "",
        component_id=(
            "component:" + hashlib.sha256(component.encode()).hexdigest()[:16]
            if kind == "broader"
            else ""
        ),
        v7_split=split,
        author_key=(component if kind == "broader" else "author:synthetic"),
        work_key=f"work:{unit_id}",
        epoch_key="16th_century",
    )


def _validation_fixture() -> list[CountedUnit]:
    rows = []
    for role in BROADER_ROLES:
        rows.append(
            _counted(
                f"{role}:shared", role=role, tokens=10, component="author:shared"
            )
        )
        rows.append(
            _counted(
                f"{role}:oversize",
                role=role,
                tokens=30,
                component=f"author:oversize:{role}",
            )
        )
        for index in range(6):
            rows.append(
                _counted(
                    f"{role}:filler:{index}",
                    role=role,
                    tokens=10,
                    component=f"work:{role}:{index}",
                )
            )
    return rows


def _encoded_document(path: Path, unit_id: str, text: str) -> EncodedDocument:
    path.write_text(text, encoding="utf-8")
    return EncodedDocument(
        unit_id=unit_id,
        logical_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        characters=len(text),
        expected_tokens=len(text) + 1,
        source_group="synthetic",
        source_id=unit_id,
        author_key=f"author:{unit_id}",
        work_key=f"work:{unit_id}",
        epoch_key="16th_century",
        text_path=path,
    )


def _encode_synthetic_pool(tmp_path: Path, directory: str) -> dict:
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir(exist_ok=True)
    pool = PoolSpec(
        pool_id="train_historical_general",
        corpus_role="historical_general",
        split="train",
        documents=(
            _encoded_document(documents_dir / "one.txt", "one", "abc"),
            _encoded_document(documents_dir / "two.txt", "two", "defg"),
        ),
    )
    return encode_pool(
        pool=pool,
        reader=TextPathOnlyReader(),
        output_dir=tmp_path / directory,
        tokenizer=FakeTokenizer(),
        tokenizer_fingerprint="f" * 64,
        eos_token_id=FakeTokenizer.eos_token_id,
        shard_target_tokens=6,
        checkpoint_interval_documents=1,
        progress_interval_documents=1,
        max_documents=None,
    )


def test_policy_locks_whole_window_alignment_and_no_gpu_authorization():
    policy = load_training_data_policy(POLICY_PATH)

    assert policy["windowing"]["context_length"] == 2048
    assert policy["windowing"]["source_span_tokens"] == 2049
    assert policy["windowing"]["budget_alignment_tokens"] == 40_960
    assert policy["activation_policy"]["requires_independent_encoded_builds"] == 2
    assert policy["activation_policy"]["gpu_work_authorized"] is False
    assert policy["activation_policy"]["cache_deletion_authorized"] is False


def test_policy_rejects_alignment_that_cannot_preserve_whole_window_ratios(tmp_path):
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload["windowing"]["budget_alignment_tokens"] = 10_240
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="alignment"):
        load_training_data_policy(path)


def test_validation_selects_global_component_and_keeps_oversize_components_training():
    counted = _validation_fixture()

    result = select_broader_validation(
        counted,
        target_fraction=0.1,
        maximum_component_fraction=0.2,
        tolerance=0.01,
        seed=1337,
    )

    assert result["selected_components"] == {"author:shared"}
    assert len(result["oversize_components"]) == 3
    assert all(
        result["unit_splits"][f"{role}:shared"] == "validation"
        for role in BROADER_ROLES
    )
    assert all(
        result["report"]["roles"][role]["validation_fraction"] == pytest.approx(0.1)
        for role in BROADER_ROLES
    )
    rows = build_broader_split_rows(counted, result)
    assert {
        row["component_decision"]
        for row in rows
        if row["oversize_component"] == "true"
    } == {"oversize_component_retained_training"}


def test_validation_fails_when_no_component_can_meet_tolerance():
    counted = [
        _counted(
            f"{role}:only", role=role, tokens=100, component=f"author:{role}"
        )
        for role in BROADER_ROLES
    ]

    with pytest.raises(ValueError, match="missed its approved tolerance"):
        select_broader_validation(
            counted,
            target_fraction=0.1,
            maximum_component_fraction=0.2,
            tolerance=0.01,
            seed=1337,
        )


def test_stage_plan_uses_exact_ratios_and_whole_2048_token_windows():
    plan = build_exact_stage_plan(
        training_role_tokens={
            "historical_general": 58_000_000,
            "historical_non_sonnet_poetry": 18_000_000,
            "nineteenth_century_bridge": 96_000_000,
        },
        v7_train_tokens=3_500_000,
        replay_tokens=2_000_000,
        composition_policy=json.loads(
            COMPOSITION_POLICY_PATH.read_text(encoding="utf-8")
        ),
        data_policy=load_training_data_policy(POLICY_PATH),
    )

    assert plan["alignment_tokens"] == 40_960
    assert plan["sampling_assignment_status"] == "pending_deterministic_training_sampler"
    for stage in plan["stages"]:
        assert stage["budget_tokens"] % 40_960 == 0
        assert sum(row["draw_tokens"] for row in stage["components"]) == stage[
            "budget_tokens"
        ]
        assert all(row["draw_tokens"] % 2048 == 0 for row in stage["components"])
        assert sum(row["draw_windows_2048"] for row in stage["components"]) == stage[
            "target_windows_2048"
        ]
        assert all(row["available_tokens"] > 0 for row in stage["components"])


def test_sonnet_counting_carries_v7_author_work_and_harmonized_epoch_metadata():
    unit = _unit(
        "sonnet:metadata",
        role="standard_sonnets",
        kind="standard_sonnet",
    )

    counted = count_canonical_units(
        reader=OneUnitReader(unit),
        v7_rows={
            unit.unit_id: {
                "include_in_v7": "true",
                "v7_split": "train",
                "v7_training_eligible": "true",
                "author_group_id": "author:resolved",
                "work_group_id": "work:resolved",
                "epoch_bucket": "500",
            }
        },
        tokenizer=FakeTokenizer(),
        eos_token_id=FakeTokenizer.eos_token_id,
        epoch_harmonization={"500": "16th_century"},
    )

    assert counted[0].author_key == "author:resolved"
    assert counted[0].work_key == "work:resolved"
    assert counted[0].epoch_key == "16th_century"


def test_encode_pool_writes_signed_int32_eos_and_document_boundaries(tmp_path):
    report = _encode_synthetic_pool(tmp_path, "encoded")

    assert report["status"] == "complete"
    assert report["documents"] == 2
    assert report["tokens"] == 9
    assert report["eos_tokens"] == 2
    assert len(report["shards"]) == 2
    first = array("i")
    first.frombytes(Path(report["shards"][0]["path"]).read_bytes())
    assert first.tolist()[-1] == FakeTokenizer.eos_token_id
    index_rows = [
        json.loads(line)
        for line in Path(report["document_index"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["unit_id"] for row in index_rows] == ["one", "two"]
    assert index_rows[0]["token_end"]["token_offset"] == 4
    assert index_rows[0]["author_key"] == "author:one"
    assert index_rows[0]["work_key"] == "work:one"
    assert index_rows[0]["epoch_key"] == "16th_century"


def test_encode_pool_resumes_at_a_completed_document_boundary(tmp_path):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    pool = PoolSpec(
        pool_id="train_historical_general",
        corpus_role="historical_general",
        split="train",
        documents=(
            _encoded_document(documents_dir / "one.txt", "one", "abc"),
            _encoded_document(documents_dir / "two.txt", "two", "defg"),
        ),
    )
    arguments = dict(
        pool=pool,
        reader=TextPathOnlyReader(),
        output_dir=tmp_path / "encoded",
        tokenizer=FakeTokenizer(),
        tokenizer_fingerprint="f" * 64,
        eos_token_id=FakeTokenizer.eos_token_id,
        shard_target_tokens=100,
        checkpoint_interval_documents=1,
        progress_interval_documents=1,
    )

    paused = encode_pool(**arguments, max_documents=1)
    completed = encode_pool(**arguments, max_documents=None)

    assert paused["status"] == "incomplete"
    assert paused["documents"] == 1
    assert completed["status"] == "complete"
    assert completed["documents"] == 2
    assert not (tmp_path / "encoded/.train_historical_general.checkpoint.json").exists()


def test_independent_encoded_builds_match_by_content_not_directory(tmp_path):
    primary_pool = _encode_synthetic_pool(tmp_path, "primary")
    reproduction_pool = _encode_synthetic_pool(tmp_path, "reproduction")
    common = {
        "broader_validation": {"identity": "same"},
        "stage_plan": {"identity": "same"},
    }
    primary = {
        **common,
        "pools": [primary_pool],
        "encoded_content_identity_sha256": encoded_content_identity([primary_pool]),
    }
    reproduction = {
        **common,
        "pools": [reproduction_pool],
        "encoded_content_identity_sha256": encoded_content_identity(
            [reproduction_pool]
        ),
    }

    comparison = compare_encoded_builds(primary, reproduction)

    assert comparison["match"] is True
    assert primary_pool["content_identity_sha256"] == reproduction_pool[
        "content_identity_sha256"
    ]


def test_pool_assignment_keeps_v7_validation_test_and_protected_units_out_of_train(
    tmp_path,
):
    counted = [
        _counted(
            f"broad:{role}", role=role, tokens=5, component=f"work:{role}"
        )
        for role in BROADER_ROLES
    ]
    counted.extend(
        _counted(
            f"broad-validation:{role}",
            role=role,
            tokens=5,
            component=f"work:validation:{role}",
        )
        for role in BROADER_ROLES
    )
    counted.extend(
        [
            _counted(
                "sonnet:train",
                role="standard_sonnets",
                kind="standard_sonnet",
                split="train",
                tokens=5,
            ),
            _counted(
                "sonnet:validation",
                role="standard_sonnets",
                kind="standard_sonnet",
                split="validation",
                tokens=5,
                training_eligible=False,
            ),
            _counted(
                "sonnet:test",
                role="standard_sonnets",
                kind="standard_sonnet",
                split="test",
                tokens=5,
                training_eligible=False,
            ),
        ]
    )
    replay = tmp_path / "replay.txt"
    replay.write_text("modern", encoding="utf-8")
    replay_report = tmp_path / "replay.json"
    replay_report.write_text(
        json.dumps({"output_sha256": hashlib.sha256(replay.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    pools = build_pool_specs(
        counted=counted,
        broader_splits={
            row.unit.unit_id: ("train" if index < 3 else "validation")
            for index, row in enumerate(counted[:6])
        },
        replay_text_path=replay,
        replay_report_path=replay_report,
        tokenizer=FakeTokenizer(),
        eos_token_id=FakeTokenizer.eos_token_id,
    )
    by_id = {pool.pool_id: pool for pool in pools}

    assert [row.unit_id for row in by_id["sonnets_train"].documents] == [
        "sonnet:train"
    ]
    assert by_id["sonnets_train"].documents[0].author_key == "author:synthetic"
    assert by_id["sonnets_train"].documents[0].epoch_key == "16th_century"
    assert [row.unit_id for row in by_id["sonnets_validation"].documents] == [
        "sonnet:validation"
    ]
    assert [row.unit_id for row in by_id["sonnets_test"].documents] == [
        "sonnet:test"
    ]


def test_split_csv_and_markdown_are_deterministic(tmp_path):
    counted = _validation_fixture()
    split = select_broader_validation(
        counted,
        target_fraction=0.1,
        maximum_component_fraction=0.2,
        tolerance=0.01,
        seed=1337,
    )
    rows = build_broader_split_rows(counted, split)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    write_broader_split_manifest(rows, first)
    write_broader_split_manifest(rows, second)
    report = {
        "status": "active_verified",
        "totals": {
            "documents": 1,
            "tokens": 2,
            "eos_tokens": 1,
            "shards": 1,
            "encoded_bytes": 8,
        },
        "broader_validation": split["report"],
        "stage_plan": build_exact_stage_plan(
            training_role_tokens={role: 1_000_000 for role in BROADER_ROLES},
            v7_train_tokens=1_000_000,
            replay_tokens=1_000_000,
            composition_policy=json.loads(
                COMPOSITION_POLICY_PATH.read_text(encoding="utf-8")
            ),
            data_policy=load_training_data_policy(POLICY_PATH),
        ),
    }

    assert first.read_bytes() == second.read_bytes()
    assert render_encoded_data_markdown(report) == render_encoded_data_markdown(report)
    assert "not sampled window assignments" in render_encoded_data_markdown(report)


def test_completed_checkpoint_artifacts_freeze_counts_budgets_and_identities():
    report_path = ROOT / "reports/minerva_7b_v7_encoded_data_v1.json"
    split_path = ROOT / "data/metadata/minerva_7b_v7_broader_splits_v1.csv"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "active_verified"
    assert report["totals"] == {
        "characters": 652_400_268,
        "documents": 26_935,
        "encoded_bytes": 723_076_816,
        "eos_tokens": 26_935,
        "shards": 30,
        "tokens": 180_769_204,
    }
    assert report["encoded_content_identity_sha256"] == (
        "e328bbb0f318800a6be9d114d9893b277a70df34c6b811b90b90bb7d65187504"
    )
    assert [stage["budget_tokens"] for stage in report["stage_plan"]["stages"]] == [
        67_665_920,
        24_903_680,
        4_423_680,
    ]
    assert report["stage_plan"]["total_budget_tokens"] == 96_993_280
    assert report["stage_plan"]["total_target_windows_2048"] == 47_360
    assert report["reproduction"]["match"] is True
    assert hashlib.sha256(split_path.read_bytes()).hexdigest() == (
        "92e56744a0e54ce959d7f9cc347b586ab8aae84c9bb104a4c6eb08d6dc683a6f"
    )
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == (
        "2acaf9c8a598e2543017b17b4b60f2d9d4a4b18520345ded1cd8712bc9304f3e"
    )
