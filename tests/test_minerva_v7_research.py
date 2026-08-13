import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from sonnet_analysis.minerva_v7_behavior import (
    analyze_matched_generations,
    build_blinded_review,
)
from sonnet_analysis.minerva_v7_embedding import (
    analyze_embedding_pair,
    count_selected_token_ids,
    resolve_verified_training_shards,
    resolve_token_registry,
)
from sonnet_analysis.minerva_v7_extraction import (
    capture_probe,
    extract_state,
    select_raw_attention_probe_hashes,
    summarize_attention,
    verify_probe_result,
)
from sonnet_analysis.minerva_v7_generation import (
    banned_next_tokens,
    generate_matched_continuation,
    generate_state_outputs,
)
from sonnet_analysis.minerva_v7_memorization import (
    REFERENCE_VERSION,
    load_verified_sonnet_train_reference,
    score_texts_against_reference,
)
from sonnet_analysis.minerva_v7_representation import (
    analyze_state_pair,
    effective_rank,
    linear_cka,
)
from sonnet_analysis.minerva_v7_runtime import load_research_config, load_verified_state
from sonnet_analysis.minerva_v7_runtime import load_verified_comparison


class ToyBlock(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        torch.nn.init.eye_(self.linear.weight)

    def forward(self, hidden):
        return (self.linear(hidden),)


class ToyProbeModel(torch.nn.Module):
    def __init__(self, vocabulary=11, hidden_size=4, blocks=2, heads=2):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(vocabulary, hidden_size)
        self.model.layers = torch.nn.ModuleList([ToyBlock(hidden_size) for _ in range(blocks)])
        self.model.norm = torch.nn.LayerNorm(hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocabulary, bias=False)
        self.heads = heads

    def forward(self, *, input_ids, attention_mask, output_attentions, use_cache, return_dict):
        hidden = self.model.embed_tokens(input_ids)
        attentions = []
        for block in self.model.layers:
            hidden = block(hidden)[0]
            length = hidden.shape[1]
            weights = torch.tril(torch.ones(length, length, device=hidden.device))
            weights = weights / weights.sum(dim=-1, keepdim=True)
            attentions.append(weights.expand(input_ids.shape[0], self.heads, -1, -1))
        hidden = self.model.norm(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden), attentions=tuple(attentions))


def _probe(index=0, domain="historical_general", length=4):
    ids = list(range(1, length + 1))
    digest = hashlib.sha256(b"".join(struct.pack("<I", value) for value in ids)).hexdigest()
    return {
        "probe_id": f"{domain}:{index}", "domain": domain,
        "source_identity": f"source:{index}", "source_split": f"validation_{domain}",
        "input_ids": ids, "attention_mask": [1] * length,
        "selected_positions": [0, length - 1], "input_ids_sha256": digest,
    }


def test_capture_probe_records_all_streams_attention_and_logits():
    model = ToyProbeModel(blocks=2)
    result = capture_probe(
        model=model, probe=_probe(length=4), device="cpu", block_count=2,
        raw_attention_layers=[0, 1], raw_attention_maximum_tokens=3,
        retain_raw_attention=True,
    )

    assert result["raw_hidden_states"].shape == (4, 4, 4)
    assert result["raw_hidden_states"].dtype == torch.bfloat16
    assert result["pooled_hidden_states"].shape == (4, 4)
    assert result["selected_hidden_states"].shape == (4, 2, 4)
    assert result["attention_entropy"].shape == (2, 2)
    assert result["raw_attention"].shape == (2, 2, 3, 3)
    assert result["top_logit_ids"].shape == (2, 11)
    assert not model.training


def test_attention_summary_is_zero_for_deterministic_diagonal_attention():
    attention = torch.eye(3).expand(2, -1, -1)
    entropy, distance = summarize_attention(attention, torch.ones(3, dtype=torch.bool))
    assert entropy.tolist() == pytest.approx([0.0, 0.0])
    assert distance.tolist() == pytest.approx([0.0, 0.0])


def test_state_extraction_is_atomic_hash_verified_and_resumable(tmp_path):
    model = ToyProbeModel(blocks=2)
    probes = [_probe(0), _probe(1, domain="standard_sonnet")]
    destination = tmp_path / "state"
    messages = []
    first = extract_state(
        model=model, probes=probes, destination=destination,
        state_metadata={"state_id": "toy"}, device="cpu", block_count=2,
        raw_attention_layers=[0], raw_attention_maximum_tokens=3,
        raw_attention_probe_hashes=select_raw_attention_probe_hashes(probes),
        progress=messages.append,
    )
    second = extract_state(
        model=model, probes=probes, destination=destination,
        state_metadata={"state_id": "toy"}, device="cpu", block_count=2,
        raw_attention_layers=[0], raw_attention_maximum_tokens=3,
        raw_attention_probe_hashes=select_raw_attention_probe_hashes(probes),
        progress=messages.append,
    )

    assert first["probe_count"] == second["probe_count"] == 2
    assert any("status=reused" in message for message in messages)
    probe_dir = destination / first["probe_results"][0]["path"]
    manifest = verify_probe_result(probe_dir, probes[0])
    assert manifest["v7_test_accessed"] is False
    (probe_dir / "aggregates.safetensors").write_bytes(b"bad")
    with pytest.raises(ValueError, match="file mismatch"):
        verify_probe_result(probe_dir, probes[0])


class ToyGenerationTokenizer:
    all_special_ids = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "PROMPT:"

    def __call__(self, text, *, add_special_tokens, return_tensors):
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        return "".join(chr(value) for value in ids)


class ToyGenerationModel:
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.training = True

    def eval(self):
        self.training = False
        return self

    def __call__(self, **kwargs):
        token = next(self.tokens)
        logits = torch.full((1, 1, 256), -100.0)
        logits[0, 0, token] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=object())


def _generation_recipe():
    return {
        "confirmatory_seed": 4099, "exploratory_replication_seeds": [4100, 4101],
        "temperature": 0.7, "top_p": 0.92, "top_k": None,
        "repetition_penalty": 1.0, "no_repeat_ngram_size": 4,
        "max_new_tokens": 20, "continuation_line_target": 1,
        "conditioning_format": "format",
    }


def test_matched_generation_keeps_token_ids_and_seed_role(tmp_path):
    result = generate_matched_continuation(
        model=ToyGenerationModel(ord(char) for char in "Seconda\n"),
        tokenizer=ToyGenerationTokenizer(), opening_line="Prima", device="cpu",
        seed=4099, max_new_tokens=20, temperature=0.7, top_k=None, top_p=0.92,
        repetition_penalty=1.0, no_repeat_ngram_size=4, continuation_line_target=1,
    )
    assert result["text"] == "Prima\nSeconda\n"
    assert result["generated_token_ids"] == [ord(char) for char in "Seconda\n"]
    assert result["conditioning_input_ids"] == [1, 2]
    completion = generate_state_outputs(
        model=ToyGenerationModel(ord(char) for char in "Seconda\n"),
        tokenizer=ToyGenerationTokenizer(), state_id="state", state_identity_sha256="a" * 64,
        prompts=[{"id": "p", "opening_line": "Prima"}], seeds=[4099],
        recipe=_generation_recipe(), output_dir=tmp_path / "generation", device="cpu",
    )
    payload = json.loads(
        (tmp_path / "generation" / completion["outputs"][0]["path"]).read_text(
            encoding="utf-8"
        )
    )
    assert payload["seed_role"] == "confirmatory"
    assert payload["v7_test_accessed"] is False


def test_no_repeat_ngram_bans_only_matching_continuations():
    assert banned_next_tokens([1, 2, 3, 1, 2], 3) == {3}
    assert banned_next_tokens([1, 2], 4) == set()


def test_linear_cka_and_representation_pair(tmp_path):
    assert linear_cka(torch.eye(3), torch.eye(3)) == pytest.approx(1.0)
    assert linear_cka(torch.ones(3, 2), torch.ones(3, 2)) is None
    assert effective_rank(torch.eye(3)) == pytest.approx(2.0)
    left = tmp_path / "left"
    right = tmp_path / "right"
    probes = [_probe(index, domain) for index, domain in enumerate(("historical_general", "standard_sonnet", "modern_instruction"))]
    extract_state(
        model=ToyProbeModel(blocks=2), probes=probes, destination=left,
        state_metadata={"state_id": "left"}, device="cpu", block_count=2,
        raw_attention_layers=[0], raw_attention_maximum_tokens=3,
        raw_attention_probe_hashes=set(),
    )
    right_model = ToyProbeModel(blocks=2)
    with torch.no_grad():
        right_model.model.embed_tokens.weight.add_(0.1)
    extract_state(
        model=right_model, probes=probes, destination=right,
        state_metadata={"state_id": "right"}, device="cpu", block_count=2,
        raw_attention_layers=[0], raw_attention_maximum_tokens=3,
        raw_attention_probe_hashes=set(),
    )
    report = analyze_state_pair(
        left_state_dir=left, right_state_dir=right,
        comparison_id="left_to_right", authoritative=False,
    )
    assert report["probe_count"] == 3
    assert report["stream_count"] == 4
    assert len(report["layer_rows"]) == 4
    assert "selected_position_linear_cka" in report["layer_rows"][0]


class ToyRegistryTokenizer:
    all_special_ids = [0]

    def encode(self, text, *, add_special_tokens):
        return [len(text)] if "split" not in text else [1, 2]


def _embedding_model(path, offset=0.0):
    path.mkdir()
    embedding = torch.arange(40, dtype=torch.float32).reshape(10, 4) + offset
    save_file(
        {"model.embed_tokens.weight": embedding, "lm_head.weight": embedding * 2},
        path / "model.safetensors",
    )
    return path


def test_embedding_registry_frequency_and_streamed_neighbors(tmp_path):
    registry = {
        "registry_version": "v1",
        "groups": {"historical": ["a", "split"], "neutral": ["bb"]},
        "policy": {"leading_space_variant": False},
    }
    resolved = resolve_token_registry(ToyRegistryTokenizer(), registry)
    assert [row["token_id"] for row in resolved["accepted"]] == [1, 2]
    assert len(resolved["rejected_fragmented_terms"]) == 1
    shard = tmp_path / "tokens.bin"
    shard.write_bytes(struct.pack("<6i", 1, 2, 2, 3, 2, 1))
    frequencies = count_selected_token_ids([shard], [1, 2], chunk_bytes=8)
    assert frequencies == {1: 2, 2: 3}
    report = analyze_embedding_pair(
        left_model_dir=_embedding_model(tmp_path / "left_model"),
        right_model_dir=_embedding_model(tmp_path / "right_model", 0.5),
        resolved_registry=resolved, frequencies=frequencies,
        top_k=2, vocabulary_chunk_rows=3,
    )
    assert report["frequency_control_recorded"]
    assert report["tensors"]["model.embed_tokens.weight"][0]["frequency"] == 2
    assert len(report["tensors"]["lm_head.weight"][0]["left_neighbors"]) == 2


def _generation_dir(path, state_id, text="Prima\nSeconda\n"):
    path.mkdir(parents=True)
    output = path / "one.json"
    payload = {
        "state_id": state_id, "v7_test_accessed": False,
        "prompt": {"id": "p", "author": "A", "period": "XIII"},
        "seed": 4099, "text": text, "opening_line": "Prima",
    }
    output.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    (path / "complete.json").write_text(
        json.dumps({"v7_test_accessed": False, "outputs": [{"path": output.name, "sha256": digest}]}),
        encoding="utf-8",
    )
    return path


def test_behavior_requires_matched_grid_and_writes_blinded_review(tmp_path):
    report = analyze_matched_generations(
        state_directories={
            "left": _generation_dir(tmp_path / "left", "left"),
            "right": _generation_dir(tmp_path / "right", "right"),
        },
        confirmatory_seed=4099,
        memorization_records=[
            {
                "record_id": "train", "source_id": "Training",
                "text": "Testo completamente diverso senza una lunga copia.",
            }
        ],
        authoritative=False,
    )
    assert len(report["rows"]) == 2
    assert all(row["analysis_role"] == "confirmatory" for row in report["rows"])
    assert report["memorization_scored"]
    artifacts = build_blinded_review(
        behavior_report=report,
        mapping_path=tmp_path / "mapping.json",
        review_path=tmp_path / "review.md",
    )
    assert artifacts["output_count"] == 2
    assert "Historical Register" in (tmp_path / "review.md").read_text(encoding="utf-8")
    assert "state_id" not in (tmp_path / "review.md").read_text(encoding="utf-8")


def test_research_config_and_state_audit_are_fail_closed(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = load_research_config(root / "configs/minerva_7b_v7_research.json")
    assert not config["authorization"]["causal_experiments_authorized"]
    model = tmp_path / "state/model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "hash_verification_performed": False,
                "states": [
                    {
                        "state_id": "state", "status": "complete",
                        "path": str(model.parent), "state_identity_sha256": "a" * 64,
                    }
                ],
            }
        ), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="full state hash audit"):
        load_verified_state(audit, "state")


def test_embedding_comparison_rejects_arbitrary_or_unregistered_states(tmp_path):
    model = tmp_path / "state/model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "hash_verification_performed": True,
                "states": [
                    {
                        "state_id": "arbitrary", "status": "complete",
                        "path": str(model.parent), "model_dir": str(model),
                        "state_identity_sha256": "a" * 64,
                    }
                ],
            }
        ), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen comparison registry"):
        load_verified_comparison(audit, "arbitrary", "arbitrary")


def test_authoritative_analysis_rejects_qualification_outputs(tmp_path):
    state = tmp_path / "state"
    extract_state(
        model=ToyProbeModel(blocks=2), probes=[_probe()], destination=state,
        state_metadata={"state_id": "state"}, device="cpu", block_count=2,
        raw_attention_layers=[0], raw_attention_maximum_tokens=3,
        raw_attention_probe_hashes=set(),
    )
    completion = json.loads((state / "complete.json").read_text(encoding="utf-8"))
    assert completion["completion_scope"] == "bounded_non_authoritative_run"
    with pytest.raises(ValueError, match="all 48 probes"):
        analyze_state_pair(
            left_state_dir=state, right_state_dir=state, comparison_id="test"
        )


def test_representation_completion_paths_are_portable_and_hash_bound(tmp_path):
    original = tmp_path / "original"
    extract_state(
        model=ToyProbeModel(blocks=2), probes=[_probe()], destination=original,
        state_metadata={"state_id": "state"}, device="cpu", block_count=2,
        raw_attention_layers=[0], raw_attention_maximum_tokens=3,
        raw_attention_probe_hashes=set(),
    )
    import shutil

    moved = tmp_path / "moved"
    shutil.move(original, moved)
    report = analyze_state_pair(
        left_state_dir=moved, right_state_dir=moved, comparison_id="portable",
        authoritative=False,
    )
    assert report["probe_count"] == 1
    completion = json.loads((moved / "complete.json").read_text(encoding="utf-8"))
    completion["probe_results"][0]["manifest_sha256"] = "0" * 64
    (moved / "complete.json").write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        analyze_state_pair(
            left_state_dir=moved, right_state_dir=moved,
            comparison_id="tampered", authoritative=False,
        )


def test_embedding_frequency_sources_use_only_hash_verified_train_pools(tmp_path):
    shard = tmp_path / "train.bin"
    shard.write_bytes(struct.pack("<2i", 1, 2))
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    report = tmp_path / "encoded.json"
    report.write_text(
        json.dumps(
            {
                "pools": [
                    {
                        "pool_id": "train_pool", "split": "train",
                        "shards": [{"path": "train.bin", "bytes": 8, "sha256": digest}],
                    },
                    {
                        "pool_id": "sonnets_test", "split": "test",
                        "shards": [{"path": "never-read-test.bin", "bytes": 1, "sha256": "x"}],
                    },
                ]
            }
        ), encoding="utf-8",
    )
    assert resolve_verified_training_shards(report, repo_root=tmp_path) == [shard]
    shard.write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity mismatch"):
        resolve_verified_training_shards(report, repo_root=tmp_path)


def test_behavior_rejects_truncated_or_hash_mismatched_output(tmp_path):
    directory = _generation_dir(tmp_path / "state", "state")
    output = directory / "one.json"
    output.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        analyze_matched_generations(
            state_directories={"state": directory}, confirmatory_seed=4099,
            authoritative=False,
        )


def test_verified_memorization_reference_is_hash_bound_and_scoring_is_bounded(
    tmp_path, monkeypatch,
):
    import sonnet_analysis.minerva_v7_memorization as module

    records_path = tmp_path / "records.jsonl"
    records = [
        {
            "record_id": "copied", "source_id": "source-a",
            "decoded_text_sha256": hashlib.sha256(("a" * 200).encode()).hexdigest(),
            "text": "a" * 200,
        },
        {
            "record_id": "other", "source_id": "source-b",
            "decoded_text_sha256": hashlib.sha256(("b" * 200).encode()).hexdigest(),
            "text": "b" * 200,
        },
    ]
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    monkeypatch.setattr(module, "EXPECTED_DOCUMENTS", 2)
    monkeypatch.setattr(module, "EXPECTED_TOKENS", 10)
    manifest = {
        "reference_version": REFERENCE_VERSION,
        "source_pool_id": "sonnets_train", "source_split": "train",
        "record_count": 2, "token_count": 10,
        "tokenizer_sha256": module.EXPECTED_TOKENIZER_SHA256,
        "document_index_sha256": module.EXPECTED_INDEX_SHA256,
        "token_shard_sha256": module.EXPECTED_SHARD_SHA256,
        "records_path": records_path.name, "records_bytes": records_path.stat().st_size,
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "v7_test_accessed": False,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded, _ = load_verified_sonnet_train_reference(manifest_path)
    scored = score_texts_against_reference(["a" * 180, "no copied span here"], loaded)
    assert scored[0]["nearest_poem_id"] == "copied"
    assert scored[0]["risk_level"] == "high"
    assert scored[1]["nearest_poem_id"] is None
    records_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="records hash mismatch"):
        load_verified_sonnet_train_reference(manifest_path)
