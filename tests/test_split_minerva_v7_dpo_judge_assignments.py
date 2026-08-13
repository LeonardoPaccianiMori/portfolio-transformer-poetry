from scripts.split_minerva_v7_dpo_judge_assignments import split_assignments


def test_split_assignments_preserves_every_judge_pair_once():
    judges = ["j1", "j2", "j3"]
    rows = [
        {
            "assignment_id": f"{judge}-{pair}",
            "judge_id": judge,
            "pair": {"pair_id": f"pair_{pair}"},
        }
        for pair in range(5)
        for judge in judges
    ]
    manifest = {
        "preference_version": "v",
        "judge_ids": judges,
        "pair_count": 5,
        "assignment_count": 15,
        "assignments": rows,
    }
    packets = split_assignments(manifest, chunk_size=2)
    assert len(packets) == 9
    assert sum(row["assignment_count"] for row in packets) == 15
    identities = [
        (row["judge_id"], row["pair"]["pair_id"])
        for packet in packets
        for row in packet["assignments"]
    ]
    assert len(identities) == len(set(identities)) == 15
    assert all(packet["assignment_count"] in {1, 2} for packet in packets)
