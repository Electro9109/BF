"""Unit/smoke tests for eval/demo_explainer.py."""

from eval.demo_explainer import create_sample_dataset, run_demo


def test_create_sample_dataset():
    df = create_sample_dataset()
    assert df.shape[0] == 60
    assert "tuyere_temp_c" in df.columns
    assert "hot_metal_si_pct" in df.columns
    assert "slag_basicity" in df.columns
    assert "blast_velocity" in df.columns
    assert "cast_id" in df.columns


def test_run_demo_dry_run_executes_without_error(capsys):
    run_demo(dry_run=True, max_examples=2)
    captured = capsys.readouterr()
    assert "PARSE QWEN EXPLAINER: END-TO-END DEMO" in captured.out
    assert "DEMO COMPLETED SUCCESSFULLY" in captured.out
