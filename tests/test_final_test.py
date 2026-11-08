"""scripts/final_test.py must refuse the test split unless the explicit test-day flag is given."""
import subprocess
import sys

import pytest

from src import config


def test_split_test_without_flag_exits_with_error():
    res = subprocess.run([sys.executable, "-m", "scripts.final_test", "--split", "test"], cwd=config.ROOT,
                         capture_output=True, text=True, timeout=60)
    assert res.returncode != 0
    assert "refusing to evaluate the TEST split" in res.stderr


def test_parse_args_rules():
    from scripts.final_test import TEST_FLAG, parse_args
    assert parse_args([]).split == "val"  # default is val
    with pytest.raises(SystemExit):
        parse_args(["--split", "test"])
    assert parse_args(["--split", "test", TEST_FLAG]).test_ok  # only parses; nothing is loaded here
