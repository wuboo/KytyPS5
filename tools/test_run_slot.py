"""CPU tests of tools/run_slot.py (no game, no GPU). Run: python3 -I tools/test_run_slot.py
RUN_SLOT_MODULE_DIR points at a directory with a mutated copy of run_slot.py (see mutants_run_slot.sh)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.environ.get("RUN_SLOT_MODULE_DIR", str(Path(__file__).resolve().parent)))
import run_slot as rs  # noqa: E402


def alive(_pid):
    return True


def dead(_pid):
    return False


class CanStart(unittest.TestCase):
    def test_image_lock_allows(self):
        self.assertTrue(rs.can_start("rt-agent 123 2026-10-12 22:04 image\n", None, alive)[0])

    def test_tempo_lock_refuses(self):
        ok, why = rs.can_start("rt-agent 123 2026-10-12 22:04\n", None, alive)
        self.assertFalse(ok)
        self.assertIn("image", why)

    def test_image_word_must_be_last(self):
        self.assertFalse(rs.can_start("rt-agent 123 image 2026-10-12 22:04\n", None, alive)[0])

    def test_no_lock_refuses(self):
        self.assertFalse(rs.can_start(None, None, alive)[0])

    def test_dead_holder_refuses(self):
        self.assertFalse(rs.can_start("rt-agent 123 d image\n", None, dead)[0])

    def test_garbled_pid_refuses(self):
        self.assertFalse(rs.can_start("rt-agent x d image\n", None, alive)[0])

    def test_second_slot_refuses(self):
        ok, why = rs.can_start("rt-agent 123 d image\n", "slot2 9 d image\n", alive)
        self.assertFalse(ok)
        self.assertIn("already", why)


class Config(unittest.TestCase):
    def test_slot_config(self):
        cfg = {"label": "A", "emulator": "/v/rt-x/kyty_emulator", "args": ["-a"], "env": {"K": "1"}}
        out = rs.make_slot_config(cfg, Path("/v/rt-x-slot2"), Path("/d"))
        self.assertEqual(out["emulator"], "/v/rt-x-slot2/kyty_emulator")
        self.assertEqual(out["label"], "A-slot2")
        self.assertEqual(out["env"], {"K": "1", "KYTY_FRAME_DUMP_DIR": "/d"})
        self.assertEqual(cfg["env"], {"K": "1"})  # the source config is not modified

    def test_slot_dir(self):
        self.assertEqual(rs.slot_dir_of(Path("/v/rt-x")), Path("/v/rt-x-slot2"))


class Prepare(unittest.TestCase):
    def test_prepare_copies_caches_and_writes_config(self):
        with tempfile.TemporaryDirectory() as t:
            v = Path(t) / "variants" / "rt-x"
            (v / "_ShaderCache").mkdir(parents=True)
            (v / "_ShaderCache" / "a.bin").write_text("x")
            (v / "kyty_emulator").write_text("#!/bin/sh\n")
            cfgdir = Path(t) / "cfg"
            cfgdir.mkdir()
            (cfgdir / "c.json").write_text(json.dumps({"label": "c", "emulator": "../variants/rt-x/kyty_emulator"}))
            self.assertEqual(rs.main(["prepare", "--variant", str(v), "--config", str(cfgdir / "c.json")]), 0)
            self.assertTrue((Path(t) / "variants" / "rt-x-slot2" / "_ShaderCache" / "a.bin").is_file())
            out = json.loads((cfgdir / "c-slot2.json").read_text())
            self.assertEqual(out["emulator"], str(Path(t).resolve() / "variants" / "rt-x-slot2" / "kyty_emulator"))
            self.assertEqual(rs.main(["prepare", "--variant", str(v)]), 2)  # exists, no --refresh


if __name__ == "__main__":
    unittest.main()
