"""Check that hosted nightly guides follow the script pins and reject drift."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github/scripts"))

import hosted_passthrough_docs
import hosted_translation_docs


class HostedNightlyDocsTest(unittest.TestCase):
    def test_guides_follow_sdk_script_pins_and_reject_stale_output(self):
        for generator, old_pin, new_pin in (
            (hosted_passthrough_docs, "openai==3.49.0", "openai==3.50.0"),
            (hosted_translation_docs, "anthropic==1.17.0", "anthropic==1.18.0"),
        ):
            with self.subTest(lane=generator.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for relative in (generator.WORKFLOW, generator.SUITE):
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text((ROOT / relative).read_text())
                guide = root / "guide.md"
                guide.write_text(generator.render())
                suite = root / generator.SUITE
                suite.write_text(suite.read_text().replace(old_pin, new_pin))
                with mock.patch.multiple(generator, ROOT=root, GUIDE=guide, __file__=str(root / Path(generator.__file__).name)), mock.patch.object(
                    sys, "argv", [generator.__file__, "--check"]
                ), mock.patch("sys.stderr"):
                    self.assertIn(new_pin.split("==")[1], generator.render())
                    with self.assertRaises(SystemExit) as error:
                        generator.main()
                    self.assertEqual(error.exception.code, 1)

    def test_missing_generation_input_is_rejected(self):
        for generator in (hosted_passthrough_docs, hosted_translation_docs):
            with self.subTest(lane=generator.__name__), self.assertRaisesRegex(ValueError, "Missing"):
                generator.field(r"^MODEL = (.+)$", "")
