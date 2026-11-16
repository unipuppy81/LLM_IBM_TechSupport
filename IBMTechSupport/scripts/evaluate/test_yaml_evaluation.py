import json
import unittest

import yaml

from .common import PROJECT_DIR


class YAMLConfigurationTests(unittest.TestCase):

    def test_same_validation_dataset(self):
        configs = []

        for name in "base/v1.yaml", "rag/v1.yaml", "rag/v2.yaml":
            path = PROJECT_DIR / "configs" / name
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            configs.append(config)

        paths = [
            config["dataset"]["path"]
            for config in configs
        ]

        self.assertEqual(len(set(paths)), 1)
        self.assertEqual(paths[0], "data/processed/val_clean.json")

        dataset = json.loads((PROJECT_DIR / paths[0]).read_text(encoding="utf-8"))

        self.assertEqual(len(dataset), 135)

        for config in configs:
            self.assertEqual(config["dataset"]["expected_count"], 135)

    def test_retrieval_configuration(self):
        versions = {}

        for version in ("v1", "v2"):
            path = PROJECT_DIR / "configs/rag" / f"{version}.yaml"
            versions[version] = yaml.safe_load(path.read_text(encoding="utf-8"))
        
        self.assertEqual(versions["v1"]["retrieval"]["method"], "reranker")
        self.assertEqual(versions["v2"]["retrieval"]["method"], "hybrid")
        self.assertEqual(versions["v1"]["retrieval"]["top_k"], 5)
        self.assertEqual(versions["v2"]["retrieval"]["top_k"], 5)


if __name__ == "__main__":
    unittest.main()