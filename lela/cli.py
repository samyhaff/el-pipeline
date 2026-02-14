import argparse
import json
from pathlib import Path

from lela.config import PipelineConfig
from lela.pipeline import ELPipeline


def main():
    parser = argparse.ArgumentParser(description="Run modular LELA.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to pipeline config JSON file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Input file paths.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional JSONL output path.",
    )
    args = parser.parse_args()

    config_data = json.loads(Path(args.config).read_text(encoding="utf-8"))

    # Resolve labels_from_kb: extract entity types from KB and use as NER labels
    ner_params = config_data.get("ner", {}).get("params", {})
    if ner_params.get("labels_from_kb"):
        kb_conf = config_data.get("knowledge_base", {})
        kb_path = kb_conf.get("params", {}).get("path")
        if kb_path:
            from lela.knowledge_bases.custom import CustomJSONLKnowledgeBase

            kb = CustomJSONLKnowledgeBase(kb_path)
            kb_types = kb.get_entity_types()
            if kb_types:
                ner_params["labels"] = kb_types
        del ner_params["labels_from_kb"]

    config = PipelineConfig.from_dict(config_data)

    pipeline = ELPipeline(config)
    pipeline.run(args.input, output_path=args.output)


if __name__ == "__main__":
    main()

