import argparse

import mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_uri", type=str, required=True, help="指定 MLflow 追蹤伺服器的 URI。")
    parser.add_argument("--run_id", type=str, required=True, help="MLflow 實驗運行的唯一標識符。")
    parser.add_argument("--report", type=str, required=True, help="Chest CT 影像文字報告。")
    parser.add_argument("--threshold", type=float, default=0.7, help="設定篩選實體分數的最低閾值。")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    model = mlflow.pytorch.load_model(f"runs:/{args.run_id}/model")
    ents = model.hf_pipeline(args.report)

    paragraphs = []
    tags = []
    point = None
    tag = None
    for ent in ents:
        if ent["score"] < args.threshold:
            continue
        if point is not None:
            paragraphs.append(args.report[point:ent["start"]].strip())
            tags.append(tag)
        point = ent["start"]
        tag = ent["entity"]

    paragraphs.append(args.report[point:].strip())
    tags.append(tag)

    results = []
    for tag, paragraph in zip(tags, paragraphs):
        print(tag, " -> ", paragraph)
        results.append((tag, paragraph))


if __name__ == "__main__":
    main()
