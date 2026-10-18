"""Build a private, read-only error review; never turn predictions into labels."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def frames_at(path, targets):
    selected = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.codec_context is None:
            raise ValueError("Missing video decoder")
        origin = float((stream.start_time or 0) * stream.time_base)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Missing frame timestamp")
            time = float(frame.pts * frame.time_base) - origin
            while len(selected) < len(targets) and time >= targets[len(selected)]:
                selected.append((frame.to_image(), time))
            if len(selected) == len(targets):
                return selected
    raise ValueError("Video ended before requested review frames")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Review evidence must not be overwritten")
    sources = read(args.manifest)["sources"]
    by_id = {row["source_id"]: row for row in sources}
    rows = []
    for fold in range(5):
        folder = args.run / f"outer_{fold}"
        calibration = read(folder / "calibration.json")
        details = read(folder / "details.json")
        for row in details["videos"]:
            if row["source_id"] in calibration["calibration_source_ids"]:
                raise ValueError("Review source overlaps calibration")
            if row["source_id"] in calibration["training_provenance"]["train_source_ids"]:
                raise ValueError("Review source overlaps training")
            rows.append({**row, "fold": fold})
    if len(rows) != len(by_id) or {row["source_id"] for row in rows} != set(by_id):
        raise ValueError("Expected exactly one held-out prediction per source")
    rows.sort(key=lambda row: (row["label"] == row["prediction"], row["label"], row["source_id"]))
    false_positives = sum(row["label"] == 0 and row["prediction"] == 1 for row in rows)
    false_negatives = sum(row["label"] == 1 and row["prediction"] == 0 for row in rows)
    args.output.mkdir(parents=True)
    records, sections = [], []
    for index, row in enumerate(rows):
        source = by_id[row["source_id"]]
        if row["label"] != source["label"]:
            raise ValueError("Prediction report and manifest labels disagree")
        path = Path(source["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Review video checksum mismatch")
        peak_index = int(np.argmax(row["window_scores"]))
        peak = (row["starts"][peak_index] + row["ends"][peak_index]) / 2
        points = source.get("events_sec", [])
        anchors = sorted(set([peak, *points]))
        targets = sorted(set(float(np.clip(t + delta, 0, max(0., source["duration_sec"] - .2)))
                             for t in anchors for delta in (-4, -2, 0, 2, 4)))
        frames = frames_at(path, targets)
        sheet = Image.new("RGB", (1000, 250 * ((len(frames) + 4) // 5)), "white")
        draw = ImageDraw.Draw(sheet)
        for number, (frame, time) in enumerate(frames):
            frame.thumbnail((196, 222))
            x, y = (number % 5) * 200, (number // 5) * 250
            sheet.paste(frame, (x + (200 - frame.width) // 2, y + 25))
            draw.text((x + 4, y + 5), f"t={time:.2f}s", fill="black")
        filename = f"case_{index + 1:02}.jpg"
        sheet.save(args.output / filename, quality=90)
        category = ("false_positive" if not row["label"] else "false_negative") if row["label"] != row["prediction"] else "correct"
        counterparts = [other["source_id"] for other in sources
                        if other["scenario_group"] == source["scenario_group"] and other["source_id"] != row["source_id"]]
        record = {"source_id": source["source_id"], "sha256": source["sha256"],
                  "video_ids": source["video_ids"], "duration_sec": source["duration_sec"],
                  "excel_label": source["label"], "excel_points_sec": points,
                  "annotation_confidence": source.get("annotation_confidence"),
                  "error": category, "outer_fold": row["fold"], "model_score": row["score"],
                  "threshold": row["threshold"], "model_peak_sec": peak,
                  "model_events": row["events"], "same_scenario_sources": counterparts,
                  "contact_sheet": filename, "review_status": "pending", "reviewer": None,
                  "observed_action": None, "event_type": None, "start_sec": None, "end_sec": None,
                  "applicable_rule": None, "rule_context": None,
                  "label_decision": "unreviewed", "notes": None}
        records.append(record)
        esc = lambda value: html.escape(str(value), quote=True)
        name = {"false_positive": "Ложная тревога", "false_negative": "Пропуск", "correct": "Верный бинарный ответ"}[category]
        timeline = " ".join(f"{80 + 820 * ((start + end) / 2) / source['duration_sec']:.2f},{170 - 140 * score:.2f}"
                            for start, end, score in zip(row["starts"], row["ends"], row["window_scores"]))
        threshold_y = 170 - 140 * row["threshold"]
        sections.append(f'<section id="case-{esc(source["source_id"])}"><h2>{index + 1}. {name} — видео {esc(source["video_ids"])}</h2>'
                        f'<p>Excel: {source["label"]}; точки {esc(points)}. Score {row["score"]:.4f}; порог {row["threshold"]:.4f}; пик {peak:.2f} s.</p>'
                        f'<p>Та же сцена: {esc(counterparts)}. <a href="{esc(path.resolve().as_uri())}">Открыть исходное видео</a></p>'
                        f'<img loading="lazy" src="{filename}" alt="Кадры вокруг пика модели и отметок Excel">'
                        f'<svg viewBox="0 0 940 200" aria-label="Оценки окон и порог"><text x="0" y="25">score 1</text><text x="0" y="175">score 0</text>'
                        f'<line x1="80" x2="900" y1="{threshold_y}" y2="{threshold_y}" stroke="red"/>'
                        f'<polyline points="{timeline}" fill="none" stroke="#185ac0" stroke-width="2"/>'
                        f'<text x="80" y="195">0 s</text><text x="840" y="195">{source["duration_sec"]:.1f} s</text></svg>'
                        '<p>Проверить: видимое действие → интервал → применимое правило и исключения. Решение пока не подтверждено.</p></section>')
        print({"reviewed_source_prepared": index + 1, "error": category}, flush=True)
    payload = {"status": "pending_human_review_not_training_labels", "records": records,
               "instructions": "Заполнить reviewer, observed_action, event_type, start_sec/end_sec, applicable_rule, rule_context, notes. label_decision: confirm_excel / dispute_excel / insufficient_evidence. Не изменять Excel-метки автоматически. Для нескольких событий указать их отдельно в notes; границы не угадывать по точке Excel."}
    (args.output / "review.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "timelines.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "index.html").write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Разбор ошибок модели</title>'
        '<style>body{max-width:1040px;margin:32px auto;font:18px/1.5 system-ui;background:#f4f5f8;color:#182030}section{background:white;padding:20px;margin:24px 0}img,svg{width:100%}a{color:#185ac0}</style>'
        f'<h1>Разбор ошибок модели</h1><p>Сначала {false_positives} ложных тревог и {false_negatives} пропусков, затем контрольные примеры. Это повторно использованная выборка разработки, не независимый тест.</p>'
        '<p>Красная линия — порог, синяя — оценки окон. Кадры показывают контекст, но не заменяют просмотр полного видео. '
        'Заполняйте <a href="review.json">review.json</a>. Метки остаются черновиком и автоматически в обучение не попадают.</p>'
        '<p>Трудовой кодекс задаёт общие обязанности, но применимость к действию требует контекста и правил работодателя. '
        'Питьё, уход из кадра, открытая дверца или подход другого человека сами по себе не доказывают проступок.</p>'
        + ''.join(sections) + '</html>', encoding="utf-8")


if __name__ == "__main__":
    main()
