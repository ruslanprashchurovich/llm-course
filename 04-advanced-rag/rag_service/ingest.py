"""Индексация базы знаний: чтение -> санитизация -> чанкинг -> эмбеддинги -> Qdrant.

Индексация — это ETL-пайплайн, и относиться к нему надо как к ETL:
детерминированные идентификаторы, идемпотентность, метрики, отчёт о том,
что было отброшено и почему.

Особенность TF-IDF-бэкенда: словарь выучивается на корпусе, поэтому полная
переиндексация = новое состояние эмбеддера = новая размерность векторов.
Коллекция в этом случае пересоздаётся (правило «одна коллекция — одна модель»
из урока 3.2 — здесь оно буквально про геометрию).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import DATA_DIR, settings
from .observability import (
    RAG_INDEX_CHUNKS,
    RAG_INDEX_LAST_SUCCESS,
    RAG_INJECTION_BLOCKED,
    RAG_PII_MASKED,
    configure_logging,
    log_event,
    stage,
)
from .security import sanitize_document, scan_injection
from .store import Chunk, KnowledgeBase

RISK_LEVELS = {"low": 0, "medium": 1, "high": 2}

# --------------------------------------------------------------------------
# Чанкинг
# --------------------------------------------------------------------------

SENTENCE_END_RE = re.compile(r"(?<=[.!?;:])\s+")


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_text(
    text: str,
    max_chars: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Режет текст по абзацам, склеивая их до max_chars, с перекрытием.

    Почему по абзацам, а не по 500 символов: разрыв внутри предложения
    убивает смысл чанка, а вместе с ним и качество поиска. Абзац — естественная
    смысловая единица документа (в коде это как «не разрывать функцию пополам»).
    """
    max_chars = max_chars or settings.chunk_max_chars
    overlap = overlap if overlap is not None else settings.chunk_overlap_chars

    pieces: list[str] = []
    for paragraph in split_paragraphs(text):
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        # слишком длинный абзац дорезаем по границам предложений
        buffer = ""
        for sentence in SENTENCE_END_RE.split(paragraph):
            if buffer and len(buffer) + len(sentence) + 1 > max_chars:
                pieces.append(buffer.strip())
                buffer = sentence
            else:
                buffer = f"{buffer} {sentence}".strip()
        if buffer:
            pieces.append(buffer.strip())

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}".strip() if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # перекрытие начинаем с границы слова, чтобы не рвать слово пополам
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            current = f"{tail}\n\n{piece}".strip() if tail else piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------
# Метаданные и сборка чанков
# --------------------------------------------------------------------------


def _to_timestamp(date_str: str) -> int:
    try:
        return int(
            datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc).timestamp()
        )
    except ValueError:
        return 0


def load_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    manifest_path = path or (DATA_DIR / "manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(payload["documents"])


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    skipped_chunks: int = 0
    quarantined_docs: list[str] = field(default_factory=list)
    pii_counts: dict[str, int] = field(default_factory=dict)
    injection_hits: dict[str, list[str]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"документов: {self.documents}",
            f"чанков в индексе: {self.chunks}",
            f"пропущено чанков: {self.skipped_chunks}",
        ]
        if self.quarantined_docs:
            lines.append("карантин: " + ", ".join(self.quarantined_docs))
        if self.pii_counts:
            lines.append(
                "замаскировано ПДн: "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.pii_counts.items()))
            )
        if self.injection_hits:
            for doc_id, rules in self.injection_hits.items():
                lines.append(f"подозрение на injection в {doc_id}: {', '.join(rules)}")
        return "\n".join(lines)


def build_chunks(
    doc: dict[str, Any],
    raw_text: str,
    *,
    tenant_id: str = "vectorika",
    sanitize: bool = True,
    mask_pii_flag: bool | None = None,
    report: IngestReport | None = None,
) -> list[Chunk]:
    """Превращает документ в список чанков с полными метаданными."""
    mask_pii_flag = (
        settings.mask_pii_in_context if mask_pii_flag is None else mask_pii_flag
    )
    trusted = doc.get("source_type", "trusted") == "trusted"

    doc_report = scan_injection(raw_text)
    if report is not None and doc_report.matched_rules:
        report.injection_hits[doc["doc_id"]] = list(doc_report.matched_rules)
    for rule in doc_report.matched_rules:
        RAG_INJECTION_BLOCKED.labels(stage="ingest", rule=rule).inc()

    if sanitize:
        cleaned = sanitize_document(
            raw_text, trusted=trusted, mask_pii_flag=mask_pii_flag
        )
        text = cleaned.text
        pii_found = cleaned.pii_found
    else:
        text = raw_text
        pii_found = {}

    for entity, count in pii_found.items():
        RAG_PII_MASKED.labels(entity=entity).inc(count)
        if report is not None:
            report.pii_counts[entity] = report.pii_counts.get(entity, 0) + count

    pieces = chunk_text(text)
    version = int(doc.get("version", 1))
    checksum = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:12]

    chunks: list[Chunk] = []
    for index, piece in enumerate(pieces):
        # риск считаем и для документа целиком, и для конкретного чанка:
        # атака может жить в одном абзаце, а чанки уезжают в промпт поштучно
        chunk_report = scan_injection(piece)
        risk = max(
            RISK_LEVELS[chunk_report.risk],
            RISK_LEVELS[doc_report.risk] if not trusted else RISK_LEVELS["low"],
        )
        metadata: dict[str, Any] = {
            "tenant_id": tenant_id,
            "doc_id": doc["doc_id"],
            "title": doc.get("title", doc["doc_id"]),
            "file": doc.get("file", ""),
            "department": doc.get("department", "all"),
            "sensitivity": int(doc.get("sensitivity", 0)),
            "version": version,
            "is_current": bool(doc.get("is_current", True)),
            "updated_at": doc.get("updated_at", ""),
            "updated_at_ts": _to_timestamp(doc.get("updated_at", "")),
            "source_type": doc.get("source_type", "trusted"),
            "owner": doc.get("owner", "unknown"),
            "chunk_index": index,
            "n_chunks": len(pieces),
            "checksum": checksum,
            "risk_level": risk,
            "risk_rules": ",".join(chunk_report.matched_rules)[:200],
            "has_pii": bool(pii_found),
        }
        chunk_id = f"{tenant_id}:{doc['doc_id']}:v{version}:{index:03d}"
        chunks.append(Chunk(id=chunk_id, text=piece, metadata=metadata))
    return chunks


def ingest_all(
    kb: KnowledgeBase | None = None,
    embedder: Any | None = None,
    *,
    data_dir: Path | None = None,
    reset: bool = False,
    sanitize: bool = True,
    include_untrusted: bool | None = None,
    verbose: bool = True,
) -> IngestReport:
    """Полная переиндексация корпуса. Возвращает отчёт."""
    from .embeddings import get_embedder

    data_dir = data_dir or DATA_DIR
    kb = kb or KnowledgeBase()
    embedder = embedder or get_embedder()
    include_untrusted = (
        settings.index_untrusted if include_untrusted is None else include_untrusted
    )

    report = IngestReport()
    all_chunks: list[Chunk] = []

    with stage("ingest.read_and_chunk"):
        for doc in load_manifest(data_dir / "manifest.json"):
            if doc.get("source_type") == "untrusted" and not include_untrusted:
                report.quarantined_docs.append(doc["doc_id"])
                continue
            raw_text = (data_dir / doc["file"]).read_text(encoding="utf-8")
            chunks = build_chunks(doc, raw_text, sanitize=sanitize, report=report)
            report.documents += 1
            all_chunks.extend(chunks)

    with stage("ingest.embed", chunks=len(all_chunks)):
        texts = [c.text for c in all_chunks]
        # TF-IDF учится на корпусе прямо здесь; у нейроэмбеддера метода fit нет
        if hasattr(embedder, "fit"):
            embedder.fit(texts)
            state_path = embedder.save()
            log_event(
                "ingest.tfidf_state_saved", path=str(state_path), dim=embedder.dim
            )
            reset = True  # размерность могла измениться — коллекцию пересоздаём
        vectors = embedder.encode_documents(texts)

    with stage("ingest.upsert", chunks=len(all_chunks)):
        kb.ensure_collection(embedder.dim, recreate=reset)
        kb.upsert(all_chunks, vectors)

    report.chunks = kb.count()
    RAG_INDEX_CHUNKS.set(report.chunks)
    RAG_INDEX_LAST_SUCCESS.set(datetime.now(tz=timezone.utc).timestamp())
    log_event(
        "ingest.done",
        documents=report.documents,
        chunks=report.chunks,
        pii=report.pii_counts,
        injection_docs=list(report.injection_hits),
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Индексация базы знаний Векторики")
    parser.add_argument(
        "--reset", action="store_true", help="удалить коллекцию перед индексацией"
    )
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="индексировать сырой текст (для демонстрации атаки, НЕ для прода)",
    )
    parser.add_argument(
        "--skip-untrusted",
        action="store_true",
        help="не индексировать документы из недоверенных источников",
    )
    args = parser.parse_args(argv)

    configure_logging(settings.log_level, json_logs=False, service="ingest")
    report = ingest_all(
        reset=args.reset,
        sanitize=not args.no_sanitize,
        include_untrusted=not args.skip_untrusted,
    )
    print(report.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
