# scripts/build_wikidata_knowledge.py
# -*- coding: utf-8 -*-

"""
从 full CSV 中抽取候选实体，并调用 Wikidata API 构建真实知识库 CSV。

输入示例：
    data/processed/politifact_full.csv
    data/processed/gossipcop_full.csv

输出示例：
    knowledge/wikidata_knowledge_politifact.csv
    knowledge/wikidata_knowledge_gossipcop.csv

输出字段：
    source_sample_id,entity_text,qid,label,description,knowledge_text,concept_uri,api_status

V2 修复内容：
    1. 增加 429 限流重试与退避等待；
    2. 429、timeout、connection_error 不再写入“失败缓存”，避免后续直接 cached_not_found；
    3. 默认 sleep 改为 1.0 秒，降低 Wikidata API 压力；
    4. 输出后打印真实成功知识数，避免 pandas 把空值读成 NaN 后误判；
    5. 继续保持轻量规则抽取实体，不依赖 NLTK 模型下载。

说明：
    当前脚本使用 Wikidata API 的 wbsearchentities 实体搜索接口。
    后续第 16 步会使用 CLIP 对 knowledge_text 编码，生成 knowledge_embeddings.pt。
"""

import argparse
import json
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm


WIKIDATA_API = "https://www.wikidata.org/w/api.php"

HEADERS = {
    "User-Agent": "KICP-Wikidata-Knowledge-Builder/0.2 (educational research; contact via the repository issue tracker)",
    "Accept": "application/json",
}


BASIC_STOP_ENTITIES = {
    "Breaking",
    "Update",
    "UPDATED",
    "BREAKING",
    "VIDEO",
    "PHOTO",
    "Photos",
    "Watch",
    "News",
    "Fake",
    "Real",
    "True",
    "False",
    "Claim",
    "Claims",
    "Says",
    "Said",
    "Report",
    "Reports",
    "Story",
    "Stories",
    "Article",
    "Articles",
    "Email",
    "Facebook",
    "Twitter",
    "YouTube",
    "Instagram",
    "Advertisement",
    "Subscribe",
    "Share",
    "Read More",
    "Associated Press",
    "Getty Images",
}


def clean_text(text) -> str:
    if text is None:
        return ""

    text = str(text)

    if text.lower() == "nan":
        return ""

    text = text.replace("\u00a0", " ")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def normalize_entity(entity: str) -> str:
    entity = clean_text(entity)
    entity = entity.strip(" ,.;:!?()[]{}'\"“”‘’")
    entity = re.sub(r"^(BREAKING|Breaking|UPDATE|Update|UPDATED|VIDEO|PHOTO)\s*[:\-]?\s*", "", entity)
    entity = clean_text(entity)
    return entity


def is_bad_entity(entity: str) -> bool:
    entity = normalize_entity(entity)

    if entity == "":
        return True

    if entity in BASIC_STOP_ENTITIES:
        return True

    if len(entity) < 3:
        return True

    if len(entity) > 80:
        return True

    if re.fullmatch(r"\d+", entity):
        return True

    letters = re.findall(r"[A-Za-z]", entity)
    if len(letters) < 3:
        return True

    words = entity.split()

    if len(words) == 1:
        word = words[0].lower()
        if word in {
            "said",
            "says",
            "claim",
            "claims",
            "news",
            "report",
            "story",
            "article",
            "photo",
            "video",
            "president",
            "senator",
            "governor",
            "court",
            "police",
            "official",
            "campaign",
            "election",
        }:
            return True

    return False


def unique_preserve_order(items: List[str]) -> List[str]:
    container = OrderedDict()

    for item in items:
        item = normalize_entity(item)

        if item and item not in container:
            container[item] = None

    return list(container.keys())


def extract_capitalized_entities(text: str) -> List[str]:
    text = clean_text(text)

    if text == "":
        return []

    acronym_pattern = r"\b[A-Z]{2,}(?:-[A-Z]{2,})?\b"

    capitalized_phrase_pattern = (
        r"\b(?:[A-Z][a-zA-Z]+|[A-Z]{2,})"
        r"(?:\s+(?:[A-Z][a-zA-Z]+|[A-Z]{2,}|of|the|and|for|in|on|at|to|from|with|by))*"
        r"\b"
    )

    candidates = []
    candidates.extend(re.findall(acronym_pattern, text))
    candidates.extend(re.findall(capitalized_phrase_pattern, text))

    cleaned = []

    for cand in candidates:
        cand = normalize_entity(cand)

        if is_bad_entity(cand):
            continue

        words = cand.split()

        if len(words) >= 2:
            if words[0] in {"The", "This", "That", "These", "Those", "There", "When", "Where", "What", "Why", "How"}:
                cand = " ".join(words[1:])
                cand = normalize_entity(cand)

        if is_bad_entity(cand):
            continue

        cleaned.append(cand)

    return unique_preserve_order(cleaned)


def extract_entities_from_row(title: str, text: str, max_entities: int) -> List[str]:
    title = clean_text(title)
    text = clean_text(text)

    title_entities = extract_capitalized_entities(title)

    text_head = text[:2500]
    text_entities = extract_capitalized_entities(text_head)

    entities = unique_preserve_order(title_entities + text_entities)

    return entities[:max_entities]


def load_cache(cache_path: Path) -> Dict:
    if not cache_path.exists():
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

        if not isinstance(cache, dict):
            return {}

        return cache

    except Exception:
        return {}


def save_cache(cache: Dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def is_transient_status(status: str) -> bool:
    transient_prefixes = [
        "http_status_429",
        "http_status_500",
        "http_status_502",
        "http_status_503",
        "timeout",
        "connection_error",
    ]

    return any(status.startswith(prefix) for prefix in transient_prefixes)


def search_wikidata_entity(
    entity_text: str,
    language: str,
    limit: int,
    timeout: int,
    cache: Dict,
    sleep_seconds: float,
    max_retries: int,
    backoff_seconds: float,
) -> Tuple[Optional[Dict], str]:
    cache_key = f"{language}::{entity_text}"

    if cache_key in cache:
        cached = cache[cache_key]

        if cached is None:
            return None, "cached_not_found"

        return cached, "cached_ok"

    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": language,
        "uselang": language,
        "type": "item",
        "limit": limit,
        "search": entity_text,
    }

    last_status = "not_started"

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                WIKIDATA_API,
                params=params,
                headers=HEADERS,
                timeout=timeout,
            )

            if response.status_code == 429:
                last_status = "http_status_429"

                retry_after = response.headers.get("Retry-After", "")
                try:
                    wait_time = float(retry_after)
                except Exception:
                    wait_time = backoff_seconds * (attempt + 1)

                print(f"[429] Wikidata 限流：等待 {wait_time:.1f} 秒后重试，entity={entity_text}")
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                last_status = f"http_status_{response.status_code}"

                # 4xx 非 429 一般不是临时问题，可以缓存为失败
                if not is_transient_status(last_status):
                    cache[cache_key] = None

                return None, last_status

            data = response.json()
            results = data.get("search", [])

            if not results:
                cache[cache_key] = None
                return None, "not_found"

            item = results[0]

            qid = item.get("id", "")
            label = item.get("label", "")
            description = item.get("description", "")
            concept_uri = item.get("concepturi", "")

            if qid == "" or label == "":
                cache[cache_key] = None
                return None, "invalid_result"

            result = {
                "qid": qid,
                "label": clean_text(label),
                "description": clean_text(description),
                "concept_uri": concept_uri,
            }

            cache[cache_key] = result

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            return result, "ok"

        except requests.exceptions.Timeout:
            last_status = "timeout"
            wait_time = backoff_seconds * (attempt + 1)
            print(f"[timeout] 等待 {wait_time:.1f} 秒后重试，entity={entity_text}")
            time.sleep(wait_time)

        except requests.exceptions.ConnectionError:
            last_status = "connection_error"
            wait_time = backoff_seconds * (attempt + 1)
            print(f"[connection_error] 等待 {wait_time:.1f} 秒后重试，entity={entity_text}")
            time.sleep(wait_time)

        except Exception as e:
            last_status = f"error_{type(e).__name__}"

            # 未知错误不缓存
            return None, last_status

    return None, last_status


def build_knowledge_text(label: str, description: str) -> str:
    label = clean_text(label)
    description = clean_text(description)

    if label and description:
        return f"{label}: {description}"

    if label:
        return label

    return description


def valid_qid_series(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    return values.str.startswith("Q")


def build_wikidata_knowledge(
    csv_path: str,
    output_path: str,
    cache_path: str,
    max_rows: int,
    max_entities_per_row: int,
    language: str,
    api_limit: int,
    timeout: int,
    sleep_seconds: float,
    max_retries: int,
    backoff_seconds: float,
) -> None:
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    cache_path = Path(cache_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到输入 CSV：{csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_columns = ["id", "title", "text"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"输入 CSV 缺少必要列：{col}，当前列名：{list(df.columns)}")

    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows).copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache = load_cache(cache_path)

    rows = []
    seen_qids = set()
    seen_sample_entity = set()

    print()
    print("========================================")
    print("开始构建 Wikidata 知识库")
    print("========================================")
    print(f"输入 CSV：{csv_path}")
    print(f"输出 CSV：{output_path}")
    print(f"缓存文件：{cache_path}")
    print(f"样本数：{len(df)}")
    print(f"每条样本最多实体数：{max_entities_per_row}")
    print(f"sleep：{sleep_seconds}")
    print(f"max_retries：{max_retries}")
    print(f"backoff_seconds：{backoff_seconds}")
    print()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Wikidata"):
        sample_id = clean_text(row["id"])
        title = clean_text(row["title"])
        text = clean_text(row["text"])

        entities = extract_entities_from_row(
            title=title,
            text=text,
            max_entities=max_entities_per_row,
        )

        for entity_text in entities:
            pair_key = f"{sample_id}::{entity_text}"

            if pair_key in seen_sample_entity:
                continue

            seen_sample_entity.add(pair_key)

            result, status = search_wikidata_entity(
                entity_text=entity_text,
                language=language,
                limit=api_limit,
                timeout=timeout,
                cache=cache,
                sleep_seconds=sleep_seconds,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
            )

            if result is None:
                rows.append(
                    {
                        "source_sample_id": sample_id,
                        "entity_text": entity_text,
                        "qid": "",
                        "label": "",
                        "description": "",
                        "knowledge_text": "",
                        "concept_uri": "",
                        "api_status": status,
                    }
                )
                continue

            qid = result["qid"]
            label = result["label"]
            description = result["description"]
            concept_uri = result["concept_uri"]
            knowledge_text = build_knowledge_text(label, description)

            seen_qids.add(qid)

            rows.append(
                {
                    "source_sample_id": sample_id,
                    "entity_text": entity_text,
                    "qid": qid,
                    "label": label,
                    "description": description,
                    "knowledge_text": knowledge_text,
                    "concept_uri": concept_uri,
                    "api_status": status,
                }
            )

    out_df = pd.DataFrame(
        rows,
        columns=[
            "source_sample_id",
            "entity_text",
            "qid",
            "label",
            "description",
            "knowledge_text",
            "concept_uri",
            "api_status",
        ],
    )

    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    save_cache(cache, cache_path)

    ok_mask = valid_qid_series(out_df["qid"]) if len(out_df) > 0 else pd.Series(dtype=bool)
    ok_df = out_df[ok_mask].copy() if len(out_df) > 0 else out_df.copy()

    print()
    print("========================================")
    print("Wikidata 知识库构建完成")
    print("========================================")
    print(f"输出文件：{output_path}")
    print(f"总知识记录数：{len(out_df)}")
    print(f"成功匹配记录数：{len(ok_df)}")
    print(f"唯一 QID 数：{ok_df['qid'].nunique() if len(ok_df) > 0 else 0}")
    print()
    print("API 状态统计：")
    if len(out_df) > 0:
        print(out_df["api_status"].value_counts())
    else:
        print("暂无记录")
    print()
    print("前 10 条成功知识：")
    if len(ok_df) > 0:
        print(ok_df[["entity_text", "qid", "label", "description"]].head(10))
    else:
        print("暂无成功知识记录")
    print("========================================")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="输入 full CSV，例如 data/processed/politifact_full.csv",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出知识库 CSV，例如 knowledge/wikidata_knowledge_politifact.csv",
    )

    parser.add_argument(
        "--cache",
        type=str,
        default="knowledge/cache/wikidata_search_cache.json",
        help="Wikidata 搜索缓存文件。",
    )

    parser.add_argument(
        "--max_rows",
        type=int,
        default=100,
        help="最多处理多少条新闻。设置为 0 表示处理全部。",
    )

    parser.add_argument(
        "--max_entities_per_row",
        type=int,
        default=5,
        help="每条新闻最多抽取多少个候选实体。",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Wikidata 搜索语言，默认 en。",
    )

    parser.add_argument(
        "--api_limit",
        type=int,
        default=3,
        help="每个实体搜索返回多少个候选，当前脚本默认取第 1 个。",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Wikidata API 请求超时时间。",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="每次 API 请求后暂停时间，避免请求过快。建议 1.0 或更大。",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="遇到 429/timeout/connection_error 时的最大重试次数。",
    )

    parser.add_argument(
        "--backoff",
        type=float,
        default=5.0,
        help="重试退避等待基数，实际等待约为 backoff * attempt。",
    )

    args = parser.parse_args()

    build_wikidata_knowledge(
        csv_path=args.csv,
        output_path=args.output,
        cache_path=args.cache,
        max_rows=args.max_rows,
        max_entities_per_row=args.max_entities_per_row,
        language=args.language,
        api_limit=args.api_limit,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
        max_retries=args.max_retries,
        backoff_seconds=args.backoff,
    )


if __name__ == "__main__":
    main()
