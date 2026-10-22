# scripts/build_full_articles.py
# -*- coding: utf-8 -*-

"""
从 FakeNewsNet minimal CSV 的 news_url 中抓取真实新闻正文和图片。

输入文件：
    data/raw/politifact/politifact_fake.csv
    data/raw/politifact/politifact_real.csv
    data/raw/gossipcop/gossipcop_fake.csv
    data/raw/gossipcop/gossipcop_real.csv

输出文件：
    data/processed/politifact_full.csv
    data/processed/gossipcop_full.csv

输出字段：
    id,title,text,article_url,image_url,image_path,label,fetch_status,error

说明：
    1. 如果正文抓取失败，text 回退为 title；
    2. 如果图片下载失败，image_path 回退为 data/assets/blank_336.png；
    3. 自动修复缺少 http/https 的 URL，避免 MissingSchema；
    4. 先小样本测试，不要一开始全量抓取；
    5. 后面训练时，可以直接把 --csv 改成 data/processed/politifact_full.csv。
"""

import argparse
import hashlib
import mimetypes
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


RAW_FILES = {
    "politifact": {
        "fake": "data/raw/politifact/politifact_fake.csv",
        "real": "data/raw/politifact/politifact_real.csv",
    },
    "gossipcop": {
        "fake": "data/raw/gossipcop/gossipcop_fake.csv",
        "real": "data/raw/gossipcop/gossipcop_real.csv",
    },
}


LABEL_MAP = {
    "real": 0,
    "fake": 1,
}


DEFAULT_BLANK_IMAGE_PATH = "data/assets/blank_336.png"


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "Connection": "close",
}


def safe_mkdir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_text(text) -> str:
    if text is None:
        return ""

    text = str(text)

    if text.lower() == "nan":
        return ""

    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def normalize_url(url: str) -> str:
    """
    修复 FakeNewsNet minimal CSV 中部分 news_url 缺少协议的问题。

    例如：
        abcnews.go.com/xxx
    修复为：
        https://abcnews.go.com/xxx

    这样可以避免 requests 报 MissingSchema。
    """
    url = clean_text(url)

    if url == "":
        return ""

    lower = url.lower()

    if lower.startswith("http://") or lower.startswith("https://"):
        return url

    if lower.startswith("//"):
        return "https:" + url

    if lower.startswith("www."):
        return "https://" + url

    # 常见情况：域名直接开头，如 abcnews.go.com/...
    if "." in url.split("/")[0]:
        return "https://" + url

    return url


def sanitize_filename(name: str) -> str:
    name = str(name)
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = name.strip()

    if len(name) == 0:
        name = "unknown"

    return name[:120]


def read_csv_safely(path: str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    return df


def find_column(df: pd.DataFrame, candidates) -> Optional[str]:
    lower_map = {}

    for col in df.columns:
        lower_map[str(col).lower()] = col

    for name in candidates:
        key = str(name).lower()
        if key in lower_map:
            return lower_map[key]

    return None


def fetch_html(url: str, timeout: int = 15) -> Tuple[Optional[str], str, str]:
    """
    返回：
        html, status, normalized_url
    """
    normalized_url = normalize_url(url)

    if normalized_url == "":
        return None, "empty_url", normalized_url

    try:
        response = requests.get(
            normalized_url,
            headers=HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )

        if response.status_code != 200:
            return None, f"http_status_{response.status_code}", normalized_url

        content_type = response.headers.get("Content-Type", "")

        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return None, f"not_html_{content_type}", normalized_url

        response.encoding = response.apparent_encoding or response.encoding

        return response.text, "ok", normalized_url

    except requests.exceptions.MissingSchema:
        return None, "missing_schema", normalized_url

    except requests.exceptions.InvalidURL:
        return None, "invalid_url", normalized_url

    except requests.exceptions.Timeout:
        return None, "timeout", normalized_url

    except requests.exceptions.SSLError:
        return None, "ssl_error", normalized_url

    except requests.exceptions.ConnectionError:
        return None, "connection_error", normalized_url

    except Exception as e:
        return None, f"error_{type(e).__name__}", normalized_url


def remove_unwanted_tags(soup: BeautifulSoup) -> None:
    unwanted_tags = [
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "form",
        "button",
        "input",
        "nav",
        "footer",
        "header",
        "aside",
    ]

    for tag in soup(unwanted_tags):
        tag.decompose()


def extract_meta_description(soup: BeautifulSoup) -> str:
    candidates = [
        ("meta", {"property": "og:description"}),
        ("meta", {"name": "description"}),
        ("meta", {"name": "twitter:description"}),
    ]

    for tag_name, attrs in candidates:
        tag = soup.find(tag_name, attrs=attrs)

        if tag and tag.get("content"):
            text = clean_text(tag.get("content"))

            if len(text) >= 50:
                return text

    return ""


def extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    remove_unwanted_tags(soup)

    containers = []

    article_tags = soup.find_all("article")
    main_tags = soup.find_all("main")

    containers.extend(article_tags)
    containers.extend(main_tags)

    common_selectors = [
        {"class_": re.compile(r"(article|story|post|entry|content|body)", re.I)},
        {"id": re.compile(r"(article|story|post|entry|content|body)", re.I)},
    ]

    for selector in common_selectors:
        found = soup.find_all(["div", "section"], **selector)
        containers.extend(found)

    if not containers:
        body = soup.find("body")
        if body:
            containers.append(body)

    best_text = ""

    for container in containers:
        paragraphs = container.find_all("p")
        pieces = []

        for p in paragraphs:
            text = clean_text(p.get_text(" ", strip=True))

            if len(text) < 30:
                continue

            lower_text = text.lower()

            bad_phrases = [
                "subscribe",
                "advertisement",
                "sign up",
                "cookie",
                "privacy policy",
                "terms of service",
                "all rights reserved",
                "follow us",
                "share this",
                "related articles",
                "read more",
            ]

            if any(phrase in lower_text for phrase in bad_phrases):
                continue

            pieces.append(text)

        article_text = clean_text(" ".join(pieces))

        if len(article_text) > len(best_text):
            best_text = article_text

    if len(best_text) < 100:
        meta_desc = extract_meta_description(soup)

        if len(meta_desc) > len(best_text):
            best_text = meta_desc

    return best_text


def extract_image_url(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    meta_candidates = [
        ("meta", {"property": "og:image"}),
        ("meta", {"name": "og:image"}),
        ("meta", {"name": "twitter:image"}),
        ("meta", {"property": "twitter:image"}),
    ]

    for tag_name, attrs in meta_candidates:
        tag = soup.find(tag_name, attrs=attrs)

        if tag and tag.get("content"):
            image_url = clean_text(tag.get("content"))

            if image_url:
                return urljoin(base_url, image_url)

    containers = []

    article = soup.find("article")
    main = soup.find("main")

    if article:
        containers.append(article)

    if main:
        containers.append(main)

    if not containers:
        body = soup.find("body")
        if body:
            containers.append(body)

    img_urls = []

    for container in containers:
        imgs = container.find_all("img")

        for img in imgs:
            src = (
                img.get("src")
                or img.get("data-src")
                or img.get("data-original")
                or img.get("data-lazy-src")
            )

            if not src:
                continue

            src = clean_text(src)

            if src.startswith("data:"):
                continue

            full_url = urljoin(base_url, src)
            img_urls.append(full_url)

    if img_urls:
        return img_urls[0]

    return ""


def guess_image_extension(image_url: str, content_type: str) -> str:
    parsed = urlparse(image_url)
    suffix = Path(parsed.path).suffix.lower()

    allowed_suffixes = [".jpg", ".jpeg", ".png", ".webp"]

    if suffix in allowed_suffixes:
        return suffix

    content_type = content_type.split(";")[0].strip()
    ext = mimetypes.guess_extension(content_type)

    if ext in allowed_suffixes:
        return ext

    return ".jpg"


def download_image(
    image_url: str,
    output_dir: Path,
    sample_id: str,
    timeout: int = 20,
) -> Tuple[str, str]:
    image_url = normalize_url(image_url)

    if image_url == "":
        return DEFAULT_BLANK_IMAGE_PATH, "empty_image_url"

    try:
        response = requests.get(
            image_url,
            headers=HEADERS,
            timeout=timeout,
            allow_redirects=True,
            stream=True,
        )

        if response.status_code != 200:
            return DEFAULT_BLANK_IMAGE_PATH, f"image_http_status_{response.status_code}"

        content_type = response.headers.get("Content-Type", "").lower()

        if "image" not in content_type:
            return DEFAULT_BLANK_IMAGE_PATH, f"not_image_{content_type}"

        content = response.content

        if len(content) < 1024:
            return DEFAULT_BLANK_IMAGE_PATH, "image_too_small"

        suffix = guess_image_extension(
            image_url=image_url,
            content_type=content_type,
        )

        safe_id = sanitize_filename(sample_id)
        url_hash = hashlib.md5(image_url.encode("utf-8")).hexdigest()[:8]
        filename = f"{safe_id}_{url_hash}{suffix}"

        output_path = output_dir / filename

        with open(output_path, "wb") as f:
            f.write(content)

        return str(output_path), "ok"

    except requests.exceptions.MissingSchema:
        return DEFAULT_BLANK_IMAGE_PATH, "image_missing_schema"

    except requests.exceptions.InvalidURL:
        return DEFAULT_BLANK_IMAGE_PATH, "image_invalid_url"

    except requests.exceptions.Timeout:
        return DEFAULT_BLANK_IMAGE_PATH, "image_timeout"

    except requests.exceptions.SSLError:
        return DEFAULT_BLANK_IMAGE_PATH, "image_ssl_error"

    except requests.exceptions.ConnectionError:
        return DEFAULT_BLANK_IMAGE_PATH, "image_connection_error"

    except Exception as e:
        return DEFAULT_BLANK_IMAGE_PATH, f"image_error_{type(e).__name__}"


def process_one_row(
    row,
    id_col: Optional[str],
    title_col: str,
    url_col: str,
    label: int,
    timeout: int,
    image_dir: Path,
) -> Dict:
    if id_col is not None:
        sample_id = clean_text(row[id_col])
    else:
        sample_id = ""

    title = clean_text(row[title_col])
    raw_article_url = clean_text(row[url_col])
    article_url = normalize_url(raw_article_url)

    if sample_id == "":
        sample_id = hashlib.md5((title + article_url).encode("utf-8")).hexdigest()[:16]

    text = title
    image_url = ""
    image_path = DEFAULT_BLANK_IMAGE_PATH
    fetch_status = "not_started"
    error = ""

    html, html_status, normalized_url = fetch_html(
        url=article_url,
        timeout=timeout,
    )

    article_url = normalized_url

    if html is None:
        fetch_status = html_status
        error = html_status

    else:
        article_text = extract_article_text(html)
        extracted_image_url = extract_image_url(
            html=html,
            base_url=article_url,
        )

        if len(article_text) >= 100:
            text = article_text
        else:
            text = title

        image_url = normalize_url(extracted_image_url)

        downloaded_path, image_status = download_image(
            image_url=image_url,
            output_dir=image_dir,
            sample_id=sample_id,
            timeout=timeout,
        )

        image_path = downloaded_path

        if image_status == "ok":
            fetch_status = "ok"
            error = ""
        else:
            fetch_status = f"text_ok_{image_status}"
            error = fetch_status

    return {
        "id": sample_id,
        "title": title,
        "text": text,
        "article_url": article_url,
        "image_url": image_url,
        "image_path": image_path,
        "label": label,
        "fetch_status": fetch_status,
        "error": error,
    }


def build_dataset(
    dataset_name: str,
    limit: int,
    sleep_seconds: float,
    timeout: int,
) -> None:
    print()
    print("========================================")
    print(f"开始处理数据集：{dataset_name}")
    print("========================================")
    print()

    output_rows = []

    image_dir = Path("data/images") / dataset_name
    image_dir.mkdir(parents=True, exist_ok=True)

    for label_name in ["fake", "real"]:
        csv_path = RAW_FILES[dataset_name][label_name]
        label = LABEL_MAP[label_name]

        df = read_csv_safely(csv_path)

        id_col = find_column(df, ["id", "news_id", "post_id"])
        title_col = find_column(df, ["title", "text", "content"])
        url_col = find_column(df, ["news_url", "url", "article_url"])

        if title_col is None:
            raise ValueError(
                f"{csv_path} 没有找到 title/text/content 列。"
                f"当前列名：{list(df.columns)}"
            )

        if url_col is None:
            raise ValueError(
                f"{csv_path} 没有找到 news_url/url/article_url 列。"
                f"当前列名：{list(df.columns)}"
            )

        if limit is not None and limit > 0:
            df = df.head(limit).copy()

        print(f"文件：{csv_path}")
        print(f"类别：{label_name}, label={label}")
        print(f"样本数：{len(df)}")
        print(f"id_col={id_col}, title_col={title_col}, url_col={url_col}")
        print()

        total = len(df)

        for local_index, (_, row) in enumerate(df.iterrows(), start=1):
            print(f"[{dataset_name}-{label_name}] {local_index}/{total}")

            result = process_one_row(
                row=row,
                id_col=id_col,
                title_col=title_col,
                url_col=url_col,
                label=label,
                timeout=timeout,
                image_dir=image_dir,
            )

            print(f"  id: {result['id']}")
            print(f"  url: {result['article_url']}")
            print(f"  status: {result['fetch_status']}")
            print(f"  text_len: {len(result['text'])}")
            print(f"  image_path: {result['image_path']}")
            print()

            output_rows.append(result)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    output_df = pd.DataFrame(
        output_rows,
        columns=[
            "id",
            "title",
            "text",
            "article_url",
            "image_url",
            "image_path",
            "label",
            "fetch_status",
            "error",
        ],
    )

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_name}_full.csv"
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    failure_df = output_df[output_df["fetch_status"] != "ok"].copy()
    failure_path = output_dir / f"{dataset_name}_full_failures.csv"
    failure_df.to_csv(failure_path, index=False, encoding="utf-8-sig")

    print()
    print("========================================")
    print(f"数据集处理完成：{dataset_name}")
    print(f"输出文件：{output_path}")
    print(f"失败记录：{failure_path}")
    print(f"总样本数：{len(output_df)}")
    print()
    print("标签统计：")
    print(output_df["label"].value_counts().sort_index())
    print()
    print("抓取状态统计：")
    print(output_df["fetch_status"].value_counts())
    print("========================================")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="politifact",
        choices=["politifact", "gossipcop", "all"],
        help="要处理的数据集。",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help=(
            "每个类别最多处理多少条。"
            "默认 10。"
            "设置为 0 表示处理全部。"
        ),
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="每条样本之间暂停多少秒，避免请求过快。",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="网页和图片请求超时时间。",
    )

    args = parser.parse_args()

    safe_mkdir("data/processed")
    safe_mkdir("data/images")
    safe_mkdir("data/assets")

    if args.dataset == "all":
        dataset_names = ["politifact", "gossipcop"]
    else:
        dataset_names = [args.dataset]

    for dataset_name in dataset_names:
        build_dataset(
            dataset_name=dataset_name,
            limit=args.limit,
            sleep_seconds=args.sleep,
            timeout=args.timeout,
        )


if __name__ == "__main__":
    main()
