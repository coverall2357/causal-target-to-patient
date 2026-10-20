"""Fetch PubMed abstracts by PMID through NCBI E-utilities.

Uses efetch (https://www.ncbi.nlm.nih.gov/books/NBK25499/). Records are cached
in a JSON file keyed by PMID, so reruns don't repeat requests. NCBI allows 3
requests a second without an API key and 10 with one; set NCBI_API_KEY in .env
to use a key.
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 100


def _text(element: ET.Element | None) -> str:
    """All text inside an element, including nested markup such as <i>."""
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse an efetch PubmedArticleSet into one record per article.

    Each record has pmid, title, journal, year, publication_types, and abstract.
    Structured abstracts keep their section labels ("RESULTS: ...").
    """
    records = []
    for article in ET.fromstring(xml_text).iter("PubmedArticle"):
        citation = article.find("MedlineCitation")
        info = citation.find("Article")
        sections = []
        for part in info.findall("Abstract/AbstractText"):
            label = part.get("Label")
            body = _text(part)
            sections.append(f"{label}: {body}" if label else body)
        year = info.findtext("Journal/JournalIssue/PubDate/Year") or info.findtext(
            "Journal/JournalIssue/PubDate/MedlineDate"
        )
        records.append({
            "pmid": citation.findtext("PMID"),
            "title": _text(info.find("ArticleTitle")),
            "journal": info.findtext("Journal/Title"),
            "year": year,
            "publication_types": [
                _text(t) for t in info.findall("PublicationTypeList/PublicationType")
            ],
            "abstract": "\n".join(sections),
        })
    return records


class PubmedClient:
    """Minimal PubMed efetch client with a JSON cache keyed by PMID."""

    def __init__(self, cache_path: Path, api_key: str | None = None):
        self.cache_path = Path(cache_path)
        self.api_key = api_key
        self.pause = 0.11 if api_key else 0.34
        self.cache = (
            json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        )

    def save(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache))

    def abstracts(self, pmids: list[str]) -> dict[str, dict]:
        """Return records for the given PMIDs, fetching only those not cached.

        PMIDs that PubMed returns no record for are absent from the result.
        """
        pmids = [str(p) for p in dict.fromkeys(pmids)]
        missing = [p for p in pmids if p not in self.cache]
        for start in range(0, len(missing), BATCH_SIZE):
            batch = missing[start:start + BATCH_SIZE]
            params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
            if self.api_key:
                params["api_key"] = self.api_key
            response = requests.get(EFETCH_URL, params=params, timeout=60)
            response.raise_for_status()
            for record in parse_pubmed_xml(response.text):
                if record["pmid"] in batch:  # ignore anything not requested
                    self.cache[record["pmid"]] = record
            time.sleep(self.pause)
        if missing:
            self.save()
        return {p: self.cache[p] for p in pmids if p in self.cache}
