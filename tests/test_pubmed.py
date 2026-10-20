"""Tests for PubMed parsing and caching, without network access."""

import json

from causal_target_to_patient.sources import pubmed
from causal_target_to_patient.sources.pubmed import PubmedClient, parse_pubmed_xml

XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <Journal><Title>Lancet</Title>
          <JournalIssue><PubDate><Year>2019</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Drug X in <i>rheumatoid</i> arthritis</ArticleTitle>
        <Abstract>
          <AbstractText Label="METHODS">Patients were randomised.</AbstractText>
          <AbstractText Label="RESULTS">ACR20 was 66% vs 30% (p&lt;0.001).</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType>Randomized Controlled Trial</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>67890</PMID>
      <Article>
        <Journal><Title>Rheumatology</Title>
          <JournalIssue><PubDate><MedlineDate>2010 Jan-Feb</MedlineDate></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>No abstract here</ArticleTitle>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def test_parse_pubmed_xml():
    first, second = parse_pubmed_xml(XML)
    assert first["pmid"] == "12345"
    assert first["title"] == "Drug X in rheumatoid arthritis"  # nested markup kept as text
    assert first["year"] == "2019"
    assert first["abstract"] == ("METHODS: Patients were randomised.\n"
                                 "RESULTS: ACR20 was 66% vs 30% (p<0.001).")
    assert first["publication_types"] == ["Randomized Controlled Trial"]
    assert second["year"] == "2010 Jan-Feb"
    assert second["abstract"] == ""


class FakeResponse:
    text = XML

    def raise_for_status(self):
        pass


def test_client_fetches_only_missing_and_caches(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params["id"])
        return FakeResponse()

    monkeypatch.setattr(pubmed.requests, "get", fake_get)
    monkeypatch.setattr(pubmed.time, "sleep", lambda s: None)
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"12345": {"pmid": "12345", "title": "cached"}}))

    client = PubmedClient(cache)
    out = client.abstracts(["12345", "67890", "99999"])
    assert calls == ["67890,99999"]          # the cached PMID isn't requested
    assert out["12345"]["title"] == "cached"
    assert "67890" in out and "99999" not in out  # no record returned for 99999
    assert "67890" in json.loads(cache.read_text())

    client.abstracts(["67890"])
    assert len(calls) == 1                   # second call served from the cache
