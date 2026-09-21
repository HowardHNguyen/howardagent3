"""Session-owned hybrid search with complete parent passages and atomic indexing."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import math
import re
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import load_document, DocumentLoaderException
from llms import provider_call
from limits import (MAX_FILES, MAX_BATCH_BYTES, MAX_BATCH_MB, MAX_CHUNKS, MAX_INDEX_CHARS,
                    EMBED_BATCH_SIZE, PARENT_CHARS, TABLE_CHARS,
                    CONTEXT_CHARS, EXPANDED_CONTEXT_CHARS)

STOPWORDS = set("a an and are as at be by can do does for from how i in is it of on or that the this to was were what which who with please tell me about".split())
NUMBERS = dict(zip("one two three four five six seven eight nine ten eleven twelve".split(), map(str, range(1, 13))))


def terms(text):
    words = re.findall(r"\w+", text.casefold())
    return [NUMBERS.get(w, w[:-1] if w.endswith("s") and len(w) > 4 else w)
            for w in words if w not in STOPWORDS]


def is_overview(query):
    return bool(re.search(r"\b(?:list|all|every|overview|summari[sz]e|what\s+are|which\s+are|\d{1,2})\b", query, re.I)
                or any(w in NUMBERS for w in query.casefold().split()))


class KeywordIndex:
    """BM25 over child chunks; built once without storing additional raw documents."""
    def __init__(self, texts):
        self.postings = defaultdict(list)
        self.lengths = []
        for index, text in enumerate(texts):
            counts = Counter(terms(text))
            self.lengths.append(sum(counts.values()))
            for word, frequency in counts.items():
                self.postings[word].append((index, frequency))
        self.average = sum(self.lengths) / max(1, len(self.lengths))

    def search(self, query, k=40):
        scores = defaultdict(float)
        for word in set(terms(query)):
            posting = self.postings.get(word, [])
            idf = math.log(1 + (len(self.lengths) - len(posting) + .5) / (len(posting) + .5))
            for index, frequency in posting:
                norm = 1.2 * (.25 + .75 * self.lengths[index] / max(1, self.average))
                scores[index] += idf * frequency * 2.2 / (frequency + norm)
        return sorted(scores, key=lambda i: (-scores[i], i))[:k]


@dataclass(frozen=True)
class Upload:
    name: str
    data: bytes


def selection_signature(uploads):
    return tuple(sorted((u.name, sha256(u.data).hexdigest()) for u in uploads))


class BuildError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("Knowledge base was not changed. Fix the listed files and rebuild.")


def parent_passages(docs):
    """Do not split ordinary tables. Bound very long text/individual table rows."""
    parents = []
    for doc in docs:
        cap = TABLE_CHARS if doc.metadata.get("block_type") == "table" else PARENT_CHARS
        splitter = RecursiveCharacterTextSplitter(chunk_size=cap, chunk_overlap=300)
        for part in splitter.split_documents([doc]):
            title = doc.metadata.get("section_title", "")
            if title and not part.page_content.startswith(title):
                part.page_content = title + "\n\n" + part.page_content
            part.metadata["parent_id"] = len(parents)
            parents.append(part)
    return parents


class DocumentRetriever:
    def __init__(self, embeddings, k=10):
        self.embeddings, self.k = embeddings, k
        self.store = None
        self.signature = None
        self.chunk_count = self.file_count = 0
        self.parents, self.chunks = [], []
        self.keyword_index = None

    @property
    def ready(self):
        return self.store is not None and self.chunk_count > 0

    def build(self, uploads, progress=None):
        report = progress or (lambda fraction, message: None)
        if not uploads or len(uploads) > MAX_FILES:
            raise BuildError([("Selection", f"Choose between 1 and {MAX_FILES} files.")])
        if sum(len(u.data) for u in uploads) > MAX_BATCH_BYTES:
            raise BuildError([("Selection", f"Total upload size must not exceed {MAX_BATCH_MB} MB.")])
        signature = selection_signature(uploads)
        if signature == self.signature and self.ready:
            report(1.0, "This selection is already indexed.")
            return False
        docs, errors, seen, chars = [], [], set(), 0
        for index, upload in enumerate(uploads):
            report(.15 * index / len(uploads), f"Reading document {index + 1} of {len(uploads)}…")
            digest = sha256(upload.data).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            try:
                file_docs = load_document(upload.name, upload.data)
                chars += sum(len(d.page_content) for d in file_docs)
                if chars > MAX_INDEX_CHARS:
                    raise BuildError([("Selection", f"Selected documents exceed {MAX_INDEX_CHARS:,} extracted characters. Build separate knowledge bases.")])
                for doc in file_docs:
                    doc.metadata["content_hash"] = digest
                docs.extend(file_docs)
            except DocumentLoaderException as exc:
                errors.append((upload.name, str(exc)))
        if errors:
            raise BuildError(errors)
        parents = parent_passages(docs)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=200)
        chunks = []
        for parent in parents:
            parts = splitter.split_documents([parent])
            title = parent.metadata.get("section_title", "")
            for part in parts:
                if title and not part.page_content.startswith(title):
                    part.page_content = title + "\n\n" + part.page_content
                part.metadata["chunk_id"] = len(chunks)
                chunks.append(part)
        if not chunks or len(chunks) > MAX_CHUNKS:
            raise BuildError([("Selection", f"Choose fewer or smaller documents (1–{MAX_CHUNKS:,} searchable chunks required).")])
        # Only commit after every batch and keyword index succeeds.
        candidate = InMemoryVectorStore(self.embeddings)
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start:start + EMBED_BATCH_SIZE]
            provider_call("OpenAI", candidate.add_documents, batch)
            done = start + len(batch)
            report(.15 + .8 * done / len(chunks), f"Indexed {done:,} of {len(chunks):,} passages…")
        keywords = KeywordIndex([c.page_content for c in chunks])
        self.store, self.parents, self.chunks, self.keyword_index = candidate, parents, chunks, keywords
        self.signature, self.chunk_count, self.file_count = signature, len(chunks), len(seen)
        report(1.0, f"Ready: {self.file_count} files, {self.chunk_count:,} searchable passages.")
        return True

    def invoke(self, query, expanded=False):
        if not self.ready:
            return []
        pool = 64 if expanded else 32
        lexical = self.keyword_index.search(query, k=pool)
        semantic_docs = provider_call("OpenAI", self.store.similarity_search, query, k=pool)
        semantic = [d.metadata["chunk_id"] for d in semantic_docs]
        scores = defaultdict(float)
        for ranking in (lexical, semantic):
            for rank, index in enumerate(ranking):
                scores[index] += 1 / (60 + rank)
        parent_scores = defaultdict(float)
        query_terms = set(terms(query))
        overview = is_overview(query)
        for index, score in scores.items():
            parent_id = self.chunks[index].metadata["parent_id"]
            parent = self.parents[parent_id]
            heading_terms = set(terms(parent.metadata.get("section_title", "")))
            heading_match = len(query_terms & heading_terms) / max(1, len(query_terms))
            score += .03 * heading_match
            if overview and parent.metadata.get("block_type") == "table":
                score += .015 * heading_match
            parent_scores[parent_id] = max(parent_scores[parent_id], score)
        ranked = sorted(parent_scores, key=lambda i: (-parent_scores[i], i))
        limit = 20 if expanded else self.k
        candidates = []
        # Expand adjacent passages within the same source for overviews and lists.
        # Sibling row groups from a matching table are preferred over unrelated hits.
        for parent_id in ranked[:limit]:
            candidates.append(parent_id)
            parent = self.parents[parent_id]
            table_id = parent.metadata.get("table_id")
            if overview and table_id:
                candidates.extend(i for i, p in enumerate(self.parents)
                                  if p.metadata.get("content_hash") == parent.metadata.get("content_hash")
                                  and p.metadata.get("table_id") == table_id)
        if overview or expanded:
            for parent_id in ranked[:limit]:
                for neighbor in (parent_id - 1, parent_id + 1):
                    if 0 <= neighbor < len(self.parents) and self.parents[neighbor].metadata.get("content_hash") == self.parents[parent_id].metadata.get("content_hash"):
                        candidates.append(neighbor)
        selected, used, seen = [], 0, set()
        budget = EXPANDED_CONTEXT_CHARS if expanded else CONTEXT_CHARS
        for parent_id in candidates:
            if parent_id in seen:
                continue
            seen.add(parent_id)
            doc = self.parents[parent_id]
            # Do not silently chop off table rows to meet the context budget.
            if used + len(doc.page_content) > budget:
                continue
            selected.append(Document(page_content=doc.page_content, metadata=dict(doc.metadata)))
            used += len(doc.page_content)
        return selected
