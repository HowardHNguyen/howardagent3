"""URL snapshots with transactional indexing; the Version 2 file path is unchanged."""
import time
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llms import provider_call
from retriever import DocumentRetriever, parent_passages, KeywordIndex
from limits import EMBED_BATCH_SIZE
from url_loader import normalize_url, load_url, URLLoadError, MAX_URLS


def url_selection(text):
    values = [line.strip() for line in text.splitlines() if line.strip()]
    if len(values) > MAX_URLS:
        raise URLLoadError(f'Enter at most {MAX_URLS} URLs, one per line.')
    return tuple(sorted(set(normalize_url(value) for value in values)))


class URLBuildError(ValueError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__('Knowledge base was not changed. Fix the listed URLs and rebuild.')


class URLRetriever(DocumentRetriever):
    def __init__(self, embeddings):
        super().__init__(embeddings)
        self.pages = []

    def build(self, urls, progress=None):
        report = progress or (lambda fraction, message: None)
        signature = url_selection('\n'.join(urls))
        if not signature:
            raise URLBuildError([('Selection', 'Enter at least one public webpage URL.')])
        pages, docs, errors, seen = [], [], [], set()
        started = time.monotonic()
        for index, url in enumerate(signature):
            if time.monotonic() - started > 120:
                raise URLBuildError([('Selection', 'Fetching exceeded two minutes. Try fewer URLs.')])
            report(.3 * index / len(signature), f'Reading page {index + 1} of {len(signature)}…')
            try:
                page = load_url(url)
                if page.url not in seen:
                    seen.add(page.url)
                    pages.append(page)
                    docs.extend(page.documents)
            except URLLoadError as exc:
                # Do not echo potentially sensitive URL query parameters in errors.
                errors.append((f'URL {index + 1}', str(exc)))
        if errors:
            raise URLBuildError(errors)
        parents = parent_passages(docs)
        chunks = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=200).split_documents(parents)
        if not chunks or len(chunks) > 3000:
            raise URLBuildError([('Selection', 'Choose fewer pages (maximum 3,000 searchable passages).')])
        for index, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = index
        candidate = InMemoryVectorStore(self.embeddings)
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start:start + EMBED_BATCH_SIZE]
            provider_call('OpenAI', candidate.add_documents, batch)
            done = start + len(batch)
            report(.3 + .65 * done / len(chunks), f'Indexed {done} of {len(chunks)} passages…')
        keywords = KeywordIndex([c.page_content for c in chunks])
        self.store, self.parents, self.chunks, self.keyword_index = candidate, parents, chunks, keywords
        self.signature, self.chunk_count, self.file_count = signature, len(chunks), len(pages)
        self.pages = pages
        report(1.0, 'Webpage knowledge base ready.')
        return True
