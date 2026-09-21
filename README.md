# Version 3 — URL knowledge assistant

Version 2 remains the file-upload app at https://howardagent2.streamlit.app/ using `streamlit_app.py`.
Version 3 is a separate Streamlit app using `streamlit_app.py`. It has no file uploader.
This repository contains the URL-only Version 3 app. Version 2 remains in the separate genai-langchain repository.

## Deploy separately

In Streamlit Community Cloud, create a **new app** with:

- Repository: `HowardHNguyen/howardagent3`
- Branch: `main`
- Main file: `streamlit_app.py`
- Custom subdomain: `howardagent3` (subject to availability)
- Python: `3.12`
- Secrets: set `OPENAI_API_KEY`, `GROQ_API_KEY`, and optionally `GROQ_MODEL` (default `openai/gpt-oss-20b`). Configure them through Streamlit's secrets editor, never GitHub.

Do not change Version 2's entry point or URL. A new Streamlit deployment needs its own secrets settings.
See [Streamlit deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Run locally

```sh
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Use

Paste up to 10 public HTTP(S) URLs, one per line, and click **Build / Refresh Knowledge Base**.
The app reads only those pages, not their links. Ask questions and inspect the cited passages and source links.
Each successful build fetches fresh snapshots and clears old conversation. Changing the selection disables
chat until a successful rebuild. Failed fetches or embedding calls leave the previous index unchanged.

Supported: readable HTML and plain text, including headings, lists, and tables. Unsupported: login-only or
paywalled pages, JavaScript-only content, images, videos, downloadable PDFs/files, and sites that block automated requests.
Limits: 10 MB per page after decompression, 200,000 extracted characters per page, 3,000 index chunks,
30-second request/read deadline per page (OS DNS resolution may take longer), four redirects, and a two-minute
budget checked before each page fetch. Indexing/model calls have separate provider timeouts.

Connections use only public IP addresses, standard ports, pinned DNS results, verified TLS with original-host SNI,
and revalidated redirects. They carry no cookies or credentials and ignore environment proxy configuration.
Gzip/deflate expansion is bounded. Only validated source URLs become clickable links; model-generated links remain disabled.
Page text is untrusted reference material, not instructions. No browsing/action tools are exposed to the model.

Each session owns its own in-memory index and history; no webpage/embedding cache is written to disk by the app.
The website receives fetch requests, OpenAI processes text for embeddings, and Groq processes relevant passages and questions.
Users should enter public URLs they may process, never secret/signed URLs. This is a public prototype, not an authenticated enterprise service.

## Validation

The 23-test suite covers URL validation, DNS/private-network rejection, pinned TLS connections,
redirect handling, compressed-size limits, extraction, complete table answers, atomic failure rollback, refresh, session isolation,
and the URL-only Streamlit build/chat/reset flow. Live fetching was also checked against Python.org and Example Domain.
