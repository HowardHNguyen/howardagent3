"""Version 3: URL-only app. Run with: streamlit run streamlit_app.py."""
import logging
import time
from uuid import uuid4
import streamlit as st
from config import Settings, ConfigurationError
from llms import create_chat_model, create_embeddings, check_groq, ServiceError
from url_loader import URLLoadError, MAX_URLS, MAX_PAGE_MB
from url_retriever import URLRetriever, URLBuildError, url_selection
from rag import ask, NO_EVIDENCE
from answer_rendering import answer_html, ANSWER_CSS

LOGGER = logging.getLogger(__name__)
st.set_page_config(page_title='AI Knowledge Platform · Version 3', page_icon='🌐', layout='wide')


def reset_session():
    for key in list(st.session_state):
        if key.startswith(('web_', 'urls_')):
            del st.session_state[key]
    st.session_state.web_schema = 2
    st.session_state.web_id = uuid4().hex
    st.session_state.web_history = []
    st.session_state.web_turns = []
    st.session_state.web_retriever = None


if 'web_id' not in st.session_state or st.session_state.get('web_schema') != 2:
    reset_session()
settings = Settings.load()
st.title('🌐 AI Knowledge Platform · Version 3')
st.caption('Chat with public webpages · Answers with source references')
with st.sidebar:
    st.subheader('Model connection')
    st.caption('Groq generation model')
    st.code(settings.model, language=None)
    if st.button('Check Groq connection'):
        now = time.monotonic()
        if now - st.session_state.get('web_last_check', 0) < 10:
            st.info('Please wait 10 seconds before checking again.')
        else:
            st.session_state.web_last_check = now
            try:
                with st.spinner('Checking model access…'):
                    st.success(check_groq(settings))
            except (ConfigurationError, ServiceError) as exc:
                st.error(str(exc))
            except Exception as exc:
                LOGGER.error('Connection check failed type=%s', type(exc).__name__)
                st.error('The connection check could not finish. The owner should check app settings.')
    st.caption('This check sends a short synthetic prompt to Groq and may incur a small API charge.')
    st.button('Clear Session', on_click=reset_session)
    st.caption('Clears this session’s webpage index and conversation. Provider records are outside this control.')
    st.link_button('Version 2 · File uploads', 'https://howardagent2.streamlit.app/')

chat, about, howto = st.tabs(['💬 Chat', 'ℹ️ About', '🧭 How to use'])
with chat:
    st.subheader('Your webpages')
    st.caption('Build fetches your URLs and sends extracted text to OpenAI for embeddings. Questions, recent conversation, and relevant passages go to Groq. Use public pages you are authorized to process; do not paste secret or signed URLs.')
    text = st.text_area('Public webpage URLs — one per line', height=140,
                        placeholder='https://example.com/article', max_chars=21000,
                        key=f'urls_{st.session_state.web_id}')
    st.caption(f'Up to {MAX_URLS} URLs · {MAX_PAGE_MB} MB downloaded and 200,000 extracted characters per page · Only the pages you enter are read')
    urls, valid = (), True
    try:
        urls = url_selection(text)
    except URLLoadError as exc:
        valid = False
        st.error(str(exc))
    retriever = st.session_state.web_retriever
    active = valid and retriever is not None and retriever.ready and retriever.signature == urls
    if urls and not active:
        st.info('Build the knowledge base for these URLs before asking a question.')
    if st.button('Build / Refresh Knowledge Base', disabled=not urls or not valid, type='primary'):
        now = time.monotonic()
        if now - st.session_state.get('web_last_build', 0) < 10:
            st.info('Please wait 10 seconds before fetching again.')
        else:
            st.session_state.web_last_build = now
            try:
                with st.spinner('Reading and indexing webpages…'):
                    if retriever is None:
                        retriever = URLRetriever(create_embeddings(settings))
                    indicator = st.progress(0, text='Preparing webpages…')
                    try:
                        retriever.build(urls, progress=lambda fraction, message: indicator.progress(fraction, text=message))
                    finally:
                        indicator.empty()
                    st.session_state.web_retriever = retriever
                    st.session_state.web_history = []
                    st.session_state.web_turns = []
                    active = True
                st.success(f'Ready: {retriever.file_count} webpages, {retriever.chunk_count} searchable passages.')
            except URLBuildError as exc:
                st.error(str(exc))
                for label, problem in exc.errors:
                    st.text(f'{label}: {problem}')
            except (URLLoadError, ConfigurationError, ServiceError) as exc:
                st.error(str(exc))
            except Exception as exc:
                LOGGER.error('Web indexing failed type=%s', type(exc).__name__)
                st.error('Indexing could not finish. The existing knowledge base was not changed. Retry with fewer URLs.')
    if active:
        st.caption(f'Knowledge base ready · {retriever.file_count} webpages · {retriever.chunk_count} passages')
        st.caption('Coverage: only the URLs listed above, not every page on those websites. Add service/product page URLs for fuller answers.')
        with st.expander('Indexed pages, fetch times, and text preview'):
            for page in retriever.pages:
                st.link_button(page.title, page.url)
                extracted = '\n\n'.join(d.page_content for d in page.documents)
                st.caption(f'Fetched {page.fetched_at} · {len(extracted):,} characters')
                if len(extracted) < 500:
                    st.warning('Very little text was extracted. This page may not contain enough detail; try a specific service or article URL.')
                st.text(extracted[:3000] + ('\n[Preview shortened]' if len(extracted) > 3000 else ''))
    st.divider()

    def show_turn(turn):
        with st.chat_message('user'):
            st.text(turn['question'])
        with st.chat_message('assistant'):
            answer = turn['answer']
            if answer == NO_EVIDENCE:
                answer = 'I could not find enough evidence in the selected webpages. Add a relevant page or ask a more specific question.'
            st.html(ANSWER_CSS + answer_html(answer))
            if turn.get('citation_warning'):
                st.warning(turn['citation_warning'])
            for source in turn.get('sources', []):
                with st.expander(f"[{source['id']}] Source passage"):
                    st.link_button('Open source webpage', source['source'])
                    st.text(source['source'])
                    if source.get('section_title'):
                        st.text(source['section_title'])
                    st.text(source['text'])

    for turn in st.session_state.web_turns:
        show_turn(turn)
    question = st.chat_input('Ask about your selected webpages…', disabled=not active, max_chars=4000)
    if question:
        now = time.monotonic()
        if now - st.session_state.get('web_last_question', 0) < 3:
            st.info('Please wait a few seconds before asking again.')
        else:
            st.session_state.web_last_question = now
            try:
                with st.spinner('Finding evidence and preparing an answer…'):
                    result = ask(retriever, create_chat_model(settings), question, st.session_state.web_history)
                turn = {'question': question, 'answer': result['answer'], 'sources': result['sources'],
                        'citation_warning': result.get('citation_warning', '')}
                st.session_state.web_turns = (st.session_state.web_turns + [turn])[-20:]
                if result['sources'] and not result.get('citation_warning'):
                    st.session_state.web_history = (st.session_state.web_history + [(question, result['answer'])])[-6:]
                show_turn(turn)
            except (ConfigurationError, ServiceError, ValueError) as exc:
                st.error(str(exc))
            except Exception as exc:
                LOGGER.error('Web answer failed type=%s', type(exc).__name__)
                st.error('The answer could not be completed. Please retry.')

with about:
    st.markdown('''
### Your webpages, your questions — knowledge assistance for everyone
Explore public webpages for work, study, research, or personal learning.
Paste the URLs, click **Build / Refresh Knowledge Base**, and chat to get answers,
summaries, and details grounded in the retrieved page text. Expand the source passages
to check the evidence or open the original webpage.

Version 3 reads URLs only. [Version 2](https://howardagent2.streamlit.app/) handles file uploads.
This app reads the specific pages you enter; it does not search or crawl an entire website.
Answers reflect the page snapshot taken when you build. Refresh the knowledge base for newer content.

### Supported pages
Public HTML webpages and plain text. Login-protected content, paywalls, downloadable files,
videos, and content rendered only by JavaScript are not supported. Some websites block automated access.
Only readable text is indexed; images and visual charts are not interpreted.
Source citations help you review answers but do not guarantee correctness.

### Data handling
Each browser session has its own in-memory index and conversation. No webpage or embedding cache
is written to disk by the application. Clear Session drops the app’s references; it does not securely
wipe memory or delete provider records. OpenAI receives extracted text and search queries for embeddings;
Groq receives questions, bounded conversation, and selected passages. Provider retention policies apply.
Fetching contacts the websites you specify. Use public, non-sensitive URLs you are authorized to process.
This demonstration has no enterprise authentication or organization-wide usage controls.
''')
with howto:
    st.markdown('''
1. Paste one or more public webpage URLs, one per line, including `https://` or `http://`.
2. Click **Build / Refresh Knowledge Base** and wait for indexing to finish.
3. Ask a question about those pages. Open the numbered source passages to check the answer.
4. Changing URLs disables chat until you rebuild. Every successful refresh starts a new conversation.
5. Use **Clear Session** when you are finished.

For a long website, choose the specific article or documentation pages you need. Adding a homepage
only reads that homepage, not all its links. Refresh explicitly when you want current page content.
If any page fails, the build keeps the previous index unchanged. Fix or remove the failed URL and try again.
''')
st.caption('© 2026 Howard Nguyen, PhD · Version 3 · URL knowledge assistant')
