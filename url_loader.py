"""Bounded public-webpage ingestion with DNS-pinned connections and safe redirects."""
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import http.client
import ipaddress
import re
import socket
import ssl
import time
import threading
import zlib
from urllib.parse import urlsplit, urlunsplit, urljoin, quote
from lxml import html, etree
from langchain_core.documents import Document

MAX_URLS = 10
MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_PAGE_CHARS = 200_000
PAGE_TIMEOUT = 20
MAX_REDIRECTS = 4


class URLLoadError(ValueError):
    """Safe to show without response bodies, credentials, or request internals."""


def normalize_url(value):
    value = value.strip()
    if not value or len(value) > 2048 or re.search(r'[\x00-\x20\x7f\\]', value):
        raise URLLoadError('Enter a valid public HTTP or HTTPS URL (at most 2,048 characters).')
    try:
        p = urlsplit(value)
        host = (p.hostname or '').encode('idna').decode('ascii').lower().rstrip('.')
        port = p.port
    except (ValueError, UnicodeError):
        raise URLLoadError('The URL has an invalid hostname or port.') from None
    if p.scheme not in ('http', 'https') or not host or p.username is not None or p.password is not None:
        raise URLLoadError('Use a public HTTP or HTTPS URL without embedded login credentials.')
    if port not in (None, 80 if p.scheme == 'http' else 443):
        raise URLLoadError('Only standard HTTP and HTTPS ports are supported.')
    if host == 'localhost' or host.endswith(('.localhost', '.local', '.internal')) or '%' in host:
        raise URLLoadError('Private or local network addresses are not allowed.')
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not public_address(address):
        raise URLLoadError('Private or local network addresses are not allowed.')
    authority = f'[{host}]' if ':' in host else host
    return urlunsplit((p.scheme, authority, quote(p.path or '/', safe="/%:@!$&'()*+,;=-._~"),
                       quote(p.query, safe="/%?:@!$&'()*+,;=-._~"), ''))


def public_address(address):
    # Also reject IPv6 transition mechanisms which can encode a non-public destination.
    return address.is_global and not (address.is_multicast or address.is_reserved
        or getattr(address, 'ipv4_mapped', None) or getattr(address, 'sixtofour', None)
        or getattr(address, 'teredo', None))


def public_addresses(host, port):
    try:
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        raise URLLoadError('The website hostname could not be resolved.') from None
    addresses = list(dict.fromkeys(record[4][0] for record in records))
    if not addresses or any(not public_address(ipaddress.ip_address(a)) for a in addresses):
        raise URLLoadError('This hostname resolves to a private or unsupported network address.')
    return addresses


@dataclass(frozen=True)
class Page:
    requested_url: str
    url: str
    title: str
    fetched_at: str
    documents: list


def _request(url, deadline):
    """Connect to a validated numeric IP; keep original Host and TLS verification/SNI.

    No second DNS lookup, environment proxies, cookies, or authorization headers.
    """
    p = urlsplit(url)
    port = 443 if p.scheme == 'https' else 80
    addresses = public_addresses(p.hostname, port)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise URLLoadError('The website took too long to respond.')
    conn = http.client.HTTPConnection(p.hostname, port, timeout=min(10, remaining))
    raw = response = timer = None
    try:
        raw = socket.create_connection((addresses[0], port), timeout=min(10, remaining))
        conn.sock = raw
        if p.scheme == 'https':
            conn.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=p.hostname)
        transport = conn.sock
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise URLLoadError('The website took too long to respond.')
        def abort():
            try:
                transport.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        timer = threading.Timer(remaining, abort)
        timer.daemon = True
        timer.start()
        conn.request('GET', urlunsplit(('', '', p.path or '/', p.query, '')),
                     headers={'User-Agent': 'HowardKnowledgeBot/3.0',
                              'Accept': 'text/html, application/xhtml+xml, text/plain',
                              'Accept-Encoding': 'identity', 'Connection': 'close'})
        response = conn.getresponse()
        headers = {k.lower(): v for k, v in response.getheaders()}
        if response.status in (301, 302, 303, 307, 308):
            return response.status, headers, b''
        if response.status != 200:
            raise URLLoadError(f'The website returned HTTP {response.status}. It may restrict automated access.')
        encoding = headers.get('content-encoding', 'identity').lower()
        if encoding not in ('identity', 'gzip', 'deflate'):
            raise URLLoadError('The website returned an unsupported compressed response.')
        kind = headers.get('content-type', '').split(';')[0].strip().lower()
        if kind not in ('text/html', 'application/xhtml+xml', 'text/plain'):
            raise URLLoadError('This URL must return a webpage or plain text. Downloads, PDFs, and media are not supported in Version 3.')
        try:
            length = int(headers.get('content-length', '0'))
        except ValueError:
            raise URLLoadError('The website returned an invalid response size.') from None
        if length > MAX_PAGE_BYTES:
            raise URLLoadError('The webpage exceeds the 2 MB download limit.')
        data = bytearray()
        while not response.isclosed():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise URLLoadError('The website took too long to respond.')
            transport.settimeout(min(10, remaining))
            part = response.read1(min(65536, MAX_PAGE_BYTES + 1 - len(data)))
            if not part:
                break
            data.extend(part)
            if len(data) > MAX_PAGE_BYTES:
                raise URLLoadError('The webpage exceeds the 2 MB download limit.')
        if (length and len(data) < length) or time.monotonic() >= deadline:
            raise URLLoadError('The website response was incomplete or timed out.')
        return response.status, headers, decode_body(bytes(data), encoding)
    finally:
        if timer:
            timer.cancel()
        if response is not None:
            response.close()
        conn.close()
        if raw:
            raw.close()


def decode_body(data, encoding):
    if encoding == 'identity':
        return data
    try:
        decoder = zlib.decompressobj(31 if encoding == 'gzip' else zlib.MAX_WBITS)
        decoded = decoder.decompress(data, MAX_PAGE_BYTES + 1)
        if len(decoded) > MAX_PAGE_BYTES or decoder.unconsumed_tail:
            raise URLLoadError('The expanded webpage exceeds the 2 MB limit.')
        if not decoder.eof or decoder.unused_data:
            raise URLLoadError('The website returned an incomplete or unsupported compressed page.')
        return decoded
    except zlib.error:
        raise URLLoadError('The website returned an invalid compressed page.') from None


def extract_page(data, content_type, requested_url, final_url):
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    metadata = {'source': final_url, 'fetched_at': timestamp,
                'content_hash': sha256(final_url.encode() + data).hexdigest()}
    if content_type.split(';')[0].strip().lower() == 'text/plain':
        text = data.decode('utf-8', errors='replace').strip()
        docs = [Document(page_content=text, metadata=metadata)] if text else []
        title = urlsplit(final_url).hostname
    else:
        try:
            charset = re.search(r'charset=[\"\']?([\w-]+)', content_type, re.I)
            parser = html.HTMLParser(no_network=True, recover=True, encoding=charset.group(1) if charset else None)
            root = html.fromstring(data, parser=parser)
        except (etree.ParserError, ValueError):
            raise URLLoadError('The webpage could not be parsed.') from None
        title = ' '.join(root.xpath('//title/text()')).strip()[:300] or urlsplit(final_url).hostname
        for node in root.xpath('//script|//style|//noscript|//template|//nav|//header|//footer|//aside|//form|//svg|//*[@hidden]|//*[@aria-hidden="true"]'):
            node.drop_tree()
        mains = root.xpath('//main|//article|//*[@role="main"]')
        body = max(mains, key=lambda n: len(n.text_content())) if mains else root
        docs, buffer, heading, table_id = [], [], title, 0
        def flush():
            if buffer:
                docs.append(Document(page_content='\n\n'.join(buffer),
                    metadata={**metadata, 'section_title': heading}))
                buffer.clear()
        for node in body.iter():
            if not isinstance(node.tag, str):
                continue
            tag = node.tag.lower()
            if any(a.tag in ('table', 'p', 'li', 'pre', 'blockquote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6') for a in node.iterancestors() if a is not body):
                continue
            text = ' '.join(node.text_content().split())
            if not text:
                continue
            if tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                flush()
                heading = text[:500]
                buffer.append(heading)
            elif tag == 'table':
                flush()
                rows = [' | '.join(' '.join(cell.text_content().split()).replace('|', '\\|')
                                  for cell in row.xpath('./th|./td')) for row in node.xpath('.//tr')]
                if rows:
                    table_id += 1
                    docs.append(Document(page_content='\n'.join(rows), metadata={**metadata,
                        'section_title': heading, 'block_type': 'table', 'table_id': str(table_id)}))
            elif tag in ('p', 'li', 'pre', 'blockquote', 'dt', 'dd'):
                buffer.append(text)
        flush()
        if not docs:
            text = ' '.join(body.text_content().split())
            if text:
                docs.append(Document(page_content=text, metadata={**metadata, 'section_title': title}))
    count = sum(len(d.page_content) for d in docs)
    if count < 80:
        raise URLLoadError('No substantial readable text was found. The page may need JavaScript, a login, or a different article URL.')
    if count > MAX_PAGE_CHARS:
        raise URLLoadError('The webpage exceeds 200,000 extracted characters. Use a more specific page URL.')
    return Page(requested_url, final_url, title, timestamp, docs)


def load_url(value):
    url = requested = normalize_url(value)
    deadline = time.monotonic() + PAGE_TIMEOUT
    visited = set()
    try:
        for _ in range(MAX_REDIRECTS + 1):
            if url in visited:
                raise URLLoadError('The website has a redirect loop.')
            visited.add(url)
            status, headers, data = _request(url, deadline)
            if status in (301, 302, 303, 307, 308):
                location = headers.get('location')
                if not location:
                    raise URLLoadError('The website returned an invalid redirect.')
                url = normalize_url(urljoin(url, location))
                continue
            return extract_page(data, headers.get('content-type', ''), requested, url)
        raise URLLoadError('The website redirected too many times.')
    except URLLoadError:
        raise
    except (OSError, http.client.HTTPException, UnicodeError, ValueError):
        raise URLLoadError('The webpage could not be fetched securely (connection, certificate, or timeout error).') from None
