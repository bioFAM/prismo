import os
import re
from html.parser import HTMLParser
from urllib import request

from platformdirs import user_cache_dir

from .._core import FeatureSets

_url = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release"
_pattern = re.compile(r"(\w.+)\.(v\d.+)\.(entrez|symbols)\.gmt")


class _MSIGDBDirectoryParser(HTMLParser):
    def __init__(self, img_alt):
        super().__init__()
        self._img_alt = img_alt
        self._have_img = False
        self._have_link = False

        self.contents = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            for attrname, attrvalue in attrs:
                if attrname == "alt" and attrvalue == self._img_alt:
                    self._have_img = True
                    break
        elif self._have_img and tag == "a":
            self._have_link = True

    def handle_data(self, data):
        if self._have_img and self._have_link:
            self.contents.append(data)
            self._have_img = self._have_link = False


def list_msigdb_versions() -> list[str]:
    """List available versions of the MSIGDB molecular signature  database."""
    parser = _MSIGDBDirectoryParser("[DIR]")
    with request.urlopen(_url) as resp:
        encoding = resp.headers.get_content_charset("utf-8")
        parser.feed(resp.read().decode(encoding))
    parser.close()
    return [v.rstrip("/") for v in parser.contents]


def list_msigdb_categories(dbver: str = "2024.1.Hs") -> list[str]:
    """List available MSIGDB categories for the given database version.

    Args:
        dbver: Version of the MSIGDB molecular signature database. Can be obtained by
            calling :func:`list_msigdb_versions`.
    """
    parser = _MSIGDBDirectoryParser("[TXT]")
    with request.urlopen(f"{_url}/{dbver}") as resp:
        encoding = resp.headers.get_content_charset("utf-8")
        parser.feed(resp.read().decode(encoding))
    parser.close()
    return sorted({match.groups()[0] for fname in parser.contents if (match := _pattern.match(fname)) is not None})


def get_msigdb_features(category: str = "h.all", dbver: str = "2024.1.Hs", entrez: bool = False) -> FeatureSets:
    """Get gene sets from the MSIGDB molecular signatures database.

    Args:
        category: An MSIGDB category. Can be obtained by calling :func:`list_msigdb_categories`.
        dbver: MSIGDB version. Can be obtained by calling :func:`list_msigdb_versions`.
        entrez: Whether to use Entrez identifiers or common gene names.
    """
    pkgname = __name__[: __name__.find(".")]
    cachedir = user_cache_dir(pkgname)
    os.makedirs(cachedir, exist_ok=True)

    ident = "entrez" if entrez else "symbols"
    name = f"{category}.v{dbver}.{ident}"
    fname = f"{name}.gmt"
    fpath = os.path.join(cachedir, fname)

    if not os.path.isfile(fpath):
        with request.urlopen(f"{_url}/{dbver}/{fname}") as resp, open(fpath, "wb") as cache:
            cache.write(resp.read())
    return FeatureSets.from_gmt(fpath, name=name)
