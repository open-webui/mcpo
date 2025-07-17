from pydantic import AnyUrl


def convert_to_route(any_url: AnyUrl):
    """Return the pathname portion of a whatever:// URI."""
    if any_url.path:
        return f"/{any_url.scheme}/{any_url.host}{any_url.path}"
    else:
        return f"/{any_url.scheme}/{any_url.host}"