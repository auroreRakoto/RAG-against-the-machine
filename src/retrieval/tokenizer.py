import re
from collections.abc import Iterator

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _normalize_token(token: str) -> str:
    if len(token) <= 3:
        return token

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"

    if token.endswith("es") and len(token) > 4:
        return token[:-2]

    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]

    return token


def tokenize(text: str, normalize: bool = True) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    if normalize:
        return [_normalize_token(token) for token in tokens]
    return tokens


def iter_tokens(text: str) -> Iterator[str]:
    for token in tokenize(text):
        yield token
