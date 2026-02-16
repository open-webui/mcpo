from fastapi import FastAPI


def create_app(*,
               host: str = "127.0.0.1",
               port: int = 8000,
               api_key: str | None = None,
               cors_allow_origins=None,
               **kwargs) -> FastAPI:
    """
    Build the FastAPI application using existing construction logic in mcpo.main.
    This introduces an app factory without changing public behavior.
    """
    if cors_allow_origins is None:
        cors_allow_origins = ["*"]

    # Delegate to mcpo.main to construct the app using the current parameters.
    from mcpo.main import build_main_app
    return build_main_app(
        host=host,
        port=port,
        api_key=api_key or "",
        cors_allow_origins=cors_allow_origins,
        **kwargs,
    )
