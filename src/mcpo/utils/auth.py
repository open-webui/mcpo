from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import base64

from passlib.context import CryptContext
from datetime import UTC, datetime, timedelta

import jwt
from typing import Optional, Union, List, Dict


ALGORITHM = "HS256"

bearer_security = HTTPBearer(auto_error=False)


def get_verify_api_key(api_key: str):
    async def verify_api_key(
        authorization: HTTPAuthorizationCredentials = Depends(bearer_security),
    ):
        if not authorization or not authorization.credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = authorization.credentials
        if token != api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key",
            )

    return verify_api_key


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces Basic or Bearer token authentication for all requests.
    """

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        # Skip authentication for OPTIONS requests
        if request.method == "OPTIONS":
            return await call_next(request)

        # Get authorization header
        authorization = request.headers.get("Authorization")

        # Verify API key
        try:
            # Use the same function that the dependency uses
            if not authorization:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid Authorization header"},
                    headers={"WWW-Authenticate": "Bearer, Basic"},
                )

            # Handle Bearer token auth
            if authorization.startswith("Bearer "):
                token = authorization[7:]  # Remove "Bearer " prefix
                if token != self.api_key:
                    return JSONResponse(
                        status_code=403, content={"detail": "Invalid API key"}
                    )
            # Handle Basic auth
            elif authorization.startswith("Basic "):
                # Decode the base64 credentials
                credentials = authorization[6:]  # Remove "Basic " prefix
                try:
                    decoded = base64.b64decode(credentials).decode("utf-8")
                    # Basic auth format is username:password
                    username, password = decoded.split(":", 1)
                    # Any username is allowed, but password must match api_key
                    if password != self.api_key:
                        return JSONResponse(
                            status_code=403, content={"detail": "Invalid credentials"}
                        )
                except Exception:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid Basic Authentication format"},
                        headers={"WWW-Authenticate": "Bearer, Basic"},
                    )
            else:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Unsupported authorization method"},
                    headers={"WWW-Authenticate": "Bearer, Basic"},
                )

            return await call_next(request)
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": str(e)})


# def create_token(data: dict, expires_delta: Union[timedelta, None] = None) -> str:
#     payload = data.copy()

#     if expires_delta:
#         expire = datetime.now(UTC) + expires_delta
#         payload.update({"exp": expire})

#     encoded_jwt = jwt.encode(payload, SESSION_SECRET, algorithm=ALGORITHM)
#     return encoded_jwt


# def decode_token(token: str) -> Optional[dict]:
#     try:
#         decoded = jwt.decode(token, SESSION_SECRET, algorithms=[ALGORITHM])
#         return decoded
#     except Exception:
#         return None
