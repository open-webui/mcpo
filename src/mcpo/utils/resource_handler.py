from typing import Dict
from uuid import uuid4
import base64
from fastapi import APIRouter, FastAPI, HTTPException

# Simulated in-memory resource registry for generated images
GEN_IMAGE_RESOURCES: Dict[str, bytes] = {}

# Generate a resource URI for base64 image data
def register_base64_image(mime_type: str, data: str) -> str:
    decoded = base64.b64decode(data)
    uri = f"image://local/{uuid4().hex}"
    GEN_IMAGE_RESOURCES[uri] = decoded
    return uri

# Serve resource APIs
def setup_resource_routes(app: FastAPI):
    router = APIRouter(tags=["Resources"])

    @router.get("/list")
    async def list_resources():
        return {
            "resources": [
                {
                    "uri": uri,
                    "name": f"Image {i+1}",
                    "mimeType": "image/png"
                }
                for i, uri in enumerate(GEN_IMAGE_RESOURCES)
            ]
        }

    @router.post("/read")
    async def read_resource(uri: str):
        if uri not in GEN_IMAGE_RESOURCES:
            raise HTTPException(status_code=404, detail="Resource not found")
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "image/png",
                    "blob": base64.b64encode(GEN_IMAGE_RESOURCES[uri]).decode('utf-8')
                }
            ]
        }

    app.include_router(router, prefix="/resources")
