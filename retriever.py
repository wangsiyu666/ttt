import abc
from pydantic import BaseModel, Field

class Chunk:
    content: str
    similarity: float

    def __init__(self, content: str, similarity: float):
        self.content = content
        self.similarity = similarity


class Document:
    id: str
    url: str | None = None
    title: str | None = None
    chunks: list[Chunk] = []

    def __init__(self, id: str, url: str | None = None, title: str | None = None, chunks: list[Chunk] = []):
        self.id = id
        self.url = url
        self.title = title
        self.chunks = chunks

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "content": "\n\n".join([chunk.content for chunk in self.chunks]),
        }
        if self.url:
            d["url"] = self.url
        if self.title:
            d["title"] = self.title
        return d


class Resource(BaseModel):

    uri: str = Field(..., description="URI of the resource")
    title: str = Field(..., description="Title of the resource")
    description: str = Field(..., description="Description of the resource")


class Retriever(abc.ABC):
    @abc.abstractmethod
    def list_resources(self, query: str | None = None) -> list[Resource]:
        pass
    @abc.abstractmethod
    def query_relevant_documents(self, query: str, resources: list[Resource] = []) -> list[Document]:
        pass















