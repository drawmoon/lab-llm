langchain 内部使用 pydantic 模块来处理数据结构，pydantic 可以非常方便的将数据转换为结构化数据。

#### pydantic 解决无法序列化协议的问题

**背景**
TODO

第一种是可以接收 CohereRerank 提供的 rerank 服务对象：
```python
from langchain_cohere.rerank import CohereRerank

rerank = CohereRerank(
    base_url="http://rerank", model="rerank-english-v2.0", cohere_api_key="abc"
)
```

第二种是接收自己封装的 rerank 服务对象：
```python
class LlmBaseRerank(BaseDocumentCompressor):
    def rerank(
        self,
        documents: Sequence[Union[str, Document, Dict[str, Any]]],
        query: str,
        *,
        rank_fields: Optional[Sequence[str]] = None,
        top_n: Optional[int] = -1,
    ) -> List[RankResult]:
        ...

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        ...
```

我新增了一个基于 rerank 服务实现对问题归类的公共组件：
```python
class RerankClassify(RunnableSerializable[str, str]):
    # 忽略对 ranker 对象的验证
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ranker: Rerankable
    categories: List[Classification]

    # 增加一个验证器，用于验证 ranker 对象是否包含 rerank 方法
    @field_validator("ranker")
    def validate_ranker(cls, data: Any):
        if not isinstance(data, Rerankable):
            raise ValueError("ranker must have a callable 'rerank' method")
        return data

    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        ...
```

增加一个 Rerankable 协议类用于约束传入的 ranker 对象包含 rerank 方法，用于兼容不同的 ranker 服务对象。

Rerankable源代码：
```python
from typing import runtime_checkable

@runtime_checkable
class Rerankable(Protocol):
    def rerank(
        self,
        documents: Sequence[Union[str, Document, Dict[str, Any]]],
        query: str,
        *,
        rank_fields: Optional[Sequence[str]] = None,
        top_n: Optional[int] = -1,
    ) -> Any: ...
```
