# LangGraph flow

> This doc will be filled in as each node is built. Skeleton below.

## Diagram (planned)

```
retrieve --> grade_documents --[good]--> generate_answer --> END
                |
                --[bad]--> rewrite_query --> retrieve   (looped, max N times)
                                                |
                                          [loops exhausted]
                                                |
                                                v
                                        answer_not_found --> END
```

## Nodes

| Node | Purpose | Status |
|---|---|---|
| `retrieve` | Embed the question, query Pinecone top-k, attach chunks + metadata to state | not yet built |
| `grade_documents` | LLM call judging whether retrieved chunks are sufficient to answer. Sets the branch. | not yet built |
| `rewrite_query` | LLM rewrites the question for a retry when grading says "bad". Increments loop counter. | not yet built |
| `generate_answer` | LLM answers strictly from retrieved chunks, with citations (chunk id + source file). | not yet built |
| `answer_not_found` | Deterministic terminal node — no chunks were good enough after max loops. | not yet built |

## Loop limit

`state.loop_count` starts at 0, incremented in `rewrite_query`. Once it hits
`MAX_RETRIEVAL_LOOPS` (default 2, see `.env.example`), the graph routes to
`answer_not_found` instead of retrying again — this is the safeguard against
infinite spinning.
