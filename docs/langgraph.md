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
| `retrieve` | Embed the question, query Pinecone top-k, attach chunks + metadata to state | built (`app/nodes.py`) |
| `grade_documents` | LLM call judging whether retrieved chunks are sufficient to answer. Sets the branch. | built |
| `rewrite_query` | LLM rewrites the question for a retry when grading says "bad". Increments loop counter. | built |
| `generate_answer` | LLM answers strictly from retrieved chunks, with citations (chunk id + source file). | built |
| `answer_not_found` | Deterministic terminal node — no chunks were good enough after max loops. | built |

Graph wiring lives in `app/graph.py`. `run_ask(question)` invokes the full
graph and returns the final state (answer + citations).

## Loop limit

`state.loop_count` starts at 0, incremented in `rewrite_query`. Once it hits
`MAX_RETRIEVAL_LOOPS` (default 2, see `.env.example`), the graph routes to
`answer_not_found` instead of retrying again — this is the safeguard against
infinite spinning.
