"""
Builds the LangGraph StateGraph for the ask flow. See docs/langgraph.md for
the diagram.

    retrieve -> grade_documents --[good]--> generate_answer -> END
                     |
                     --[bad, loops left]--> rewrite_query -> retrieve (loop)
                     |
                     --[bad, loops exhausted]--> answer_not_found -> END
"""

from langgraph.graph import StateGraph, START, END

from app.nodes import (
    GraphState,
    retrieve,
    grade_documents,
    rewrite_query,
    generate_answer,
    answer_not_found,
    route_after_grading,
)


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("answer_not_found", answer_not_found)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "answer_not_found": "answer_not_found",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate_answer", END)
    graph.add_edge("answer_not_found", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_ask(question: str) -> dict:
    """Runs the full graph for a question. Returns the final state dict."""
    graph = get_graph()
    initial_state = {
        "question": question,
        "original_question": question,
        "chunks": [],
        "loop_count": 0,
        "grading_sufficient": False,
        "answer": "",
        "citations": [],
    }
    return graph.invoke(initial_state)
