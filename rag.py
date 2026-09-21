"""LangGraph RAG pipeline (max compatibility, ASCII-only)."""

from typing import Annotated, List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from llms import chat_model
from retriever import DocumentRetriever

# Shared retriever instance
retriever = DocumentRetriever()

# Graph runtime config (thread_id can be any stable value)
config = {"configurable": {"thread_id": "abc123"}}


class State(TypedDict, total=False):
    # LangGraph message accumulator
    messages: Annotated[List[BaseMessage], add_messages]
    # Retrieved docs
    docs: List[Document]
    # Final answer text
    answer: str


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful corporate documentation assistant. "
            "Use the provided context to answer the user. "
            "If the context is insufficient, say what is missing and ask a clarifying question.",
        ),
        ("human", "{question}"),
        ("system", "Context:\n{context}"),
    ]
)


def retrieve(state: State) -> State:
    question = state["messages"][-1].content
    docs = retriever.invoke(question)
    return {"docs": docs}


def generate(state: State) -> State:
    question = state["messages"][-1].content
    docs = state.get("docs", [])
    context = "\n\n".join([d.page_content for d in docs])

    chain = PROMPT | chat_model
    response = chain.invoke({"question": question, "context": context})
    return {"answer": response.content}


def finalize(state: State) -> State:
    answer = state.get("answer", "I could not generate an answer. Please try again.")
    return {"messages": [AIMessage(content=answer)]}


# Build graph using widely supported API calls
builder = StateGraph(State)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("finalize", finalize)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "finalize")
builder.add_edge("finalize", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
