from typing import TypedDict
from ddgs import DDGS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def call_agent(query: str) -> str:
    response = client.responses.create(
            model="gpt-4o-mini",
            input=query,
        )
    return response.output_text


class ResearchState(TypedDict):
    question: str
    sources: list[dict]
    summary: str
    verified: bool
    report: str


def search_web(query: str) -> list[dict]:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        return list(results)


def research(state: ResearchState) -> dict:
    print("Researching...")

    results = search_web(state["question"])

    return {"sources": results}


def summarize(state: ResearchState) -> dict:
    print("Summarizing...")

    source_text = "\n".join(
        f"Title: {source.get('title', '')}\n"
        f"URL: {source.get('href', '')}\n"
        f"Content: {source.get('body', '')}"
        for source in state["sources"]
    )

    summary = call_agent(
        f"""
        Answer the research question using the sources below.
        Cite sources using their URLs.
        If the sources do not provide enough evidence, say so.

        Question: {state['question']}

        Sources:
        {source_text}
        """
    )

    return {"summary": summary}


def fact_check(state: ResearchState) -> dict:
    print("Fact-checking...")

    source_text = "\n".join(
        f"URL: {source.get('href', '')}\n"
        f"Content: {source.get('body', '')}"
        for source in state["sources"]
    )

    result = call_agent(
        f"""
        Evaluate whether the summary is supported by the provided sources.
        Return exactly PASS if the main claims are supported.
        Otherwise, return FAIL and briefly explain why.

        Question: {state['question']}
        Summary: {state['summary']}
        Sources:
        {source_text}
        """
    )

    verified = result.strip().upper().startswith("PASS")

    return {"verified": verified}


def write_report(state: ResearchState) -> dict:
    print("Writing report...")

    report = call_agent(
        f"""
        Write a concise research report answering this question.

        Question: {state['question']}
        Summary: {state['summary']}
        Verified: {state['verified']}

        If verification failed, clearly explain that the findings
        could not be fully verified.
        """
    )

    return {"report": report}


def route_after_check(state: ResearchState) -> str:
    if state["verified"]:
        return "write_report"

    return "failed"


def run_workflow(question: str) -> ResearchState:
    state: ResearchState = {
        "question": question,
        "sources": [],
        "summary": "",
        "verified": False,
        "report": "",
    }

    state.update(research(state))
    state.update(summarize(state))
    state.update(fact_check(state))

    next_step = route_after_check(state)

    if next_step == "write_report":
        state.update(write_report(state))
    else:
        state["report"] = (
            "The research could not be verified.\n\n"
            + state["summary"]
        )
        print("Verification failed.")

    return state


result = run_workflow("How does solar power work?")

print("\nFinal report:")
print(result["report"])

