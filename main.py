from dotenv import load_dotenv
from utils.audio import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items,extract_key_decisions,extract_conclusions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


def run_pipeline(source: str, language :str = "english") -> dict:
    print("starting AI Video Assitant")

    chunks = process_input(source)

    transcript = transcribe_all(chunks,language = language)
    print(f"raw transcription(first 300 chracters): {transcript[:300]}")

    title = generate_title(transcript)

    summary = summarize(transcript)

    action_item = extract_action_items(transcript)

    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain
    }

if __name__ == "__main__":
    #CLI entry point
    source = input("Enter Youtube URL or local file path:").strip()
    language = input("Language (english/hinglish):").strip() or "english"
    results = run_pipeline(source,language)

    print("\n" + "=" *60)
    print(f"title: {results['title']}")
    print(f"\n summary: {results['summary']}")
    print(f"\n action_items: {results['action_items']}")
    print(f"\n key_decisions: {results['key_decisions']}")
    print(f"\n open_questions: {results['open_questions']}")
    print("=" *60)

    #phase 2 - Chat with your meeting voa RAG system
    print("\n chat with your meeting voa RAG system")

    rag_chain = result["rag_chain"]

    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit","quit","q"]:
            print("Exiting chat. Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n Assistana : {answer}\n")
