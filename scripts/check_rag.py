"""문서 읽기·출처 검증·Streamlit 화면 테스트. --live는 실제 OpenAI API를 호출합니다."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest


def main() -> None:
    docs, report = app.load_documents(app.file_signature())
    chunks = app.split_documents(docs)
    for item in report:
        print(item)
    print(f"Read {len(docs)} nonempty pages; {len(chunks)} chunks")
    assert len(report) == len(app.file_signature())
    assert all(chunk.metadata.get("source") for chunk in chunks)
    source = Document(page_content="국내출장의 여비는 규정에 따라 지급합니다.", metadata={"source": "test.pdf", "page": 1})
    valid = app.GroundedAnswer(supported=True, claims=[app.Claim(
        text="국내출장 여비는 규정에 따라 지급합니다.",
        evidence=[app.Evidence(source_id=1, quote=source.page_content)],
    )])
    assert app.validate_answer(valid, [source])["sources"][0]["file"] == "test.pdf"
    for evidence in [app.Evidence(source_id=99, quote=source.page_content), app.Evidence(source_id=1, quote="문서에 존재하지 않는 문장을 인용했습니다.")]:
        invalid = app.GroundedAnswer(supported=True, claims=[app.Claim(text="가짜 답변", evidence=[evidence])])
        assert app.validate_answer(invalid, [source])["text"] == app.UNKNOWN
    assert app.validate_answer(app.GroundedAnswer(supported=False, claims=[]), [source])["text"] == app.UNKNOWN

    # 화면은 Streamlit의 실제 앱 실행 도구로 검증하고, 외부 통신만 대체합니다.
    at = AppTest.from_file(str(app.ROOT / "app.py"), default_timeout=90)
    with patch("dotenv.dotenv_values", return_value={"OPENAI_API_KEY": ""}):
        at.run()
        assert not at.exception, at.exception
        assert at.info
    with patch("dotenv.dotenv_values", return_value={"OPENAI_API_KEY": "test-key"}):
        at.run()
        assert not at.exception, at.exception
        assert len(at.chat_input) == 1
    if "--live" in sys.argv:
        key = app.read_api_key()
        if not key:
            raise RuntimeError("Live test requires OPENAI_API_KEY in .env")
        store = app.build_store(docs, key)
        for question, should_answer in [
            ("근무지 내 출장에서 원칙적으로 숙박비를 지급할 수 있나요?", True),
            ("화성에서 키우는 토마토의 적정 비료 배합을 알려주세요.", False),
        ]:
            result = app.answer_question(question, store, key)
            print("QUESTION:", question)
            print("ANSWER:", result["text"])
            print("CITATIONS:", len(result["sources"]))
            assert bool(result["sources"]) == should_answer, result
        # 실제 임베딩/검색/생성 함수를 연결해 채팅 입력부터 출처 렌더링까지 실행합니다.
        with patch("dotenv.dotenv_values", return_value={"OPENAI_API_KEY": key}):
            at.run()
            at.session_state["store"] = store
            at.chat_input[0].set_value("근무지 내 출장에서 원칙적으로 숙박비를 지급할 수 있나요?").run(timeout=120)
            assert not at.exception, at.exception
            assert not at.error, at.error
            assert at.session_state["messages"][-1]["role"] == "assistant"
            assert at.session_state["messages"][-1]["result"]["sources"]
    print("PASS: document loading, citation validation, and Streamlit execution")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Streamlit 테스트 도구의 모듈 단위 임시 폴더를 명시적으로 정리합니다.
        # Python 종료 시 ResourceWarning이 생기는 것을 방지합니다.
        from streamlit.testing.v1.app_test import TMP_DIR
        TMP_DIR.cleanup()
