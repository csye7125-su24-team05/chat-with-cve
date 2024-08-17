import streamlit as st
from main import conversational_rag_chain
from time import sleep

st.title("CVE-GPT")

def _get_session():
    from streamlit.runtime import get_instance
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    runtime = get_instance()
    session_id = get_script_run_ctx().session_id
    session_info = runtime._session_mgr.get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return session_info.session

# client = qa_chain

if "openai_model" not in st.session_state:
    st.session_state["ollama_model"] = "gemma2:2b"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        res = conversational_rag_chain.stream(
            {"input": prompt},
            config={
                "configurable": {"session_id": "abc123"}
            },
        )

        def ans_gen():
            for item in res:
                if "answer" in item:
                    yield item["answer"]
                    sleep(0.05)
        response = st.write_stream(ans_gen())
        
    st.session_state.messages.append({"role": "assistant", "content": response})