import streamlit as st
from src.helper import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    get_conversational_chain
)

# =========================
# CHAT FUNCTION
# =========================
def user_input(user_question):

    if st.session_state.conversation is None:
        st.warning("Please upload and process PDF first!")
        return

    response = st.session_state.conversation({'question': user_question})

    st.session_state.chatHistory = response['chat_history']

    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("🧑 User: ", message.content)
        else:
            st.write("🤖 Bot: ", message.content)


# =========================
# MAIN APP
# =========================
def main():

    st.set_page_config(page_title="Meow Retrieval")
    st.header("🐱 Meow-Retrieval-System")

    user_question = st.text_input("Ask a Question from the PDF Files")

    # SESSION STATE INIT
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []


    # HANDLE QUESTION
    if user_question:
        user_input(user_question)


    # SIDEBAR
    with st.sidebar:
        st.title("Menu 📄")

        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):

            if not pdf_docs:
                st.error("Please upload at least one PDF!")
                return

            with st.spinner("Processing..."):

                # STEP 1: Extract text
                raw_text = get_pdf_text(pdf_docs)

                # STEP 2: Chunk text
                text_chunks = get_text_chunks(raw_text)

                # STEP 3: Vector store
                vector_store = get_vector_store(text_chunks)

                # STEP 4: Create conversation chain
                st.session_state.conversation = get_conversational_chain(vector_store)

                st.success("Processing Done ✅")


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    main()