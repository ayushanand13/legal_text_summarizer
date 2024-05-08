import nltk
import requests
import streamlit as st
import validators
from PIL import Image
from rouge import Rouge
from transformers import AutoTokenizer, pipeline

# local modules
from extractive_summarizer.model_processors import Summarizer
from utils import clean_text, fetch_article_text, preprocess_text_for_abstractive_summarization, read_text_from_file

if __name__ == "__main__":
    # ---------------------------------
    # Main Application
    # ---------------------------------
    col1, col2 = st.columns([1, 3])  # Adjust the ratio based on your layout preference

    # Use the first column to display the image
    with col1:
        image_url = "https://upload.wikimedia.org/wikipedia/en/5/52/Indian_Institute_of_Technology%2C_Patna.svg"
        st.image(image_url, use_column_width=True)

    # You can use the second column for other parts of your app
    with col2:
        st.title("Artificial Intelligence for Legal Assistance Using Summarization")
    # ---

    # Styling the header
    st.markdown(
        """
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Display title and project details
    st.title("Legal Text Summarizer 📝")

    # Personal and Supervisor information
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="big-font">Name: Ayush Anand</p>', unsafe_allow_html=True)
        st.markdown("**Roll No:** 2211ai18")

    with col2:
        st.markdown('<p class="big-font">Supervisor: Dr. Pradeepika Verma</p>', unsafe_allow_html=True)
        st.write(" ")

    # Use additional Markdown or write functions for content
    st.header("Overview")
    st.write(
        "This project utilizes state-of-the-art AI techniques to summarize legal texts, \
            aiming to assist students and professionals in the legal field."
    )

    # Additional content can be added here
    st.subheader("Project Description")
    st.write(
        "The aim of this project is to develop a robust text summarizer that can efficiently \
            process and condense extensive legal documents into concise summaries. The application \
            leverages techniques like BERT and BART for extractive and abstractive summarization, \
            respectively, ensuring the preservation of critical legal information while providing \
            clear, accessible summaries."
    )

    # Footer
    st.write("Thank you for visiting my presentation!")
    # ----
    summarize_type = st.sidebar.selectbox("Summarization type", options=["Extractive", "Abstractive"])

    st.markdown(
        "Enter a text or a url to get a concise summary of the article while conserving the overall meaning. This app supports text in the following formats:"
    )
    st.markdown(
        """- Raw text in text box 
- URL of article/news to be summarized 
- .txt, .pdf, .docx file formats"""
    )
    st.markdown(
        """This app supports two type of summarization:

1. **Extractive Summarization**: The extractive approach involves picking up the most important phrases and lines from the documents. It then combines all the important lines to create the summary. So, in this case, every line and word of the summary actually belongs to the original document which is summarized.
2. **Abstractive Summarization**: The abstractive approach involves rephrasing the complete document while capturing the complete meaning of the document. This type of summarization provides more human-like summary"""
    )
    st.markdown("---")
    # ---------------------------
    # SETUP & Constants
    nltk.download("punkt")
    abs_tokenizer_name = "facebook/bart-large-cnn"
    abs_model_name = "facebook/bart-large-cnn"
    abs_tokenizer = AutoTokenizer.from_pretrained(abs_tokenizer_name)
    abs_max_length = 90
    abs_min_length = 30
    # ---------------------------

    inp_text = st.text_input("Enter text or a url here")
    st.markdown(
        "<h3 style='text-align: center; color: green;'>OR</h3>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload a .txt, .pdf, .docx file for summarization")

    is_url = validators.url(inp_text)
    if is_url:
        # complete text, chunks to summarize (list of sentences for long docs)
        text, cleaned_txt = fetch_article_text(url=inp_text)
    elif uploaded_file:
        cleaned_txt = read_text_from_file(uploaded_file)
        cleaned_txt = clean_text(cleaned_txt)
    else:
        cleaned_txt = clean_text(inp_text)

    # view summarized text (expander)
    with st.expander("View input text"):
        if is_url:
            st.write(cleaned_txt[0])
        else:
            st.write(cleaned_txt)
    summarize = st.button("Summarize")

    # called on toggle button [summarize]
    if summarize:
        if summarize_type == "Extractive":
            if is_url:
                text_to_summarize = " ".join([txt for txt in cleaned_txt])
            else:
                text_to_summarize = cleaned_txt
            # extractive summarizer

            with st.spinner(text="Creating extractive summary. This might take a few seconds ..."):
                ext_model = Summarizer()
                summarized_text = ext_model(text_to_summarize, num_sentences=5)

        elif summarize_type == "Abstractive":
            with st.spinner(text="Creating abstractive summary. This might take a few seconds ..."):
                text_to_summarize = cleaned_txt
                abs_summarizer = pipeline("summarization", model=abs_model_name, tokenizer=abs_tokenizer_name)

                if is_url is False:
                    # list of chunks
                    text_to_summarize = preprocess_text_for_abstractive_summarization(
                        tokenizer=abs_tokenizer, text=cleaned_txt
                    )

                tmp_sum = abs_summarizer(
                    text_to_summarize,
                    max_length=abs_max_length,
                    min_length=abs_min_length,
                    do_sample=False,
                )

                summarized_text = " ".join([summ["summary_text"] for summ in tmp_sum])

        # final summarized output
        st.subheader("Summarized text")
        st.info(summarized_text)

        # st.subheader("Rogue Scores")
        # rouge_sc = Rouge()
        # ground_truth = cleaned_txt[0] if is_url else cleaned_txt
        # score = rouge_sc.get_scores(summarized_text, ground_truth, avg=True)
        # st.code(score)
