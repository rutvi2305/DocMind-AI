"""
summarizer.py
-------------
Generates summaries and extracts structured insights from PDF documents
using LangChain's map-reduce summarization chain with Groq (FREE).
"""

from typing import List, Optional

from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_groq import ChatGroq


# ── Prompt Templates ──────────────────────────────────────────────────────────

MAP_PROMPT_TEMPLATE = """You are an expert document analyst.
Analyze the following section of a document and extract:
1. The main topics covered
2. Key facts, figures, or statistics mentioned
3. Any conclusions or recommendations

Document section:
{text}

Provide a concise analysis (3-5 sentences):"""

REDUCE_PROMPT_TEMPLATE = """You are an expert at synthesizing information from multiple document sections.
Below are analyses of different sections of the same document.

Analyses:
{text}

Create a comprehensive summary that:
1. Covers the document's main purpose and scope
2. Highlights the most important findings or arguments
3. Lists 5-7 key insights as bullet points
4. Ends with a one-sentence conclusion

Format your response as:
SUMMARY:
[2-3 paragraph summary]

KEY INSIGHTS:
• [insight 1]
• [insight 2]
• [insight 3]
• [insight 4]
• [insight 5]

CONCLUSION:
[one sentence]"""

INSIGHT_EXTRACTION_TEMPLATE = """You are a business analyst extracting structured insights from a document.

Document content:
{text}

Extract and return a JSON-like structured analysis with these fields:
- main_topic: What is this document primarily about?
- document_type: (e.g., research paper, business report, manual, article)
- key_themes: List of 3-5 major themes
- important_facts: List of 5 specific facts, numbers, or data points
- action_items: List of any recommendations or next steps mentioned
- sentiment: Overall tone (positive/neutral/negative/mixed)
- target_audience: Who is this document written for?

Respond clearly with each field labeled."""


class DocumentSummarizer:
    """
    Summarizes documents and extracts key insights using Groq (FREE).
    Uses map-reduce strategy for long documents to stay within context limits.
    Groq supports: llama-3.1-8b-instant, llama-3.1-70b-versatile, mixtral-8x7b-32768
    """

    def __init__(self, model: str = "llama-3.1-8b-instant", temperature: float = 0.3):
        self.llm = ChatGroq(model=model, temperature=temperature)

        self.map_prompt = PromptTemplate(
            input_variables=["text"],
            template=MAP_PROMPT_TEMPLATE,
        )
        self.reduce_prompt = PromptTemplate(
            input_variables=["text"],
            template=REDUCE_PROMPT_TEMPLATE,
        )
        self.insight_prompt = PromptTemplate(
            input_variables=["text"],
            template=INSIGHT_EXTRACTION_TEMPLATE,
        )

    def summarize(self, documents: List[Document]) -> str:
        """
        Generate a full summary of all document chunks.
        Uses map-reduce: summarize each chunk (map), then combine (reduce).
        """
        if not documents:
            return "No documents to summarize."

        # For short documents, use "stuff" (single call). For long ones, use map_reduce.
        if len(documents) <= 3:
            chain = load_summarize_chain(
                self.llm,
                chain_type="stuff",
                prompt=self.reduce_prompt,
            )
        else:
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                map_prompt=self.map_prompt,
                combine_prompt=self.reduce_prompt,
                verbose=False,
            )

        result = chain.invoke({"input_documents": documents})
        return result.get("output_text", "Summary generation failed.")

    def extract_insights(self, documents: List[Document]) -> str:
        """
        Extract structured key insights from the document content.
        Combines all document text and runs a targeted extraction prompt.
        """
        if not documents:
            return "No documents provided for insight extraction."

        # Combine text from all chunks (limit to avoid token overflow)
        combined_text = "\n\n".join(
            doc.page_content for doc in documents[:15]
        )

        # Trim to ~12000 characters to stay well within token limits
        if len(combined_text) > 12000:
            combined_text = combined_text[:12000] + "\n\n[Document truncated for analysis]"

        prompt = self.insight_prompt.format(text=combined_text)
        response = self.llm.invoke(prompt)
        return response.content

    def quick_summary(self, text: str, max_words: int = 100) -> str:
        """
        Generate a short one-paragraph summary of raw text.
        Useful for summarizing individual pages or sections.
        """
        prompt = f"""Summarize the following text in {max_words} words or fewer.
Be concise and focus on the most important information.

Text:
{text}

Summary:"""
        response = self.llm.invoke(prompt)
        return response.content

    def compare_documents(self, doc_summaries: List[str]) -> str:
        """
        Compare multiple documents and highlight similarities and differences.
        """
        if len(doc_summaries) < 2:
            return "Need at least 2 documents to compare."

        formatted = "\n\n".join(
            f"DOCUMENT {i+1}:\n{s}" for i, s in enumerate(doc_summaries)
        )
        prompt = f"""You have been given summaries of {len(doc_summaries)} documents.
Compare these documents and identify:
1. Common themes and shared information
2. Key differences in content, approach, or conclusions
3. Which document covers what uniquely

{formatted}

Provide a structured comparison:"""
        response = self.llm.invoke(prompt)
        return response.content