"""RAG Pipeline integrating retriever with Gemini LLM."""

import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai

try:
    from langfuse import observe
except ImportError:
    # Fallback dummy decorator
    def observe(*args, **kwargs):
        def decorator(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return decorator


from app.rag.retriever import HybridRetriever
from app.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline combining hybrid retrieval and Gemini generation."""

    def __init__(self, retriever: HybridRetriever):
        """
        Initialize RAG pipeline.

        Args:
            retriever: HybridRetriever instance
        """
        self.retriever = retriever

        # Configure Gemini
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)

        logger.info(
            f"Initialized RAG Pipeline with Gemini model: {settings.gemini_model}"
        )

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string.

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", {})

            # Extract key information
            country = content.get("country", "Unknown")
            main_content = content.get("content", "")

            # Format visa requirements
            visa_info = ""
            if "visa_requirements" in content:
                visa_req = content["visa_requirements"]
                if "indian_citizens" in visa_req:
                    indian_visa = visa_req["indian_citizens"]
                    visa_info = f"\n\nVisa Requirements for Indian Citizens:\n"
                    visa_info += (
                        f"- Visa Required: {indian_visa.get('visa_required', 'N/A')}\n"
                    )
                    visa_info += f"- Visa Type: {indian_visa.get('visa_type', 'N/A')}\n"
                    visa_info += f"- Processing Time: {indian_visa.get('processing_time', 'N/A')}\n"
                    visa_info += (
                        f"- Stay Duration: {indian_visa.get('stay_duration', 'N/A')}\n"
                    )

                    if "documents_needed" in indian_visa:
                        visa_info += f"- Documents Needed: {', '.join(indian_visa['documents_needed'][:5])}\n"

                    if "note" in indian_visa:
                        visa_info += f"- Note: {indian_visa['note']}\n"

            # Format attractions
            attractions = ""
            if "attractions" in content and content["attractions"]:
                attractions = (
                    f"\n\nTop Attractions: {', '.join(content['attractions'][:5])}"
                )

            # Format other info
            other_info = ""
            if "best_time_to_visit" in content:
                other_info += f"\n\nBest Time to Visit: {content['best_time_to_visit']}"
            if "climate" in content:
                other_info += f"\nClimate: {content['climate']}"
            if "currency" in content:
                other_info += f"\nCurrency: {content['currency']}"

            # Combine all parts
            doc_text = f"[Source {i}: {country}]\n{main_content}{visa_info}{attractions}{other_info}"
            context_parts.append(doc_text)

        return "\n\n---\n\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for Gemini.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        prompt = f"""You are a friendly travel assistant helping users plan their trips. 
Your goal is to provide SIMPLE, CLEAR, and EASY-TO-UNDERSTAND answers.

Use the following context to answer the user's question.

Context:
{context}

User Question: {query}

IMPORTANT INSTRUCTIONS for your response:
1. **Use simple language** - Avoid complex words, write like you're talking to a friend
2. **Be concise** - Keep answers short and to the point (max 300 words)
3. **Use bullet points** - Make information scannable
4. **Highlight key info** - Put the most important details first
5. **Use emojis** - Add relevant emojis (✈️ 🌍 💰 📅) to make it friendly

FORMAT YOUR ANSWER LIKE THIS:

**[Country Name]** 🌍

**Do I need a visa?**
• Yes/No - [visa type]
• Processing time: [time]
• Stay allowed: [duration]

**Documents needed:** 📄
• List only the TOP 3-4 most important documents
• Use simple terms (passport, photo, bank statement, etc.)

**Best time to go:** 📅
• [Month to Month] - [reason in simple words]

**Must-see places:** ✈️
• [Top 3 attractions only]

**Quick tips:** 💡
• Currency: [currency]
• Language: [language]
• [One helpful tip]

Keep it SHORT, SIMPLE, and HELPFUL!"""

        return prompt

    @observe(name="rag_pipeline", as_type="chain")
    def query(
        self,
        query: str,
        top_k: int = None,
        temperature: float = None,
        max_tokens: int = None,
        return_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute RAG query: retrieve documents and generate response.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            return_sources: Whether to return source documents

        Returns:
            Dict containing response and metadata
        """
        try:
            logger.info(f"Processing RAG query: '{query}'")

            # Step 1: Retrieve relevant documents
            documents = self.retriever.hybrid_search(query, top_k=top_k)
            logger.info(f"Retrieved {len(documents)} documents")

            if not documents:
                logger.warning("No documents retrieved")
                return {
                    "answer": "I couldn't find specific information about that. Could you please rephrase your question or ask about a specific destination?",
                    "sources": [],
                    "query": query,
                }

            # Step 2: Format context
            context = self._format_context(documents)

            # Step 3: Create prompt
            prompt = self._create_prompt(query, context)

            # Step 4: Generate response with Gemini
            generation_config = {
                "temperature": temperature or settings.gemini_temperature,
                "max_output_tokens": max_tokens or settings.gemini_max_tokens,
            }

            logger.info("Generating response with Gemini...")
            response = self.model.generate_content(
                prompt, generation_config=generation_config
            )

            answer = response.text

            # Track token usage for LangFuse with model information
            usage_metadata = {}
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                usage_metadata = {
                    "model": "gemini-2.0-flash-exp",  # Model name for LangFuse
                    "input_tokens": getattr(usage, "prompt_token_count", 0),
                    "output_tokens": getattr(usage, "candidates_token_count", 0),
                    "total_tokens": getattr(usage, "total_token_count", 0),
                }
                logger.info(f"Token usage: {usage_metadata}")

                # Report usage to LangFuse
                try:
                    from langfuse.decorators import langfuse_context

                    if langfuse_context.get_current_observation_id():
                        langfuse_context.update_current_observation(
                            model=usage_metadata["model"],
                            usage={
                                "input": usage_metadata["input_tokens"],
                                "output": usage_metadata["output_tokens"],
                                "total": usage_metadata["total_tokens"],
                            },
                        )
                except ImportError:
                    logger.warning(
                        "LangFuse decorators not available, skipping usage reporting"
                    )

            logger.info("Response generated successfully")

            # Step 5: Prepare result
            result = {"answer": answer, "query": query, "sources_count": len(documents)}

            # Add usage metadata if available
            if usage_metadata:
                result["usage"] = usage_metadata

            if return_sources:
                result["sources"] = [
                    {
                        "country": doc["content"].get("country", "Unknown"),
                        "title": doc["content"].get("title", ""),
                        "score": doc.get("score", 0),
                        "id": doc.get("id"),
                    }
                    for doc in documents
                ]

            return result

        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "sources": [],
                "query": query,
                "error": str(e),
            }

    def query_with_chat_history(
        self, query: str, chat_history: List[Dict[str, str]], top_k: int = None
    ) -> Dict[str, Any]:
        """
        Execute RAG query with chat history context.

        Args:
            query: User query
            chat_history: Previous chat messages [{"role": "user/assistant", "content": "..."}]
            top_k: Number of documents to retrieve

        Returns:
            Dict containing response and metadata
        """
        try:
            # Enhance query with chat history context for better retrieval
            enhanced_query = query
            if chat_history:
                recent_context = " ".join(
                    [
                        msg["content"]
                        for msg in chat_history[-3:]
                        if msg.get("role") == "user"
                    ]
                )
                enhanced_query = f"{recent_context} {query}"

            # Retrieve documents with enhanced query
            documents = self.retriever.hybrid_search(enhanced_query, top_k=top_k)

            if not documents:
                return {
                    "answer": "I couldn't find specific information about that. Could you please provide more details?",
                    "sources": [],
                    "query": query,
                }

            # Format context
            context = self._format_context(documents)

            # Create chat-aware prompt
            chat_context = "\n".join(
                [
                    f"{msg['role'].capitalize()}: {msg['content']}"
                    for msg in chat_history[-5:]
                ]
            )

            prompt = f"""You are a knowledgeable travel assistant in an ongoing conversation.

Previous Conversation:
{chat_context}

Retrieved Information:
{context}

Current Question: {query}

Provide a helpful response that considers the conversation history and retrieved information."""

            # Generate response
            response = self.model.generate_content(prompt)

            return {
                "answer": response.text,
                "query": query,
                "sources_count": len(documents),
                "sources": [
                    {
                        "country": doc["content"].get("country", "Unknown"),
                        "title": doc["content"].get("title", ""),
                        "score": doc.get("score", 0),
                    }
                    for doc in documents
                ],
            }

        except Exception as e:
            logger.error(f"Error in chat query: {e}")
            return {
                "answer": f"Error processing your request: {str(e)}",
                "sources": [],
                "query": query,
                "error": str(e),
            }
