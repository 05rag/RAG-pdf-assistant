# ___ Advanced RAG Pipeline: Streaming, Citations, History, Summarization ___
from src.engine import RAGRetriever
from typing import List, Dict, Any
import time

class AdvancedRAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = [] # Store query history

    def query(self, question: str, source_file: str = None, top_k: int = 8, min_score =0.3, stream: bool =False, summarize: bool = False) -> Dict:
        
        rewrite_prompt = f"Identify and fix only typos, grammatical error in this query. Query: {question}"
        search_query = self.llm.invoke(rewrite_prompt).content
        print(f"Searching for: {search_query}")

        #Retrieve relevant document
        results = self.retriever.retrieve(search_query, source_file=source_file ,top_k = top_k, score_threshold = min_score)
        
        if not results:
            return{
                'question': question,
                'answer' : "No relevant context found.",
                'sources' : [],
                'summary' : None,
            }
        
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page' : doc['metadata'].get('page', 'unknown'),
                'score' : doc['similarity_score'],
                'preview': doc['content'][:300] + '...'
            } for doc in results]


            #Streaming anser simulation
            prompt = f"""
            You are a document-based assistant.

            Answer the question using the provided context.

            RULES:
            - Use the context as the primary source.
            - If exact data (email, link, name, date) is present, extract it precisely.
            - If partial information is available, answer based on it clearly.
            - If the answer is not present, say: "Not found in document."

            FORMAT:
            - Use bullet points for lists.
            - Keep answers concise and clear.
            - Highlight key terms when useful.

            Context:
            {context}

            Question: {question}
            """
            if stream:
                print("Streaming answer: ")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end= '', flush = True)
                    time.sleep(0.05)
                print()

            response = self.llm.invoke([prompt])
            answer = response.content


            # Add this temporary debug line:
            print(f"DEBUG: Keys in source dictionary: {sources[0].keys()}")

            #Add citations to answer
            citations = [
                f"[{i+1}] {src.get('source', 'Unknown')} (page {src.get('page', 'N/A')})" 
                for i, src in enumerate(sources)
            ]

            answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

            # Optionally summarize answer
            summary = None
            if summarize and answer:
                summary_prompt = f"Summarize the folling answer in 2 sentences:\n{answer}"
                summary_resp = self.llm.invoke([summary_prompt])
                summary = summary_resp.content
    
            #Store query history
            self.history.append({
                'question' : question,
                'answer' : answer,
                'sources' : sources,
                'summary' : summary
            })

            return {
                'question' : question,
                'answer' : answer_with_citations,
                'sources' : sources,
                'summary' : summary,
                'history': self.history
            }
