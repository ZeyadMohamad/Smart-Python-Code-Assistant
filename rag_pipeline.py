# rag_pipeline.py
import os
from typing import List, Dict, Any
from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

class RAGPipeline:
    """RAG system using LangChain, embeddings, FAISS, and RetrievalQA"""
    
    def __init__(self, top_k=3):
        self.top_k = top_k
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'}  # Changed to CPU for better compatibility
        )
        self.vectorstore = None
        self.retrieval_qa = None
        self.setup_rag()
    
    def setup_rag(self):
        """Load datasets and create vector store with RetrievalQA for evaluation"""
        documents = []
        
        # Load HumanEval dataset
        try:
            print("Loading HumanEval dataset...")
            humaneval = load_dataset("openai/openai_humaneval", split="test")
            for item in humaneval:
                content = f"Problem: {item['prompt']}\nSolution:\n{item['canonical_solution']}"
                doc = Document(
                    page_content=content,
                    metadata={
                        'task_id': item['task_id'],
                        'prompt': item['prompt'],
                        'solution': item['canonical_solution'],
                        'source': 'humaneval'
                    }
                )
                documents.append(doc)
            print(f"Loaded {len(documents)} HumanEval examples")
        except Exception as e:
            print(f"Error loading HumanEval: {e}")
        
        # Load MBPP training data
        try:
            print("Loading MBPP training dataset...")
            mbpp = load_dataset("mbpp", split="train")
            for item in mbpp.select(range(200)):  # Use subset for efficiency
                content = f"Problem: {item['text']}\nSolution:\n{item['code']}"
                doc = Document(
                    page_content=content,
                    metadata={
                        'task_id': str(item['task_id']),
                        'prompt': item['text'],
                        'solution': item['code'],
                        'source': 'mbpp'
                    }
                )
                documents.append(doc)
            print(f"Added MBPP examples, total: {len(documents)}")
        except Exception as e:
            print(f"Error loading MBPP: {e}")
        
        # Create vector store and RAG pipeline
        if documents:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Initialize LLM for RetrievalQA (for evaluation purposes)
            llm = ChatOpenAI(
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                model="qwen/qwen-2.5-72b-instruct:free",  # Using different model for RAG evaluation
                temperature=0.1,
                max_tokens=512
            )
            
            # Create custom prompt template for evaluation
            prompt_template = """You are a code evaluation assistant. Based on the following retrieved examples, analyze the given query and provide insights about code similarity and relevance.

Retrieved Examples:
{context}

Query: {question}

Provide a brief analysis of:
1. How relevant the retrieved examples are to the query
2. Quality of the retrieved solutions
3. Similarity score (0-1) indicating how well the examples match the query

Analysis:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create RetrievalQA chain for evaluation
            self.retrieval_qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": self.top_k}
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            print("RAG pipeline with RetrievalQA created successfully for evaluation")
    
    def retrieve_examples(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve similar examples using FAISS similarity search"""
        if not self.vectorstore:
            return []
        
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            )
            docs = retriever.get_relevant_documents(query)
            
            return [{
                'task_id': doc.metadata['task_id'],
                'prompt': doc.metadata['prompt'],
                'solution': doc.metadata['solution'],
                'source': doc.metadata['source']
            } for doc in docs]
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
    
    def evaluate_retrieval_quality(self, query: str, expected_solution: str) -> Dict[str, Any]:
        """Evaluate RAG retrieval quality using RetrievalQA chain"""
        if not self.retrieval_qa:
            return {"retrieval_score": 0.0, "retrieved_examples": [], "rag_analysis": "RAG not initialized"}
        
        try:
            # Get RAG-based evaluation
            rag_result = self.retrieval_qa({"query": query})
            rag_analysis = rag_result["result"]
            source_docs = rag_result.get("source_documents", [])
            
            # Extract retrieved examples from source documents
            retrieved_examples = []
            for doc in source_docs:
                retrieved_examples.append({
                    'task_id': doc.metadata['task_id'],
                    'prompt': doc.metadata['prompt'],
                    'solution': doc.metadata['solution'],
                    'source': doc.metadata['source']
                })
            
            # Simple scoring based on number of retrieved examples and diversity
            retrieval_score = len(retrieved_examples) / self.top_k
            
            # Bonus for source diversity
            sources = set(ex['source'] for ex in retrieved_examples)
            diversity_bonus = len(sources) / 2.0  # Max 2 sources (humaneval, mbpp)
            final_score = min(1.0, retrieval_score + (diversity_bonus * 0.1))
            
            return {
                "retrieval_score": final_score,
                "retrieved_examples": retrieved_examples,
                "num_retrieved": len(retrieved_examples),
                "rag_analysis": rag_analysis,
                "source_diversity": len(sources)
            }
            
        except Exception as e:
            print(f"RAG evaluation error: {e}")
            # Fallback to simple retrieval
            retrieved_examples = self.retrieve_examples(query)
            return {
                "retrieval_score": len(retrieved_examples) / self.top_k,
                "retrieved_examples": retrieved_examples,
                "num_retrieved": len(retrieved_examples),
                "rag_analysis": f"RAG evaluation failed: {str(e)}",
                "source_diversity": 0
            }

# RAG Evaluation System
class RAGEvaluator:
    """Evaluate RAG system on MBPP dataset"""
    
    def __init__(self, assistant):
        self.assistant = assistant
        
    def evaluate_on_mbpp(self, num_examples=10):
        """Evaluate RAG system on MBPP test examples"""
        try:
            # Load MBPP test dataset
            mbpp_test = load_dataset("mbpp", split="test")
            results = []
            
            print(f"Evaluating on {num_examples} MBPP examples...")
            
            for i, example in enumerate(mbpp_test.select(range(num_examples))):
                print(f"Evaluating example {i+1}/{num_examples}")
                
                # Get RAG-based response from assistant
                result = self.assistant.process(example['text'])
                
                # Use RAG pipeline to evaluate retrieval quality
                rag_eval = self.assistant.rag_pipeline.evaluate_retrieval_quality(
                    query=example['text'],
                    expected_solution=example['code']
                )
                
                # Evaluate retrieval quality using RAG evaluation
                retrieval_quality = rag_eval.get("retrieval_score", 0.0)
                
                # Simple pass/fail based on whether code was generated
                generated_code = result['generated_response']
                has_function = 'def ' in generated_code
                
                eval_result = {
                    'task_id': example['task_id'],
                    'prompt': example['text'],
                    'expected_code': example['code'],
                    'generated_code': generated_code,
                    'retrieval_quality': retrieval_quality,
                    'num_retrieved': rag_eval.get("num_retrieved", 0),
                    'has_function': has_function,
                    'intent': result['intent'],
                    'rag_retrieved_examples': rag_eval.get("retrieved_examples", [])
                }
                
                results.append(eval_result)
            
            # Calculate overall metrics
            avg_retrieval_quality = sum(r['retrieval_quality'] for r in results) / len(results)
            function_generation_rate = sum(r['has_function'] for r in results) / len(results)
            
            summary = {
                'total_examples': num_examples,
                'avg_retrieval_quality': avg_retrieval_quality,
                'function_generation_rate': function_generation_rate,
                'detailed_results': results
            }
            
            return summary
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return {"error": str(e)}