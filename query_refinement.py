# query_refinement.py (Fixed)
import re
from typing import List, Dict, Any

class QueryRefiner:
    def __init__(self, llm):
        self.llm = llm
    
    def suggest_follow_up_questions(self, query: str, context: str) -> List[str]:
        """Suggest follow-up questions based on the current query and context"""
        prompt = f"""
        Based on the following query and context, suggest 3-5 relevant follow-up questions that would help deepen the research:
        
        Original query: {query}
        Context: {context[:1000]}...
        
        Return only the questions as a numbered list, without any additional text.
        """
        
        response = self.llm(prompt, max_tokens=300, temperature=0.3)
        questions = [q.strip() for q in response['choices'][0]['text'].split('\n') if q.strip()]
        
        # Clean up numbering
        questions = [re.sub(r'^\d+[\.\)]\s*', '', q) for q in questions]
        
        return questions
    
    def refine_query(self, query: str, feedback: str) -> str:
        """Refine a query based on user feedback"""
        prompt = f"""
        Refine the following research query based on the user's feedback:
        
        Original query: {query}
        User feedback: {feedback}
        
        Provide a refined version of the query that addresses the feedback while maintaining the original intent.
        Return only the refined query without additional text.
        """
        
        response = self.llm(prompt, max_tokens=150, temperature=0.2)
        refined_query = response['choices'][0]['text'].strip()
        
        return refined_query
    
    def expand_query(self, query: str) -> List[str]:
        """Expand a query into related queries for comprehensive research"""
        prompt = f"""
        Expand the following research query into 3-5 related queries that would provide a more comprehensive understanding:
        
        Original query: {query}
        
        Return only the related queries as a numbered list, without any additional text.
        """
        
        response = self.llm(prompt, max_tokens=300, temperature=0.3)
        queries = [q.strip() for q in response['choices'][0]['text'].split('\n') if q.strip()]
        
        # Clean up numbering
        queries = [re.sub(r'^\d+[\.\)]\s*', '', q) for q in queries]
        
        return queries