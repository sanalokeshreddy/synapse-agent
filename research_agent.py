# research_agent.py
import re
from typing import List, Dict, Any
from enum import Enum

class ResearchStepType(Enum):
    QUERY_DECOMPOSITION = "query_decomposition"
    SOURCE_IDENTIFICATION = "source_identification"
    INFORMATION_EXTRACTION = "information_extraction"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"

class ResearchStep:
    def __init__(self, step_type: ResearchStepType, description: str, result: str = ""):
        self.step_type = step_type
        self.description = description
        self.result = result

class DeepResearchAgent:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.research_steps = []
        
    def decompose_query(self, query: str) -> List[str]:
        """Break down a complex query into sub-questions"""
        prompt = f"""
        Decompose the following research query into 3-5 specific sub-questions that would help answer it comprehensively.
        Query: {query}
        
        Return only the sub-questions as a numbered list, without any additional text.
        """
        
        response = self.llm(prompt, max_tokens=300, temperature=0.1, stop=["\n\n"])
        sub_questions = [q.strip() for q in response['choices'][0]['text'].split('\n') if q.strip()]
        
        # Clean up numbering
        sub_questions = [re.sub(r'^\d+[\.\)]\s*', '', q) for q in sub_questions]
        
        step = ResearchStep(
            ResearchStepType.QUERY_DECOMPOSITION,
            "Breaking down the main query into sub-questions for comprehensive research",
            "\n".join([f"{i+1}. {q}" for i, q in enumerate(sub_questions)])
        )
        self.research_steps.append(step)
        
        return sub_questions
    
    def research_sub_question(self, sub_question: str, context_chunks: List[str]) -> Dict[str, Any]:
        """Research a specific sub-question using the provided context"""
        # First, identify the most relevant chunks for this sub-question
        prompt_identify = f"""
        Identify which of the following context chunks are most relevant to answer: "{sub_question}"
        
        Context chunks:
        {chr(10).join([f'[Chunk {i+1}] {chunk}' for i, chunk in enumerate(context_chunks)])}
        
        Return only the numbers of the most relevant chunks (e.g., "1, 3, 5") without additional text.
        """
        
        response = self.llm(prompt_identify, max_tokens=50, temperature=0.1)
        relevant_indices = []
        try:
            relevant_indices = [int(i.strip()) - 1 for i in response['choices'][0]['text'].split(',')]
        except:
            # If parsing fails, use all chunks
            relevant_indices = list(range(len(context_chunks)))
        
        relevant_chunks = [context_chunks[i] for i in relevant_indices if i < len(context_chunks)]
        
        step = ResearchStep(
            ResearchStepType.SOURCE_IDENTIFICATION,
            f"Identifying relevant sources for sub-question: '{sub_question}'",
            f"Selected chunks: {', '.join([str(i+1) for i in relevant_indices if i < len(context_chunks)])}"
        )
        self.research_steps.append(step)
        
        # Extract information from relevant chunks
        prompt_extract = f"""
        Based on the following context, extract information relevant to: "{sub_question}"
        
        Context:
        {chr(10).join([f'[Source {i+1}] {chunk}' for i, chunk in enumerate(relevant_chunks)])}
        
        Extract key facts, insights, and information that helps answer the question.
        Organize your response with clear bullet points.
        """
        
        response = self.llm(prompt_extract, max_tokens=500, temperature=0.2)
        extracted_info = response['choices'][0]['text']
        
        step = ResearchStep(
            ResearchStepType.INFORMATION_EXTRACTION,
            f"Extracting relevant information from sources for: '{sub_question}'",
            extracted_info
        )
        self.research_steps.append(step)
        
        return {
            "sub_question": sub_question,
            "relevant_chunks": relevant_chunks,
            "extracted_info": extracted_info
        }
    
    def synthesize_findings(self, research_results: List[Dict[str, Any]], main_query: str) -> str:
        """Synthesize all research findings into a coherent answer"""
        research_summary = "\n\n".join([
            f"Sub-question: {result['sub_question']}\nFindings: {result['extracted_info']}"
            for result in research_results
        ])
        
        prompt_synthesize = f"""
        Synthesize the following research findings to provide a comprehensive answer to the query: "{main_query}"
        
        Research findings:
        {research_summary}
        
        Provide a well-structured, coherent response that:
        1. Directly answers the main query
        2. Incorporates evidence from the research findings
        3. Acknowledges any limitations or gaps in the information
        4. Is organized with clear sections if appropriate
        """
        
        response = self.llm(prompt_synthesize, max_tokens=1000, temperature=0.2)
        synthesized_answer = response['choices'][0]['text']
        
        step = ResearchStep(
            ResearchStepType.SYNTHESIS,
            "Synthesizing all research findings into a comprehensive answer",
            synthesized_answer
        )
        self.research_steps.append(step)
        
        return synthesized_answer
    
    def validate_answer(self, answer: str, context_chunks: List[str]) -> str:
        """Validate the answer against the source material"""
        prompt_validate = f"""
        Validate whether the following answer is consistent with the provided source material:
        
        Answer to validate:
        {answer}
        
        Source material:
        {chr(10).join([f'[Source {i+1}] {chunk}' for i, chunk in enumerate(context_chunks)])}
        
        Identify any:
        1. Claims that are not supported by the sources
        2. Potential misinterpretations of the source material
        3. Important information from the sources that was omitted
        
        Provide a concise validation report.
        """
        
        response = self.llm(prompt_validate, max_tokens=400, temperature=0.1)
        validation_report = response['choices'][0]['text']
        
        step = ResearchStep(
            ResearchStepType.VALIDATION,
            "Validating the answer against source material for accuracy",
            validation_report
        )
        self.research_steps.append(step)
        
        return validation_report
    
    def conduct_research(self, query: str, context_chunks: List[str]) -> Dict[str, Any]:
        """Conduct multi-step research on a query"""
        self.research_steps = []
        
        # Step 1: Decompose the query
        sub_questions = self.decompose_query(query)
        
        # Step 2: Research each sub-question
        research_results = []
        for sub_q in sub_questions:
            result = self.research_sub_question(sub_q, context_chunks)
            research_results.append(result)
        
        # Step 3: Synthesize findings
        answer = self.synthesize_findings(research_results, query)
        
        # Step 4: Validate answer
        validation = self.validate_answer(answer, context_chunks)
        
        return {
            "answer": answer,
            "validation": validation,
            "research_steps": [{"type": step.step_type.value, "description": step.description, "result": step.result} 
                              for step in self.research_steps],
            "sub_questions": sub_questions,
            "research_results": research_results
        }