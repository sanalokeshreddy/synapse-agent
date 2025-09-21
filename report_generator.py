# report_generator.py
from fpdf import FPDF
import markdown
from typing import List, Dict, Any

class ReportGenerator:
    def __init__(self):
        pass
    
    def generate_pdf_report(self, research_data: Dict[str, Any], output_path: str) -> None:
        """Generate a PDF research report"""
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, txt=f"Research Report: {research_data['query']}", ln=True, align='C')
        pdf.ln(10)
        
        # Main answer
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt="Executive Summary", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 8, research_data['answer'])
        pdf.ln(5)
        
        # Research steps
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt="Research Methodology", ln=True)
        pdf.set_font("Arial", '', 12)
        
        for i, step in enumerate(research_data['research_steps']):
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, txt=f"Step {i+1}: {step['description']}", ln=True)
            pdf.set_font("Arial", '', 10)
            pdf.multi_cell(0, 6, step['result'])
            pdf.ln(2)
        
        # Sources
        if 'sources' in research_data:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, txt="Reference Sources", ln=True)
            pdf.set_font("Arial", '', 10)
            
            for i, source in enumerate(research_data['sources']):
                pdf.multi_cell(0, 6, f"Source {i+1}: {source[:200]}...")
                pdf.ln(2)
        
        # Validation
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt="Validation Report", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 8, research_data['validation'])
        
        pdf.output(output_path)
    
    def generate_markdown_report(self, research_data: Dict[str, Any], output_path: str) -> None:
        """Generate a Markdown research report"""
        md_content = f"# Research Report: {research_data['query']}\n\n"
        
        md_content += "## Executive Summary\n\n"
        md_content += f"{research_data['answer']}\n\n"
        
        md_content += "## Research Methodology\n\n"
        for i, step in enumerate(research_data['research_steps']):
            md_content += f"### Step {i+1}: {step['description']}\n\n"
            md_content += f"{step['result']}\n\n"
        
        if 'sources' in research_data:
            md_content += "## Reference Sources\n\n"
            for i, source in enumerate(research_data['sources']):
                md_content += f"{i+1}. {source[:200]}...\n\n"
        
        md_content += "## Validation Report\n\n"
        md_content += f"{research_data['validation']}\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def generate_html_report(self, research_data: Dict[str, Any], output_path: str) -> None:
        """Generate an HTML research report"""
        md_content = f"# Research Report: {research_data['query']}\n\n"
        
        md_content += "## Executive Summary\n\n"
        md_content += f"{research_data['answer']}\n\n"
        
        md_content += "## Research Methodology\n\n"
        for i, step in enumerate(research_data['research_steps']):
            md_content += f"### Step {i+1}: {step['description']}\n\n"
            md_content += f"{step['result']}\n\n"
        
        if 'sources' in research_data:
            md_content += "## Reference Sources\n\n"
            for i, source in enumerate(research_data['sources']):
                md_content += f"{i+1}. {source[:200]}...\n\n"
        
        md_content += "## Validation Report\n\n"
        md_content += f"{research_data['validation']}\n"
        
        html_content = markdown.markdown(md_content)
        
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Research Report: {research_data['query']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
                h3 {{ color: #7f8c8d; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)