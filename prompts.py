from langchain.prompts import PromptTemplate

import templates

rag_prompt = PromptTemplate.from_template(templates.rag_template)
competitor_prompt = PromptTemplate.from_template(templates.competitor_template)