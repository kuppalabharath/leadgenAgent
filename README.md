LeadGen Agent

LeadGen Agent is a streamlined, cross-device chatbot designed to answer questions, provide relevant information, and help turn casual conversations into qualified leads. It blends a custom knowledge base with a powerful language model, ensuring accurate answers when the data is available and smooth, natural conversation when it’s not.

What It Does
	•	Answers with context – Pulls information from a curated knowledge base using FAISS and sentence-transformer embeddings.
	•	Keeps the conversation going – Falls back to Groq’s LLaMA model for general or open-ended questions.
	•	Identifies potential leads – Detects when a user is showing interest and prompts for contact details.
	•	Saves useful information – Stores collected leads locally for follow-up or integration into other systems.

How It Works
	1.	The user asks a question.
	2.	The system searches the knowledge base for the most relevant context.
	3.	If a match is found, it crafts an accurate, clear response.
	4.	If not, the conversation continues through the language model.
	5.	When intent suggests real interest, the agent requests contact details and logs them for future use.




