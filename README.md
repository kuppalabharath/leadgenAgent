# leadgen-chatbot
An interactive, cross-device chatbot that blends natural conversation with accurate, context-aware answers. Detects user interest, captures leads, and works seamlessly as a shareable prototype — perfect for turning casual chats into meaningful connections.

LeadGen – Conversational Lead Generation Prototype

LeadGen is a smart, domain-adaptable chatbot designed to engage users, provide precise information, and identify potential leads. It combines a Retrieval-Augmented Generation (RAG) pipeline for answering domain-specific questions with a general language model for broader, open-ended conversations.

Features
	•	Domain-specific accuracy – Retrieves relevant answers from a custom knowledge base.
	•	Seamless fallback – Handles general queries outside the stored data.
	•	Lead identification – Detects interest and requests user contact details for follow-up.
	•	Structured retrieval – Powered by FAISS and sentence-transformer embeddings for fast and accurate matches.

How It Works
	1.	Context Retrieval – Finds the most relevant knowledge for a given query.
	2.	Answer Generation – Uses Groq’s LLaMA model to produce clear, natural responses.
	3.	Lead Capture – Prompts for contact details when user intent indicates strong interest.
	4.	Data Logging – Stores captured lead details locally for follow-up actions.

Potential Applications
	•	Educational course inquiries
	•	Product or service pre-sales
	•	Customer onboarding assistance
	•	Event registration and engagement
