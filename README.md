# RAG Chatbot Project

A lightweight Retrieval-Augmented Generation (RAG) chatbot implementation using a local vector database, sentence embeddings, and a small language model.

## Introduction

This project implements a simple chatbot that uses retrieval-augmented generation to provide accurate and context-aware responses. The chatbot leverages a local vector database to store and retrieve relevant documents based on user queries.

## Features

- **Vector Database**: Manages a local database of text embeddings for efficient retrieval.
- **Sentence Embedding**: Converts text into vector embeddings using a pre-trained model.
- **Small Language Model**: Generates responses using a compact language model.
- **Retrieval-Augmented Generation**: Combines retrieved information with generated responses for accurate answers.

## Components

### VectorDatabase

Manages storing, retrieving, and searching text embeddings in a local database.

### SentenceEmbedder

Converts text into vector embeddings using a pre-trained sentence transformer.

### SmolLM2 Model

Generates responses and integrates with the retrieval system using a small language model.

### RAG ChatBot

Orchestrates the interaction between the user, the language model, and the vector database to provide relevant responses.

## Usage

To use the chatbot:

```python
from rag import RAG_ChatBot

chatbot = RAG_ChatBot()

response = chatbot("Your question here")
print(response)
```

## Example Usage

Here's an example of how to use the chatbot in a Jupyter notebook:

```python
from rag import RAG_ChatBot

# Initialize the chatbot
chatbot = RAG_ChatBot()

# Ask a question
response = chatbot("I want to find out who is the discoverer behind insulin resistance medicine and the year discovered.")
print(response)
```

**Output:**

```
The discoverer behind insulin resistance medicine is Frederick Banting, and the year discovered is 1921.
```

You can continue the conversation:

```python
response = chatbot("How do you know so much?")
print(response)
```

**Output:**

```
I was trained on a vast amount of text data, which allows me to provide information on a wide range of topics.
```