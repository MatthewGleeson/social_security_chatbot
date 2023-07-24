# Creating a Social Security Question answering Chatbot with Beautiful Soup, Pinecone, ChatGPT, Langchain


While reading a book about homeless healthcare I was surprised to hear of the difficulty that people have with determining whether they qualify for social security in the US. The "Blue Book" of social security, which the US Social Security Agency publishes for the intention of doctors and nurses to use to determine whether their patients qualify, uses a lot of medical jargon that may be difficult for the normal person to understand. After doing some digging, I realized that the blue book is published online. I wanted to explore the idea of scraping all blue book content using Beautiful Soup, then using that content to create a vector database and store it in Pinecone with semantic embeddings. Once I have my database that I can use to ground the language model, I want to use Langchain and ChatGPT to make a chatbot which is able to look to the Blue Book as a primary source(aka [retrieval augmented generation](https://arxiv.org/abs/2005.11401)). 


Navigate to the python notebook to view and download the notebook to try it out for yourself!