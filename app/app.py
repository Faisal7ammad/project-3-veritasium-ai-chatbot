from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import re
from dotenv import load_dotenv
from collections import defaultdict
import wikipedia
import nltk
from nltk.corpus import stopwords
from pinecone import Pinecone
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains import RetrievalQA, LLMChain, ConversationChain
from langchain.vectorstores import Pinecone as LCPinecone
from langchain.memory import ConversationSummaryBufferMemory
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
OPENAI_API_KEY = os.getenv('IH_OPENAI_API_KEY')
PC_API_KEY = os.getenv('PC_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')

if not OPENAI_API_KEY or not PC_API_KEY:
    logging.error("OPENAI_API_KEY or PC_API_KEY not set in environment variables")

# Set the environment variables
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['HF_TOKEN'] = HF_TOKEN

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PC_API_KEY)
    # Initialize the Pinecone index
    index_name = "veritasium-vs-final"
    pinecone_index = pc.Index(index_name)
    logging.debug(f"Pinecone index '{index_name}' initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing Pinecone index: {e}")
    pinecone_index = None

# Initialize embeddings
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model='text-embedding-ada-002')

# Initialize LangChain Pinecone vector store with the summary as text_key
vector_store = LCPinecone(
    index=pinecone_index,
    embedding=embeddings_model,
    text_key="transcription"
)

# Initialize the Chat LLM with model_kwargs
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
chat_model = llm

# Create the retriever
retriever = vector_store.as_retriever()

# Set up the retrieval-based QA chain using RetrievalQA.from_chain_type
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Load the pre-trained model from the local directory
model_path = "sentence_transformer_model"  # Use relative path instead of absolute
try:
    model = SentenceTransformer(model_path)
    logging.debug(f"Model loaded successfully from {model_path}.")
except Exception as e:
    logging.error(f"Error loading model from {model_path}: {e}")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Set model to HF path if loading fails 


# Define keywords for different query types
fetch_keywords = [
    "share a video", "give me a video", "video explaining", "fetch a video",
    "find a video", "video about", "recommend a video", "video on", "video link",
    "YouTube video", "video URL", "get me a video", "video recommendation",
    "suggest a video", "can you show me a video", "video to watch", "find me a video",
    "share a link to a video", "suggest something to watch", "show me a video",
    "video related to", "share a YouTube link", "video on topic", "video to explain",
    "help me find a video", "show a video on", "suggest a YouTube video",
    "give me a YouTube link", "video related to this", "recommend a link",
    "suggest a clip", "share something to watch", "video link for", "YouTube link for",
    "a video about", "get a video link", "find a YouTube video", "share a clip",
    "provide a video", "show a link to a video", "fetch me a video", "video suggestion",
    "suggest a clip to watch", "recommend a video link", "YouTube link about",
    "can you share a video", "find a clip", "give a video link", "video search for",
    "help me get a video"
]

summarization_keywords = [
    "summarize this video", "summarize the video", "summarize", "summary",
    "give me a summary", "summary for", "summary about", "video summary",
    "summarize video content", "summarize the following video", "video overview",
    "what is this video about", "explain the video", "video explanation",
    "brief the video", "video briefing", "summarize the content",
    "content summary of video", "give a video summary", "summarize what is in the video",
    "overview of the video", "summarize the main points", "what does the video say",
    "explain the content", "what is in the video", "key points of the video",
    "video content explanation", "video highlights", "summarize the highlights",
    "video recap", "recap the video", "give an overview of the video",
    "content overview", "brief summary of the video", "video in short",
    "summarize the YouTube video", "explain the video content", "summarize YouTube video",
    "summarize the main ideas", "summary of the video content", "summarize key points",
    "quick summary of the video", "brief the content", "what's in this video",
    "what is covered in the video", "main points of the video", "highlight the video content",
    "summarize the important points", "summarize the video overview", "video summary explanation",
    "quick recap of the video"
]

# Compute embeddings for the keyword lists
fetch_keywords_embeddings = model.encode(fetch_keywords)
summarization_keywords_embeddings = model.encode(summarization_keywords)

# Expanded list of phrases indicating lack of information
lack_of_information_phrases = [
    "I'm sorry,", "I'm sorry, but", "i don't have information", "i don't have the information",
    "i don't have the specific information", "i don't have any information", "i don't have details about",
    "i don't have any details on", "i don't have any specific details", "i'm not sure",
    "i'm not certain", "i cannot provide information", "i cannot provide details", "I don't have information on",
    "i don't have specific information", "i don't have enough information", "I don't have relevant information",
    "i'm sorry, but the retrieved information does not contain", "i'm sorry, but i don't have relevant details", "i couldn't find specific details on",
    "i couldn't find relevant information on", "there is no information available on", "there is no relevant information on",
    "the retrieved data does not contain", "the retrieved information does not mention", "the retrieved information does not provide details on",
    "the retrieved documents do not have specific details", "there is nothing on", "no information found on",
    "no relevant details found on", "no specific details found on", "i don't have any relevant details about",
    "i don't have any useful information about", "i don't have any additional details on", "i'm afraid i don't have information on",
    "unfortunately, there is no information on", "unfortunately, the data does not include", "i cannot find any information on",
    "i'm unable to find details on", "no useful information found on", "i don't have the necessary details",
    "i don't have the needed information about", "there's nothing relevant about", "there's no useful data on",
    "there's no relevant content", "i cannot locate any details", "i have no information about",
    "i have no relevant data on", "the search results do not include", "i'm sorry, but i couldn't find anything on",
    "the search didn't return any details on", "i couldn't retrieve information on", "i couldn't gather any details on",
    "the retrieved data lacks information", "does not contain any details", "does not provide any relevant details",
    "does not contain any relevant details about", "I don't have any relevant information", "I don't have any relevant details",
    "I do not have relevant information", "information provided does not mention", "I don't have details",
    "I don't have any details", "I don't have any specific details", "I couldn't find specific details",
    "There is no information available", "There is no relevant information",
    "The retrieved information does not provide details", "The retrieved documents do not have specific details", "No information found",
    "I cannot provide an answer"
]

# Define the function to ask GPT with retriever
def ask_gpt_with_retriever(query, context=""):
    # Use the qa_chain to get the response and source documents
    result = qa_chain({"query": query})
    response = result["result"]
    source_documents = result["source_documents"]

    # Log retrieved documents for verification
    retrieved_texts = "\n\n".join(doc.page_content for doc in source_documents)
    logging.debug("Retrieved Documents:\n%s", retrieved_texts)

    # Combine retrieved texts with the existing context
    combined_context = context + "\n\nRetrieved documents:\n" + retrieved_texts

    messages = [
        SystemMessage(content="You are an assistant for question-answering tasks. Use the following pieces of retrieved info from Veritasium videos to answer the question. If the info doesn't help, just say that you don't know and be concise in your response. else if the retrieved info is helpful, be as verbose and educational in your response as possible."),
        HumanMessage(content="Here is some info retrieved from Veritasium videos:\n" + combined_context),
        HumanMessage(content="Based on this info, please answer the following question':"),
        HumanMessage(content=query)
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    llm_chain = LLMChain(llm=chat_model, prompt=prompt)
    gpt_response = llm_chain.run({})
    return gpt_response

class FetchAgent:
    def __init__(self, pinecone_index):
        self.pinecone_index = pinecone_index
        self.vectorizer = TfidfVectorizer()

    def fetch_all_video_ids(self):
        try:
            query_response = self.pinecone_index.query(
                vector=[0] * 1536,
                top_k=1500,
                include_metadata=True
            )
            all_ids = [match['id'] for match in query_response['matches']]
            logging.debug(f"Fetched {len(all_ids)} video IDs.")
            return all_ids
        except Exception as e:
            logging.error(f"An error occurred while fetching video IDs: {e}")
            return []

    def fetch_video_metadata(self, video_ids):
        try:
            video_data = self.pinecone_index.fetch(ids=video_ids)
            return video_data['vectors']
        except Exception as e:
            logging.error(f"An error occurred while fetching video metadata: {e}")
            return {}

    def fetch_video_urls(self, keyword_phrase, all_ids):
        results = defaultdict(list)
        if not all_ids:
            logging.debug("No video IDs found.")
            return results

        # Vectorize the query
        query_vector = self.vectorizer.fit_transform([keyword_phrase]).toarray()

        # Fetch metadata in batches
        batch_size = 100
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i:i+batch_size]
            video_metadata_batch = self.fetch_video_metadata(batch_ids)
            if not video_metadata_batch:
                logging.debug("No video metadata found.")
                continue

            for chunk_id, video_metadata in video_metadata_batch.items():
                metadata = video_metadata.get('metadata', {})
                title = metadata.get('title', '')
                description = metadata.get('description', '')
                transcription = metadata.get('transcription', '')
                base_video_id = chunk_id.split('_')[0]

                # Combine title, description, and transcription
                combined_text = f"{title} {description} {transcription}"

                # Vectorize the combined text
                text_vector = self.vectorizer.transform([combined_text]).toarray()

                # Calculate cosine similarity
                relevance_score = cosine_similarity(query_vector, text_vector)[0][0]

                if relevance_score > 0:
                    if base_video_id not in results or results[base_video_id][2] < relevance_score:
                        results[base_video_id] = (title, metadata['url'], relevance_score)

            logging.debug(f"Processed batch {i // batch_size + 1}/{len(all_ids) // batch_size + 1}")

        # Sort results by relevance score in descending order
        sorted_results = sorted(results.values(), key=lambda x: x[2], reverse=True)
        return [(title, url) for title, url, _ in sorted_results]

    def extract_keywords(self, query):
        # Use NLTK stopwords and additional custom stopwords
        stop_words = set(stopwords.words('english'))
        custom_stopwords = {'can', 'you', 'share', 'a', 'video', 'url', 'explaining', 'the', 'about', 'is', 'are', 'and', 'in'}
        all_stopwords = stop_words.union(custom_stopwords)

        # Simple keyword extraction using regular expression and common words filtering
        query = re.sub(r'[^\w\s]', '', query)  # Remove punctuation
        words = query.lower().split()
        keywords = [word for word in words if word not in all_stopwords]
        return ' '.join(keywords)  # Join keywords into a single phrase

    def run(self, query):
        keyword_phrase = self.extract_keywords(query)
        logging.debug(f"Extracted Keyword Phrase: {keyword_phrase}")  # Print extracted keywords for debugging

        if not keyword_phrase:
            return "No relevant keywords found in the query."

        all_ids = self.fetch_all_video_ids()
        if not all_ids:
            return "No video IDs found."

        results = self.fetch_video_urls(keyword_phrase, all_ids)

        if results:
            unique_results = list(dict.fromkeys(results))  # Remove duplicates while maintaining order
            logging.debug(f"Number of unique video matches: {len(unique_results)}")  # Log the number of unique results
            response = "Here are the top 5 video recommendations (while the video might not be strictly about your topic, it might be related):<br>"
            for title, url in unique_results[:5]:  # Limit to top 5 results
                response += f"{title}: {url}<br><br>"  # Add an extra <br> for better separation
        else:
            response = "No videos found for your query."
        return response.strip()

class VideoSummarizerAgent:
    def __init__(self, fetch_agent, qa_chain):
        self.fetch_agent = fetch_agent
        self.qa_chain = qa_chain
        self.vectorizer = TfidfVectorizer()

    def fetch_video_chunks(self, base_video_id):
        try:
            chunk_ids = [f"{base_video_id}_{i}" for i in range(30)]  # Adjust as needed
            video_metadata_batch = self.fetch_agent.fetch_video_metadata(chunk_ids)
            combined_transcriptions = []

            for chunk_id in sorted(video_metadata_batch.keys(), key=lambda x: int(x.split('_')[-1])):
                video_metadata = video_metadata_batch[chunk_id]['metadata']
                transcription = video_metadata.get('transcription', "")
                if isinstance(transcription, str):
                    combined_transcriptions.append(transcription)
                else:
                    combined_transcriptions.extend(transcription)

            combined_text = " ".join(combined_transcriptions)
            logging.debug(f"Fetched and combined transcription for video ID {base_video_id}")
            return combined_text
        except Exception as e:
            logging.error(f"An error occurred while fetching video chunks for video ID {base_video_id}: {e}")
            return ""

    def filter_content(self, text):
        filtered_text = re.sub(r'This video is sponsored by.*?$', '', text, flags=re.MULTILINE)
        filtered_text = re.sub(r'Check out .* for more information', '', filtered_text, flags=re.MULTILINE)
        # Add more filtering rules as needed
        return filtered_text

    def extract_keywords(self, query):
        stop_words = set(stopwords.words('english'))
        custom_stopwords = {'can', 'you', 'share', 'a', 'video', 'url', 'explaining', 'about', 'is', 'are', 'and', 'in'}
        all_stopwords = stop_words.union(custom_stopwords)

        query = re.sub(r'[^\w\s]', '', query)  # Remove punctuation
        words = query.lower().split()
        keywords = [word for word in words if word not in all_stopwords]
        return ' '.join(keywords)  # Join keywords into a single phrase

    def search_similar_videos(self, query, all_ids):
        keyword_phrase = self.extract_keywords(query)
        query_vector = self.vectorizer.fit_transform([keyword_phrase]).toarray()

        best_match_id = None
        highest_similarity = 0

        for batch_start in range(0, len(all_ids), 100):
            batch_ids = all_ids[batch_start:batch_start+100]
            video_metadata_batch = self.fetch_agent.fetch_video_metadata(batch_ids)

            for chunk_id, video_metadata in video_metadata_batch.items():
                metadata = video_metadata.get('metadata', {})
                title = metadata.get('title', '')
                description = metadata.get('description', '')
                transcription = metadata.get('transcription', '')
                combined_text = f"{title} {description} {transcription}"

                text_vector = self.vectorizer.transform([combined_text]).toarray()
                similarity = cosine_similarity(query_vector, text_vector)[0][0]

                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_id = chunk_id.split('_')[0]

        return best_match_id

    def extract_video_id(self, video_url_or_title):
        match = re.search(r'(?:v=|video id |youtu\.be/)([\w-]+)', video_url_or_title)
        if match:
            video_id = match.group(1)
            logging.debug(f"Extracted video ID from query: {video_id}")
            return video_id
        else:
            all_ids = self.fetch_agent.fetch_all_video_ids()
            if not all_ids:
                return None

            best_match_id = self.search_similar_videos(video_url_or_title, all_ids)
            if best_match_id:
                logging.debug(f"Extracted video ID from title search: {best_match_id}")
                return best_match_id
            else:
                logging.debug("No matching video found.")
                return None

    def summarize_video(self, video_url_or_title):
        try:
            video_id = self.extract_video_id(video_url_or_title)
            if not video_id:
                return "Could not find the video in Veritasium's channel. Please make sure to send the video URL or title."

            combined_text = self.fetch_video_chunks(video_id)
            if not combined_text:
                return "Could not find the video in Veritasium's channel. Please make sure to send the video URL or title."

            # Filter out unnecessary content
            filtered_text = self.filter_content(combined_text)

            # Generate summary using ask_gpt_with_retriever with a refined prompt
            summary_prompt = f"Provide a comprehensive and concise summary of the following video, removing any promotional content or irrelevant details:\n\n{filtered_text}"
            summary = ask_gpt_with_retriever(summary_prompt)
            logging.debug(f"Generated summary for video ID {video_id}")
            return summary.strip()
        except Exception as e:
            logging.error(f"An unexpected error occurred for video {video_url_or_title}: {e}")
            return "I was unable to summarize the video due to an unexpected issue. Try again."

    def run(self, query):
        summary = self.summarize_video(query)
        return summary

class ExternalKnowledgeRetrievalAgent:
    def __init__(self):
        pass

    def fetch_wikipedia_info(self, query):
        try:
            search_results = wikipedia.search(query)
            logging.debug(f"Wikipedia search results for '{query}': {search_results}")
            if not search_results:
                logging.debug(f"No relevant search results found for query: '{query}'")
                return "No relevant information found. Please provide more details or check your query.", "None"

            page = wikipedia.page(search_results[0])
            summary = page.summary
            logging.debug(f"Successfully fetched information for '{query}' from Wikipedia: {summary}")
            return f"I couldn't find relevant information in Veritasium's YouTube channel, but here's some information from Wikipedia:\n\n{summary}", "Wikipedia"

        except wikipedia.exceptions.DisambiguationError as e:
            logging.error(f"Disambiguation error while fetching Wikipedia info for query '{query}': {e}")
            options = '\n'.join(e.options[:5])  # Show top 5 options for disambiguation
            return f"Wikipedia: Your query is ambiguous. Please provide more specific information. Possible options:\n{options}", "Wikipedia"

        except wikipedia.exceptions.PageError as e:
            logging.error(f"Page error while fetching Wikipedia info for query '{query}': {e}")
            return "The page does not exist on Wikipedia. Please check your query and try again.", "Wikipedia"

        except Exception as e:
            logging.error(f"An error occurred while fetching Wikipedia info for query '{query}': {e}")
            return "I was Unable to retrieve information from Wikipedia due to an unexpected issue.", "Wikipedia"

    def answer_query(self, query):
        response, source = self.fetch_wikipedia_info(query)
        logging.debug(f"Wikipedia answer for query '{query}': {response}")
        return response, source

# Initialize Agents
fetch_agent = FetchAgent(pinecone_index)
summarizer_agent = VideoSummarizerAgent(fetch_agent, qa_chain)
external_knowledge_agent = ExternalKnowledgeRetrievalAgent()

class OrchestrationAgent:
    def __init__(self, fetch_agent, summarizer_agent, external_knowledge_agent, memory):
        self.fetch_agent = fetch_agent
        self.summarizer_agent = summarizer_agent
        self.external_knowledge_agent = external_knowledge_agent
        self.memory = memory

    def identify_task(self, query):
        query_embedding = model.encode([query])

        # Calculate cosine similarities
        fetch_similarities = cosine_similarity(query_embedding, fetch_keywords_embeddings)
        summarization_similarities = cosine_similarity(query_embedding, summarization_keywords_embeddings)

        max_fetch_similarity = max(fetch_similarities[0])
        max_summarization_similarity = max(summarization_similarities[0])

        # Improved optimal threshold
        optimal_threshold = 0.32

        if max_fetch_similarity > max_summarization_similarity and max_fetch_similarity > optimal_threshold:
            return 'fetch'
        elif max_summarization_similarity > max_fetch_similarity and max_summarization_similarity > optimal_threshold:
            return 'summarize'
        else:
            return 'general'

    def allocate_agent(self, query):
        # Load memory variables
        memory_variables = self.memory.load_memory_variables({})
        memory_history = memory_variables.get('history', '')

        # Debug: Print memory_history structure
        logging.debug("Memory History: %s", memory_history)

        # Encode the new query
        query_embedding = model.encode([query])

        # Initialize best similarity and response
        best_similarity = 0
        best_response = None

        # Define threshold for high similarity to reuse the same response
        high_similarity_threshold = 0.95 

        # Check similarity with stored queries in memory
        if memory_history:
            memory_entries = memory_history.split('\nHuman: ')
            for entry in memory_entries:
                if entry:
                    parts = entry.split('\nAI: ')
                    if len(parts) == 2:
                        stored_query = parts[0].strip()
                        stored_response = parts[1].strip()

                        # Encode the stored query
                        stored_query_embedding = model.encode([stored_query])

                        # Compute cosine similarity
                        similarity_score = cosine_similarity(query_embedding, stored_query_embedding)[0][0]

                        # If similarity is high enough, reuse the previous response
                        if similarity_score > high_similarity_threshold:
                            logging.debug(f"High similarity found: {similarity_score}")
                            response = f"From Memory: {stored_response}"
                            source = "Memory"
                            return response, source

                        # Update best similarity and response if it's the best seen so far
                        if similarity_score > best_similarity:
                            best_similarity = similarity_score
                            best_response = stored_response

        # Identify the task type
        task_type = self.identify_task(query)
        logging.debug(f"Identified task: {task_type}")

        # Initialize response and source variables
        response = ""
        source = ""

        # Allocate to the correct agent based on the task type
        if task_type == 'fetch':
            response = self.fetch_agent.run(query)
            source = "FetchAgent"
        elif task_type == 'summarize':
            response = self.summarizer_agent.run(query)
            source = "VideoSummarizerAgent"
        else:
            response = ask_gpt_with_retriever(query)
            source = "Retriever"

            # Check if the response lacks specific information
            if source == "Retriever":
                response_embedding = model.encode([response])
                lack_of_information_embeddings = model.encode(lack_of_information_phrases)
                similarity_scores = cosine_similarity(response_embedding, lack_of_information_embeddings)
                max_similarity = max(similarity_scores[0])

                if max_similarity > 0.2:  # Use a similarity threshold to detect lack of information
                    if any(phrase in response.lower() for phrase in lack_of_information_phrases):
                        logging.debug(f"Lack of information detected, switching to Wikipedia for query: {query}")
                        response, source = self.external_knowledge_agent.answer_query(query)
                        if any(phrase in response.lower() for phrase in lack_of_information_phrases):
                            logging.error("Wikipedia also failed to provide useful information.")
                            response = "I don't know. Something might have gone wrong! Please rephrase or ask another question :)"
                            source = "None"
                        else:
                            source = "Wikipedia"

        # Store the response in memory only if it's from FetchAgent, VideoSummarizerAgent, or ExternalKnowledgeAgent and it's informative
        if source in ["FetchAgent", "VideoSummarizerAgent", "Wikipedia"] and not response.startswith("From Memory:"):
            self.memory.save_context({"input": query}, {"output": response})

        logging.debug(f"Response: {response}, Source: {source}")
        return response, source

# Initialize memory and conversation chain
memory = ConversationSummaryBufferMemory(llm=chat_model)
conversation_chain = ConversationChain(llm=chat_model, memory=memory)

# Pass the memory to the OrchestrationAgent
orchestration_agent = OrchestrationAgent(fetch_agent, summarizer_agent, external_knowledge_agent, memory)

# Initialize Flask app
app = Flask(__name__)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    logging.debug(f"Received query request with data: {data}")
    query = data.get('query')
    context = data.get('context', "")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Pass the query through the orchestrator
    response, source = orchestration_agent.allocate_agent(query)

    logging.debug(f"Response: {response}, Source: {source}")

    return jsonify({"response": response, "source": source})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)