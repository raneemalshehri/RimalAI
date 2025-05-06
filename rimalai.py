import os
import json
import pandas as pd
from dotenv import load_dotenv
import langid
import tempfile
import re
import ast
import speech_recognition as sr
from gtts import gTTS
from IPython.display import Audio, display, HTML
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import folium
from mapboxgl.utils import create_color_stops, df_to_geojson
from mapboxgl.viz import CircleViz

# Initialize OpenAI client
client = OpenAI()

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")

# --- Enhanced Data Loading and Preparation ---
def load_data():
    """Load and preprocess datasets with error handling"""
    try:
        # Load Rimal dataset
        with open(r"/Users/raneem/Desktop/RIMAL-AI/Rimal_AI_dataset.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        rimal_df = pd.json_normalize(data) if isinstance(data, dict) else pd.DataFrame(data)
        
        # Load Arabic poems dataset
        with open(r"/Users/raneem/Desktop/RIMAL-AI/arabic_poems_dataset.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        poems_df = pd.json_normalize(data) if isinstance(data, dict) else pd.DataFrame(data)
        
        return rimal_df, poems_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

rimal_df, poems_df = load_data()

# --- Improved Vector Store Creation ---
def create_vector_stores(rimal_df, poems_df):
    """Create FAISS vector stores with enhanced metadata handling"""
    def combine_rimal_fields(row):
        """Enhanced field combination with error handling"""
        try:
            parts = []
            for field in ['type', 'name', 'description', 'vision2030']:
                val = row.get(field) if isinstance(row, dict) else getattr(row, field, None)
                if pd.notnull(val) and not isinstance(val, (float, int)):
                    parts.append(str(val))
            
            # Handle nested fields more robustly
            for list_field in ['notable_sites', 'activities', 'sustainable_initiatives']:
                items = row.get(list_field, []) if isinstance(row, dict) else getattr(row, list_field, [])
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            name = item.get('name') or item.get('project_name') or item.get('project', '')
                            desc = item.get('description', '')
                            if name or desc:
                                parts.append(f"{list_field[:-1].replace('_',' ').title()}: {name}. {desc}")
            
            # Handle values and beliefs
            values_beliefs = row.get('values_and_beliefs', {}) if isinstance(row, dict) else getattr(row, 'values_and_beliefs', {})
            if isinstance(values_beliefs, dict):
                values = values_beliefs.get('values', [])
                if isinstance(values, list):
                    parts.append("Values: " + ", ".join(str(v) for v in values))
            
            # Handle rituals
            rituals = row.get('rituals', []) if isinstance(row, dict) else getattr(row, 'rituals', [])
            if isinstance(rituals, list):
                parts.append("Rituals: " + ", ".join(str(r) for r in rituals))
            
            # Handle media
            media = row.get('media', {}) if isinstance(row, dict) else getattr(row, 'media', {})
            if isinstance(media, str):
                try:
                    media = json.loads(media)
                except:
                    media = {}
            
            for media_type in ['images', 'videos', 'audio']:
                urls = media.get(media_type, [])
                if isinstance(urls, list) and urls:
                    parts.append(f"{media_type.title()}: {', '.join(str(url) for url in urls[:3])}")
            
            return " | ".join(parts) if parts else ""
        except Exception as e:
            print(f"Error processing row: {e}")
            return ""

    try:
        # Create embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Process Rimal data
        rimal_texts = rimal_df.apply(combine_rimal_fields, axis=1).tolist()
        rimal_metadatas = rimal_df.to_dict(orient="records")
        
        # Process Poems data
        poems_df['combined'] = poems_df.apply(
            lambda x: f"{x.get('title', '')} {x.get('poet', '')} {x.get('text', '')} {x.get('translation', '')}", 
            axis=1
        )
        poems_texts = poems_df['combined'].tolist()
        poems_metadatas = poems_df.to_dict(orient="records")
        
        # Create and save FAISS stores
        rimal_vectordb = FAISS.from_texts(rimal_texts, embedding_model, metadatas=rimal_metadatas)
        rimal_vectordb.save_local("rimalai_faiss")
        
        poems_vectordb = FAISS.from_texts(poems_texts, embedding_model, metadatas=poems_metadatas)
        poems_vectordb.save_local("poem_faiss")
        
        return rimal_vectordb, poems_vectordb
    except Exception as e:
        print(f"Error creating vector stores: {e}")
        return None, None

rimal_vectordb, poems_vectordb = create_vector_stores(rimal_df, poems_df)

# --- Enhanced Map Visualization ---
def extract_coordinates_from_metadata(metadata_list):
    """Improved coordinate extraction with validation"""
    coordinates = []
    for doc in metadata_list:
        try:
            coords = doc.get('coordinates')
            if not coords:
                continue
                
            if isinstance(coords, str):
                coords = ast.literal_eval(coords)
            
            lat = coords.get('latitude')
            lon = coords.get('longitude')
            if lat is None or lon is None:
                continue
                
            coordinates.append({
                'name': doc.get('name', 'Unknown Location'),
                'description': (doc.get('description', '')[:150] + '...') if doc.get('description') else '',
                'type': doc.get('type', 'Landmark'),
                'latitude': float(lat),
                'longitude': float(lon)
            })
        except Exception as e:
            continue
    return coordinates

def create_map(locations):
    """Create interactive map with enhanced features"""
    if not locations:
        return None
    
    # Center map on Saudi Arabia with better default zoom
    m = folium.Map(location=[23.8859, 45.0792], zoom_start=6, tiles='cartodbpositron')
    
    # Add markers with custom icons
    for loc in locations:
        popup_text = f"""
        <b>{loc['name']}</b><br>
        <i>Type: {loc['type']}</i><br>
        {loc['description']}
        """
        
        # Different icons for different types
        icon_color = 'red'
        if 'heritage' in loc['type'].lower():
            icon_color = 'green'
        elif 'cultural' in loc['type'].lower():
            icon_color = 'blue'
            
        folium.Marker(
            [loc['latitude'], loc['longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=loc['name'],
            icon=folium.Icon(color=icon_color, icon='info-sign')
        ).add_to(m)
    
    # Add measure control
    folium.plugins.MeasureControl().add_to(m)
    
    return m._repr_html_()

# --- Modernized Tools Setup ---
def setup_tools():
    """Initialize tools with updated OpenAI API usage"""
    try:
        # Initialize LLM with modern OpenAI client
        llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Enhanced QA Chain for Saudi culture
        rimal_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            As an expert on Saudi Arabian culture, landmarks, and Vision 2030, provide detailed answers.
            Always respond in the same language as the question.
            Include geographical info when available and format responses clearly.
            
            Context: {context}
            
            Question: {question}
            
            Detailed Answer:
            """
        )
        
        rimal_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=rimal_vectordb.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": rimal_prompt}
        )
        
        def query_with_media_and_map(query: str):
            """Enhanced query function with better media handling"""
            try:
                response = rimal_qa({"query": query})
                answer = response["result"]
                
                # Process source documents for media and locations
                locations = extract_coordinates_from_metadata(
                    [doc.metadata for doc in response["source_documents"]]
                )
                map_html = create_map(locations) if locations else None
                
                # Collect unique media items
                media_items = []
                seen_media = set()
                
                for doc in response["source_documents"]:
                    media = doc.metadata.get("media", {})
                    if isinstance(media, str):
                        try:
                            media = json.loads(media)
                        except:
                            media = {}
                    
                    # Process images
                    for img in media.get("images", [])[:3]:  # Limit to 3 images per doc
                        if img and img not in seen_media:
                            media_items.append({"type": "image", "url": img})
                            seen_media.add(img)
                    
                    # Process videos
                    for vid in media.get("videos", [])[:1]:  # Limit to 1 video per doc
                        if vid and vid not in seen_media:
                            media_items.append({"type": "video", "url": vid})
                            seen_media.add(vid)
                
                return {
                    "answer": answer,
                    "media": media_items,
                    "map": map_html,
                    "sources": [doc.metadata.get("name", "") for doc in response["source_documents"]]
                }
            except Exception as e:
                return {"error": f"Failed to process query: {str(e)}"}
        
        # Modernized Arabic poetry QA
        poems_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            As an Arabic poetry expert, answer questions about poems and poets.
            Match the question's language (Arabic/English) in your response.
            Include poem text and translation when relevant.
            
            Context: {context}
            
            Question: {question}
            
            Expert Answer:
            """
        )
        
        poems_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=poems_vectordb.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": poems_prompt}
        )
        
        # Updated Arabic culture poem generator
        def generate_arabic_poem(prompt):
            """Generate poem using modern OpenAI API"""
            try:
                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": "You're a professional Arabic poet. Write authentic Arabic poems in traditional styles."
                        },
                        {
                            "role": "user",
                            "content": f"Write an Arabic poem about: {prompt}\nInclude an English translation labeled 'Translation:'"
                        }
                    ],
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error generating poem: {str(e)}"
        
        # Modernized image generation
        def generate_image(prompt, n=1, size="1024x1024"):
            """Generate images using DALL-E 3"""
            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=n,
                    size=size,
                    quality="standard"
                )
                return [img.url for img in response.data]
            except Exception as e:
                print(f"Image generation error: {e}")
                return []
        
        # Enhanced text-to-speech with language detection
        def tts_tool_func(text: str) -> str:
            """Convert text to speech with improved language handling"""
            try:
                lang = langid.classify(text)[0]
                supported_langs = {'ar': 'ar', 'en': 'en'}
                lang_code = supported_langs.get(lang, 'en')
                
                tts = gTTS(text, lang=lang_code)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    return fp.name
            except Exception as e:
                print(f"TTS error: {e}")
                return ""
        
        # Enhanced location visualization
        def location_visualization(query=None):
            """Improved location finder with fuzzy matching"""
            try:
                locations = extract_coordinates_from_metadata(rimal_df.to_dict('records'))
                
                if query:
                    # Case-insensitive search across multiple fields
                    query_lower = query.lower()
                    filtered = [
                        loc for loc in locations 
                        if (query_lower in loc['name'].lower() or 
                            query_lower in loc['description'].lower() or
                            query_lower in loc['type'].lower())
                    ]
                    locations = filtered or locations  # Fallback to all if no matches
                
                map_html = create_map(locations)
                if not map_html:
                    return {"message": "No location data available"}
                
                return {
                    "message": f"Found {len(locations)} locations" + (f" matching '{query}'" if query else ""),
                    "map": map_html,
                    "locations": locations[:10]  # Limit to 10 results
                }
            except Exception as e:
                return {"error": f"Location search failed: {str(e)}"}
        
        # Define tools with enhanced descriptions
        tools = [
            Tool(
                name="SaudiCultureExpert",
                func=query_with_media_and_map,
                description=(
                    "Answers questions about Saudi landmarks, culture and Vision 2030. "
                    "Returns detailed answer with media links and map when available. "
                    "Input should be a clear question about Saudi Arabia."
                )
            ),
            Tool(
                name="ArabicPoetryExpert",
                func=poems_qa.run,
                description=(
                    "Answers questions about Arabic poetry including poems, poets and translations. "
                    "Maintains the question's language (Arabic/English) in responses."
                )
            ),
            Tool(
                name="ArabicPoemGenerator",
                func=generate_arabic_poem,
                description=(
                    "Generates original Arabic poems with English translations. "
                    "Input should be a topic or theme for the poem."
                )
            ),
            Tool(
                name="TextToSpeech",
                func=tts_tool_func,
                description=(
                    "Converts text to speech in Arabic or English. "
                    "Returns path to MP3 file. Input should be text to speak."
                )
            ),
            Tool(
                name="LocationFinder",
                func=location_visualization,
                description=(
                    "Finds and displays Saudi locations on an interactive map. "
                    "Optional input can filter locations by name, type or description."
                )
            )
        ]
        
        return tools, llm
    except Exception as e:
        print(f"Failed to initialize tools: {e}")
        return [], None

tools, llm = setup_tools()

# --- Modern Agent Initialization ---
try:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
except Exception as e:
    print(f"Failed to initialize agent: {e}")
    agent = None

# --- Enhanced Voice Interaction ---
def record_audio(duration=5):
    """Record audio with improved error handling"""
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening... (Speak now)")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=duration)
            return audio
    except sr.WaitTimeoutError:
        print("Listening timed out")
        return None
    except Exception as e:
        print(f"Recording error: {e}")
        return None

def transcribe_audio(audio):
    """Transcribe using Whisper with modern API"""
    try:
        # Get audio data as bytes
        audio_data = audio.get_wav_data()
        with tempfile.NamedTemporaryFile(suffix=".wav") as fp:
            fp.write(audio_data)
            fp.seek(0)
            audio_file = open(fp.name, "rb")
        
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1"
        )
        audio_file.close()
        return transcript.text
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

def voice_query_agent(agent, max_attempts=3):
    """Enhanced voice interaction with retries"""
    for attempt in range(max_attempts):
        audio = record_audio()
        if not audio:
            if attempt < max_attempts - 1:
                print("Please try speaking again")
                continue
            return None
        
        transcript = transcribe_audio(audio)
        if not transcript:
            if attempt < max_attempts - 1:
                print("Couldn't understand. Please try again")
                continue
            return None
        
        print(f"\nYou said: {transcript}")
        return process_query(agent, transcript, is_voice=True)
    
    return None

# --- Improved Text Interaction ---
def text_query_agent(agent):
    """Enhanced text interaction with input validation"""
    while True:
        user_input = input("\nYour question (or 'back' to cancel): ").strip()
        if user_input.lower() == 'back':
            return None
        if user_input:
            return process_query(agent, user_input, is_voice=False)
        print("Please enter a question")

def process_query(agent, query, is_voice=False):
    """Process queries with better response handling"""
    try:
        if not agent:
            print("Agent not available")
            return None
        
        result = agent.run(query)
        
        # Handle different response types
        if isinstance(result, dict):
            # Structured response from tools
            response_text = result.get("answer", result.get("message", "Here's what I found:"))
            print(f"\n{response_text}")
            
            # Display map if available
            if result.get("map"):
                display(HTML(result["map"]))
            
            # Play audio response for voice queries
            if is_voice and response_text:
                play_audio_response(response_text)
            
            # Show media if available
            if result.get("media"):
                print("\nRelated Media:")
                for item in result["media"][:3]:  # Limit to 3 items
                    print(f"- {item['type'].title()}: {item['url']}")
        
        else:
            # Simple text response
            print(f"\n{result}")
            if is_voice:
                play_audio_response(str(result))
        
        return result
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        if is_voice:
            play_audio_response("Sorry, I encountered an error processing your request")
        return None

def play_audio_response(text):
    """Play audio response with improved error handling"""
    try:
        lang = langid.classify(text)[0]
        lang_code = 'ar' if lang == 'ar' else 'en'
        
        tts = gTTS(text, lang=lang_code)
        with tempfile.NamedTemporaryFile(suffix=".mp3") as fp:
            tts.save(fp.name)
            print("\nPlaying response...")
            display(Audio(fp.name, autoplay=True))
    except Exception as e:
        print(f"Failed to play audio: {e}")

# --- Modern Main Interaction Loop ---
def main():
    """Enhanced main interaction loop with better UX"""
    print("\n" + "="*50)
    print("🌟 Welcome to RimalAI - Your Saudi Culture Assistant")
    print("="*50)
    print("\nFeatures:")
    print("- Ask about Saudi landmarks, culture & Vision 2030")
    print("- Get information about Arabic poetry")
    print("- Generate original Arabic poems with translations")
    print("- Find locations on interactive maps")
    print("- Voice and text interaction modes")
    
    if not agent:
        print("\n⚠️ Initialization failed. Some features may not work.")
    
    while True:
        print("\nOptions:")
        print("v - Voice interaction")
        print("t - Text interaction")
        print("q - Quit")
        
        choice = input("\nChoose mode (v/t/q): ").strip().lower()
        
        if choice == 'q':
            print("\nThank you for using RimalAI! Goodbye.")
            break
        elif choice == 'v':
            print("\nEntering voice mode...")
            voice_query_agent(agent)
        elif choice == 't':
            print("\nEntering text mode...")
            text_query_agent(agent)
        else:
            print("Invalid choice. Please select v, t, or q.")

if __name__ == "__main__":
    main()