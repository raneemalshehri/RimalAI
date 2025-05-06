import mapboxgl
from mapboxgl.utils import create_color_stops, df_to_geojson
from mapboxgl.viz import CircleViz
import streamlit as st
from streamlit_chat import message
import os
import json
import pandas as pd
from dotenv import load_dotenv
import langid
from io import BytesIO
import base64
import requests
import asyncio
import warnings
from openai import OpenAI
from gtts import gTTS
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")

# Constants
DEFAULT_LOCATIONS = {
    "Al-Ula": {"latitude": 26.6141, "longitude": 37.9219},
    "Riyadh": {"latitude": 24.7136, "longitude": 46.6753},
    "Jeddah": {"latitude": 21.5433, "longitude": 39.1728}
}

# Session State Management
def initialize_session_state():
    defaults = {
        'generated': [],
        'past': [],
        'messages': [
            {"role": "system", "content": "You're a Saudi culture and Arabic poetry expert."}
        ],
        'audio_file': None,
        'map_html': None,
        'media_items': [],
        'generated_images': [],
        'mode': 'Text'
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# Data Loading
@st.cache_resource
def load_data():
    try:
        # Load Rimal dataset
        with open("Rimal_AI_dataset.json", 'r', encoding='utf-8') as f:
            rimal_data = json.load(f)
        rimal_df = pd.json_normalize(rimal_data)
        
        # Load Arabic poems dataset
        with open("arabic_poems_dataset.json", 'r', encoding='utf-8') as f:
            poems_data = json.load(f)
        poems_df = pd.json_normalize(poems_data)
        
        # Create embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Process and create Rimal vector store
        rimal_texts = rimal_df.apply(lambda x: " | ".join([
            str(x.get(field, "")) for field in ['type', 'name', 'description', 'vision2030'] 
            if pd.notnull(x.get(field, ""))
        ]), axis=1).tolist()
        rimal_vectordb = FAISS.from_texts(rimal_texts, embedding_model, metadatas=rimal_df.to_dict(orient="records"))
        
        # Process and create Poems vector store
        poems_df['combined'] = poems_df.apply(
            lambda x: f"{x.get('title', '')} {x.get('poet', '')} {x.get('text', '')} {x.get('translation', '')}", 
            axis=1
        )
        poems_vectordb = FAISS.from_texts(
            poems_df['combined'].tolist(), 
            embedding_model, 
            metadatas=poems_df.to_dict(orient="records")
        )
        
        return rimal_vectordb, poems_vectordb
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Geocoding Service
def geocode_location(location_name):
    """Unified location lookup with fallbacks"""
    if location_name in DEFAULT_LOCATIONS:
        return {
            'name': location_name,
            **DEFAULT_LOCATIONS[location_name],
            'place_name': f"{location_name}, Saudi Arabia"
        }
    
    if not MAPBOX_API_KEY:
        return None
        
    try:
        response = requests.get(
            f"https://api.mapbox.com/geocoding/v5/mapbox.places/{location_name}.json",
            params={
                'access_token': MAPBOX_API_KEY,
                'country': 'sa',
                'types': 'poi,place',
                'limit': 1
            }
        )
        response.raise_for_status()
        data = response.json()
        
        if data['features']:
            feature = data['features'][0]
            return {
                'name': feature.get('text', location_name),
                'latitude': feature['center'][1],
                'longitude': feature['center'][0],
                'place_name': feature.get('place_name', location_name)
            }
    except Exception:
        return None

# Visualization
def create_map_viz(locations):
    """Create map visualization for any valid locations"""
    valid_locations = [loc for loc in locations if loc]
    if not valid_locations:
        return None
    
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [loc['longitude'], loc['latitude']]
                },
                "properties": {
                    "name": loc['name'],
                    "description": loc.get('place_name', '')
                }
            } for loc in valid_locations
        ]
    }
    
    center = {
        'lon': sum(loc['longitude'] for loc in valid_locations)/len(valid_locations),
        'lat': sum(loc['latitude'] for loc in valid_locations)/len(valid_locations)
    }
    
    return CircleViz(
        access_token=MAPBOX_API_KEY,
        data=geojson,
        label_property="name",
        center=center,
        zoom=10 if len(valid_locations) == 1 else 7,
        radius=10,
        color="#FF6D00",
        stroke_width=2
    ).as_iframe(width=800, height=600)

def extract_media_from_qa(qa_result):
    """Extract media items from QA results"""
    media_items = []
    for doc in qa_result["source_documents"]:
        media = doc.metadata.get('media', {})
        if isinstance(media, str):
            try:
                media = json.loads(media)
            except:
                media = {}
        
        for img in media.get('images', [])[:3]:
            media_items.append({'type': 'image', 'url': img})
        for vid in media.get('videos', [])[:1]:
            media_items.append({'type': 'video', 'url': vid})
    
    return media_items[:3]

def extract_location_names(text):
    """Extract potential location names from text"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "Extract Saudi Arabian city or landmark names from the text. Return only the names separated by commas."},
                {"role": "user", "content": text}
            ],
            max_tokens=50,
            temperature=0
        )
        names = response.choices[0].message.content.strip()
        return [name.strip() for name in names.split(",") if name.strip()]
    except Exception as e:
        print(f"Error extracting location names: {e}")
        return []

# Poem Generation with Image
def generate_poem_with_image(topic):
    """Generate poem and corresponding image"""
    # Generate poem
    poem_response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "Generate beautiful Arabic poems with English translations"},
            {"role": "user", "content": f"Create a poem about {topic} with Arabic text and English translation"}
        ],
        max_tokens=500
    )
    poem = poem_response.choices[0].message.content
    
    # Generate image
    image_response = client.images.generate(
        model="dall-e-3",
        prompt=f"Arabic poetry scene about {topic}: traditional setting with calligraphy, desert landscape, Arabic coffee cups, elegant and cultural, digital art",
        size="1024x1024",
        quality="standard",
        n=1
    )
    image_url = image_response.data[0].url
    
    return {
        "answer": poem,
        "image_url": image_url,
        "content_type": "poem_with_image"
    }



# Agent Initialization
@st.cache_resource(show_spinner="Initializing AI agent...")
def _initialize_agent_resources():
    """Initialize all resources needed for the agent (cached separately)"""
    rimal_vectordb, poems_vectordb = load_data()
    if not rimal_vectordb or not poems_vectordb:
        return None, None, None

    llm = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        temperature=0.3,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Initialize QA chains
    rimal_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=rimal_vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    poems_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=poems_vectordb.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )

    return llm, rimal_qa, poems_qa

def create_agent_tools(llm, rimal_qa, poems_qa):
    """Create tools list with improved response formatting"""
    return [
        Tool(
            name="SaudiCultureExpert",
            func=lambda q: str(rimal_qa.run(q)),  # Convert to string
            description="Answers about Saudi landmarks and culture"
        ),
        Tool(
            name="ArabicPoetryExpert",
            func=lambda q: str(poems_qa.run(q)),  # Convert to string
            description="Provides information about Arabic poetry"
        ),
        Tool(
            name="PoemGenerator",
            func=lambda q: generate_poem_with_image(q),
            description="Generates Arabic poems with English translations and images"
        ),
        Tool(
            name="LocationFinder",
            func=lambda q: {
                "locations": [geocode_location(name) for name in extract_location_names(q)],
                "query": q
            },
            description="Finds locations in Saudi Arabia"
        ),
        Tool(
            name="TextToSpeech",
            func=lambda t: (gTTS(t, lang='ar' if langid.classify(t)[0] == 'ar' else 'en')
                          .write_to_fp(audio := BytesIO()) or audio.seek(0) or audio),
            description="Converts text to speech in Arabic or English"
        )
    ]

def create_agent():
    """Main agent initialization function"""
    try:
        from langchain.agents import initialize_agent as langchain_initialize_agent
        
        llm, rimal_qa, poems_qa = _initialize_agent_resources()
        if not llm:
            return None

        tools = create_agent_tools(llm, rimal_qa, poems_qa)

        # Initialize the agent using the renamed function
        return langchain_initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            max_iterations=5,
            verbose=False
        )
    except Exception as e:
        st.error(f"Agent initialization failed: {str(e)}")
        return None
    
# Core Processing
def process_query(query, agent):
    """Unified query processing pipeline with improved response handling"""
    try:
        result = agent.run(query)  # Changed from invoke() to run()
        
        # Handle different response formats
        if isinstance(result, dict):
            # Handle map responses
            if result.get("locations"):
                if map_html := create_map_viz(result["locations"]):
                    st.session_state['map_html'] = map_html
            
            # Handle media responses
            if result.get("media"):
                st.session_state['media_items'] = result["media"]
                
            # Handle generated images
            if result.get("image_url"):
                st.session_state['generated_images'] = [result["image_url"]]
            
            # Return the actual answer or default message
            response = result.get("answer", "Here's what I found:")
            
        elif isinstance(result, str):
            response = result
        else:
            response = "Here's what I found: " + str(result)
        
        # Generate audio for voice mode
        if st.session_state.get('mode') == "Voice":
            audio = BytesIO()
            lang = 'ar' if langid.classify(response)[0] == 'ar' else 'en'
            gTTS(response, lang=lang).write_to_fp(audio)
            audio.seek(0)
            st.session_state['audio_file'] = audio
        
        return response
        
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Main Application
def main():
    initialize_session_state()
    
    # Initialize agent using the new function name
    agent = create_agent()
    if not agent:
        st.error("Failed to initialize agent. Please check your configuration.")
        return
  
    # Sidebar
    with st.sidebar:
        st.title("RimalAI")
        st.markdown("""
        Saudi Culture & Poetry Assistant
        - Landmark information
        - Poetry generation
        - Location mapping
        """)
        st.session_state['mode'] = st.radio(
            "Input Mode:", 
            ("Text", "Voice"), 
            horizontal=True
        )
    
    # Main Interface
    st.title("🌍 RimalAI Cultural Assistant")
    
    # Voice Input
    if st.session_state['mode'] == "Voice" and st.button("🎤 Start Recording"):
        with st.spinner("Listening..."):
            try:
                import speech_recognition as sr
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    audio = r.listen(source, timeout=5)
                    query = r.recognize_google(audio)
                    st.session_state['past'].append(f"🎤: {query}")
                    response = process_query(query, agent)
                    st.session_state['generated'].append(response)
            except Exception as e:
                st.error(f"Voice processing error: {str(e)}")
    
    # Text Input
    if user_input := st.chat_input("Ask about Saudi culture..."):
        st.session_state['past'].append(user_input)
        response = process_query(user_input, agent)
        st.session_state['generated'].append(response)
    
    # Display Conversation
    for i, (user_msg, bot_msg) in enumerate(zip(st.session_state['past'], st.session_state['generated'])):
        message(user_msg, is_user=True, key=f"{i}_user")
        message(bot_msg, key=f"{i}")
        
        # Play audio for last voice response
        if i == len(st.session_state['past']) - 1 and st.session_state.get('audio_file'):
            audio_base64 = base64.b64encode(st.session_state['audio_file'].read()).decode()
            st.components.v1.html(f"""
                <audio controls autoplay>
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                </audio>
            """, height=50)
    
    # Display Map
    if st.session_state.get('map_html'):
        with st.expander("📍 Location Map", expanded=True):
            st.components.v1.html(st.session_state['map_html'], height=600)
    
    # Display Media
    if st.session_state.get('media_items'):
        with st.expander("🖼️ Related Media", expanded=True):
            cols = st.columns(min(3, len(st.session_state['media_items'])))
            for i, item in enumerate(st.session_state['media_items'][:3]):
                with cols[i]:
                    st.image(item['url']) if item['type'] == 'image' else st.video(item['url'])
    
    # Display Generated Images
    if st.session_state.get('generated_images'):
        with st.expander("🎨 Generated Art", expanded=True):
            for img_url in st.session_state['generated_images']:
                st.image(img_url, use_column_width=True)

if __name__ == "__main__":
    # Create a new event loop for Streamlit
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main()