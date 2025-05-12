import os
import base64 
import json
import tempfile
import streamlit as st
from gtts import gTTS
from openai import OpenAI
import speech_recognition as sr
from langdetect import detect
from PIL import Image
import requests
from io import BytesIO
import folium
from streamlit_folium import folium_static
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="RimalAI - Saudi Culture Assistant",
    page_icon="🐪🏜️",
    layout="wide"
)

# Load environment variables
load_dotenv(override=True)

# Initialize OpenAI client
client = OpenAI()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")

# Verify API keys
if not OPENAI_API_KEY:
    st.error("OpenAI key missing from .env file")
    st.stop()

if not MAPBOX_API_KEY:
    st.error("Mapbox key missing from .env file")
    st.stop()

# Initialize LangChain components
@st.cache_resource
def initialize_agent_resources():
    try:
        # Load datasets
        with open("Rimal_AI_dataset.json", 'r', encoding='utf-8') as f:
            rimal_data = json.load(f)
        rimal_df = pd.json_normalize(rimal_data)
        
        with open("arabic_poems_dataset.json", 'r', encoding='utf-8') as f:
            poems_data = json.load(f)
        poems_df = pd.json_normalize(poems_data)
        
        # Create embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create vector stores
        rimal_texts = rimal_df.apply(lambda x: " | ".join([
            str(x.get(field, "")) for field in ['type', 'name', 'description', 'vision2030'] 
            if pd.notnull(x.get(field, ""))
        ]), axis=1).tolist()
        rimal_vectordb = FAISS.from_texts(rimal_texts, embedding_model, metadatas=rimal_df.to_dict(orient="records"))
        
        poems_df['combined'] = poems_df.apply(
            lambda x: f"{x.get('title', '')} {x.get('poet', '')} {x.get('text', '')} {x.get('translation', '')}", 
            axis=1
        )
        poems_vectordb = FAISS.from_texts(
            poems_df['combined'].tolist(), 
            embedding_model, 
            metadatas=poems_df.to_dict(orient="records")
        )
        
        # Initialize LLM
        llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Create QA chains
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
        
        # Create tools with proper output handling
        tools = [
            Tool(
                name="SaudiCultureExpert",
                func=lambda q: {
                    "answer": rimal_qa({"query": q})["result"],
                    "source_documents": rimal_qa({"query": q})["source_documents"]
                },
                description="Answers about Saudi landmarks and culture"
            ),
            Tool(
                name="ArabicPoetryExpert",
                func=lambda q: {
                    "answer": poems_qa({"query": q})["result"],
                    "source_documents": poems_qa({"query": q})["source_documents"]
                },
                description="Provides information about Arabic poetry"
            ),
            Tool(
                name="PoemGenerator",
                func=lambda q: generate_arabic_poem(q),
                description="Generates Arabic poems with English translations"
            ),
            Tool(
                name="LocationFinder",
                func=lambda q: {
                    "locations": [get_location_coordinates(name) for name in extract_location_names(q)],
                    "query": q
                },
                description="Finds locations in Saudi Arabia"
            )
        ]
        
        # Initialize agent
        return initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            max_iterations=5,
            verbose=False
        )
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

# Initialize agent
agent = initialize_agent_resources()

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

def extract_location_from_query(query):
    """Extract potential location names from a query"""
    try:
        # Common Saudi cities and landmarks
        known_locations = [
            "Riyadh", "Jeddah", "Mecca", "Medina", "Dammam", "Khobar", "Dhahran",
            "Al-Ula", "Neom", "Abha", "Taif", "Jubail", "Yanbu", "Tabuk",
            "Mada'in Saleh", "Al-Ahsa", "Al-Khobar", "Al-Khafji", "Al-Qassim",
            "Al-Baha", "Al-Jouf", "Al-Madinah", "Al-Qunfudhah", "Al-Wajh",
            "Buraidah", "Hail", "Jizan", "Khamis Mushait", "Najran", "Sakakah"
        ]
        
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Check for exact matches first
        for location in known_locations:
            if location.lower() in query_lower:
                return location
        
        # If no exact match, try to extract using GPT
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "Extract a Saudi Arabian city or landmark name from the text. Return only the name, nothing else. If no clear location is mentioned, return 'none'."},
                {"role": "user", "content": query}
            ],
            max_tokens=50,
            temperature=0
        )
        
        extracted_location = response.choices[0].message.content.strip()
        if extracted_location.lower() != 'none':
            return extracted_location
            
        return None
    except Exception as e:
        st.error(f"Error extracting location: {e}")
        return None

def create_map(location_data):
    """Create an interactive map centered on the location with enhanced styling"""
    if not location_data:
        return None
    
    # Create base map with a nicer style
    m = folium.Map(
        location=[location_data['latitude'], location_data['longitude']],
        zoom_start=12,
        tiles="https://api.mapbox.com/styles/v1/mapbox/light-v11/tiles/{z}/{x}/{y}?access_token=" + MAPBOX_API_KEY,
        attr='Mapbox'
    )
    
    # Add satellite layer as an option
    folium.TileLayer(
        tiles="https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}?access_token=" + MAPBOX_API_KEY,
        attr='Mapbox Satellite',
        name='Satellite View'
    ).add_to(m)
    
    # Custom icon for landmarks
    landmark_icon = folium.Icon(
        color='red',
        icon='landmark',
        prefix='fa',
        icon_color='white'
    )
    
    # Add marker with custom styling
    folium.Marker(
        [location_data['latitude'], location_data['longitude']],
        popup=folium.Popup(
            f"""
            <div style='font-family: Arial, sans-serif; padding: 10px;'>
                <h3 style='color: #2c3e50; margin: 0 0 10px 0;'>{location_data['name']}</h3>
                <p style='color: #34495e; margin: 0;'>{location_data['full_name']}</p>
            </div>
            """,
            max_width=300
        ),
        tooltip=location_data['name'],
        icon=landmark_icon
    ).add_to(m)
    
    # Add a circle to highlight the area
    folium.Circle(
        location=[location_data['latitude'], location_data['longitude']],
        radius=1000,  # 1km radius
        color='#e74c3c',
        fill=True,
        fill_color='#e74c3c',
        fill_opacity=0.1
    ).add_to(m)
    
    # Add measure control
    folium.plugins.MeasureControl(
        position='topleft',
        primary_length_unit='kilometers',
        secondary_length_unit='miles',
        primary_area_unit='square kilometers',
        secondary_area_unit='acres'
    ).add_to(m)
    
    # Add fullscreen control
    folium.plugins.Fullscreen().add_to(m)
    
    # Add minimap
    folium.plugins.MiniMap().add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m._repr_html_()

def display_location_info(location_data):
    """Display location information in a nicely formatted panel"""
    if not location_data:
        return
    
    with st.sidebar:
        st.markdown("""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h2 style='color: #2c3e50; margin-bottom: 15px;'>📍 Location Information</h2>
        """, unsafe_allow_html=True)
        
        # Display location name and type
        st.markdown(f"### {location_data['name']}")
        st.markdown(f"*{location_data.get('type', 'Landmark').title()}*")
        
        # Display description if available
        if location_data.get('description'):
            st.markdown("---")
            st.markdown("#### About")
            st.markdown(location_data['description'])
        
        # Display additional information if available
        if 'climate' in location_data:
            st.markdown("---")
            st.markdown("#### Climate")
            climate = location_data['climate']
            if isinstance(climate, dict):
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Summer:**")
                    st.markdown(climate.get('summer', 'N/A'))
                with cols[1]:
                    st.markdown("**Winter:**")
                    st.markdown(climate.get('winter', 'N/A'))
        
        # Display notable sites if available
        if 'notable_sites' in location_data:
            st.markdown("---")
            st.markdown("#### Notable Sites")
            for site in location_data['notable_sites']:
                with st.expander(site.get('name', 'Site')):
                    st.markdown(site.get('description', 'No description available'))
        
        st.markdown("</div>", unsafe_allow_html=True)

def search_poems(query):
    """Search for poems in the dataset"""
    try:
        with open("arabic_poems_dataset.json", 'r', encoding='utf-8') as f:
            poems_data = json.load(f)
        
        # Convert query to lowercase for case-insensitive search
        query_lower = query.lower()
        matching_poems = []
        
        for poem in poems_data:
            # Search in title, poet, and text
            if (query_lower in poem.get('title', '').lower() or
                query_lower in poem.get('poet', '').lower() or
                query_lower in poem.get('text', '').lower() or
                query_lower in poem.get('translation', '').lower()):
                matching_poems.append(poem)
        
        return matching_poems
    except Exception as e:
        st.error(f"Error searching poems: {str(e)}")
        return []

def process_query(query, agent):
    """Process user query using the agent"""
    try:
        if not agent:
            return "Error: Agent not initialized properly."
        
        # First try to extract location from query
        location_name = extract_location_from_query(query)
        if location_name:
            # Try to find location in our dataset first for media
            with open("Rimal_AI_dataset.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                location_data = None
                media_items = []
                
                # Find location and its media in dataset
                for item in data:
                    if location_name.lower() in item.get('name', '').lower():
                        location_data = {
                            'name': item['name'],
                            'longitude': item['coordinates']['longitude'],
                            'latitude': item['coordinates']['latitude'],
                            'full_name': f"{item['name']}, Saudi Arabia",
                            'type': item.get('type', 'landmark'),
                            'description': item.get('description', ''),
                            'climate': item.get('climate'),
                            'notable_sites': item.get('notable_sites', [])
                        }
                        
                        # Extract media items
                        media = item.get('media', {})
                        if isinstance(media, str):
                            try:
                                media = json.loads(media)
                            except:
                                media = {}
                        
                        # Add images
                        for img in media.get('images', [])[:3]:
                            if img:
                                media_items.append({'type': 'image', 'url': img})
                        
                        # Add videos
                        for vid in media.get('videos', [])[:1]:
                            if vid:
                                media_items.append({'type': 'video', 'url': vid})
                        
                        break
                
                # If not found in dataset, try Mapbox
                if not location_data:
                    location_data = get_location_coordinates(location_name)
                
                if location_data:
                    # Create map for the location
                    map_html = create_map(location_data)
                    if map_html:
                        st.session_state['map_html'] = map_html
                    
                    # Display location information
                    display_location_info(location_data)
                    
                    # Store media items in session state
                    if media_items:
                        st.session_state['media_items'] = media_items
                    
                    # Create response with location info and media
                    response = f"I found information about {location_name}."
                    if location_data.get('description'):
                        response += f"\n\n{location_data['description']}"
                    
                    return response
        
        # Check if this is a poetry-related query
        poetry_keywords = ['poem', 'poetry', 'poet', 'verse', 'قصيدة', 'شعر', 'شاعر']
        if any(keyword in query.lower() for keyword in poetry_keywords):
            # Search for poems
            matching_poems = search_poems(query)
            if matching_poems:
                response = "I found some poems that match your query:\n\n"
                for poem in matching_poems[:3]:  # Show top 3 matches
                    response += f"**{poem.get('title', 'Untitled')}** by {poem.get('poet', 'Unknown')}\n"
                    response += f"{poem.get('text', '')}\n\n"
                    response += f"*Translation:*\n{poem.get('translation', '')}\n\n---\n\n"
                return response
        
        # If no location found or location search failed, proceed with normal query processing
        result = agent.run(query)
        
        # Handle different response formats
        if isinstance(result, dict):
            # Handle map responses
            if result.get("locations"):
                if map_html := create_map(result["locations"][0]):
                    st.session_state['map_html'] = map_html
            
            # Handle media responses
            if result.get("source_documents"):
                media_items = []
                for doc in result["source_documents"]:
                    # Extract media from metadata
                    media = doc.metadata.get('media', {})
                    if isinstance(media, str):
                        try:
                            media = json.loads(media)
                        except:
                            media = {}
                    
                    # Add images
                    for img in media.get('images', [])[:3]:
                        if img:
                            media_items.append({'type': 'image', 'url': img})
                    
                    # Add videos
                    for vid in media.get('videos', [])[:1]:
                        if vid:
                            media_items.append({'type': 'video', 'url': vid})
                
                if media_items:
                    st.session_state['media_items'] = media_items
                
            # Handle generated images
            if result.get("image_url"):
                st.session_state['generated_images'] = [result["image_url"]]
            
            # Return the actual answer or default message
            response = result.get("answer", "Here's what I found:")
            
            # Add source information if available
            if result.get("source_documents"):
                sources = [doc.metadata.get('name', '') for doc in result["source_documents"] if doc.metadata.get('name')]
                if sources:
                    response += "\n\nSources: " + ", ".join(sources)
            
        elif isinstance(result, str):
            response = result
        else:
            response = "Here's what I found: " + str(result)
        
        return response
        
    except Exception as e:
        return f"Error processing query: {str(e)}"
    
    import base64
    
def load_css_with_background():
    # Read SVG and convert to base64
    with open("RimalAI.svg", "rb") as f:
        b64_svg = base64.b64encode(f.read()).decode()
    
    # Read CSS and replace placeholder
    with open("style.css", "r") as f:
        css = f.read().replace("{{b64_svg}}", b64_svg)
    
    # Inject CSS
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Call this after imports
load_css_with_background()

# --- App Header ---
st.title("🐪🏜️ RimalAI - Saudi Culture Assistant")
st.markdown("""
    <div style='background-color:#f0f2f6;padding:20px;border-radius:10px;margin-bottom:20px;'>
        <h3 style='color:#333;'>Explore Saudi Arabia's Rich Culture</h3>
        <p style='color:#555;'>
            Ask about landmarks, generate Arabic poetry with translations, visualize locations on maps, 
            and experience Saudi culture through AI.
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'map_html' not in st.session_state:
    st.session_state.map_html = None
if 'generated_poem' not in st.session_state:
    st.session_state.generated_poem = ""
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None

# --- Helper Functions ---
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'

def text_to_speech(text, lang='ar'):
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")
        return None

def record_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak now... (listening for 5 seconds)")
        audio = r.listen(source, timeout=5)
        return audio

def transcribe_audio(audio):
    try:
        # Save audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            fp.write(audio.get_wav_data())
            fp.flush()
            
            # Open the file in binary mode for the API
            with open(fp.name, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1"
                )
            return transcript.text
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None
    finally:
        # Clean up the temporary file
        if 'fp' in locals():
            os.unlink(fp.name)

def generate_arabic_poem(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You're a professional Arabic poet. Write authentic Arabic poems in traditional styles with English translations."},
                {"role": "user", "content": f"Write an Arabic poem about: {prompt}\n\nInclude:\n1. The Arabic poem with proper formatting\n2. An English translation labeled 'Translation:'\n3. Brief context about the poem's style and themes"}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Poem generation error: {e}")
        return None

def generate_image(prompt):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=f"High-quality image showing: {prompt}. Cultural, authentic, detailed.",
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url
    except Exception as e:
        st.error(f"Image generation error: {e}")
        return None

def get_location_coordinates(location_name):
    """Get coordinates for a location name using local dataset and Mapbox Geocoding API"""
    try:
        # First try to find in our local dataset
        with open("Rimal_AI_dataset.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Case-insensitive search in local dataset
        location_name_lower = location_name.lower()
        for item in data:
            # Only consider landmarks, historical places, and main cities
            if (item.get('type', '').lower() in ['landmark', 'historical', 'city'] and 
                location_name_lower in item.get('name', '').lower()):
                coords = item.get('coordinates', {})
                if coords and 'latitude' in coords and 'longitude' in coords:
                    return {
                        'name': item['name'],
                        'longitude': coords['longitude'],
                        'latitude': coords['latitude'],
                        'full_name': f"{item['name']}, Saudi Arabia",
                        'type': item.get('type', 'landmark'),
                        'description': item.get('description', '')
                    }
        
        # If not found in local dataset, try Mapbox with more specific parameters
        base_url = "https://api.mapbox.com/geocoding/v5/mapbox.places"
        endpoint = f"{base_url}/{location_name}.json"
        params = {
            "access_token": MAPBOX_API_KEY,
            "country": "SA",  # Limit to Saudi Arabia
            "types": "poi,place",  # Focus on points of interest and places
            "limit": 5,
            "language": "en"
        }
        
        response = requests.get(endpoint, params=params)
        data = response.json()
        
        if not data.get('features'):
            # Try a broader search without country restriction
            params.pop('country', None)
            response = requests.get(endpoint, params=params)
            data = response.json()
            
            if not data.get('features'):
                st.warning(f"Could not find exact location for '{location_name}'. Please try a different name or check the spelling.")
                return None
        
        # Find the best match by comparing relevance scores
        best_match = max(data['features'], key=lambda x: x.get('relevance', 0))
        coordinates = best_match['center']
        
        # Verify if the location is in Saudi Arabia
        is_in_saudi = False
        if 'context' in best_match:
            # Check if any context item has Saudi Arabia
            is_in_saudi = any(
                context.get('text', '').lower() == 'saudi arabia' or
                context.get('short_code', '').lower() == 'sa'
                for context in best_match['context']
            )
        
        if not is_in_saudi:
            st.warning(f"Found '{best_match['text']}' but it's not in Saudi Arabia. Please try a different location.")
            return None
        
        return {
            'name': best_match.get('text', location_name),
            'longitude': coordinates[0],
            'latitude': coordinates[1],
            'full_name': best_match['place_name'],
            'type': 'landmark',
            'description': best_match.get('text', '')
        }
    except Exception as e:
        st.error(f"Location lookup error: {str(e)}")
        return None

# --- Main App Sections ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🗣️ Voice Interaction", 
    "✍️ Text Interaction", 
    "🎭 Poem Generator", 
    "🗺️ Location Finder"
])

with tab1:
    st.header("Voice Interaction")
    st.markdown("Ask questions about Saudi culture using your voice")
    
    if st.button("🎤 Start Recording", key="voice_record"):
        audio = record_audio()
        if audio:
            transcript = transcribe_audio(audio)
            if transcript:
                st.session_state.voice_query = transcript
    
    if 'voice_query' in st.session_state:
        st.text_area("Your voice query:", value=st.session_state.voice_query, height=100)
        
        if st.button("🔍 Process Voice Query"):
            with st.spinner("Processing your query..."):
                # Process the query using the actual processing logic
                response = process_query(st.session_state.voice_query, agent)
                
                # Display response
                st.subheader("Response:")
                st.write(response)
                
                # Convert to speech if in Arabic
                lang = detect_language(response)
                audio_file = text_to_speech(response, lang='ar' if lang == 'ar' else 'en')
                if audio_file:
                    st.audio(audio_file, format='audio/mp3')
                    st.session_state.audio_file = audio_file

with tab2:
    st.header("Text Interaction")
    st.markdown("Type your questions about Saudi culture, landmarks, or Vision 2030")
    
    text_query = st.text_area("Your question:", height=100)
    
    if st.button("🔍 Submit Text Query"):
        with st.spinner("Researching your question..."):
            # Process the query using the actual processing logic
            response = process_query(text_query, agent)
            
            st.subheader("Answer:")
            st.write(response)
            
            # Check if response contains location names
            if any(word in response.lower() for word in ['riyadh', 'jeddah', 'mecca', 'medina', 'dammam']):
                location_name = next(word for word in ['Riyadh', 'Jeddah', 'Mecca', 'Medina', 'Dammam'] 
                                  if word.lower() in response.lower())
                location_data = get_location_coordinates(location_name)
                if location_data:
                    map_html = create_map(location_data)
                    st.subheader(f"Map of {location_name}")
                    folium_static(folium.Map(
                        location=[location_data['latitude'], location_data['longitude']],
                        zoom_start=12
                    ))

with tab3:
    st.header("Arabic Poem Generator")
    st.markdown("Generate authentic Arabic poems with English translations and matching images")
    
    poem_prompt = st.text_input("Enter a theme or topic for your poem:")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✍️ Generate Poem"):
            with st.spinner("Composing your poem..."):
                poem = generate_arabic_poem(poem_prompt)
                if poem:
                    st.session_state.generated_poem = poem
                    st.subheader("Generated Poem:")
                    st.write(poem)
                    
                    # Generate matching image
                    image_url = generate_image(f"Arabic cultural theme: {poem_prompt}")
                    if image_url:
                        image_response = requests.get(image_url)
                        st.session_state.generated_image = Image.open(BytesIO(image_response.content))
                        st.session_state.image_data = image_response.content
    
    with col2:
        if st.session_state.generated_poem:
            st.download_button(
                label="📥 Download Poem",
                data=st.session_state.generated_poem,
                file_name="arabic_poem.txt",
                mime="text/plain"
            )
            
            if st.session_state.generated_image:
                st.image(st.session_state.generated_image, caption="Generated Image for your poem")
                if hasattr(st.session_state, 'image_data'):
                    st.download_button(
                        label="📥 Download Image",
                        data=BytesIO(st.session_state.image_data),
                        file_name="poem_image.png",
                        mime="image/png"
                    )

with tab4:
    st.header("Location Finder")
    st.markdown("Find and visualize locations in Saudi Arabia")
    
    location_query = st.text_input("Enter a landmark or city in Saudi Arabia:")
    
    if st.button("📍 Find Location"):
        with st.spinner("Locating..."):
            location_data = get_location_coordinates(location_query)
            if location_data:
                st.success(f"Found: {location_data['full_name']}")
                
                # Create and display map
                map_html = create_map(location_data)
                folium_static(folium.Map(
                    location=[location_data['latitude'], location_data['longitude']],
                    zoom_start=12,
                    tiles="https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{z}/{x}/{y}?access_token=" + MAPBOX_API_KEY,
                    attr='Mapbox'
                ))
                
                # Add marker
                m = folium.Map(
                    location=[location_data['latitude'], location_data['longitude']],
                    zoom_start=12
                )
                folium.Marker(
                    [location_data['latitude'], location_data['longitude']],
                    popup=location_data['full_name'],
                    tooltip=location_data['name']
                ).add_to(m)
                folium_static(m)
            else:
                st.error("Location not found. Try another name.")

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
        # RimalAI
        Your AI assistant for exploring Saudi Arabian culture, landmarks, and heritage.
        
        **Features:**
        - Voice and text interaction
        - Arabic poem generation
        - Location visualization
        - Cultural information
        
        [GitHub Repository](#) | [Contact Us](#)
    """)
    
    st.markdown("---")
    st.markdown("### Settings")
    voice_enabled = st.checkbox("Enable Voice Responses", True)
    show_translations = st.checkbox("Always Show English Translations", True)

# --- Footer ---
st.markdown("---")
st.markdown("""
    <footer>
        © 2023 RimalAI | Powered by OpenAI and Mapbox
    </footer>
""", unsafe_allow_html=True)

# Add this after the main content display
# Display Media
if st.session_state.get('media_items'):
    with st.expander("🖼️ Related Media", expanded=True):
        cols = st.columns(min(3, len(st.session_state['media_items'])))
        for i, item in enumerate(st.session_state['media_items'][:3]):
            with cols[i]:
                if item['type'] == 'image':
                    st.image(item['url'])
                elif item['type'] == 'video':
                    st.video(item['url'])

# Display Generated Images
if st.session_state.get('generated_images'):
    with st.expander("🎨 Generated Art", expanded=True):
        for img_url in st.session_state['generated_images']:
            st.image(img_url, use_column_width=True)