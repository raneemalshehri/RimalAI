# RimalAI - Saudi Culture Assistant 🐪

RimalAI is an innovative AI-powered application that serves as a comprehensive guide to Saudi Arabian culture, landmarks, and heritage. It combines advanced AI technologies with interactive features to provide an engaging cultural experience.

## 🌟 Features

### 1. Voice Interaction
- Real-time voice input processing
- Automatic language detection (Arabic/English)
- Text-to-speech response generation
- Natural conversation flow

### 2. Text-Based Interaction
- Natural language query processing
- Context-aware responses
- Source attribution
- Rich media integration

### 3. Location Services
- Interactive map visualization
- Detailed location information panels
- Climate and notable sites information
- Satellite view and measurement tools
- Custom landmark markers

### 4. Poetry Features
- Arabic poem generation
- Poem search functionality
- Bilingual support (Arabic with English translations)
- Image generation for poems
- Historical poem database

## 📋 Prerequisites

Before you begin, ensure you have the following:
- Python 3.8 or higher
- pip (Python package installer)
- Git
- OpenAI API key
- Mapbox API key

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RimalAI.git
cd RimalAI
```

2. Create and activate a virtual environment:
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## ⚙️ Environment Setup

1. Create a `.env` file in the root directory:
```bash
touch .env
```

2. Add your API keys to the `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
MAPBOX_API_KEY=your_mapbox_api_key_here
```

3. Verify the environment variables are loaded:
```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('OpenAI Key:', bool(os.getenv('OPENAI_API_KEY'))); print('Mapbox Key:', bool(os.getenv('MAPBOX_API_KEY')))"
```

## 📁 Project Structure

```
RimalAI/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── style.css            # Custom styling
├── Rimal_AI_dataset.json # Cultural and landmark data
├── arabic_poems_dataset.json # Poetry database
└── README.md            # Project documentation
```

### Key Files Description

1. **app.py**
   - Main application logic
   - Streamlit interface implementation
   - AI agent configuration
   - Location and poetry processing

2. **Rimal_AI_dataset.json**
   - Contains information about Saudi landmarks
   - Includes coordinates, descriptions, and media
   - Structured data for location search

3. **arabic_poems_dataset.json**
   - Collection of Arabic poems
   - Includes translations and metadata
   - Used for poem search and generation

4. **style.css**
   - Custom styling for the Streamlit interface
   - Responsive design elements
   - Theme customization

## 🎮 How to Use

### 1. Starting the Application
```bash
streamlit run app.py
```
The application will be available at `http://localhost:8501`

### 2. Voice Interaction
1. Click the "🎤 Start Recording" button
2. Speak your query (5-second recording window)
3. Wait for the transcription
4. Click "🔍 Process Voice Query" to get the response

### 3. Text Interaction
1. Type your question in the text area
2. Click "🔍 Submit Text Query"
3. View the response with any related media

### 4. Poem Generator
1. Enter a theme or topic
2. Click "✍️ Generate Poem"
3. View the generated poem with translation
4. Download the poem and associated image

### 5. Location Finder
1. Enter a landmark or city name
2. Click "📍 Find Location"
3. Explore the interactive map
4. View detailed information in the sidebar

## 🔧 Configuration Options

### Customizing the Interface
Edit `style.css` to modify:
- Color schemes
- Font styles
- Layout spacing
- Component styling

### Modifying the Dataset
1. Update `Rimal_AI_dataset.json` to add:
   - New landmarks
   - Additional information
   - Media content

2. Update `arabic_poems_dataset.json` to add:
   - New poems
   - Translations
   - Poet information

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify API keys in `.env`
   - Check key format and validity
   - Ensure environment variables are loaded

2. **Voice Recognition Issues**
   - Check microphone permissions
   - Verify internet connection
   - Ensure proper audio input device

3. **Map Display Problems**
   - Verify Mapbox API key
   - Check internet connection
   - Clear browser cache

4. **Poem Generation Errors**
   - Verify OpenAI API key
   - Check API rate limits
   - Ensure proper prompt formatting

## 📚 API Documentation

### OpenAI API
- Used for natural language processing
- Poem generation
- Image generation
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)

### Mapbox API
- Location geocoding
- Map visualization
- [Mapbox API Documentation](https://docs.mapbox.com/api/)

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Contribution Areas
- Content enhancement
- UI/UX improvements
- Performance optimization
- Bug fixes
- Feature additions
-chatbot capabilities

