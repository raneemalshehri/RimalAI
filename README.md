# RimalAI - Saudi Culture Assistant 🐪

RimalAI is an innovative AI-powered application that serves as a comprehensive guide to Saudi Arabian culture, landmarks, and heritage.

## 🌟 Features

### 1. Voice Interaction
- Real-time voice input processing
- Automatic language detection (Arabic/English)
- Text-to-speech response generation

### 2. Text-Based Interaction
- Natural language query processing
- Context-aware responses
- Rich media integration

### 3. Location Services
- Interactive map visualization
- Detailed location information
- Satellite view and measurement tools

### 4. Poetry Features
- Arabic poem generation
- Bilingual support (Arabic/English)
- Image generation for poems

## 📋 Prerequisites

- Python 3.8+
- pip
- Git
- OpenAI API key
- Mapbox API key

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/raneemalshehri/RimalAI.git
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
├── app.py                 # Streamlit interface implementation
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── style.css            # Custom styling
├── Rimal_AI_dataset.json # Cultural and landmark data
├── AI_Rimal_Lastmain.ipynb # Main application logic(Experiments & evaluations)
├── arabic_poems_dataset.json # Poetry database
└── README.md            # Project documentation
```

## 🎮 How to Use

### 1. Starting the Application
```bash
streamlit run app.py
```

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

## Project Documentation
1.Final Report: [https://docs.google.com/document/d/1KPVlPrv1wuG5STKbaz0zNkiIdKR-QNR3_s-dfz0xbMo/edit?usp=sharing](https://docs.google.com/document/d/1KPVlPrv1wuG5STKbaz0zNkiIdKR-QNR3_s-dfz0xbMo/edit?usp=drive_link)

2.Project Presentation: https://www.canva.com/design/DAGmkTc4CWc/z2DzYUSV1_ppXePkEfnFFw/view?utm_content=DAGmkTc4CWc&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h3a718a4ad4

3.Demo Video (included in Presentation)

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
- chatbot capabilities


